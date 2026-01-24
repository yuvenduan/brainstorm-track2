import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.fft
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 2. Dataset (With Coordinate Inversion)
# ==========================================
class NeuralDataset(Dataset):
    def __init__(self, data_path, gt_path, window_size=100, grid_dim=(32, 32)):
        # Load Parquet
        self.data_array = pd.read_parquet(data_path).values.astype(np.float32)
        self.gt_df = pd.read_parquet(gt_path)
        
        self.window_size = window_size
        self.H, self.W = grid_dim
        
        # Normalize (Global Z-Score)
        self.data_norm = (self.data_array - self.data_array.mean()) / (self.data_array.std() + 1e-6)
        self.spatial_data = self.data_norm.reshape(-1, self.H, self.W)
        self.spot_names = ['vx_pos', 'vx_neg', 'vy_pos', 'vy_neg', 'click']

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        # Handle start of file padding
        if idx < self.window_size:
            idx = self.window_size

        window = self.spatial_data[idx - self.window_size : idx]
        
        target_map = np.zeros((self.H, self.W), dtype=np.float32)
        
        for spot in self.spot_names:
            raw_col = self.gt_df.iloc[idx][f'{spot}_center_col']
            raw_row = self.gt_df.iloc[idx][f'{spot}_center_row']
            
            if not np.isnan(raw_col) and not np.isnan(raw_row):
                # --- COORDINATE INVERSION ---
                c_col = (self.W - 1) - raw_col
                c_row = (self.H - 1) - raw_row
                
                if 0 <= c_col < self.W and 0 <= c_row < self.H:
                    self._draw_gaussian(target_map, c_col, c_row)
                
        return torch.from_numpy(window), torch.from_numpy(target_map).unsqueeze(0)

    def _draw_gaussian(self, heatmap, x, y, sigma=2.0):
        xx, yy = np.meshgrid(np.arange(self.W), np.arange(self.H))
        dist = (xx - x)**2 + (yy - y)**2
        heatmap += np.exp(-dist / (2 * sigma**2))

# ==========================================
# 3. Visualization Loop
# ==========================================
def create_full_dataset_gif(model, dataset, device, filename="medium_test.gif", stride=50):
    """
    stride=50 means we visualize every 50th sample (100ms at 500Hz).
    For a 30s recording, this results in 300 frames.
    """
    print(f"Generating GIF for full dataset (Stride: {stride})...")
    model.eval()
    
    inputs_viz, preds_viz, targets_viz = [], [], []
    
    # Calculate indices to visit
    indices = range(dataset.window_size, len(dataset), stride)
    total_frames = len(indices)
    print(f"Total Frames to generate: {total_frames}")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            inp, tgt = dataset[idx]
            inp = inp.unsqueeze(0).to(device)
            
            # Inference
            logit = model(inp)
            prob = torch.sigmoid(logit)
            
            # Save for plotting
            # Input: Use last time step
            inputs_viz.append(inp[0, -1, :, :].cpu().numpy()) 
            preds_viz.append(prob[0, 0, :, :].cpu().numpy())
            targets_viz.append(tgt[0, :, :].cpu().numpy())
            
            if i % 50 == 0:
                print(f"Processed {i}/{total_frames} frames...")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ims = []
    
    print("Stitching animation...")
    for i in range(len(inputs_viz)):
        im1 = axes[0].imshow(inputs_viz[i], animated=True, cmap='viridis', vmin=-2, vmax=2)
        if i==0: axes[0].set_title("Input (Live)")
        
        im2 = axes[1].imshow(preds_viz[i], animated=True, cmap='magma', vmin=0, vmax=1)
        if i==0: axes[1].set_title("Prediction")
        
        im3 = axes[2].imshow(targets_viz[i], animated=True, cmap='magma', vmin=0, vmax=1)
        if i==0: axes[2].set_title("Ground Truth")
        
        ims.append([im1, im2, im3])

    # FPS=15 with Stride=50 (100ms) means playback is ~1.5x real-time speed
    ani = animation.ArtistAnimation(fig, ims, interval=66, blit=True)
    ani.save(filename, writer='pillow', fps=15)
    print(f"GIF saved to {filename}")
    plt.close()

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # 1. Load Data
    print("Loading Hard Dataset...")
    ds = NeuralDataset(
        "data/hard/track2_data.parquet", 
        "data/hard/ground_truth.parquet",
        window_size=100
    )
    
    # 2. Load Model
    # Note: We load the whole object. Ensure SpectralHotSpotFinder is defined above!
    print("Loading Model...")
    model = torch.load("hotspot_finder_model.pth", map_location=device)
    model.to(device)
    
    # 3. Generate GIF
    # stride=50 (100ms) is a good balance for seeing the whole file
    create_full_dataset_gif(model, ds, device, filename="hard_results.gif", stride=50)

if __name__ == "__main__":
    main()