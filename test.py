import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.fft
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. Model Definition (Required for torch.load)
# ==========================================
class HotSpotFinder(nn.Module):
    def __init__(self, time_depth=100, sampling_rate=500):
        super().__init__()
        # Bands: Theta/Alpha, Beta, Low Gamma, High Gamma
        self.n_fft = time_depth
        self.bands = [(1, 3), (3, 7), (7, 15), (15, 31)]

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(128, 1, kernel_size=1)

    def compute_features(self, x):
        fft_out = torch.fft.rfft(x, n=self.n_fft, dim=1)
        power_spectrum = fft_out.abs().pow(2)
        band_powers = []
        for start, end in self.bands:
            band_p = power_spectrum[:, start:end, :, :].mean(dim=1)
            band_powers.append(band_p)
        features = torch.stack(band_powers, dim=1)
        features = torch.log1p(features)
        return features

    def forward(self, x):
        x_spectral = self.compute_features(x)
        x_enc = self.encoder(x_spectral)
        out = self.final(x_enc)
        return out

# ==========================================
# 2. Helpers (CoM & Thresholding)
# ==========================================
def apply_top_k_threshold(heatmap, k=100):
    """
    Keeps only the top k pixels in the heatmap. Sets others to 0.
    This removes background noise and makes the blob cleaner.
    """
    if k >= heatmap.size:
        return heatmap
        
    # Flatten to find the k-th largest value
    flat = heatmap.flatten()
    # partition returns the k-th element in the correct sorted position
    # We negate 'flat' because partition sorts ascending
    kth_largest_val = -np.partition(-flat, k-1)[k-1]
    
    # Apply threshold
    heatmap_clean = heatmap.copy()
    heatmap_clean[heatmap_clean < kth_largest_val] = 0
    return heatmap_clean

def get_center_of_mass(heatmap):
    """
    Computes the weighted center of mass (x, y).
    """
    total_mass = np.sum(heatmap)
    if total_mass < 1e-4: return None

    H, W = heatmap.shape
    y_grid, x_grid = np.indices((H, W))

    y_center = np.sum(y_grid * heatmap) / total_mass
    x_center = np.sum(x_grid * heatmap) / total_mass
    
    return x_center, y_center

# ==========================================
# 3. Dataset
# ==========================================
class NeuralDataset(Dataset):
    def __init__(self, data_path, gt_path, window_size=100, grid_dim=(32, 32)):
        self.data_array = pd.read_parquet(data_path).values.astype(np.float32)
        self.gt_df = pd.read_parquet(gt_path)
        self.window_size = window_size
        self.H, self.W = grid_dim
        
        self.data_norm = self.data_array # Raw data is fine given the Log1p in model
        self.spatial_data = self.data_norm.reshape(-1, self.H, self.W)
        self.spot_names = ['vx_pos', 'vx_neg', 'vy_pos', 'vy_neg', 'click']

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        if idx < self.window_size: idx = self.window_size

        window = self.spatial_data[idx - self.window_size : idx]
        target_map = np.zeros((self.H, self.W), dtype=np.float32)
        
        for spot in self.spot_names:
            raw_col = self.gt_df.iloc[idx][f'{spot}_center_col']
            raw_row = self.gt_df.iloc[idx][f'{spot}_center_row']
            if not np.isnan(raw_col) and not np.isnan(raw_row):
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
# 4. Visualization Loop
# ==========================================
def create_dataset_gif(model, dataset, device, filename, stride=50, top_k=100):
    print(f"Generating GIF: {filename} ...")
    model.eval()
    
    inputs_viz, preds_viz, targets_viz = [], [], []
    coms_viz = [] 
    
    indices = range(dataset.window_size, len(dataset), stride)
    total_frames = len(indices)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            inp, tgt = dataset[idx]
            inp = inp.unsqueeze(0).to(device)
            
            logit = model(inp)
            prob = torch.sigmoid(logit)
            
            # --- CLEANUP STEP ---
            pred_map = prob[0, 0, :, :].cpu().numpy()
            
            # 1. Apply Threshold (Keep only top K pixels)
            pred_map_clean = apply_top_k_threshold(pred_map, k=top_k)
            
            # 2. Compute CoM on CLEAN map
            com = get_center_of_mass(pred_map_clean)
            
            # Store
            inputs_viz.append(inp[0, -1, :, :].cpu().numpy()) 
            preds_viz.append(pred_map_clean) # Show the cleaned map!
            targets_viz.append(tgt[0, :, :].cpu().numpy())
            coms_viz.append(com)
            
            if i % 100 == 0:
                print(f"  Processed {i}/{total_frames} frames...")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ims = []
    
    print("  Stitching animation...")
    for i in range(len(inputs_viz)):
        # 1. Input
        im1 = axes[0].imshow(inputs_viz[i], animated=True, cmap='viridis', vmin=-2, vmax=2)
        if i==0: axes[0].set_title("Input (Live)")
        
        # 2. Prediction (Cleaned) + Red X
        im2 = axes[1].imshow(preds_viz[i], animated=True, cmap='magma', vmin=0, vmax=1)
        artists = [im1, im2]
        
        if coms_viz[i] is not None:
            cx, cy = coms_viz[i]
            # Red X marker
            marker, = axes[1].plot(cx, cy, 'rx', markersize=10, markeredgewidth=2, animated=True)
            artists.append(marker)
        if i==0: axes[1].set_title(f"Prediction (Top {top_k}) + CoM")
        
        # 3. Ground Truth
        im3 = axes[2].imshow(targets_viz[i], animated=True, cmap='magma', vmin=0, vmax=1)
        artists.append(im3)
        if i==0: axes[2].set_title("Ground Truth")
        
        ims.append(artists)

    ani = animation.ArtistAnimation(fig, ims, interval=66, blit=True)
    ani.save(filename, writer='pillow', fps=15)
    print(f"  -> Saved to {filename}\n")
    plt.close()

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # 1. Setup Output Folder
    output_folder = "testing"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Results will be saved to: {output_folder}/")

    # 2. Load Model
    model_path = "hotspot_finder_model.pth" # or your specific model name
    print(f"Loading Model from {model_path}...")
    try:
        model = torch.load(model_path, map_location=device)
        model.to(device)
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        return

    # 3. Loop over datasets
    difficulties = ['super_easy', 'easy', 'medium', 'hard']
    
    for diff in difficulties:
        print(f"--- Processing {diff.upper()} Dataset ---")
        
        data_path = f"data/{diff}/track2_data.parquet"
        gt_path = f"data/{diff}/ground_truth.parquet"
        
        # Check file existence
        if not os.path.exists(data_path):
            print(f"Skipping {diff}: File not found.")
            continue
            
        # Load Dataset
        ds = NeuralDataset(data_path, gt_path, window_size=100)
        
        # Generate GIF
        save_path = f"{output_folder}/{diff}_result.gif"
        
        # Stride=50 (100ms) for speed, Top_k=100 pixels for cleanliness
        create_dataset_gif(model, ds, device, filename=save_path, stride=50, top_k=100)

if __name__ == "__main__":
    main()