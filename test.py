import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.fft
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. Model Definition
# ==========================================
class HotSpotFinder(nn.Module):
    def __init__(self, time_depth=100, sampling_rate=500):
        super().__init__()
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
        return self.final(x_enc)

# ==========================================
# 2. Optimized Dataset (Cached Grid)
# ==========================================
class FastNeuralDataset(Dataset):
    def __init__(self, data_path, gt_path, window_size=100, grid_dim=(32, 32)):
        # Load data once
        self.data_array = pd.read_parquet(data_path).values.astype(np.float32)
        self.gt_df = pd.read_parquet(gt_path)
        self.window_size = window_size
        self.H, self.W = grid_dim
        
        # Pre-reshape data
        self.spatial_data = self.data_array.reshape(-1, self.H, self.W)
        self.spot_names = ['vx_pos', 'vx_neg', 'vy_pos', 'vy_neg', 'click']

        # OPTIMIZATION: Pre-calculate the coordinate grid once!
        # This prevents creating 32x32 arrays 10,000 times per second
        self.xx, self.yy = np.meshgrid(np.arange(self.W), np.arange(self.H))

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        if idx < self.window_size: idx = self.window_size

        # Input
        window = self.spatial_data[idx - self.window_size : idx]
        
        # Target
        target_map = np.zeros((self.H, self.W), dtype=np.float32)
        for spot in self.spot_names:
            raw_col = self.gt_df.iloc[idx][f'{spot}_center_col']
            raw_row = self.gt_df.iloc[idx][f'{spot}_center_row']
            
            if not np.isnan(raw_col) and not np.isnan(raw_row):
                c_col = (self.W - 1) - raw_col
                c_row = (self.H - 1) - raw_row
                
                if 0 <= c_col < self.W and 0 <= c_row < self.H:
                    # Use cached grid
                    dist = (self.xx - c_col)**2 + (self.yy - c_row)**2
                    target_map += np.exp(-dist / (2 * 2.0**2))
                
        return torch.from_numpy(window), torch.from_numpy(target_map).unsqueeze(0)

# ==========================================
# 3. Helpers (Vectorized CoM)
# ==========================================
def apply_top_k_threshold_batch(batch_heatmap, k=100):
    """
    Optimized to handle a batch of heatmaps (B, H, W)
    """
    B, H, W = batch_heatmap.shape
    flat = batch_heatmap.reshape(B, -1)
    
    # Find k-th largest value per image
    # We partition around the (N - k)th element
    kth_vals = np.partition(flat, -k, axis=1)[:, -k]
    kth_vals = kth_vals[:, None, None] # Broadcast back to (B, 1, 1)
    
    # Threshold
    mask = batch_heatmap >= kth_vals
    return batch_heatmap * mask

def get_center_of_mass_batch(heatmaps):
    """
    Computes CoM for a batch of maps (B, H, W).
    """
    B, H, W = heatmaps.shape
    total_mass = np.sum(heatmaps, axis=(1, 2))
    
    # Avoid division by zero
    valid = total_mass > 1e-4
    
    # Grids
    y_grid, x_grid = np.indices((H, W))
    
    # Weighted sums
    # heatmaps: (B, H, W) * grid: (H, W) -> (B, H, W) -> sum -> (B,)
    y_center = np.sum(y_grid[None, :, :] * heatmaps, axis=(1, 2))
    x_center = np.sum(x_grid[None, :, :] * heatmaps, axis=(1, 2))
    
    y_center[valid] /= total_mass[valid]
    x_center[valid] /= total_mass[valid]
    
    # Return list of tuples or Nones
    results = []
    for i in range(B):
        if valid[i]:
            results.append((x_center[i], y_center[i]))
        else:
            results.append(None)
    return results

# ==========================================
# 4. Batched Visualization Loop
# ==========================================
def create_dataset_gif(model, dataset, device, filename, stride=50, top_k=100):
    print(f"Generating GIF: {filename} ...")
    model.eval()
    
    # 1. Create a Subset of indices to visualize (Striding)
    indices = list(range(dataset.window_size, len(dataset), stride))
    subset = Subset(dataset, indices)
    
    # 2. Use DataLoader for Batched Inference (Major Speedup!)
    # num_workers=2 allows CPU to prepare Ground Truths while GPU predicts
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)
    
    inputs_viz, preds_viz, targets_viz = [], [], []
    coms_viz = [] 

    print(f"  Running Batched Inference on {len(indices)} frames...")
    
    with torch.no_grad():
        for batch_idx, (bx, by) in enumerate(loader):
            bx = bx.to(device)
            
            # GPU Forward Pass
            logits = model(bx)
            probs = torch.sigmoid(logits)
            
            # Move to CPU as numpy batch
            # bx shape: (B, Time, H, W) -> We want last frame: (B, H, W)
            b_in = bx[:, -1, :, :].cpu().numpy()
            b_pred = probs[:, 0, :, :].cpu().numpy()
            b_tgt = by[:, 0, :, :].cpu().numpy()
            
            # --- Vectorized Post-Processing ---
            # 1. Threshold Batch
            b_pred_clean = apply_top_k_threshold_batch(b_pred, k=top_k)
            
            # 2. CoM Batch
            b_coms = get_center_of_mass_batch(b_pred_clean)
            
            # Store results
            inputs_viz.extend(list(b_in))
            preds_viz.extend(list(b_pred_clean))
            targets_viz.extend(list(b_tgt))
            coms_viz.extend(b_coms)

    # 3. Plotting (Sequential)
    # Matplotlib animation is inherently slow, but inference is now instant.
    print("  Stitching animation (this part is CPU bound)...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ims = []
    
    for i in range(len(inputs_viz)):
        im1 = axes[0].imshow(inputs_viz[i], animated=True, cmap='viridis', vmin=-2, vmax=2)
        if i==0: axes[0].set_title("Input (Live)")
        
        im2 = axes[1].imshow(preds_viz[i], animated=True, cmap='magma', vmin=0, vmax=1)
        artists = [im1, im2]
        
        if coms_viz[i] is not None:
            cx, cy = coms_viz[i]
            marker, = axes[1].plot(cx, cy, 'rx', markersize=10, markeredgewidth=2, animated=True)
            artists.append(marker)
        if i==0: axes[1].set_title(f"Pred (Top {top_k}) + CoM")
        
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
    
    os.makedirs("testing", exist_ok=True)

    model_path = "hotspot_finder_model.pth"
    try:
        model = torch.load(model_path, map_location=device)
        model.to(device)
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        return

    difficulties = ['super_easy', 'easy', 'medium', 'hard']
    
    for diff in difficulties:
        print(f"--- Processing {diff.upper()} Dataset ---")
        data_path = f"data/{diff}/track2_data.parquet"
        gt_path = f"data/{diff}/ground_truth.parquet"
        
        if not os.path.exists(data_path):
            print(f"Skipping {diff} (File not found)")
            continue
            
        ds = FastNeuralDataset(data_path, gt_path, window_size=100)
        
        create_dataset_gif(
            model, ds, device, 
            filename=f"testing/{diff}_result.gif", 
            stride=500, 
            top_k=100
        )

if __name__ == "__main__":
    main()