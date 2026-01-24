"""
Data Ingestion & Splitting: The script loads raw 1024-channel neural data and reshapes it into a 32x32 spatial grid to preserve electrode geometry, then splits it chronologically into 80% training and 20% testing sets.

On-the-Fly Preprocessing: For each training step, the dataloader extracts a 200ms history window and generates a target heatmap by drawing a Gaussian blob around the inverted ground truth coordinates.

Spectral Feature Extraction: Inside the model's forward pass, a Fast Fourier Transform (FFT) converts the time-domain window into frequency power across four specific bands (Theta, Beta, Low Gamma, High Gamma), which are then log-transformed to stabilize variance.

Spatial Pattern Recognition: The resulting 4-channel spectral image is processed by a 4-layer Convolutional Neural Network (CNN) that learns to identify spatial clusters of activity and projects them down to a single probability heatmap.

Training & Monitoring: The model optimizes a Binary Cross Entropy loss function (weighted 10x for hot spot pixels) using the Adam optimizer, while a separate evaluation loop periodically generates GIF animations to visually verify performance on the test set.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.fft

class HotSpotFinder(nn.Module):
    def __init__(self, time_depth=100, sampling_rate=500):
        super().__init__()
        
        # We are extracting 4 distinct frequency bands
        num_bands = 4 
        
        # --- Spectral Feature Extraction Settings ---
        self.n_fft = time_depth
        self.freq_res = sampling_rate / self.n_fft  # 500 / 100 = 5 Hz per bin
        
        # Pre-calculate indices for the bands (approximate)
        # 1. Theta/Alpha (4-12 Hz) -> Bins 1 to 2
        # 2. Beta (12-30 Hz) -> Bins 3 to 6
        # 3. Low Gamma (30-70 Hz) -> Bins 7 to 14
        # 4. High Gamma (70-150 Hz) -> Bins 15 to 30
        self.bands = [
            (1, 3),   # 5-10 Hz (Python slice 1:3 excludes upper bound)
            (3, 7),   # 15-30 Hz
            (7, 15),  # 35-70 Hz
            (15, 31)  # 75-150 Hz
        ]

        # --- CNN Architecture ---
        # Note: in_channels is now 4 (the bands), not time_depth!
        self.encoder = nn.Sequential(
            # L1
            nn.Conv2d(num_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # L2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # L3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # L4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(128, 1, kernel_size=1)

    def compute_features(self, x):
        """
        x shape: (Batch, Time, H, W)
        Returns: (Batch, Bands, H, W)
        """
        # 1. FFT along the time dimension (dim=1)
        # rfft returns complex tensor for positive frequencies only
        fft_out = torch.fft.rfft(x, n=self.n_fft, dim=1)
        
        # 2. Compute Power Spectral Density (PSD)
        # shape: (Batch, Freq_Bins, H, W)
        power_spectrum = fft_out.abs().pow(2)
        
        # 3. Aggregate into bands
        band_powers = []
        for start, end in self.bands:
            # Average power in this frequency band
            # shape: (Batch, H, W)
            band_p = power_spectrum[:, start:end, :, :].mean(dim=1)
            band_powers.append(band_p)
            
        # Stack to create channel dimension: (Batch, 4, H, W)
        features = torch.stack(band_powers, dim=1)
        
        # 4. Log-Transform (Crucial for Neural Power)
        # Neural power follows a 1/f distribution; log stabilizes variance
        features = torch.log1p(features)
        
        return features

    def forward(self, x):
        # x: (Batch, 100, 32, 32)
        
        # Extract Frequency Bands on the fly
        x_spectral = self.compute_features(x) # -> (Batch, 4, 32, 32)
        
        # Pass through CNN
        x_enc = self.encoder(x_spectral)
        out = self.final(x_enc)
        
        return out

class NeuralDataset(Dataset):
    def __init__(self, data_array, gt_df, indices, window_size=100, grid_dim=(32, 32)):
        self.window_size = window_size
        self.H, self.W = grid_dim
        self.indices = indices
        
        self.data_array = data_array
        self.gt_df = gt_df
        
        # Normalize
        # self.data_norm = (self.data_array - self.data_array.mean()) / (self.data_array.std() + 1e-6)
        self.data_norm = self.data_array # no normalization for now
        self.spatial_data = self.data_norm.reshape(-1, self.H, self.W)
        self.spot_names = ['vx_pos', 'vx_neg', 'vy_pos', 'vy_neg', 'click']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_t = self.indices[idx]
        
        if actual_t < self.window_size:
            actual_t = self.window_size

        window = self.spatial_data[actual_t - self.window_size : actual_t]
        
        target_map = np.zeros((self.H, self.W), dtype=np.float32)
        
        for spot in self.spot_names:
            # Raw Ground Truth
            raw_col = self.gt_df.iloc[actual_t][f'{spot}_center_col']
            raw_row = self.gt_df.iloc[actual_t][f'{spot}_center_row']
            
            if not np.isnan(raw_col) and not np.isnan(raw_row):
                # --- APPLY INVERSION HERE ---
                # We subtract from W and H to flip the axes.
                # We use (W-1) and (H-1) to keep 0-indexed bounds safe, 
                # though pure (W - col) works fine for the Gaussian function too.
                
                c_col = (self.W - 1) - raw_col
                c_row = (self.H - 1) - raw_row
                
                # Check bounds just in case the flip pushes it slightly off
                if 0 <= c_col < self.W and 0 <= c_row < self.H:
                    self._draw_gaussian(target_map, c_col, c_row)
                
        return torch.from_numpy(window), torch.from_numpy(target_map).unsqueeze(0)

    def _draw_gaussian(self, heatmap, x, y, sigma=2.0):
        xx, yy = np.meshgrid(np.arange(self.W), np.arange(self.H))
        dist = (xx - x)**2 + (yy - y)**2
        heatmap += np.exp(-dist / (2 * sigma**2))

# --- 3. Visualization ---
def create_strided_gif(model, dataset, device, filename="training_result.gif", frames=100, stride=1):
    print(f"Generating GIF ({frames} frames)...")
    model.eval()
    inputs_viz, preds_viz, targets_viz = [], [], []
    
    with torch.no_grad():
        for i in range(frames):
            idx = i * stride
            if idx >= len(dataset): break
            
            inp, tgt = dataset[idx]
            inp = inp.unsqueeze(0).to(device)
            
            prob = torch.sigmoid(model(inp))
            
            inputs_viz.append(inp[0, -1, :, :].cpu().numpy()) 
            preds_viz.append(prob[0, 0, :, :].cpu().numpy())
            targets_viz.append(tgt[0, :, :].cpu().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ims = []
    for i in range(len(inputs_viz)):
        im1 = axes[0].imshow(inputs_viz[i], animated=True, cmap='viridis', vmin=-2, vmax=2)
        if i==0: axes[0].set_title("Input (Live)")
        im2 = axes[1].imshow(preds_viz[i], animated=True, cmap='magma', vmin=0, vmax=1)
        if i==0: axes[1].set_title("Prediction")
        im3 = axes[2].imshow(targets_viz[i], animated=True, cmap='magma', vmin=0, vmax=1)
        if i==0: axes[2].set_title("Ground Truth")
        ims.append([im1, im2, im3])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    ani.save(filename, writer='pillow', fps=10)
    print(f"GIF saved to {filename}")
    plt.close()

# --- 4. Main Training Loop ---
def main():
    # --- Config ---
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 2
    WINDOW = 100         # 200ms @ 500Hz
    TEST_STRIDE_MS = 30  # 30ms stride for testing
    FREQ = 500
    
    # Calculate Stride in Samples
    test_step = int((TEST_STRIDE_MS / 1000) * FREQ) # 25 samples
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # --- Data Loading ---
    print("Loading Data...")
    df_data_full = pd.read_parquet("data/hard/track2_data.parquet")
    df_gt_full = pd.read_parquet("data/hard/ground_truth.parquet")
    
    raw_data = df_data_full.values.astype(np.float32)
    total_len = len(raw_data)
    
    # --- Split Indices ---
    split_idx = int(total_len * 0.8)
    
    # Train: Stride 1 (Use every sample)
    train_indices = range(WINDOW, split_idx)
    
    # Test: Stride 25 (Every 50ms)
    # We use range(start, stop, step) to create the strided indices efficiently
    test_indices = range(split_idx, total_len, test_step)
    
    print(f"Dataset Sizes:")
    print(f"  Training Samples: {len(train_indices)}")
    print(f"  Test Samples:     {len(test_indices)} (Strided every {TEST_STRIDE_MS}ms)")

    train_ds = NeuralDataset(raw_data, df_gt_full, train_indices, window_size=WINDOW)
    test_ds = NeuralDataset(raw_data, df_gt_full, test_indices, window_size=WINDOW)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- Setup ---
    model = HotSpotFinder(time_depth=WINDOW).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # --- Loop ---
    print("\nStarting Training...")
    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # --- Periodic Evaluation (Every 200 Batches) ---
            if batch_idx > 0 and batch_idx % 200 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for v_in, v_tgt in test_loader:
                        v_in, v_tgt = v_in.to(device), v_tgt.to(device)
                        v_out = model(v_in)
                        v_loss = criterion(v_out, v_tgt)
                        val_loss += v_loss.item()

                    # Generate visualization from the Test Set
                    # We use stride=1 here for the GIF so the animation is smooth, but limit frames
                    create_strided_gif(model, test_ds, device, filename="prediction.gif", frames=100, stride=1)
                
                avg_val = val_loss / len(test_loader)
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Train Loss: {loss.item():.4f} | Test Loss: {avg_val:.4f}")
                
                # Switch back to train mode!
                model.train()

    create_strided_gif(model, test_ds, device, filename="final_prediction.gif", frames=100, stride=1)
    print("Training Complete.")

    # --- Save Model ---
    torch.save(model.cpu(), "hotspot_finder_model.pth")

if __name__ == "__main__":
    main()