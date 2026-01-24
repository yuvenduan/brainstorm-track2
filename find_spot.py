"""
1. Data Ingestion & Splitting: The script loads raw 1024-channel neural data and reshapes it into a 32x32 spatial grid to preserve electrode geometry, then splits it chronologically into 80% training and 20% testing sets.
2. On-the-Fly Preprocessing: For each training step, the dataloader extracts a 200ms history window and generates a target heatmap by drawing a Gaussian blob around the inverted ground truth coordinates.
3. Spectral Feature Extraction: Inside the model's forward pass, a Fast Fourier Transform (FFT) converts the time-domain window into frequency power across four specific bands (Theta, Beta, Low Gamma, High Gamma), which are then log-transformed to stabilize variance.
4. Spatial Pattern Recognition: The resulting 4-channel spectral image is processed by a 4-layer Convolutional Neural Network (CNN) that learns to identify spatial clusters of activity and projects them down to a single probability heatmap.
5. Training & Monitoring: The model optimizes a Binary Cross Entropy loss function (weighted 10x for hot spot pixels) using the Adam optimizer, while a separate evaluation loop periodically generates GIF animations to visually verify performance on the test set.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.fft

# ==========================================
# 1. Model Definition
# ==========================================
class HotSpotFinder(nn.Module):
    def __init__(self, time_depth=100, sampling_rate=500):
        super().__init__()
        num_bands = 4 
        self.n_fft = time_depth
        self.bands = [(1, 3), (3, 7), (7, 15), (15, 31)]

        self.encoder = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=3, padding=1),
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
# 2. Dataset Definition
# ==========================================
class NeuralDataset(Dataset):
    def __init__(self, data_array, gt_df, indices, window_size=100, grid_dim=(32, 32)):
        self.window_size = window_size
        self.H, self.W = grid_dim
        self.indices = indices
        self.data_array = data_array
        self.gt_df = gt_df
        self.data_norm = self.data_array 
        self.spatial_data = self.data_norm.reshape(-1, self.H, self.W)
        self.spot_names = ['vx_pos', 'vx_neg', 'vy_pos', 'vy_neg', 'click']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_t = self.indices[idx]
        if actual_t < self.window_size: actual_t = self.window_size

        window = self.spatial_data[actual_t - self.window_size : actual_t]
        target_map = np.zeros((self.H, self.W), dtype=np.float32)
        
        for spot in self.spot_names:
            raw_col = self.gt_df.iloc[actual_t][f'{spot}_center_col']
            raw_row = self.gt_df.iloc[actual_t][f'{spot}_center_row']
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
# 3. Visualization Helper
# ==========================================
def create_strided_gif(model, dataset, device, filename, frames=100, stride=1):
    """
    Generates a GIF for a SPECIFIC dataset (e.g., just the 'hard' one).
    """
    model.eval()
    inputs_viz, preds_viz, targets_viz = [], [], []
    
    # Pick a random start point, but ensure we don't go out of bounds
    max_start = max(0, len(dataset) - (frames * stride))
    start_idx = np.random.randint(0, max_start + 1)
    
    with torch.no_grad():
        for i in range(frames):
            idx = start_idx + (i * stride)
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
        if i==0: axes[0].set_title("Input")
        im2 = axes[1].imshow(preds_viz[i], animated=True, cmap='magma', vmin=0, vmax=1)
        if i==0: axes[1].set_title("Pred")
        im3 = axes[2].imshow(targets_viz[i], animated=True, cmap='magma', vmin=0, vmax=1)
        if i==0: axes[2].set_title("Truth")
        ims.append([im1, im2, im3])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    ani.save(filename, writer='pillow', fps=10)
    plt.close()
    print(f"  -> Saved {filename}")

# ==========================================
# 4. Main Training Loop
# ==========================================
def main():
    # --- Config ---
    BATCH_SIZE = 64
    LR = 7e-4
    EPOCHS = 2
    WINDOW = 100         
    TEST_STRIDE_MS = 30  
    FREQ = 500
    test_step = int((TEST_STRIDE_MS / 1000) * FREQ) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # --- Load Separate Datasets ---
    difficulties = ["super_easy", "easy", "medium", "hard"]
    
    # We store test sets in a DICTIONARY so we can access them by name later
    test_datasets_dict = {} 
    train_datasets_list = []
    
    print("Loading datasets...")
    for diff in difficulties:
        print(f"  Processing: {diff}...")
        d_path = f"data/{diff}/track2_data.parquet"
        g_path = f"data/{diff}/ground_truth.parquet"
        
        df_data = pd.read_parquet(d_path)
        df_gt = pd.read_parquet(g_path)
        raw_values = df_data.values.astype(np.float32)
        
        total_len = len(raw_values)
        split_idx = int(total_len * 0.8)
        
        ds_train = NeuralDataset(raw_values, df_gt, range(WINDOW, split_idx), window_size=WINDOW)
        ds_test = NeuralDataset(raw_values, df_gt, range(split_idx, total_len, test_step), window_size=WINDOW)
        
        train_datasets_list.append(ds_train)
        test_datasets_dict[diff] = ds_test  # Save separately for viz!

    # Combine for the main training loader
    full_train_ds = ConcatDataset(train_datasets_list)
    # Combine test sets just for calculating the global loss metric
    full_test_ds = ConcatDataset(list(test_datasets_dict.values()))
    
    train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(full_test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = HotSpotFinder(time_depth=WINDOW).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

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
            
            # --- EVALUATION BLOCK ---
            if batch_idx > 0 and batch_idx % 2000 == 0:
                model.eval()
                
                # 1. Calc Global Loss (Fast)
                val_loss = 0.0
                with torch.no_grad():
                    for v_in, v_tgt in test_loader:
                        v_in, v_tgt = v_in.to(device), v_tgt.to(device)
                        v_out = model(v_in)
                        val_loss += criterion(v_out, v_tgt).item()
                avg_val = val_loss / len(test_loader)
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Train: {loss.item():.4f} | Val: {avg_val:.4f}")

                # 2. Generate SPECIFIC GIFs for each difficulty
                print("  Generating visualizations for each dataset...")
                for diff_name, ds in test_datasets_dict.items():
                    # Create a gif for 'medium', 'hard', etc explicitly
                    create_strided_gif(
                        model, ds, device, 
                        filename=f"training/viz_{diff_name}_step{global_step}.gif", 
                        frames=60, stride=5
                    )
                
                model.train()

        torch.save(model.cpu(), f"training/hotspot_finder_epoch{epoch+1}.pth")

    print("Training Complete. Saving model...")
    torch.save(model.cpu(), "hotspot_finder_model.pth")

if __name__ == "__main__":
    main()