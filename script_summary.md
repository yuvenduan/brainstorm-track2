# BCI Hot Spot Detection Pipeline Summary

## Script 1: Multi-Dataset Training (`find_spot.py`)

* **Data Pipeline:** Loads raw neural recordings (1024 channels) and ground truth coordinates from `super_easy`, `easy`, `medium`, and `hard` folders. It reshapes channels into a 32x32 grid, performs an 80/20 chronological split, and combines them using `ConcatDataset`.
* **On-the-Fly Processing:** The `NeuralDataset` class extracts a 200ms history window (100 samples) for each step and generates a target heatmap by drawing a Gaussian blob around the *inverted* ground truth coordinates (`Width - x`, `Height - y`).
* **Model Architecture:** The `HotSpotFinder` module uses an internal FFT layer to extract power in 4 specific bands (Theta, Beta, Low Gamma, High Gamma). It applies a Log1p transform before passing the 4-channel spectral image into a 4-layer CNN.
* **Training Loop:** Optimizes `BCEWithLogitsLoss` (weighted 10x for hot spot pixels) using the Adam optimizer. Every 2000 steps, it runs a validation loop that calculates loss and generates separate GIF visualizations for each dataset difficulty.

## Script 2: Inference & Visualization (`test.py`)

* **Initialization:** Loads the trained `HotSpotFinder` model and iterates through all three difficulty levels ('super_easy', 'easy', 'medium', 'hard'), initializing a `NeuralDataset` for each.
* **Inference Loop:** Runs the model on the full recordings with a 50-sample stride (100ms) to generate probability heatmaps.
* **Post-Processing:** Applies a **Top-K Threshold** (keeping only the top 100 strongest pixels) to remove background noise and calculates the **Weighted Center of Mass (CoM)** of the remaining blob.
* **Visualization:** Generates comparative GIFs saved to a `testing/` folder. The visualization displays the live Neural Input, the Cleaned Prediction (marked with a Red 'X' for the CoM), and the Ground Truth side-by-side.