# Neural Data Video Export Pipeline

## Overview

Python script to process neural data from BrainStorm Track 2 challenge:
1. Bad channel detection (per-video, not generalizing across datasets)
2. Median interpolation imputation from good neighbors
3. Artifact detection and removal (60Hz line noise, white noise)
4. Frequency band decomposition (Theta/Alpha, Beta, Low Gamma, High Gamma)
5. Multi-band visualization with amplitude colorbar
6. Export to mp4/avi video file

## Key Design Decisions

1. **Bad channel detection is per-video.** Use the first 1000 frames of each video to identify bad channels as channels that do not change over time (variance-based). There can be anywhere between 0 and 16 bad channels (never more than 16). If a channel is "stuck" at the same value but >75% of the first 1000 frames are also having that value, treat as no bad channels, skip interpolation, and move on to standardization. For calculating standard deviation, only consider pixels with values different from the mean in the first 1000 frames.

2. **Artifact detection and removal.** After bad channel imputation, detect and remove 60Hz line noise (may or may not exist) and isolate white noise component across space and time if present. Report these artifacts in verbose mode before subtracting them.

3. **Frequency band decomposition.** Break the signal down into 4 frequency bands based on physiological relevance:
   - Theta/Alpha: 4-12 Hz (movement planning, attention)
   - Beta: 12-30 Hz (movement preparation/suppression)
   - Low Gamma: 30-70 Hz (local cortical processing)
   - High Gamma: 70-150 Hz (motor execution, local firing)

4. **Multi-panel visualization.** Display 4 panels (one per frequency band) showing changing image intensities with amplitude information displayed on a colorbar to the right.

5. **Maximum intensity projection smoothing.** After frequency decomposition and standardizing each frame, take the maximum intensity projection every 50 frames using half-overlapping windows to smooth out the signal.

6. **Frame range display.** Print the range of frames on the top right corner of each video output.

7. **Deliverables:**
   - Save the Python script used to perform these actions
   - Generate and save output MP4 videos for super-easy and hard datasets

## Script Location

`scripts/export_video.py`

## Usage

```bash
# Generate videos for deliverables
python scripts/export_video.py -d super_easy -o /store1/candy/bci/super_easy.mp4 --verbose
python scripts/export_video.py -d hard -o /store1/candy/bci/hard.mp4 --verbose

# Adjust detection parameters if needed
python scripts/export_video.py -d hard -o /store1/candy/bci/hard.mp4 --detection-frames 2000 --variance-threshold 1e-7

# Skip imputation to see raw bad channels
python scripts/export_video.py -d hard -o /store1/candy/bci/hard_raw.mp4 --skip-imputation

# Adjust MIP window/stride
python scripts/export_video.py -d super_easy -o /store1/candy/bci/super_easy.mp4 --mip-window 100 --mip-stride 50
```

## Processing Pipeline

```
Load parquet data (150000 samples x 1024 channels)
    ↓
Detect bad channels using first 1000 frames (variance < threshold, 0-16 channels max)
    ↓
Impute bad channels if any found (median of 8-connected good neighbors)
    ↓
Detect and remove artifacts:
  - 60Hz line noise detection (may or may not exist)
  - White noise isolation across space and time
  - Report in verbose mode and subtract
    ↓
Frequency band decomposition (bandpass filtering):
  - Theta/Alpha: 4-12 Hz
  - Beta: 12-30 Hz
  - Low Gamma: 30-70 Hz
  - High Gamma: 70-150 Hz
    ↓
Compute normalization stats (mean/std from first 100 frames of THIS video, per band)
    ↓
Generate frames for each band:
  - Average ~17 samples per frame (30 FPS from 500 Hz)
  - Standardize: z = (x - mean) / std
  - Clip to ±3 std
  - Reshape to 32x32 grid
    ↓
Apply maximum intensity projection (MIP) per band:
  - 50-frame windows with half-overlap (25 frame stride)
  - Take max across each window
    ↓
Render final frames:
  - 4 panels (2x2 grid), one per frequency band
  - Apply Magma colormap
  - Amplitude colorbar on right side
  - Add frame range text (top right corner)
    ↓
Export to mp4
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --difficulty` | hard | Dataset: super_easy, easy, medium, hard |
| `-o, --output` | required | Output path (.mp4 or .avi) |
| `--fps` | 30.0 | Video frame rate |
| `--frame-size` | 512 | Output frame size per panel (pixels) |
| `--variance-threshold` | 1e-6 | Bad channel detection threshold |
| `--detection-frames` | 1000 | Frames for bad channel detection |
| `--connectivity` | 8 | Neighbor connectivity (4 or 8) |
| `--n-std` | 3.0 | Std deviations for colormap range |
| `--stats-frames` | 100 | Frames for normalization stats |
| `--mip-window` | 50 | MIP window size (frames) |
| `--mip-stride` | 25 | MIP stride (frames, half-overlap default) |
| `--skip-imputation` | false | Skip bad channel imputation |
| `--skip-artifact-removal` | false | Skip 60Hz and white noise removal |
| `-v, --verbose` | false | Print detailed statistics including artifacts |

## Dependencies

Added to `pyproject.toml`:
- `opencv-python>=4.8.0`
- `matplotlib>=3.7.0`
- `scipy>=1.11.0` (for bandpass filtering)

## Output Location

Videos saved to `/store1/candy/bci/`
