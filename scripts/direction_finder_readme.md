# Neural Compass Visualization

A neural-only compass visualization tool that processes 1024-channel neural data into a 32×32 heatmap, tracks activity hotspots, and generates an animated guidance compass.

## Overview

- **Input**: 1024 channels of neural data (parquet format)
- **Processing**: Converts channels → 32×32 spatial heatmap per frame
- **Tracking**: Detects and tracks the strongest activity blob (hotspot)
- **Temporal Smoothing**: Aggregates last 20 frames for stability
- **Output**: Animated GIF/MP4 showing arrow from array center toward hotspot

## Requirements

```bash
pip install numpy pandas matplotlib scipy
```

Optional (for MP4 export):
- FFmpeg (system installation)

## Usage

```bash
python compass_neural.py
```

## Configuration

Edit settings in the script:

- **Data Path**: `NEURAL_PARQUET` - Path to input parquet file
- **Output**: `OUT_FILE` - Output filename (default: `compass_neural_only.gif`)
- **Sampling**: `FS = 500` Hz
- **Window**: `WIN_MS = 100` ms, `STEP_MS = 100` ms
- **Feature**: `FEATURE = "rms"` (Root Mean Square)
- **Smoothing**: `SPATIAL_SMOOTH_SIGMA = 0.9`
- **Detection**: `TOP_PCT = 98.5%` threshold, `MIN_AREA = 3` pixels
- **Temporal**: `AGG_FRAMES = 20` frames for aggregation
- **Lock Distance**: `LOCK_DIST = 2.5` pixels (center proximity threshold)

## Output

- **Heatmap**: 32×32 spatial activity map (inferno colormap)
- **Center Marker**: White dot at array center (15.5, 15.5)
- **Hotspot**: Cyan dot marking detected activity centroid
- **Arrow**: Cyan vector from center toward hotspot
- **Guidance Text**: Directional instruction (e.g., "MOVE UP-RIGHT", "LOCKED", "STABLE")

## Guidance States

- **STABLE**: No significant activity detected
- **LOCKED**: Hotspot within 2.5 pixels of center
- **MOVE [DIRECTION]**: 8-directional guidance (UP, DOWN, LEFT, RIGHT, diagonals)

## Notes

- Processes up to 5000 time samples by default
- Automatically selects electrode columns from parquet file
- Falls back to GIF if MP4 export fails (FFmpeg not available)
