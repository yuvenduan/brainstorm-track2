# Center of Mass Tracking - Usage Guide

## Overview

The group processor can track the intensity-weighted center of mass (CoM) of neural activity in real-time and visualize it with a bright yellow arrow pointing from the grid center to the CoM location.

## Features

- **Real-time CoM computation**: Intensity-weighted spatial averaging per frame
- **Terminal output**: Print CoM coordinates (row, col) for each batch
- **Visual indicator**: Bright yellow arrow from [16,16] to CoM on web visualization
- **CSV export**: Save all CoM values with timestamps to file on exit

## Quick Start

### 1. Start Data Stream
```bash
uv run brainstorm-stream --from-file data/hard --port 8765
```

### 2. Start Processor with CoM Tracking
```bash
.venv/bin/python3 scripts/process_stream_group.py \
  --no-log-power \
  --track-com
```

### 3. Open Web Visualization
Navigate to `http://localhost:8001` in your browser and click **Connect**.

You should now see:
- Neural activity heatmap updating in real-time
- **Bright yellow arrow** pointing from center to center of mass
- CoM coordinates printing to terminal
- CSV file will be saved when you stop the processor

## CLI Options

### Basic Usage
```bash
python scripts/process_stream_group.py [OPTIONS]
```

### Center of Mass Options
| Flag | Default | Description |
|------|---------|-------------|
| `--track-com` | Off | Enable center of mass tracking and CSV export |
| `--no-track-com` | Default | Disable center of mass tracking |

### Complete Example with All Options
```bash
.venv/bin/python3 scripts/process_stream_group.py \
  --source ws://localhost:8765 \
  --port 8766 \
  --calibration 500 \
  --temporal-frames 10 \
  --gaussian-sigma 0.9 \
  --output-band high_gamma \
  --no-log-power \
  --track-com \
  --verbose
```

### Alternative Data Sources
```bash
# Connect to remote device
.venv/bin/python3 scripts/process_stream_group.py \
  --source ws://192.168.1.50:9000 \
  --track-com \
  --no-log-power

# Change output port
.venv/bin/python3 scripts/process_stream_group.py \
  --source ws://localhost:8765 \
  --port 8800 \
  --track-com
```

## Terminal Output Example

When tracking is enabled, you'll see real-time CoM coordinates:

```
✓ Calibration complete, starting real-time processing
Streaming High Gamma band with Hilbert envelope

CoM: t=10.638s, row=13.68, col=15.41
CoM: t=10.658s, row=15.65, col=15.80
CoM: t=10.678s, row=15.20, col=14.34
CoM: t=10.698s, row=14.44, col=14.15
CoM: t=10.718s, row=15.00, col=14.31
...
```

## CSV Output

### File Location
```
output/center_of_mass.csv
```

The output directory is created automatically if it doesn't exist.

### File Format
```csv
time_s,com_row,com_col
10.638000,13.681501,15.413550
10.658000,15.646763,15.800860
10.678000,15.195999,14.344120
...
```

- **time_s**: Timestamp in seconds
- **com_row**: Row coordinate in grid space [0, 31]
- **com_col**: Column coordinate in grid space [0, 31]

Grid center is at [16, 16].

### When CSV is Saved
The CSV file is automatically saved when you:
- Press **Ctrl+C** to stop the processor
- The processor exits for any reason

A confirmation message will be printed:
```
✓ Saved 3413 center of mass values to: output/center_of_mass.csv
```

## Visualization Details

### Yellow Arrow
- **Color**: Bright yellow (#FFEB3B)
- **Start**: Grid center [16, 16]
- **End**: Current center of mass location
- **Style**: Solid line with arrowhead pointing to CoM
- **Width**: 3 pixels

### Center of Mass Calculation
The CoM is computed using intensity-weighted spatial averaging:

```python
# Shift values to non-negative for weighting
weights = grid - np.min(grid)

# Weighted average of row/col coordinates
com_row = Σ(row_i × weight_i) / Σ(weight_i)
com_col = Σ(col_i × weight_i) / Σ(weight_i)
```

If total weight is near zero (no activity), CoM defaults to grid center [16, 16].

## Use Cases

### 1. Hotspot Tracking
Track the location of peak neural activity over time:
```bash
.venv/bin/python3 scripts/process_stream_group.py \
  --track-com \
  --temporal-frames 5 \
  --output-band high_gamma
```

### 2. Movement Analysis
Analyze how activity center moves across the electrode array:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('output/center_of_mass.csv')

# Plot trajectory
plt.figure(figsize=(8, 8))
plt.plot(df['com_col'], df['com_row'], 'o-', alpha=0.5)
plt.xlim(0, 31)
plt.ylim(0, 31)
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Center of Mass Trajectory')
plt.gca().invert_yaxis()  # Match screen coordinates
plt.grid(True)
plt.show()
```

### 3. Direction Detection
Calculate direction vector from center:
```python
import numpy as np

# Load data
df = pd.read_csv('output/center_of_mass.csv')

# Compute vectors from center [16, 16]
df['delta_row'] = df['com_row'] - 16
df['delta_col'] = df['com_col'] - 16

# Compute angle (radians)
df['angle'] = np.arctan2(df['delta_row'], df['delta_col'])

# Compute distance from center
df['distance'] = np.sqrt(df['delta_row']**2 + df['delta_col']**2)

print(df[['time_s', 'angle', 'distance']].head())
```

## Performance Notes

- CoM computation adds ~1-2ms per batch (negligible)
- Terminal printing can be suppressed by redirecting stderr
- CSV writing only happens on exit (no runtime overhead)
- Arrow rendering in browser has minimal performance impact

## Troubleshooting

### Arrow not visible
- Ensure `--track-com` flag is set
- Check that web app connected after processor started
- Refresh browser page

### CSV file not created
- Check that processor exited cleanly (Ctrl+C, not kill -9)
- Verify write permissions for `output/` directory
- Look for error messages in terminal

### CoM stuck at center [16, 16]
- Activity may be very low (all weights ~0)
- Check that calibration completed successfully
- Try different frequency band or temporal smoothing

## Related Files

- **Processor**: `scripts/process_stream_group.py`
- **Web App**: `example_app/app.js`
- **Documentation**: `GROUP_PIPELINE.md`

## Complete Workflow Example

```bash
# Terminal 1: Start data stream
uv run brainstorm-stream --from-file data/hard --port 8765

# Terminal 2: Start processor with CoM tracking
.venv/bin/python3 scripts/process_stream_group.py \
  --track-com \
  --no-log-power \
  --temporal-frames 10 \
  --verbose

# Browser: Open http://localhost:8001 and click Connect

# Watch the yellow arrow track neural activity!

# When done: Press Ctrl+C in Terminal 2
# CSV will be saved to: output/center_of_mass.csv
```
