# Group Real-time Neural Processing Pipeline

## Overview

The group pipeline is a real-time neural data processor designed for low-latency streaming with multi-band frequency decomposition and log-power features.

## Key Features

- **Bad Channel Detection & Imputation** (max 16 channels)
- **60Hz Notch Filter** (Q=30)
- **4-Band Frequency Decomposition**
  - Theta/Alpha: 4-12 Hz
  - Beta: 12-30 Hz
  - Low Gamma: 30-70 Hz
  - High Gamma: 70-150 Hz
- **Log-Power Features**: Bandpass → Hilbert envelope → square → log(power)
- **Gaussian Spatial Smoothing** (sigma=0.9, before temporal)
- **Configurable Temporal Smoothing** (1-20 frames, after feature extraction)
- **Robust Normalization**: Median + std from calibration
- **Min-Max Stretch**: Output scaled to [-0.02, +0.02]

## Processing Pipeline

### Calibration Phase (First 500 Samples)
1. Detect bad channels (variance < 1e-6)
2. Report bad channels to terminal
3. Impute bad channels using 8-connected median
4. Design 60Hz notch filter
5. Design 4 bandpass filters
6. Extract features for all bands from calibration data
7. Compute normalization stats per band (median + std)
8. Compute min/max for output stretching

### Real-time Processing Phase (Per Batch)
1. **Impute bad channels** (8-connected median)
2. **Apply 60Hz notch filter** (causal, stateful)
3. **Extract band features** (bandpass → Hilbert → square → log)
4. **Reshape to 32x32 grid**
5. **Gaussian spatial smoothing** (per frame)
6. **Temporal smoothing** (buffer average of N frames)
7. **Normalize** (median + std)
8. **Min-max stretch** to [-0.02, +0.02]
9. **Stream to clients**

## Architecture

```
Data Stream (ws://localhost:8765)
         ↓
  Group Processor (processes data)
         ↓
  Your Web App (ws://localhost:8766)
```

## Usage

### Start the data stream
```bash
uv run brainstorm-stream hard
```

### Start the group processor (in another terminal)
```bash
# Default: high gamma band, 10 frames temporal smoothing
python scripts/process_stream_group.py

# Custom temporal smoothing
python scripts/process_stream_group.py --temporal-frames 5

# Different frequency band
python scripts/process_stream_group.py --output-band beta

# Adjust Gaussian smoothing
python scripts/process_stream_group.py --gaussian-sigma 1.2

# Verbose mode
python scripts/process_stream_group.py --verbose
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source`, `-s` | `ws://localhost:8765` | Source WebSocket URL |
| `--port`, `-p` | `8766` | Output port for processed stream |
| `--calibration`, `-c` | `500` | Calibration frames |
| `--temporal-frames`, `-t` | `10` | Temporal smoothing (1-20) |
| `--gaussian-sigma`, `-g` | `0.9` | Spatial smoothing sigma |
| `--output-band`, `-b` | `high_gamma` | Output band (theta_alpha, beta, low_gamma, high_gamma) |
| `--verbose`, `-v` | `False` | Detailed stats |

## Output Format

The processor outputs neural data in the same format as the input stream:

```json
{
  "type": "sample_batch",
  "neural_data": [[...], ...],  // Shape: (batch_size, 1024)
  "start_time_s": 0.0,
  "sample_count": 10,
  "fs": 500.0
}
```

All values are scaled to the range **[-0.02, +0.02]**.

## Processing Latency

- **Calibration**: 500 frames @ 500Hz = 1 second
- **Per-batch latency**: ~5-10ms (depends on temporal smoothing)
  - Temporal smoothing = 1 frame: minimal latency
  - Temporal smoothing = 20 frames: 40ms latency @ 500Hz

## Comparison with Other Pipelines

| Feature | Davy | Yuduan | Akhil | Candy | **Group** |
|---------|------|--------|-------|-------|-----------|
| Bad Channels | Zero out | None | None | Impute | **Impute (max 16)** |
| Notch Filter | 60+120 Hz | None | None | None | **60 Hz only** |
| Bands | 1 (70-150) | 4 FFT | None | 4 (Hilbert) | **4 (Hilbert + log-power)** |
| Features | Log-power | FFT power | RMS | Hilbert | **Hilbert → square → log** |
| Spatial | Gaussian bg sub | CNN | Gaussian | Median | **Gaussian** |
| Temporal | EMA | 200ms window | 20-frame buffer | None | **Configurable (1-20)** |
| Normalization | Z-score + baseline | Log1p | Percentile | Median + std | **Median + std** |
| Output Range | ±3 std | [0, 1] | [0, 1] | [-0.2, 0.25] | **[-0.02, +0.02]** |

## Key Design Choices

1. **Log-power on top of Hilbert envelope**: Captures both amplitude modulation AND power, more stable than raw power
2. **Gaussian before temporal**: Reduces spatial noise before aggregation, prevents noise from dominating the temporal buffer
3. **Median + std normalization**: More robust to outliers than mean-based z-score
4. **60Hz only notch**: Avoids over-filtering while removing primary line noise
5. **Max 16 bad channels**: Prevents excessive imputation which can introduce artifacts
6. **[-0.02, +0.02] range**: Tight range for precise visualization with minimal saturation

## Terminal Output Example

```
╭─────────────────────────────────────────────────────────╮
│ Group Real-time Neural Data Processor                  │
│ Multi-band decomposition with log-power features       │
╰─────────────────────────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Setting            ┃ Value                      ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Source URL         │ ws://localhost:8765        │
│ Output Port        │ ws://localhost:8766        │
│ Calibration Frames │ 500                        │
│ Temporal Smoothing │ 10 frames                  │
│ Gaussian Sigma     │ 0.90                       │
│ Output Band        │ High Gamma                 │
│ Output Range       │ [-0.02, +0.02]             │
└────────────────────┴────────────────────────────┘

Connecting to source: ws://localhost:8765...
✓ Connected to source
✓ Initialized: 1024 channels, grid 32x32, fs=500.0 Hz

Calibrating with 500 samples...
Bad channels detected: [51, 232, 419, 582]
Extracting frequency band features...

┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric                 ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Bad channels detected  │ 4       │
│ Calibration frames     │ 500     │
│ Temporal smoothing     │ 10      │
│ Gaussian sigma         │ 0.90    │
│ Output band            │ High    │
│                        │ Gamma   │
└────────────────────────┴─────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Band                       ┃ Median  ┃ Std    ┃ Min (norm) ┃ Max (norm) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Theta Alpha (4-12Hz)       │ -2.1234 │ 1.4567 │ -3.2145    │ 2.8976     │
│ Beta (12-30Hz)             │ -1.8765 │ 1.2345 │ -2.9876    │ 2.6543     │
│ Low Gamma (30-70Hz)        │ -1.6543 │ 1.1234 │ -2.7654    │ 2.4321     │
│ High Gamma (70-150Hz)      │ -1.4321 │ 0.9876 │ -2.5432    │ 2.2109     │
└────────────────────────────┴─────────┴────────┴────────────┴────────────┘

✓ Calibration complete, starting real-time processing
Streaming High Gamma band with log-power features

Client connected (total: 1)
```


  The processor is now streaming High Gamma band log-power features in real-time with all preprocessing applied (bad channel imputation → 60Hz notch → bandpass → Hilbert → square → log → Gaussian spatial smooth → temporal smooth → normalization → min-max stretch)
  