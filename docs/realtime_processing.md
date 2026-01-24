# Real-time Processing Pipeline

Simple real-time neural data processing that sits between the data stream and your web app.

## Architecture

```
┌──────────────────┐                          ┌──────────────────┐
│  stream_data.py  │    WebSocket (500 Hz)    │ process_stream.py│
│  ws://:8765      │ ────────────────────────▶│  (Processor)     │
└──────────────────┘                          └────────┬─────────┘
                                                       │ ws://:8766
                                                       ▼ processed data
                                              ┌──────────────────┐
                                              │    Web App       │
                                              │   (Browser)      │
                                              └──────────────────┘
```

## Processing Steps

1. **Bad Channel Detection** (first 1000 frames)
   - Detects channels with variance < 1e-6 (dead/stuck channels)
   - Limits to max 16 bad channels

2. **Bad Channel Imputation**
   - Replaces bad channel values with median of 8-connected neighbors
   - Falls back to global median if no good neighbors

3. **Spatial Smoothing** (optional)
   - 3x3 median filter to reduce spatially uncorrelated noise
   - Preserves spatial patterns while reducing single-pixel artifacts

## Usage

### 1. Start the Data Stream

```bash
# Terminal 1: Start streaming data
uv run brainstorm-stream hard
```

### 2. Start the Processor

```bash
# Terminal 2: Start the processing pipeline
python scripts/process_stream.py
```

The processor will:
- Connect to `ws://localhost:8765` (data stream)
- Serve processed data on `ws://localhost:8766`
- Calibrate using first 1000 frames
- Start real-time processing after calibration

### 3. Connect Your Web App

Update the WebSocket URL in your frontend:

**Before:**
```typescript
const [serverUrl, setServerUrl] = useState('ws://localhost:8765');
```

**After:**
```typescript
const [serverUrl, setServerUrl] = useState('ws://localhost:8766');
```

Or just type `ws://localhost:8766` in the UI input field.

## Options

```bash
python scripts/process_stream.py --help
```

### Common Options

**Change output port:**
```bash
python scripts/process_stream.py --port 9000
```

**Disable spatial smoothing:**
```bash
python scripts/process_stream.py --no-spatial-smooth
```

**Adjust calibration length:**
```bash
python scripts/process_stream.py --calibration 500
```

**Verbose logging:**
```bash
python scripts/process_stream.py --verbose
```

**Connect to final server:**
```bash
python scripts/process_stream.py --source ws://<server-ip>:8765/stream
```

## Output Format

The processor outputs data in the **same format** as the original stream:

**Init message:**
```json
{
  "type": "init",
  "channels_coords": [[1, 1], [1, 2], ..., [32, 32]],
  "grid_size": 32,
  "fs": 500.0,
  "batch_size": 10
}
```

**Sample batch message:**
```json
{
  "type": "sample_batch",
  "neural_data": [[0.1, -0.2, ...], ...],
  "start_time_s": 1.234,
  "sample_count": 10,
  "fs": 500.0
}
```

Your existing frontend code will work without modification!

## Performance

- **Latency**: ~2-3 seconds during calibration, then <10ms per batch
- **Throughput**: Handles 500 Hz stream with 1024 channels in real-time
- **Memory**: ~100 MB for calibration buffer

## Troubleshooting

**"Connection refused" error:**
- Make sure the data stream is running first (`uv run brainstorm-stream hard`)
- Check that port 8765 is not blocked by firewall

**No data appearing in web app:**
- Check that web app is connecting to port 8766 (not 8765)
- Look for calibration message in processor terminal
- Try `--verbose` flag to see processing stats

**High latency:**
- Latency during first 1000 frames is normal (calibration)
- After calibration, processing should be near real-time
- Reduce `--calibration` frames if you can accept less accurate bad channel detection

## Advanced: Custom Processing

To add custom processing, edit `scripts/process_stream.py`:

```python
async def process_batch(self, batch: np.ndarray) -> np.ndarray:
    """Add your processing here."""
    processed = batch.copy()

    # Your custom processing
    processed = my_custom_filter(processed)

    return processed
```

The processor maintains compatibility with your frontend by keeping the same JSON message format.
