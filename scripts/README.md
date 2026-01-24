# Scripts

Tools for visualizing and exporting neural data from the BrainStorm Track 2 challenge.

## Available Scripts

### Export Video (MP4/AVI)

Export neural data to video files with bad channel detection, imputation, and MIP smoothing.

**Generate deliverables (super_easy and hard):**
```bash
# Super easy dataset
uv run python scripts/export_video.py -d super_easy -o /store1/candy/bci/super_easy.mp4 --verbose

# Hard dataset
uv run python scripts/export_video.py -d hard -o /store1/candy/bci/hard.mp4 --verbose
```

**Features:**
- Bad channel detection using first 1000 frames (variance analysis)
- Automatic median interpolation from good neighbors (skipped if no bad channels)
- Standardized visualization (mean/std from first 100 frames)
- Maximum intensity projection (MIP) with 50-frame half-overlapping windows
- Frame range annotation (top right corner)
- Export to MP4 or AVI format

**Common options:**
```bash
# Adjust detection parameters
--detection-frames 2000          # Use more frames for bad channel detection
--variance-threshold 1e-7        # Adjust sensitivity

# Adjust MIP smoothing
--mip-window 100                 # Larger window for more smoothing
--mip-stride 50                  # Adjust overlap (half of window = 50% overlap)

# Skip imputation to see raw bad channels
--skip-imputation

# Other options
--fps 60                         # Higher frame rate
--frame-size 1024               # Larger output resolution
--verbose                        # Print detailed statistics
```

**See also:** [Claude.md](../Claude.md) for complete pipeline details

---

### Super Easy Dataset (Crystal-Clear Signals)

**Python Script:**
```bash
uv run python scripts/visualize_super_easy.py
```

**Shell Script:**
```bash
./scripts/view_super_easy.sh
```

**Features:**
- Crystal-clear signals with no noise
- Single centered tuned region
- Perfect for understanding the neural signals

---

### Hard Dataset (Matches Live Evaluation)

**Python Script:**
```bash
uv run python scripts/visualize_hard.py
```

**Shell Script:**
```bash
./scripts/view_hard.sh
```

**Features:**
- Challenging conditions with noise and artifacts
- 15 scattered tuned spots
- 8 bad channels (4 dead, 4 with artifacts)
- 60 Hz line noise + white noise
- **Matches live evaluation conditions**

---

## Script Options

All Python scripts support:

```bash
# Don't open browser automatically
--no-browser

# Custom ports
--ws-port 9000 --web-port 9001

# Custom data directory
--data-dir /path/to/data

# Help
--help
```

## How They Work

Both scripts:
1. Check if the dataset is downloaded
2. Load the neural data
3. Start the WebSocket streaming server (port 8765)
4. Start the web viewer server (port 8000)
5. Open http://localhost:8000 in your browser
6. Stream data continuously (looping)

Press `Ctrl+C` to stop both servers.

## Other Scripts

### Download Data
```bash
# Download specific difficulty
uv run python -m scripts.download super_easy
uv run python -m scripts.download easy
uv run python -m scripts.download medium
uv run python -m scripts.download hard
```

### Manual Server Control

If you prefer to run servers separately:

**Terminal 1 - Data Streaming:**
```bash
uv run brainstorm-stream --from-file data/hard/
```

**Terminal 2 - Web Viewer:**
```bash
uv run brainstorm-serve
```

### Control Client (Live Evaluation)
```bash
# Send keyboard controls during live evaluation
uv run python -m scripts.control_client
```

## Troubleshooting

**Dataset not found:**
```bash
uv run python -m scripts.download hard
```

**Port already in use:**
```bash
# Kill processes on ports
pkill -f brainstorm-stream
pkill -f brainstorm-serve

# Or use different ports
uv run python scripts/visualize_hard.py --ws-port 9000 --web-port 9001
```

**Browser doesn't open:**
- Manually navigate to http://localhost:8000
- Or use `--no-browser` flag

## Recommended Workflow

1. **Understand the signals** with super_easy:
   ```bash
   ./scripts/view_super_easy.sh
   ```

2. **Develop your solution** with hard:
   ```bash
   ./scripts/view_hard.sh
   ```

3. **Customize the visualization** in `example_app/`

4. **Test with streaming data** to ensure real-time performance
