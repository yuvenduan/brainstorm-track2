# Quick Start: Visualizing Neural Data

This guide shows you how to quickly visualize the datasets with the web viewer.

## Prerequisites

1. Make sure you have installed the project:
   ```bash
   make install
   ```

2. Download datasets:
   ```bash
   # Start with super_easy to understand the signals
   uv run python -m scripts.download super_easy

   # Download hard for development (matches live evaluation)
   uv run python -m scripts.download hard
   ```

## Option 1: Python Script (Recommended)

The Python script launches both servers in a single process and automatically opens your browser.

### Super Easy Dataset (Crystal-Clear Signals)

```bash
uv run python scripts/visualize_super_easy.py
```

### Hard Dataset (Matches Live Evaluation)

```bash
uv run python scripts/visualize_hard.py
```

### Options

```bash
# Don't open browser automatically
uv run python scripts/visualize_hard.py --no-browser

# Use custom ports
uv run python scripts/visualize_hard.py --ws-port 9000 --web-port 9001

# See all options
uv run python scripts/visualize_hard.py --help
```

## Option 2: Shell Script

The bash script launches both servers as background processes.

### Super Easy Dataset

```bash
./scripts/view_super_easy.sh
```

### Hard Dataset

```bash
./scripts/view_hard.sh
```

Both scripts automatically:
- Start the WebSocket streaming server (port 8765)
- Start the web viewer server (port 8000)
- Open http://localhost:8000 in your browser

Press `Ctrl+C` to stop both servers.

## Option 3: Manual (Two Terminals)

If you prefer more control, run each server in its own terminal:

**Terminal 1 - Data Streaming:**
```bash
# Super easy
uv run brainstorm-stream --from-file data/super_easy/

# Or hard
uv run brainstorm-stream --from-file data/hard/
```

**Terminal 2 - Web Viewer:**
```bash
uv run brainstorm-serve
```

Then open http://localhost:8000 in your browser.

## What You'll See

### Super Easy Dataset
- **Crystal-clear signals** with no noise
- **Single centered tuned region** for easy understanding
- **Clean movement patterns** showing velocity-tuned neural activity

Perfect for:
- Understanding how the neural signals work
- Testing your visualization ideas
- Initial algorithm development

### Hard Dataset
- **Challenging conditions** with noise and artifacts
- **Multiple scattered tuned regions** (15 spots)
- **Bad channels** (4 dead, 4 with artifacts)
- **60 Hz line noise** and white noise

Perfect for:
- **Final development and testing** (matches live evaluation)
- Validating robustness to noise
- Testing bad channel detection
- Real-world performance evaluation

## Next Steps

### Recommended Workflow

1. **Start with super_easy** to understand the signals:
   ```bash
   ./scripts/view_super_easy.sh
   ```

2. **Switch to hard** for real development:
   ```bash
   ./scripts/view_hard.sh
   ```
   The hard dataset matches the live evaluation conditions with noise and artifacts.

3. **Customize the visualization:**
   - Modify `example_app/app.js` for different visualizations
   - Edit `example_app/style.css` for styling
   - See `docs/getting_started.md` for signal processing guidance

4. **Build your solution:**
   - The example app is just a basic heatmap
   - Your solution should go far beyond this
   - See `docs/overview.md` for judging criteria

## Troubleshooting

**Data not found:**
```bash
uv run python -m scripts.download super_easy
```

**Port already in use:**
```bash
# Use different ports
uv run python scripts/visualize_super_easy.py --ws-port 9000 --web-port 9001
```

**Browser doesn't open:**
- Manually navigate to http://localhost:8000
- Or use `--no-browser` flag and open it yourself

**Servers won't stop:**
```bash
# Find and kill processes
pkill -f brainstorm-stream
pkill -f brainstorm-serve
```
