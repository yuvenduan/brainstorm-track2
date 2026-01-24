#!/usr/bin/env bash
# Quick launcher for viewing the hard dataset
#
# This script launches both the WebSocket streaming server and the web viewer
# in the background, then opens your browser.
#
# Usage:
#   ./scripts/view_hard.sh
#
#   # Or with bash
#   bash scripts/view_hard.sh

set -e

# Configuration
WS_PORT=8765
WEB_PORT=8000
DATA_DIR="data/hard"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m' # No Color

echo ""
echo -e "${RED}Hard Dataset Viewer${NC}"
echo -e "${DIM}Challenging conditions - matches live evaluation${NC}"
echo ""

# Check if data exists
if [ ! -f "$DATA_DIR/track2_data.parquet" ]; then
    echo -e "${RED}Error:${NC} hard dataset not found!"
    echo ""
    echo "Download it with:"
    echo -e "  ${CYAN}uv run python -m scripts.download hard${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Found hard dataset"
echo ""
echo -e "${DIM}Starting servers...${NC}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"

    # Kill background processes
    if [ ! -z "$STREAM_PID" ]; then
        kill $STREAM_PID 2>/dev/null || true
    fi
    if [ ! -z "$WEB_PID" ]; then
        kill $WEB_PID 2>/dev/null || true
    fi

    echo -e "${GREEN}✓${NC} Servers stopped"
    echo ""
    exit 0
}

# Set up trap for Ctrl+C
trap cleanup INT TERM

# Start streaming server in background
echo -e "${DIM}Starting WebSocket streaming server on port $WS_PORT...${NC}"
uv run brainstorm-stream --from-file "$DATA_DIR" --port $WS_PORT > /dev/null 2>&1 &
STREAM_PID=$!

# Give it time to start
sleep 2

# Start web viewer in background
echo -e "${DIM}Starting web viewer on port $WEB_PORT...${NC}"
uv run brainstorm-serve --port $WEB_PORT > /dev/null 2>&1 &
WEB_PID=$!

# Give it time to start
sleep 2

echo ""
echo -e "${GREEN}Servers Running${NC}"
echo ""
echo -e "  ${CYAN}Web Viewer:${NC}      http://localhost:$WEB_PORT"
echo -e "  ${CYAN}WebSocket Stream:${NC} ws://localhost:$WS_PORT"
echo -e "  ${CYAN}Dataset:${NC}          hard (challenging - matches live eval)"
echo ""
echo -e "${YELLOW}Note:${NC} This dataset has noise, artifacts, and bad channels"
echo -e "${DIM}Press Ctrl+C to stop all servers${NC}"
echo ""

# Open browser
URL="http://localhost:$WEB_PORT"
echo -e "${DIM}Opening browser to $URL...${NC}"
echo ""

# Try to open browser (works on macOS and most Linux systems)
if command -v xdg-open > /dev/null; then
    xdg-open "$URL" 2>/dev/null
elif command -v open > /dev/null; then
    open "$URL" 2>/dev/null
elif command -v python3 > /dev/null; then
    python3 -m webbrowser "$URL" 2>/dev/null
fi

# Wait for processes
wait
