#!/usr/bin/env python3
"""
Quick launcher for visualizing the super_easy dataset.

This script:
1. Checks if super_easy data is downloaded
2. Launches the WebSocket streaming server
3. Launches the web viewer server
4. Opens the viewer in your default browser

Usage:
    python scripts/visualize_super_easy.py

    # Or via uv
    uv run python scripts/visualize_super_easy.py

    # With custom ports
    python scripts/visualize_super_easy.py --ws-port 8765 --web-port 8000

    # Don't open browser automatically
    python scripts/visualize_super_easy.py --no-browser
"""

import asyncio
import json
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import websockets
from aiohttp import web
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from websockets.server import WebSocketServerProtocol

# Import from existing scripts
from scripts.stream_data import (
    FileSampleProvider,
    StreamingServer,
    generate_channel_coords,
)

app = typer.Typer()
console = Console()

VIEWER_DIR = Path(__file__).parent.parent / "example_app"


def check_super_easy_data(data_dir: Path) -> bool:
    """Check if super_easy data is downloaded."""
    data_file = data_dir / "super_easy" / "track2_data.parquet"
    return data_file.exists()


def load_super_easy_data(data_dir: Path) -> tuple[np.ndarray, float]:
    """Load super_easy dataset."""
    data_path = data_dir / "super_easy" / "track2_data.parquet"

    console.print(f"[dim]Loading data from {data_path}...[/dim]")
    data_df = pd.read_parquet(data_path)
    data = data_df.values.astype(np.float32)

    # Assume 500 Hz sampling rate
    fs = 500.0

    console.print(
        f"[dim]Loaded {data.shape[0]:,} samples ({data.shape[0] / fs:.1f}s) "
        f"with {data.shape[1]} channels[/dim]"
    )

    return data, fs


def print_header() -> None:
    """Print welcome header."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Super Easy Dataset Viewer[/bold cyan]\n"
            "[dim]Crystal-clear signals for understanding neural activity[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def print_instructions(web_host: str, web_port: int, ws_port: int) -> None:
    """Print access instructions."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row("Web Viewer:", f"http://{web_host}:{web_port}")
    table.add_row("WebSocket Stream:", f"ws://localhost:{ws_port}")
    table.add_row("Dataset:", "super_easy (crystal-clear signals)")

    console.print(Panel(table, title="[bold green]Servers Running[/bold green]", border_style="green"))
    console.print()
    console.print("[dim]Press Ctrl+C to stop all servers[/dim]")
    console.print()


async def run_web_server(host: str, port: int) -> None:
    """Run the web viewer server."""

    async def index_handler(request: web.Request) -> web.FileResponse:
        """Serve index.html."""
        return web.FileResponse(VIEWER_DIR / "index.html")

    # Create web app
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_static("/", VIEWER_DIR, show_index=True)

    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    # Keep running
    await asyncio.Event().wait()


async def run_streaming_server(
    data: np.ndarray,
    fs: float,
    host: str,
    port: int,
    batch_size: int,
) -> None:
    """Run the WebSocket streaming server."""

    # Create channel coordinates
    grid_size = int(np.sqrt(data.shape[1]))
    channels_coords = generate_channel_coords(grid_size)

    # Create sample provider
    sample_provider = FileSampleProvider(data=data, fs=fs, loop=True)

    # Create streaming server
    server = StreamingServer(
        host=host,
        port=port,
        channels_coords=channels_coords,
        sample_provider=sample_provider,
        fs=fs,
        batch_size=batch_size,
    )

    # Start streaming with live display
    with Live(
        server.make_status_panel(),
        console=console,
        refresh_per_second=10,
        transient=False,
    ) as live:
        await server.start(live)


@app.command()
def main(
    ws_port: int = typer.Option(8765, help="WebSocket server port"),
    web_port: int = typer.Option(8000, help="Web viewer port"),
    web_host: str = typer.Option("localhost", help="Web viewer host"),
    open_browser: bool = typer.Option(True, "--browser/--no-browser", help="Open browser automatically"),
    data_dir: Path = typer.Option(Path("data"), help="Data directory path"),
    batch_size: int = typer.Option(10, help="Samples per WebSocket message"),
) -> None:
    """
    Launch the super_easy dataset viewer.

    This starts both the WebSocket streaming server and the web viewer,
    then optionally opens your browser to visualize the crystal-clear signals.
    """
    print_header()

    # Check if data exists
    if not check_super_easy_data(data_dir):
        console.print("[red]Error:[/red] super_easy dataset not found!")
        console.print()
        console.print("Download it with:")
        console.print("  [cyan]uv run python -m scripts.download super_easy[/cyan]")
        console.print()
        raise typer.Exit(code=1)

    console.print("[green]✓[/green] Found super_easy dataset")
    console.print()

    # Load data
    try:
        data, fs = load_super_easy_data(data_dir)
    except Exception as e:
        console.print(f"[red]Error loading data:[/red] {e}")
        raise typer.Exit(code=1)

    console.print()
    console.print("[dim]Starting servers...[/dim]")
    console.print()

    # Print instructions
    print_instructions(web_host, web_port, ws_port)

    # Run both servers
    async def run_servers():
        # Open browser after a short delay
        if open_browser:
            url = f"http://{web_host}:{web_port}"

            async def delayed_browser_open():
                await asyncio.sleep(2)
                console.print(f"[dim]Opening browser to {url}...[/dim]")
                console.print()
                webbrowser.open(url)

            # Schedule browser open
            asyncio.create_task(delayed_browser_open())

        # Run both servers concurrently
        await asyncio.gather(
            run_web_server(web_host, web_port),
            run_streaming_server(data, fs, "localhost", ws_port, batch_size),
        )

    try:
        asyncio.run(run_servers())
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Shutting down servers...[/yellow]")
        console.print("[green]✓[/green] Servers stopped")
        console.print()


if __name__ == "__main__":
    app()
