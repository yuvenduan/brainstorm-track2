#!/usr/bin/env python3
"""
Real-time Neural Data Processor

Sits between the data stream and your web app, applying simple processing:
1. Bad channel detection (first 1000 frames)
2. Bad channel imputation using neighbor median
3. Simple spatial smoothing to reduce noise
4. Re-streams processed data in the same format

Architecture:
    Data Stream (ws://localhost:8765)
         ↓
    This Processor (processes data)
         ↓
    Your Web App (ws://localhost:8766) ← connects here instead

Usage:
    # Start the data stream (in one terminal)
    uv run brainstorm-stream hard

    # Start this processor (in another terminal)
    python scripts/process_stream.py

    # Update your web app to connect to ws://localhost:8766
"""

import asyncio
import json
from collections import deque
from typing import Optional

import numpy as np
import typer
import websockets
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer()

# Constants
GRID_SIZE = 32
N_CHANNELS = GRID_SIZE * GRID_SIZE  # 1024


class RealtimeProcessor:
    """Real-time neural data processor with bad channel imputation."""

    def __init__(
        self,
        source_url: str,
        output_port: int,
        calibration_frames: int = 1000,
        spatial_smooth: bool = True,
        verbose: bool = False,
    ):
        self.source_url = source_url
        self.output_port = output_port
        self.calibration_frames = calibration_frames
        self.spatial_smooth = spatial_smooth
        self.verbose = verbose

        # State
        self.fs = 500.0
        self.channels_coords = None
        self.grid_size = GRID_SIZE
        self.batch_size = 10

        # Calibration
        self.is_calibrated = False
        self.calibration_buffer = []
        self.bad_mask = None
        self.global_mean = 0.0
        self.global_std = 1.0

        # High gamma band stats
        self.high_gamma_median = 0.0
        self.high_gamma_std = 1.0
        self.high_gamma_min = -0.25  # Min value after initial normalization
        self.high_gamma_max = 0.25   # Max value after initial normalization

        # Bandpass filter coefficients and state
        self.high_gamma_filter = None
        self.filter_states = None  # Filter state per channel

        # Connected clients
        self.clients = set()

        # Stats
        self.frames_processed = 0
        self.samples_processed = 0

    # =========================================================================
    # Server: Send processed data to clients
    # =========================================================================

    async def serve_clients(self, websocket):
        """Handle incoming client connections."""
        self.clients.add(websocket)
        console.print(f"[green]Client connected[/green] (total: {len(self.clients)})")

        try:
            # Send init message to new client
            if self.channels_coords is not None:
                init_msg = {
                    "type": "init",
                    "channels_coords": self.channels_coords.tolist(),
                    "grid_size": self.grid_size,
                    "fs": self.fs,
                    "batch_size": self.batch_size,
                }
                await websocket.send(json.dumps(init_msg))

            # Keep connection alive
            async for message in websocket:
                pass  # Clients don't send us anything
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            console.print(f"[yellow]Client disconnected[/yellow] (total: {len(self.clients)})")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        msg_json = json.dumps(message)
        await asyncio.gather(
            *[client.send(msg_json) for client in self.clients],
            return_exceptions=True,
        )

    # =========================================================================
    # Client: Receive data from source stream
    # =========================================================================

    async def receive_from_source(self):
        """Connect to source stream and process data."""
        console.print(f"[dim]Connecting to source: {self.source_url}[/dim]")

        try:
            async with websockets.connect(self.source_url) as ws:
                console.print(f"[green]✓[/green] Connected to source")

                async for message in ws:
                    data = json.loads(message)

                    if data["type"] == "init":
                        await self.handle_init(data)
                    elif data["type"] == "sample_batch":
                        await self.handle_sample_batch(data)

        except Exception as e:
            console.print(f"[red]Error connecting to source:[/red] {e}")
            raise

    async def handle_init(self, data: dict):
        """Handle initialization message from source."""
        self.channels_coords = np.array(data["channels_coords"])
        self.grid_size = data["grid_size"]
        self.fs = data["fs"]
        self.batch_size = data.get("batch_size", 10)

        console.print(
            f"[green]✓[/green] Initialized: {len(self.channels_coords)} channels, "
            f"grid {self.grid_size}x{self.grid_size}, fs={self.fs} Hz"
        )

        # Broadcast init to clients
        await self.broadcast(data)

    async def handle_sample_batch(self, data: dict):
        """Process and rebroadcast sample batch."""
        neural_data = np.array(data["neural_data"])  # Shape: (batch_size, 1024)
        start_time_s = data["start_time_s"]
        sample_count = data["sample_count"]

        # Process the batch
        processed_data = await self.process_batch(neural_data)

        # Update stats
        self.samples_processed += sample_count

        # Rebroadcast processed data
        output_msg = {
            "type": "sample_batch",
            "neural_data": processed_data.tolist(),
            "start_time_s": start_time_s,
            "sample_count": sample_count,
            "fs": self.fs,
        }
        await self.broadcast(output_msg)

        # Verbose logging
        if self.verbose and self.frames_processed % 100 == 0:
            console.print(
                f"[dim]Processed {self.frames_processed} batches, "
                f"{self.samples_processed} samples[/dim]"
            )

        self.frames_processed += 1

    # =========================================================================
    # Processing Pipeline
    # =========================================================================

    async def process_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Apply processing pipeline to batch of samples.

        Steps:
        1. Calibration phase (first N batches) - detect bad channels, compute stats
        2. Bad channel imputation
        3. Standardization (z-score)
        4. High gamma bandpass filtering + envelope extraction
        5. Normalization using calibration stats
        6. Rescale to [-0.25, +0.25] range

        Args:
            batch: Array of shape (batch_size, 1024)

        Returns:
            Processed batch of same shape, rescaled to [-0.25, +0.25]
        """
        # === CALIBRATION PHASE ===
        if not self.is_calibrated:
            # Buffer samples for calibration
            for sample in batch:
                self.calibration_buffer.append(sample)

            # Check if we have enough
            if len(self.calibration_buffer) >= self.calibration_frames:
                self.calibrate()
            else:
                # Still calibrating - pass through unprocessed
                return batch

        # === PROCESSING PHASE ===
        processed = batch.copy()

        # Step 1: Impute bad channels (sample-by-sample due to neighbor dependencies)
        if self.bad_mask is not None and self.bad_mask.any():
            for i in range(len(processed)):
                processed[i] = self.impute_sample(processed[i])

        # Step 2: Standardization (z-score normalization) - vectorized
        processed = (processed - self.global_mean) / self.global_std

        # Step 3: Extract high gamma envelope for entire batch
        processed = self.extract_high_gamma_envelope_batch(processed)

        # Step 4: Normalize using high gamma stats - vectorized
        processed = (processed - self.high_gamma_median) / self.high_gamma_std
        processed = np.clip(processed, -3, 3)

        # Step 5: Rescale from [-3, 3] to intermediate range
        processed = processed * (0.25 / 3.0)

        # Step 6: Min-max scaling to stretch to [-0.2, +0.25]
        # Formula: output = -0.2 + (input - min) / (max - min) * (0.25 - (-0.2))
        new_min = -0.2
        new_max = 0.25
        old_range = self.high_gamma_max - self.high_gamma_min
        new_range = new_max - new_min

        if old_range > 1e-10:
            processed = new_min + (processed - self.high_gamma_min) / old_range * new_range
        else:
            # If no range, center at 0
            processed = np.zeros_like(processed)

        # Clip to ensure we stay in bounds
        processed = np.clip(processed, new_min, new_max)

        # Step 7: Optional spatial smoothing (sample-by-sample)
        if self.spatial_smooth:
            for i in range(len(processed)):
                processed[i] = self.smooth_sample(processed[i])

        return processed

    def calibrate(self):
        """Run calibration using buffered samples."""
        calibration_data = np.array(self.calibration_buffer)  # Shape: (n_samples, 1024)

        console.print(f"[cyan]Calibrating with {len(calibration_data)} samples...[/cyan]")

        # Step 1: Detect bad channels
        self.bad_mask = self.detect_bad_channels(calibration_data)
        n_bad = self.bad_mask.sum() if self.bad_mask is not None else 0

        # Step 2: Impute bad channels in calibration data
        if self.bad_mask is not None and self.bad_mask.any():
            for i in range(len(calibration_data)):
                calibration_data[i] = self.impute_sample(calibration_data[i])

        # Step 3: Compute global stats for standardization
        self.global_mean = float(np.mean(calibration_data))
        self.global_std = float(np.std(calibration_data))
        if self.global_std < 1e-10:
            self.global_std = 1.0

        # Step 4: Design high gamma bandpass filter
        self.design_high_gamma_filter()

        # Step 5: Extract high gamma band from calibration data and compute normalization stats
        console.print("[dim]Extracting high gamma band for normalization...[/dim]")

        # Standardize all calibration data
        standardized_calibration = (calibration_data - self.global_mean) / self.global_std

        # Extract high gamma envelope for the entire batch
        high_gamma_calibration = self.extract_high_gamma_envelope_batch(standardized_calibration)

        # Compute high gamma stats (median and std from above-median values)
        self.high_gamma_median = float(np.median(high_gamma_calibration))
        above_median = high_gamma_calibration[high_gamma_calibration > self.high_gamma_median]
        if len(above_median) > 0:
            self.high_gamma_std = float(np.std(above_median))
        else:
            self.high_gamma_std = 1.0
        if self.high_gamma_std < 1e-10:
            self.high_gamma_std = 1.0

        # Step 6: Apply normalization and rescaling to calibration data to find min/max
        normalized_calibration = (high_gamma_calibration - self.high_gamma_median) / self.high_gamma_std
        clipped_calibration = np.clip(normalized_calibration, -3, 3)
        rescaled_calibration = clipped_calibration * (0.25 / 3.0)

        # Compute min/max for stretching
        self.high_gamma_min = float(np.min(rescaled_calibration))
        self.high_gamma_max = float(np.max(rescaled_calibration))

        # Prevent division by zero
        if abs(self.high_gamma_max - self.high_gamma_min) < 1e-10:
            self.high_gamma_min = -0.25
            self.high_gamma_max = 0.25

        # Print results
        table = Table(title="Calibration Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Bad channels detected", str(n_bad))
        table.add_row("Global mean", f"{self.global_mean:.6f}")
        table.add_row("Global std", f"{self.global_std:.6f}")
        table.add_row("High Gamma median", f"{self.high_gamma_median:.6f}")
        table.add_row("High Gamma std", f"{self.high_gamma_std:.6f}")
        table.add_row("High Gamma min (observed)", f"{self.high_gamma_min:.6f}")
        table.add_row("High Gamma max (observed)", f"{self.high_gamma_max:.6f}")
        console.print(table)
        console.print()

        if n_bad > 0:
            bad_indices = np.where(self.bad_mask)[0].tolist()
            console.print(f"[yellow]Bad channels:[/yellow] {bad_indices[:20]}{'...' if n_bad > 20 else ''}")
            console.print()

        self.is_calibrated = True
        console.print("[green]✓ Calibration complete, starting real-time processing[/green]")
        console.print("[cyan]Streaming High Gamma (70-150Hz) band only[/cyan]")
        console.print()

    def design_high_gamma_filter(self):
        """Design bandpass filter for high gamma band (70-150Hz)."""
        from scipy.signal import butter

        low_freq = 70.0
        high_freq = 150.0
        nyquist = self.fs / 2.0

        # Normalize to Nyquist
        low = max(0.001, min(low_freq / nyquist, 0.999))
        high = max(low + 0.001, min(high_freq / nyquist, 0.999))

        # Design 4th order Butterworth bandpass filter
        b, a = butter(4, [low, high], btype='band')
        self.high_gamma_filter = (b, a)

    def extract_high_gamma_envelope_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Extract high gamma envelope from a batch using causal bandpass filter + Hilbert.

        Uses stateful filtering to maintain continuity across batches.

        Args:
            batch: Array of shape (batch_size, 1024)

        Returns:
            High gamma envelope, same shape as input
        """
        from scipy.signal import hilbert, lfilter, lfilter_zi

        if self.high_gamma_filter is None:
            return np.abs(batch)

        b, a = self.high_gamma_filter

        # Initialize filter states if first time
        if self.filter_states is None:
            zi = lfilter_zi(b, a)
            # One filter state per channel
            self.filter_states = np.zeros((batch.shape[1], len(zi)))
            for ch in range(batch.shape[1]):
                self.filter_states[ch] = zi

        envelope_batch = np.zeros_like(batch)

        # Process each channel across the batch with state
        for ch in range(batch.shape[1]):
            channel_signal = batch[:, ch]

            # Apply causal bandpass filter with state
            filtered, self.filter_states[ch] = lfilter(
                b, a, channel_signal, zi=self.filter_states[ch]
            )

            # Hilbert transform to get envelope
            analytic = hilbert(filtered)
            envelope_batch[:, ch] = np.abs(analytic)

        return envelope_batch

    def detect_bad_channels(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect bad channels based on variance.

        A channel is "bad" if:
        - Variance < 1e-6 (stuck/dead channel)

        Args:
            data: Array of shape (n_samples, 1024)

        Returns:
            Boolean mask of shape (1024,), True = bad channel
        """
        # Compute variance for each channel
        channel_variance = np.var(data, axis=0)

        # Dead channels
        bad_mask = channel_variance < 1e-6

        # Limit to max 16 bad channels (keep worst ones)
        if bad_mask.sum() > 16:
            bad_indices = np.where(bad_mask)[0]
            bad_variances = channel_variance[bad_indices]
            sorted_indices = bad_indices[np.argsort(bad_variances)]

            # Reset and keep only 16 worst
            bad_mask = np.zeros(N_CHANNELS, dtype=bool)
            bad_mask[sorted_indices[:16]] = True

        return bad_mask if bad_mask.any() else None

    def impute_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Impute bad channels using median of 8-connected neighbors.

        Args:
            sample: Array of shape (1024,)

        Returns:
            Imputed sample
        """
        imputed = sample.copy()
        bad_channels = np.where(self.bad_mask)[0]

        for ch in bad_channels:
            neighbors = self.get_neighbors(ch)
            good_neighbors = [n for n in neighbors if not self.bad_mask[n]]

            if good_neighbors:
                # Median of good neighbors
                imputed[ch] = np.median(imputed[good_neighbors])
            else:
                # Fallback: global median of all good channels
                imputed[ch] = np.median(imputed[~self.bad_mask])

        return imputed

    def smooth_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Apply 3x3 median filter for spatial smoothing.

        Args:
            sample: Array of shape (1024,)

        Returns:
            Smoothed sample
        """
        from scipy.ndimage import median_filter

        # Reshape to grid
        grid = sample.reshape(GRID_SIZE, GRID_SIZE)

        # Apply median filter
        smoothed_grid = median_filter(grid, size=3, mode="reflect")

        # Reshape back
        return smoothed_grid.reshape(N_CHANNELS)

    def get_neighbors(self, channel_idx: int) -> list[int]:
        """
        Get 8-connected neighbors for a channel.

        Args:
            channel_idx: Flat channel index (0-1023)

        Returns:
            List of neighbor indices
        """
        row = channel_idx // GRID_SIZE
        col = channel_idx % GRID_SIZE

        # 8-connected neighbors
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        neighbors = []
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                neighbors.append(nr * GRID_SIZE + nc)

        return neighbors

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def run(self):
        """Run the processor: start server and connect to source."""
        # Start WebSocket server for clients
        server = await websockets.serve(
            self.serve_clients,
            "localhost",
            self.output_port,
        )

        console.print(
            f"[green]✓[/green] Processing server started on ws://localhost:{self.output_port}"
        )
        console.print()

        # Connect to source and start processing
        await self.receive_from_source()


# =============================================================================
# CLI
# =============================================================================


@app.command()
def main(
    source_url: str = typer.Option(
        "ws://localhost:8765",
        "--source",
        "-s",
        help="Source WebSocket URL to read from",
    ),
    output_port: int = typer.Option(
        8766,
        "--port",
        "-p",
        help="Port to serve processed stream on",
    ),
    calibration_frames: int = typer.Option(
        1000,
        "--calibration",
        "-c",
        help="Number of frames to use for calibration",
    ),
    no_spatial_smooth: bool = typer.Option(
        False,
        "--no-spatial-smooth",
        help="Disable spatial smoothing (3x3 median filter)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print detailed processing stats",
    ),
) -> None:
    """
    Real-time neural data processor.

    Connects to a data stream, applies processing, and restreams to clients.

    Example:
        # Start data stream
        uv run brainstorm-stream hard

        # Start processor (in another terminal)
        python scripts/process_stream.py

        # Connect your web app to ws://localhost:8766 instead of ws://localhost:8765
    """
    # Print header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Real-time Neural Data Processor[/bold cyan]\n"
            "[dim]Bad channel imputation + spatial smoothing[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Config table
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Source URL", source_url)
    table.add_row("Output Port", f"ws://localhost:{output_port}")
    table.add_row("Calibration Frames", str(calibration_frames))
    table.add_row("Spatial Smoothing", "Disabled" if no_spatial_smooth else "Enabled (3x3 median)")
    console.print(table)
    console.print()

    # Create processor
    processor = RealtimeProcessor(
        source_url=source_url,
        output_port=output_port,
        calibration_frames=calibration_frames,
        spatial_smooth=not no_spatial_smooth,
        verbose=verbose,
    )

    # Run
    try:
        asyncio.run(processor.run())
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Stopped by user[/yellow]")


if __name__ == "__main__":
    app()
