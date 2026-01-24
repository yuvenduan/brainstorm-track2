#!/usr/bin/env python3
"""
Group Real-time Neural Data Processor

Multi-band frequency decomposition pipeline with log-power features.
Designed for low-latency streaming with configurable temporal smoothing.

Processing Pipeline:
1. Bad channel detection (first 500 frames) - max 16 channels
2. Bad channel imputation (8-connected median)
3. 60Hz notch filter
4. 4-band frequency decomposition (Theta/Alpha, Beta, Low Gamma, High Gamma)
5. Hilbert envelope → square → log(power)
6. Gaussian spatial smoothing (per frame, before temporal)
7. Temporal smoothing (configurable 1-20 frames, after feature extraction)
8. Normalization (median + std from calibration)
9. Min-max stretch to [-0.02, +0.02]

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
    python scripts/process_stream_group.py --temporal-frames 10

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

# Frequency band definitions
FREQUENCY_BANDS = {
    "theta_alpha": (4.0, 12.0),
    "beta": (12.0, 30.0),
    "low_gamma": (30.0, 70.0),
    "high_gamma": (70.0, 150.0),
}

BAND_NAMES = ["theta_alpha", "beta", "low_gamma", "high_gamma"]


class GroupRealtimeProcessor:
    """Real-time neural data processor with multi-band decomposition."""

    def __init__(
        self,
        source_url: str,
        output_port: int,
        calibration_frames: int = 500,
        temporal_frames: int = 10,
        gaussian_sigma: float = 0.9,
        output_band: str = "high_gamma",
        verbose: bool = False,
    ):
        self.source_url = source_url
        self.output_port = output_port
        self.calibration_frames = calibration_frames
        self.temporal_frames = max(1, min(20, temporal_frames))  # Clamp to [1, 20]
        self.gaussian_sigma = gaussian_sigma
        self.output_band = output_band
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

        # Normalization stats (per band)
        self.band_medians = {}
        self.band_stds = {}
        self.band_mins = {}
        self.band_maxs = {}

        # Filter states
        self.notch_filter = None
        self.notch_states = None
        self.band_filters = {}
        self.band_filter_states = {}

        # Temporal smoothing buffer
        self.temporal_buffer = deque(maxlen=self.temporal_frames)

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
        1. Calibration phase (first 500 batches) - detect bad channels, compute stats
        2. Bad channel imputation
        3. 60Hz notch filter
        4. 4-band frequency decomposition
        5. Hilbert envelope → square → log(power)
        6. Gaussian spatial smoothing (per frame)
        7. Temporal smoothing (buffer of N frames)
        8. Normalization (median + std)
        9. Min-max stretch to [-0.02, +0.02]

        Args:
            batch: Array of shape (batch_size, 1024)

        Returns:
            Processed batch of same shape, values in [-0.02, +0.02]
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

        # Step 2: Apply 60Hz notch filter (causal, stateful)
        processed = self.apply_notch_filter_batch(processed)

        # Step 3: Extract features for output band
        # Apply bandpass → Hilbert → square → log for the selected band
        band_features = self.extract_band_features_batch(processed, self.output_band)

        # Step 4-9: Per-sample processing (normalize → spatial → temporal → stretch)
        median_val = self.band_medians[self.output_band]
        std_val = self.band_stds[self.output_band]
        old_min = self.band_mins[self.output_band]
        old_max = self.band_maxs[self.output_band]
        new_min = -0.02
        new_max = 0.02
        old_range = old_max - old_min

        output_batch = np.zeros_like(processed)
        for i in range(len(band_features)):
            sample = band_features[i]  # Shape: (1024,)

            # Step 4: Normalize (median + std) - BEFORE smoothing
            normalized = (sample - median_val) / std_val

            # Step 5: Reshape to grid for spatial processing
            grid = normalized.reshape(GRID_SIZE, GRID_SIZE)

            # Step 6: Gaussian spatial smoothing
            smoothed_grid = self.gaussian_smooth(grid)

            # Step 7: Temporal smoothing (accumulate in buffer)
            self.temporal_buffer.append(smoothed_grid)
            temporal_avg = np.mean(self.temporal_buffer, axis=0)

            # Step 8: Min-max stretch to [-0.02, +0.02]
            if old_range > 1e-10:
                stretched = new_min + (temporal_avg - old_min) / old_range * (new_max - new_min)
            else:
                stretched = np.zeros_like(temporal_avg)

            # Clip to ensure bounds
            stretched = np.clip(stretched, new_min, new_max)

            # Flatten back to 1024
            output_batch[i] = stretched.reshape(N_CHANNELS)

        return output_batch

    def calibrate(self):
        """Run calibration using buffered samples."""
        calibration_data = np.array(self.calibration_buffer)  # Shape: (n_samples, 1024)

        console.print(f"[cyan]Calibrating with {len(calibration_data)} samples...[/cyan]")

        # Step 1: Detect bad channels
        self.bad_mask = self.detect_bad_channels(calibration_data)
        n_bad = self.bad_mask.sum() if self.bad_mask is not None else 0

        if n_bad > 0:
            bad_indices = np.where(self.bad_mask)[0].tolist()
            console.print(f"[yellow]Bad channels detected:[/yellow] {bad_indices}")

        # Step 2: Impute bad channels in calibration data
        if self.bad_mask is not None and self.bad_mask.any():
            for i in range(len(calibration_data)):
                calibration_data[i] = self.impute_sample(calibration_data[i])

        # Step 3: Compute global stats for standardization
        self.global_mean = float(np.mean(calibration_data))
        self.global_std = float(np.std(calibration_data))
        if self.global_std < 1e-10:
            self.global_std = 1.0

        # Step 4: Design filters
        self.design_notch_filter()
        self.design_bandpass_filters()

        # Step 5: Extract features for all bands from calibration data
        console.print("[dim]Extracting frequency band features...[/dim]")
        for band_name in BAND_NAMES:
            # Apply notch filter first
            notch_filtered = self.apply_notch_filter_batch_offline(calibration_data)

            # Extract band features
            band_features = self.extract_band_features_batch_offline(notch_filtered, band_name)

            # Compute normalization stats (median + std from all pixels)
            median_val = float(np.median(band_features))
            std_val = float(np.std(band_features))
            if std_val < 1e-10:
                std_val = 1.0

            self.band_medians[band_name] = median_val
            self.band_stds[band_name] = std_val

            # Compute min/max for stretching (after normalization)
            normalized = (band_features - median_val) / std_val
            self.band_mins[band_name] = float(np.min(normalized))
            self.band_maxs[band_name] = float(np.max(normalized))

        # Print results
        table = Table(title="Calibration Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Bad channels detected", str(n_bad))
        table.add_row("Calibration frames", str(self.calibration_frames))
        table.add_row("Temporal smoothing frames", str(self.temporal_frames))
        table.add_row("Gaussian sigma", f"{self.gaussian_sigma:.2f}")
        table.add_row("Output band", self.output_band.replace("_", " ").title())
        console.print(table)
        console.print()

        # Band stats table
        band_table = Table(title="Frequency Band Statistics")
        band_table.add_column("Band", style="cyan")
        band_table.add_column("Median", style="white")
        band_table.add_column("Std", style="white")
        band_table.add_column("Min (norm)", style="white")
        band_table.add_column("Max (norm)", style="white")

        for band_name in BAND_NAMES:
            band_label = band_name.replace("_", " ").title()
            low, high = FREQUENCY_BANDS[band_name]
            band_label += f" ({low:.0f}-{high:.0f}Hz)"

            band_table.add_row(
                band_label,
                f"{self.band_medians[band_name]:.4f}",
                f"{self.band_stds[band_name]:.4f}",
                f"{self.band_mins[band_name]:.4f}",
                f"{self.band_maxs[band_name]:.4f}",
            )

        console.print(band_table)
        console.print()

        self.is_calibrated = True
        console.print("[green]✓ Calibration complete, starting real-time processing[/green]")
        console.print(f"[cyan]Streaming {self.output_band.replace('_', ' ').title()} band with log-power features[/cyan]")
        console.print()

    def design_notch_filter(self):
        """Design 60Hz notch filter."""
        from scipy.signal import iirnotch

        notch_freq = 60.0
        quality_factor = 30.0

        b, a = iirnotch(notch_freq, quality_factor, self.fs)
        self.notch_filter = (b, a)

    def design_bandpass_filters(self):
        """Design bandpass filters for all 4 frequency bands."""
        from scipy.signal import butter

        nyquist = self.fs / 2.0

        for band_name in BAND_NAMES:
            low_freq, high_freq = FREQUENCY_BANDS[band_name]

            # Normalize to Nyquist
            low = max(0.001, min(low_freq / nyquist, 0.999))
            high = max(low + 0.001, min(high_freq / nyquist, 0.999))

            # Design 4th order Butterworth bandpass filter
            b, a = butter(4, [low, high], btype='band')
            self.band_filters[band_name] = (b, a)

    def apply_notch_filter_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Apply causal 60Hz notch filter to batch using stateful filtering.

        Args:
            batch: Array of shape (batch_size, 1024)

        Returns:
            Filtered batch, same shape
        """
        from scipy.signal import lfilter, lfilter_zi

        if self.notch_filter is None:
            return batch

        b, a = self.notch_filter

        # Initialize filter states if first time
        if self.notch_states is None:
            zi = lfilter_zi(b, a)
            # One filter state per channel
            self.notch_states = np.zeros((batch.shape[1], len(zi)))
            for ch in range(batch.shape[1]):
                self.notch_states[ch] = zi

        filtered_batch = np.zeros_like(batch)

        # Process each channel across the batch with state
        for ch in range(batch.shape[1]):
            channel_signal = batch[:, ch]

            # Apply causal notch filter with state
            filtered, self.notch_states[ch] = lfilter(
                b, a, channel_signal, zi=self.notch_states[ch]
            )

            filtered_batch[:, ch] = filtered

        return filtered_batch

    def apply_notch_filter_batch_offline(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter without state (for calibration)."""
        from scipy.signal import filtfilt

        if self.notch_filter is None:
            return data

        b, a = self.notch_filter
        filtered = np.zeros_like(data)

        for ch in range(data.shape[1]):
            filtered[:, ch] = filtfilt(b, a, data[:, ch])

        return filtered

    def extract_band_features_batch(self, batch: np.ndarray, band_name: str) -> np.ndarray:
        """
        Extract log-power features for a frequency band using causal filtering.

        Pipeline: bandpass → Hilbert envelope → square → log(power + eps)

        Args:
            batch: Array of shape (batch_size, 1024)
            band_name: Name of frequency band

        Returns:
            Log-power features, same shape as input
        """
        from scipy.signal import hilbert, lfilter, lfilter_zi

        if band_name not in self.band_filters:
            return np.abs(batch)

        b, a = self.band_filters[band_name]

        # Initialize filter states if first time for this band
        if band_name not in self.band_filter_states:
            zi = lfilter_zi(b, a)
            # One filter state per channel
            self.band_filter_states[band_name] = np.zeros((batch.shape[1], len(zi)))
            for ch in range(batch.shape[1]):
                self.band_filter_states[band_name][ch] = zi

        features_batch = np.zeros_like(batch)

        # Process each channel across the batch with state
        for ch in range(batch.shape[1]):
            channel_signal = batch[:, ch]

            # Apply causal bandpass filter with state
            filtered, self.band_filter_states[band_name][ch] = lfilter(
                b, a, channel_signal, zi=self.band_filter_states[band_name][ch]
            )

            # Hilbert transform to get envelope
            analytic = hilbert(filtered)
            envelope = np.abs(analytic)

            # Square and log transform
            power = envelope ** 2
            log_power = np.log(power + 1e-12)

            features_batch[:, ch] = log_power

        return features_batch

    def extract_band_features_batch_offline(self, data: np.ndarray, band_name: str) -> np.ndarray:
        """Extract band features without state (for calibration)."""
        from scipy.signal import filtfilt, hilbert

        if band_name not in self.band_filters:
            return np.abs(data)

        b, a = self.band_filters[band_name]
        features = np.zeros_like(data)

        for ch in range(data.shape[1]):
            # Bandpass filter
            filtered = filtfilt(b, a, data[:, ch])

            # Hilbert envelope
            analytic = hilbert(filtered)
            envelope = np.abs(analytic)

            # Square and log
            power = envelope ** 2
            log_power = np.log(power + 1e-12)

            features[:, ch] = log_power

        return features

    def detect_bad_channels(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect bad channels based on variance.

        A channel is "bad" if:
        - Variance < 1e-6 (stuck/dead channel)

        Args:
            data: Array of shape (n_samples, 1024)

        Returns:
            Boolean mask of shape (1024,), True = bad channel
            Maximum of 16 bad channels will be returned
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

    def gaussian_smooth(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian spatial smoothing to a grid.

        Args:
            grid: Array of shape (32, 32)

        Returns:
            Smoothed grid, same shape
        """
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(grid, sigma=self.gaussian_sigma, mode="reflect")

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
        500,
        "--calibration",
        "-c",
        help="Number of frames to use for calibration",
    ),
    temporal_frames: int = typer.Option(
        10,
        "--temporal-frames",
        "-t",
        help="Number of frames for temporal smoothing (1-20)",
    ),
    gaussian_sigma: float = typer.Option(
        0.9,
        "--gaussian-sigma",
        "-g",
        help="Sigma for Gaussian spatial smoothing",
    ),
    output_band: str = typer.Option(
        "high_gamma",
        "--output-band",
        "-b",
        help="Frequency band to output (theta_alpha, beta, low_gamma, high_gamma)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print detailed processing stats",
    ),
) -> None:
    """
    Group real-time neural data processor with multi-band decomposition.

    Processes neural data with 4-band frequency decomposition, log-power features,
    and configurable temporal smoothing for low-latency streaming.

    Example:
        # Start data stream
        uv run brainstorm-stream hard

        # Start processor (in another terminal)
        python scripts/process_stream_group.py --temporal-frames 10

        # Connect your web app to ws://localhost:8766
    """
    # Validate output band
    if output_band not in BAND_NAMES:
        console.print(f"[red]Error:[/red] Invalid output band: {output_band}")
        console.print(f"Valid options: {', '.join(BAND_NAMES)}")
        raise typer.Exit(code=1)

    # Validate temporal frames
    if temporal_frames < 1 or temporal_frames > 20:
        console.print(f"[red]Error:[/red] temporal_frames must be between 1 and 20")
        raise typer.Exit(code=1)

    # Print header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Group Real-time Neural Data Processor[/bold cyan]\n"
            "[dim]Multi-band decomposition with log-power features[/dim]",
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
    table.add_row("Temporal Smoothing", f"{temporal_frames} frames")
    table.add_row("Gaussian Sigma", f"{gaussian_sigma:.2f}")
    table.add_row("Output Band", output_band.replace("_", " ").title())
    table.add_row("Output Range", "[-0.02, +0.02]")
    console.print(table)
    console.print()

    # Create processor
    processor = GroupRealtimeProcessor(
        source_url=source_url,
        output_port=output_port,
        calibration_frames=calibration_frames,
        temporal_frames=temporal_frames,
        gaussian_sigma=gaussian_sigma,
        output_band=output_band,
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
