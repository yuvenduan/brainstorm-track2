#!/usr/bin/env python3
"""
Neural Data Video Exporter for BrainStorm Track 2

Processes neural data with bad channel detection, imputation, artifact removal,
frequency band decomposition, MIP smoothing, and exports to multi-panel video.

Features:
- Bad channel detection using first 1000 frames (variance analysis)
  - Skips imputation if >75% of samples are stuck at same value
  - Max 16 bad channels (never more)
- Median interpolation from good 8-connected neighbors
- Artifact detection and removal:
  - 60Hz line noise (notch filter)
  - White noise (spatial smoothing)
- Frequency band decomposition with Hilbert envelope extraction:
  - Theta/Alpha: 4-12 Hz (movement planning, attention)
  - Beta: 12-30 Hz (movement preparation/suppression)
  - Low Gamma: 30-70 Hz (local cortical processing)
  - High Gamma: 70-150 Hz (motor execution, local firing)
- Per-band normalization (median + std from above-median pixels, first 1000 frames)
- Mean intensity projection with half-overlapping windows (500 frames, 250 stride)
- Multi-panel visualization (2x2 grid with amplitude colorbar)
- Frame range annotation in video
- Export to mp4 or avi format

Usage:
    python scripts/export_video.py -d hard -o /store1/candy/bci/hard.mp4
    python scripts/export_video.py -d super_easy -o /store1/candy/bci/super_easy.mp4 --verbose
    python scripts/export_video.py -d hard -o output.mp4 --skip-artifact-removal
"""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import typer
from matplotlib import colormaps
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(help="Export neural data to video with bad channel imputation")
console = Console()

# Constants
GRID_SIZE = 32
N_CHANNELS = GRID_SIZE * GRID_SIZE  # 1024
SAMPLING_RATE = 500.0  # Hz


# =============================================================================
# Channel Coordinate Utilities
# =============================================================================


def channel_to_grid_coords(channel_idx: int) -> tuple[int, int]:
    """
    Convert flat channel index to grid coordinates (0-indexed).

    Channel layout: row-major order
    - Channel 0 -> (0, 0) top-left
    - Channel 31 -> (0, 31) top-right
    - Channel 1023 -> (31, 31) bottom-right
    """
    row = channel_idx // GRID_SIZE
    col = channel_idx % GRID_SIZE
    return row, col


def grid_coords_to_channel(row: int, col: int) -> int:
    """Convert grid coordinates back to flat channel index."""
    return row * GRID_SIZE + col


def get_neighbors(channel_idx: int, connectivity: int = 8) -> list[int]:
    """
    Get neighboring channel indices for a given channel.

    Args:
        channel_idx: The channel to find neighbors for
        connectivity: 4 (cardinal) or 8 (include diagonals)

    Returns:
        List of valid neighbor channel indices
    """
    row, col = channel_to_grid_coords(channel_idx)

    # Cardinal directions (4-connectivity)
    offsets_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Add diagonals for 8-connectivity
    offsets_8 = offsets_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    offsets = offsets_8 if connectivity == 8 else offsets_4

    neighbors = []
    for dr, dc in offsets:
        nr, nc = row + dr, col + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            neighbors.append(grid_coords_to_channel(nr, nc))

    return neighbors


# =============================================================================
# Bad Channel Detection
# =============================================================================


def detect_bad_channels(
    data: np.ndarray,
    variance_threshold: float = 1e-6,
    fs: float = SAMPLING_RATE,
    target_fps: float = 30.0,
    detection_frames: int = 1000,
    stuck_threshold: float = 0.75,
) -> tuple[np.ndarray, dict, bool]:
    """
    Detect bad channels based on variance analysis using first N frames.

    Bad channel criteria:
    1. DEAD: Variance < variance_threshold (stuck at constant value)
    2. SATURATED: Mean very close to min/max of data range AND low variance

    Special case: If >75% of first N frames have the same value globally,
    treat as no bad channels and skip interpolation.

    For std calculation, only consider pixels with values different from
    the mean in the first N frames.

    Args:
        data: Neural data array of shape (n_samples, 1024)
        variance_threshold: Channels with variance below this are considered dead
        fs: Sampling rate in Hz
        target_fps: Target frame rate for video
        detection_frames: Number of frames to use for detection
        stuck_threshold: Fraction of samples that must be stuck to skip imputation

    Returns:
        bad_mask: Boolean array of shape (1024,), True = bad channel
        stats: Dictionary with detection statistics
        skip_imputation: True if imputation should be skipped (>75% stuck)
    """
    # Use only first N frames for detection
    samples_per_frame = int(fs / target_fps)
    n_samples_for_detection = samples_per_frame * detection_frames
    n_samples_for_detection = min(n_samples_for_detection, data.shape[0])
    detection_data = data[:n_samples_for_detection]

    # Check for globally stuck values (>75% of samples have same value)
    # This indicates a global issue, not individual bad channels
    unique_vals, counts = np.unique(detection_data.ravel(), return_counts=True)
    total_samples = detection_data.size
    most_common_ratio = counts.max() / total_samples

    if most_common_ratio > stuck_threshold:
        # Skip imputation - too many stuck values globally
        stats = {
            "n_dead": 0,
            "n_saturated": 0,
            "n_total_bad": 0,
            "dead_channels": [],
            "saturated_channels": [],
            "variance_stats": {
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
            },
            "skip_reason": f">{stuck_threshold*100:.0f}% samples stuck at same value",
            "stuck_ratio": float(most_common_ratio),
        }
        return np.zeros(N_CHANNELS, dtype=bool), stats, True

    # Compute per-channel statistics
    # Modified: only consider pixels with values different from the mean
    channel_mean = np.mean(detection_data, axis=0)
    channel_variance = np.zeros(N_CHANNELS, dtype=np.float64)

    for ch_idx in range(N_CHANNELS):
        ch_data = detection_data[:, ch_idx]
        ch_mean = channel_mean[ch_idx]
        # Only include values that differ from mean
        diff_mask = np.abs(ch_data - ch_mean) > 1e-10
        if diff_mask.sum() > 0:
            channel_variance[ch_idx] = np.var(ch_data[diff_mask])
        else:
            channel_variance[ch_idx] = 0.0  # All values same as mean

    channel_min = np.min(detection_data, axis=0)
    channel_max = np.max(detection_data, axis=0)

    # Detect dead channels (near-zero variance)
    dead_mask = channel_variance < variance_threshold

    # Detect saturated channels (constant at extreme values)
    data_min, data_max = detection_data.min(), detection_data.max()
    data_range = data_max - data_min

    if data_range > 0:
        is_at_min = np.abs(channel_mean - data_min) < 0.01 * data_range
        is_at_max = np.abs(channel_mean - data_max) < 0.01 * data_range
        low_dynamic_range = (channel_max - channel_min) < 0.01 * data_range
        saturated_mask = (is_at_min | is_at_max) & low_dynamic_range
    else:
        saturated_mask = np.zeros(N_CHANNELS, dtype=bool)

    bad_mask = dead_mask | saturated_mask

    # Enforce maximum of 16 bad channels
    if bad_mask.sum() > 16:
        # Keep only the 16 channels with lowest variance
        bad_indices = np.where(bad_mask)[0]
        bad_variances = channel_variance[bad_indices]
        sorted_indices = bad_indices[np.argsort(bad_variances)]
        # Reset mask and keep only worst 16
        bad_mask = np.zeros(N_CHANNELS, dtype=bool)
        bad_mask[sorted_indices[:16]] = True
        dead_mask = dead_mask & bad_mask
        saturated_mask = saturated_mask & bad_mask

    stats = {
        "n_dead": int(dead_mask.sum()),
        "n_saturated": int(saturated_mask.sum()),
        "n_total_bad": int(bad_mask.sum()),
        "dead_channels": np.where(dead_mask)[0].tolist(),
        "saturated_channels": np.where(saturated_mask)[0].tolist(),
        "variance_stats": {
            "min": float(channel_variance.min()),
            "max": float(channel_variance.max()),
            "median": float(np.median(channel_variance)),
        },
    }

    return bad_mask, stats, False


# =============================================================================
# Channel Imputation
# =============================================================================


def impute_bad_channels(
    data: np.ndarray,
    bad_mask: np.ndarray,
    connectivity: int = 8,
) -> np.ndarray:
    """
    Impute bad channels using median of good neighbors.

    For each bad channel:
    1. Find its neighbors (4 or 8-connected)
    2. Filter to only good neighbors
    3. Take median of good neighbor values at each time point
    4. If no good neighbors, use global median of all good channels

    Args:
        data: Neural data array of shape (n_samples, 1024)
        bad_mask: Boolean array of shape (1024,), True = bad channel
        connectivity: 4 or 8-connected neighbors

    Returns:
        Imputed data array with bad channels filled in
    """
    imputed = data.copy()
    bad_channels = np.where(bad_mask)[0]
    good_mask = ~bad_mask

    for ch in bad_channels:
        neighbors = get_neighbors(ch, connectivity=connectivity)
        good_neighbors = [n for n in neighbors if not bad_mask[n]]

        if len(good_neighbors) > 0:
            # Median of good neighbors at each time point
            neighbor_values = imputed[:, good_neighbors]
            imputed[:, ch] = np.median(neighbor_values, axis=1)
        else:
            # Fallback: use global median of all good channels at each time point
            imputed[:, ch] = np.median(imputed[:, good_mask], axis=1)

    return imputed


# =============================================================================
# Artifact Detection and Removal
# =============================================================================


def detect_60hz_line_noise(
    data: np.ndarray,
    fs: float = SAMPLING_RATE,
    notch_freq: float = 60.0,
    detection_threshold: float = 3.0,
) -> tuple[bool, float]:
    """
    Detect presence of 60Hz line noise using FFT power analysis.

    Compares power at 60Hz (+/- 2Hz) to surrounding frequency bands.

    Args:
        data: Neural data array of shape (n_samples, n_channels)
        fs: Sampling rate in Hz
        notch_freq: Line noise frequency (60Hz in US, 50Hz in Europe)
        detection_threshold: Ratio of 60Hz power to surrounding bands

    Returns:
        detected: True if 60Hz noise is significant
        relative_power: Power at 60Hz relative to surrounding frequencies
    """
    # Use a subset of data for efficiency
    n_samples = min(data.shape[0], int(fs * 10))  # 10 seconds max
    subset = data[:n_samples]

    # Average across channels for detection
    avg_signal = np.mean(subset, axis=1)

    # Compute FFT
    fft_result = np.fft.rfft(avg_signal)
    freqs = np.fft.rfftfreq(len(avg_signal), 1 / fs)
    power = np.abs(fft_result) ** 2

    # Find power at 60Hz band (+/- 2Hz)
    notch_mask = (freqs >= notch_freq - 2) & (freqs <= notch_freq + 2)
    notch_power = np.mean(power[notch_mask]) if notch_mask.any() else 0

    # Find power at surrounding bands (40-55Hz and 65-80Hz)
    lower_mask = (freqs >= 40) & (freqs <= 55)
    upper_mask = (freqs >= 65) & (freqs <= 80)
    surround_mask = lower_mask | upper_mask
    surround_power = np.mean(power[surround_mask]) if surround_mask.any() else 1

    # Compute relative power
    relative_power = notch_power / surround_power if surround_power > 0 else 0

    detected = relative_power > detection_threshold
    return detected, float(relative_power)


def remove_60hz_line_noise(
    data: np.ndarray,
    fs: float = SAMPLING_RATE,
    notch_freq: float = 60.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Remove 60Hz line noise using IIR notch filter.

    Uses scipy.signal.iirnotch + filtfilt for zero-phase filtering.

    Args:
        data: Neural data array of shape (n_samples, n_channels)
        fs: Sampling rate in Hz
        notch_freq: Line noise frequency to remove
        quality_factor: Q factor for notch filter (higher = narrower notch)

    Returns:
        Filtered data with 60Hz removed
    """
    from scipy.signal import filtfilt, iirnotch

    # Design notch filter
    b, a = iirnotch(notch_freq, quality_factor, fs)

    # Apply to each channel
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = filtfilt(b, a, data[:, ch])

    return filtered


def detect_white_noise(
    data: np.ndarray,
    fs: float = SAMPLING_RATE,
    high_freq_band: tuple[float, float] = (180.0, 240.0),
    detection_threshold: float = 0.1,
) -> tuple[bool, float]:
    """
    Detect white noise component via high-frequency power analysis.

    White noise has flat spectrum; estimate by comparing high-frequency
    power to overall signal power.

    Args:
        data: Neural data array of shape (n_samples, n_channels)
        fs: Sampling rate in Hz
        high_freq_band: Frequency range to analyze (near Nyquist)
        detection_threshold: Ratio threshold for detection

    Returns:
        detected: True if significant white noise component found
        estimated_std: Estimated standard deviation of white noise
    """
    # Use a subset of data for efficiency
    n_samples = min(data.shape[0], int(fs * 10))
    subset = data[:n_samples]

    # Average across channels
    avg_signal = np.mean(subset, axis=1)

    # Compute FFT
    fft_result = np.fft.rfft(avg_signal)
    freqs = np.fft.rfftfreq(len(avg_signal), 1 / fs)
    power = np.abs(fft_result) ** 2

    # High frequency band power (near Nyquist - typically noise-dominated)
    nyquist = fs / 2
    high_freq_band = (min(high_freq_band[0], nyquist - 10), min(high_freq_band[1], nyquist - 1))
    high_mask = (freqs >= high_freq_band[0]) & (freqs <= high_freq_band[1])
    high_power = np.mean(power[high_mask]) if high_mask.any() else 0

    # Total signal power
    total_power = np.mean(power)

    # Estimate noise std from high-frequency content
    if high_mask.any():
        # Noise std estimation from power spectral density
        noise_variance = high_power * len(avg_signal) / fs
        estimated_std = np.sqrt(max(0, noise_variance))
    else:
        estimated_std = 0.0

    # Detection based on ratio
    ratio = high_power / total_power if total_power > 0 else 0
    detected = ratio > detection_threshold

    return detected, float(estimated_std)


def estimate_and_remove_white_noise(
    data: np.ndarray,
    fs: float = SAMPLING_RATE,
) -> tuple[np.ndarray, float]:
    """
    Estimate and reduce white noise component using spatial smoothing.

    Uses median filtering across channels at each time point to reduce
    spatially uncorrelated noise while preserving spatial patterns.

    Args:
        data: Neural data array of shape (n_samples, n_channels)
        fs: Sampling rate in Hz

    Returns:
        denoised_data: Data with white noise reduced
        noise_std: Estimated noise standard deviation
    """
    from scipy.ndimage import median_filter

    # Estimate noise std from temporal differences
    # White noise should dominate high-frequency temporal variations
    temporal_diff = np.diff(data, axis=0)
    noise_std = float(np.median(np.abs(temporal_diff)) / 0.6745)  # MAD estimator

    # Reshape to grid for spatial smoothing
    n_samples = data.shape[0]
    grids = data.reshape(n_samples, GRID_SIZE, GRID_SIZE)

    # Apply mild spatial median filter (3x3) to reduce uncorrelated noise
    denoised_grids = np.zeros_like(grids)
    for t in range(n_samples):
        denoised_grids[t] = median_filter(grids[t], size=3, mode="reflect")

    # Reshape back
    denoised = denoised_grids.reshape(n_samples, N_CHANNELS)

    return denoised, noise_std


def remove_artifacts(
    data: np.ndarray,
    fs: float = SAMPLING_RATE,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Main artifact removal pipeline.

    Steps:
    1. Detect and remove 60Hz line noise (if present)
    2. Detect and reduce white noise component (if significant)

    Args:
        data: Neural data array of shape (n_samples, n_channels)
        fs: Sampling rate in Hz
        verbose: Print detection/removal statistics

    Returns:
        cleaned_data: Data with artifacts removed
        artifact_stats: Dictionary with detection/removal statistics
    """
    artifact_stats = {
        "line_noise_60hz": {"detected": False, "removed": False, "relative_power": 0.0},
        "white_noise": {"detected": False, "removed": False, "estimated_std": 0.0},
    }

    cleaned = data.copy()

    # Step 1: 60Hz line noise
    detected_60hz, power_60hz = detect_60hz_line_noise(cleaned, fs)
    artifact_stats["line_noise_60hz"]["detected"] = detected_60hz
    artifact_stats["line_noise_60hz"]["relative_power"] = power_60hz

    if detected_60hz:
        cleaned = remove_60hz_line_noise(cleaned, fs)
        artifact_stats["line_noise_60hz"]["removed"] = True
        if verbose:
            console.print(
                f"[yellow]60Hz line noise detected (relative power={power_60hz:.2f}x), removed[/yellow]"
            )

    # Step 2: White noise
    detected_white, white_std = detect_white_noise(cleaned, fs)
    artifact_stats["white_noise"]["detected"] = detected_white
    artifact_stats["white_noise"]["estimated_std"] = white_std

    if detected_white:
        cleaned, actual_std = estimate_and_remove_white_noise(cleaned, fs)
        artifact_stats["white_noise"]["removed"] = True
        artifact_stats["white_noise"]["estimated_std"] = actual_std
        if verbose:
            console.print(
                f"[yellow]White noise detected (estimated std={actual_std:.4f}), reduced[/yellow]"
            )

    return cleaned, artifact_stats


# =============================================================================
# Frequency Band Decomposition
# =============================================================================

# Frequency band definitions based on physiological relevance
FREQUENCY_BANDS = {
    "theta_alpha": (4.0, 12.0),  # Theta/Alpha: movement planning, attention
    "beta": (12.0, 30.0),  # Beta: movement preparation/suppression
    "low_gamma": (30.0, 70.0),  # Low Gamma: local cortical processing
    "high_gamma": (70.0, 150.0),  # High Gamma: motor execution, local firing
}

BAND_NAMES = ["theta_alpha", "beta", "low_gamma", "high_gamma"]
BAND_LABELS = {
    "theta_alpha": "Theta/Alpha (4-12Hz)",
    "beta": "Beta (12-30Hz)",
    "low_gamma": "Low Gamma (30-70Hz)",
    "high_gamma": "High Gamma (70-150Hz)",
}


def bandpass_filter_with_envelope(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    fs: float = SAMPLING_RATE,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase bandpass filter and extract amplitude envelope.

    Uses Hilbert transform to extract the instantaneous amplitude (envelope)
    of the bandpass-filtered signal. This is essential because averaging
    oscillatory signals over time would cancel to zero.

    Args:
        data: Neural data array of shape (n_samples, n_channels)
        low_freq: Lower cutoff frequency (Hz)
        high_freq: Upper cutoff frequency (Hz)
        fs: Sampling rate (Hz)
        order: Filter order

    Returns:
        Amplitude envelope of filtered data, same shape as input
    """
    from scipy.signal import butter, filtfilt, hilbert

    nyquist = fs / 2.0

    # Normalize frequencies to Nyquist
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Clamp to valid range (0, 1) exclusive
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))

    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype="band")

    # Apply bandpass filter and extract envelope for each channel
    envelope = np.zeros_like(data)
    for ch in range(data.shape[1]):
        # Bandpass filter
        filtered = filtfilt(b, a, data[:, ch])
        # Hilbert transform to get analytic signal, then absolute value for envelope
        analytic = hilbert(filtered)
        envelope[:, ch] = np.abs(analytic)

    return envelope


def decompose_frequency_bands(
    data: np.ndarray,
    fs: float = SAMPLING_RATE,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """
    Decompose signal into 4 frequency bands and extract amplitude envelopes.

    For each band, applies bandpass filter then Hilbert transform to get
    the instantaneous amplitude envelope. This allows meaningful averaging
    over time without oscillation cancellation.

    Args:
        data: Neural data array of shape (n_samples, n_channels)
        fs: Sampling rate (Hz)
        verbose: Print band statistics

    Returns:
        Dictionary mapping band name to amplitude envelope array.
        Each array has same shape as input, values are non-negative.
    """
    bands_data = {}

    for band_name in BAND_NAMES:
        low_freq, high_freq = FREQUENCY_BANDS[band_name]
        envelope = bandpass_filter_with_envelope(data, low_freq, high_freq, fs)
        bands_data[band_name] = envelope

        if verbose:
            band_std = np.std(envelope)
            band_mean = np.mean(envelope)
            console.print(f"[dim]  {BAND_LABELS[band_name]}: mean={band_mean:.4f}, std={band_std:.4f}[/dim]")

    return bands_data


# =============================================================================
# Normalization Statistics
# =============================================================================


def compute_normalization_stats(
    data: np.ndarray,
    fs: float = SAMPLING_RATE,
    target_fps: float = 30.0,
    n_frames: int = 100,
) -> tuple[float, float]:
    """
    Compute global mean and std from first n_frames.

    Args:
        data: Neural data array of shape (n_samples, 1024)
        fs: Sampling rate in Hz
        target_fps: Target frame rate for video
        n_frames: Number of frames to use for computing statistics

    Returns:
        Tuple of (global_mean, global_std)
    """
    samples_per_frame = int(fs / target_fps)
    n_samples_for_stats = samples_per_frame * n_frames

    # Ensure we don't exceed data length
    n_samples_for_stats = min(n_samples_for_stats, data.shape[0])

    stats_data = data[:n_samples_for_stats]

    global_mean = float(np.mean(stats_data))
    global_std = float(np.std(stats_data))

    # Prevent division by zero
    if global_std < 1e-10:
        global_std = 1.0

    return global_mean, global_std


def compute_normalization_stats_per_band(
    bands_data: dict[str, np.ndarray],
    fs: float = SAMPLING_RATE,
    target_fps: float = 30.0,
    n_frames: int = 1000,
    samples_per_frame: int | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Compute normalization stats from first n_frames for each frequency band.

    For each band:
    1. Find the median value across all pixels in first n_frames
    2. Compute std only from pixels that are above the median

    Args:
        bands_data: Dictionary mapping band name to data array
        fs: Sampling rate in Hz
        target_fps: Target frame rate for video
        n_frames: Number of frames to use for computing statistics
        samples_per_frame: Samples to average per frame (default: fs/target_fps)

    Returns:
        Dictionary mapping band name to (median, std_above_median) tuple
    """
    if samples_per_frame is None:
        samples_per_frame = int(fs / target_fps)
    n_samples_for_stats = samples_per_frame * n_frames

    stats = {}
    for band_name, data in bands_data.items():
        # Ensure we don't exceed data length
        n_samples = min(n_samples_for_stats, data.shape[0])
        stats_data = data[:n_samples]

        # Compute median across all pixels
        median_val = float(np.median(stats_data))

        # Compute std only from pixels above the median
        above_median = stats_data[stats_data > median_val]
        if len(above_median) > 0:
            std_val = float(np.std(above_median))
        else:
            std_val = 1.0

        # Prevent division by zero
        if std_val < 1e-10:
            std_val = 1.0

        stats[band_name] = (median_val, std_val)

    return stats


# =============================================================================
# Frame Generation
# =============================================================================


def generate_frames(
    data: np.ndarray,
    fs: float = SAMPLING_RATE,
    target_fps: float = 30.0,
    global_mean: float = 0.0,
    global_std: float = 1.0,
    n_std: float = 3.0,
) -> Iterator[tuple[np.ndarray, int, int]]:
    """
    Generate standardized frames from neural data.

    Args:
        data: Neural data array of shape (n_samples, 1024)
        fs: Sampling rate in Hz
        target_fps: Target frame rate for video
        global_mean: Mean for standardization
        global_std: Std for standardization
        n_std: Number of std deviations for clipping

    Yields:
        Tuple of (standardized_grid, start_sample, end_sample)
        - standardized_grid: numpy array of shape (32, 32), float32, clipped to ±n_std
        - start_sample: Starting sample index for this frame
        - end_sample: Ending sample index for this frame
    """
    samples_per_frame = int(fs / target_fps)
    n_frames = data.shape[0] // samples_per_frame

    for frame_idx in range(n_frames):
        start_sample = frame_idx * samples_per_frame
        end_sample = start_sample + samples_per_frame

        # Average samples within frame (temporal smoothing)
        frame_data = data[start_sample:end_sample].mean(axis=0)

        # Standardize: z = (x - mean) / std
        standardized = (frame_data - global_mean) / global_std

        # Clip to ±n_std
        clipped = np.clip(standardized, -n_std, n_std)

        # Reshape to grid
        grid = clipped.reshape(GRID_SIZE, GRID_SIZE)

        yield grid, start_sample, end_sample


def apply_mip(
    frames: Iterator[tuple[np.ndarray, int, int]],
    window_size: int = 50,
    stride: int = 25,
) -> Iterator[tuple[np.ndarray, int, int]]:
    """
    Apply maximum intensity projection (MIP) with overlapping windows.

    Args:
        frames: Iterator of (grid, start_sample, end_sample) tuples
        window_size: Number of frames per MIP window
        stride: Number of frames to advance between windows (half-overlap = window_size/2)

    Yields:
        Tuple of (mip_grid, window_start_sample, window_end_sample)
        - mip_grid: Max projection of frames in window, shape (32, 32)
        - window_start_sample: Starting sample of first frame in window
        - window_end_sample: Ending sample of last frame in window
    """
    buffer = []

    for grid, start_sample, end_sample in frames:
        buffer.append((grid, start_sample, end_sample))

        # When buffer reaches window size, compute MIP and yield
        if len(buffer) == window_size:
            # Stack all grids and take max across frames
            grids = np.stack([g for g, _, _ in buffer], axis=0)
            mip_grid = np.max(grids, axis=0)

            # Frame range is from first to last frame in window
            window_start = buffer[0][1]
            window_end = buffer[-1][2]

            yield mip_grid, window_start, window_end

            # Slide window by stride
            buffer = buffer[stride:]

    # Handle remaining frames in buffer (partial window at end)
    if len(buffer) > 0:
        grids = np.stack([g for g, _, _ in buffer], axis=0)
        mip_grid = np.max(grids, axis=0)
        window_start = buffer[0][1]
        window_end = buffer[-1][2]
        yield mip_grid, window_start, window_end


def generate_frames_per_band(
    bands_data: dict[str, np.ndarray],
    bands_stats: dict[str, tuple[float, float]],
    fs: float = SAMPLING_RATE,
    target_fps: float = 30.0,
    n_std: float = 3.0,
    samples_per_frame: int | None = None,
) -> Iterator[tuple[dict[str, np.ndarray], int, int]]:
    """
    Generate standardized frames for all frequency bands simultaneously.

    Args:
        bands_data: Dictionary mapping band name to data array
        bands_stats: Dictionary mapping band name to (median, std_above_median) tuple
        fs: Sampling rate in Hz
        target_fps: Target frame rate for video
        n_std: Number of std deviations for clipping
        samples_per_frame: Samples to average per frame (default: fs/target_fps)

    Yields:
        Tuple of (grids_dict, start_sample, end_sample)
        - grids_dict: Dictionary mapping band name to 32x32 grid
        - start_sample: Starting sample index for this frame
        - end_sample: Ending sample index for this frame
    """
    if samples_per_frame is None:
        samples_per_frame = int(fs / target_fps)
    # Get number of samples from first band
    first_band = list(bands_data.values())[0]
    n_frames = first_band.shape[0] // samples_per_frame

    for frame_idx in range(n_frames):
        start_sample = frame_idx * samples_per_frame
        end_sample = start_sample + samples_per_frame

        grids = {}
        for band_name, data in bands_data.items():
            median_val, std_above = bands_stats[band_name]

            # Average samples within frame
            frame_data = data[start_sample:end_sample].mean(axis=0)

            # Standardize using median and std from above-median pixels
            standardized = (frame_data - median_val) / std_above

            # Clip to ±n_std
            clipped = np.clip(standardized, -n_std, n_std)

            # Reshape to grid
            grids[band_name] = clipped.reshape(GRID_SIZE, GRID_SIZE)

        yield grids, start_sample, end_sample


def apply_mean_intensity_projection_per_band(
    frames: Iterator[tuple[dict[str, np.ndarray], int, int]],
    window_size: int = 500,
    stride: int = 250,
) -> Iterator[tuple[dict[str, np.ndarray], int, int]]:
    """
    Apply mean intensity projection to all frequency bands.

    Uses half-overlapping windows (default: 500 frames with 250 stride).

    Args:
        frames: Iterator of (grids_dict, start_sample, end_sample) tuples
        window_size: Number of frames per window
        stride: Number of frames to advance between windows

    Yields:
        Tuple of (mean_grids_dict, window_start_sample, window_end_sample)
        - mean_grids_dict: Dictionary mapping band name to mean projection grid
        - window_start_sample: Starting sample of first frame in window
        - window_end_sample: Ending sample of last frame in window
    """
    buffer = []

    for grids, start_sample, end_sample in frames:
        buffer.append((grids, start_sample, end_sample))

        if len(buffer) == window_size:
            mean_grids = {}
            for band_name in BAND_NAMES:
                stacked = np.stack([b[0][band_name] for b in buffer], axis=0)
                mean_grids[band_name] = np.mean(stacked, axis=0)

            window_start = buffer[0][1]
            window_end = buffer[-1][2]

            yield mean_grids, window_start, window_end
            buffer = buffer[stride:]

    # Handle remaining frames
    if len(buffer) > 0:
        mean_grids = {}
        for band_name in BAND_NAMES:
            stacked = np.stack([b[0][band_name] for b in buffer], axis=0)
            mean_grids[band_name] = np.mean(stacked, axis=0)

        window_start = buffer[0][1]
        window_end = buffer[-1][2]
        yield mean_grids, window_start, window_end


def render_frames(
    mip_frames: Iterator[tuple[np.ndarray, int, int]],
    n_std: float = 3.0,
    frame_size: int = 512,
) -> Iterator[np.ndarray]:
    """
    Render MIP frames with colormap and frame range annotation.

    Args:
        mip_frames: Iterator of (mip_grid, start_sample, end_sample) tuples
        n_std: Number of std deviations for colormap range
        frame_size: Output frame size in pixels (square)

    Yields:
        Rendered frame as numpy array of shape (frame_size, frame_size, 3), uint8 RGB
    """
    # Get Magma colormap
    magma = colormaps["magma"]

    for grid, start_sample, end_sample in mip_frames:
        # Normalize to [0, 1] for colormap (grid is already clipped to ±n_std)
        normalized = (grid + n_std) / (2 * n_std)

        # Apply colormap (returns RGBA float in [0, 1])
        colored = magma(normalized)[:, :, :3]  # Drop alpha channel

        # Convert to uint8
        frame = np.uint8(colored * 255)

        # Resize to target frame size using nearest neighbor
        frame = cv2.resize(
            frame, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST
        )

        # Add frame range text to top right corner
        # Convert RGB to BGR for OpenCV text rendering
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        text = f"{start_sample}-{end_sample}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)  # White in BGR

        # Get text size to position it correctly
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Position: top-right with 10px padding
        x = frame_size - text_width - 10
        y = text_height + 10

        # Add black background for better readability
        cv2.rectangle(
            frame_bgr,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            (0, 0, 0),  # Black background
            -1,
        )

        # Add text
        cv2.putText(
            frame_bgr,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Convert back to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        yield frame_rgb


def create_colorbar_image(
    width: int,
    height: int,
    n_std: float,
    cmap,
) -> np.ndarray:
    """
    Create a vertical colorbar image.

    Args:
        width: Width of colorbar in pixels
        height: Height of colorbar in pixels
        n_std: Number of std deviations for range
        cmap: Matplotlib colormap

    Returns:
        RGB uint8 array of shape (height, width, 3)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import io

    # Create figure with tight layout
    fig_height = height / 100  # inches at 100 dpi
    fig_width = width / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    fig.patch.set_facecolor("black")

    # Create colorbar
    norm = Normalize(vmin=-n_std, vmax=n_std)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation="vertical")
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.set_label("Amplitude (std)", color="white", fontsize=8)

    # Render to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black", bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)

    # Read image from buffer
    from PIL import Image

    img = Image.open(buf)
    colorbar_img = np.array(img.convert("RGB"))

    plt.close(fig)

    # Resize to exact dimensions
    colorbar_img = cv2.resize(colorbar_img, (width, height))

    return colorbar_img


def add_band_label(panel: np.ndarray, band_name: str) -> np.ndarray:
    """
    Add frequency band label to bottom of panel.

    Args:
        panel: RGB uint8 array
        band_name: Band name key

    Returns:
        Panel with label added
    """
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)

    text = BAND_LABELS[band_name]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Position at bottom center
    x = (panel.shape[1] - text_w) // 2
    y = panel.shape[0] - 8

    # Background rectangle
    cv2.rectangle(panel_bgr, (x - 3, y - text_h - 3), (x + text_w + 3, y + 3), (0, 0, 0), -1)
    cv2.putText(panel_bgr, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    return cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)


def render_frames_multipanel(
    mip_frames: Iterator[tuple[dict[str, np.ndarray], int, int]],
    n_std: float = 3.0,
    panel_size: int = 256,
    colorbar_width: int = 60,
) -> Iterator[np.ndarray]:
    """
    Render 2x2 grid of frequency bands with colorbar on right.

    Layout:
    +------------------+------------------+--------+
    | Theta/Alpha      | Beta             |        |
    | (4-12Hz)         | (12-30Hz)        | Color  |
    +------------------+------------------+  bar   |
    | Low Gamma        | High Gamma       |        |
    | (30-70Hz)        | (70-150Hz)       |        |
    +------------------+------------------+--------+

    Args:
        mip_frames: Iterator of (grids_dict, start_sample, end_sample) tuples
        n_std: Number of std deviations for colormap normalization
        panel_size: Size of each panel in pixels
        colorbar_width: Width of colorbar region in pixels

    Yields:
        Rendered frame as RGB uint8 array
    """
    magma = colormaps["magma"]

    # Pre-render colorbar once
    grid_height = panel_size * 2
    colorbar_img = create_colorbar_image(colorbar_width, grid_height, n_std, magma)

    # Total frame dimensions
    grid_width = panel_size * 2
    total_width = grid_width + colorbar_width
    total_height = grid_height

    # Panel positions: 2x2 grid
    # (band_name, row_offset, col_offset)
    positions = [
        ("theta_alpha", 0, 0),  # Top-left
        ("beta", 0, panel_size),  # Top-right
        ("low_gamma", panel_size, 0),  # Bottom-left
        ("high_gamma", panel_size, panel_size),  # Bottom-right
    ]

    for grids, start_sample, end_sample in mip_frames:
        # Create output frame with black background
        frame = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        for band_name, row_offset, col_offset in positions:
            grid = grids[band_name]

            # Normalize to [0, 1]
            normalized = (grid + n_std) / (2 * n_std)

            # Apply colormap
            colored = magma(normalized)[:, :, :3]
            panel = np.uint8(colored * 255)

            # Resize to panel size
            panel = cv2.resize(panel, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST)

            # Add band label
            panel = add_band_label(panel, band_name)

            # Place in frame
            frame[row_offset : row_offset + panel_size, col_offset : col_offset + panel_size] = (
                panel
            )

        # Add colorbar on right side
        frame[:, grid_width:] = colorbar_img

        # Add frame range text (top-right corner of the grid area)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        text = f"{start_sample}-{end_sample}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Position at top-right of the grid (before colorbar)
        x = grid_width - text_width - 10
        y = text_height + 10

        # Background rectangle
        cv2.rectangle(
            frame_bgr,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame_bgr, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        yield frame_rgb


# =============================================================================
# Video Export
# =============================================================================


def export_video(
    frames: Iterator[np.ndarray],
    output_path: Path,
    fps: float = 30.0,
    frame_width: int = 512,
    frame_height: int = 512,
    total_frames: int | None = None,
) -> None:
    """
    Export frames to video file.

    Args:
        frames: Iterator of frames (RGB uint8 arrays)
        output_path: Output video path (.mp4 or .avi)
        fps: Frame rate
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        total_frames: Total number of frames (for progress bar)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine codec from extension
    ext = output_path.suffix.lower()
    if ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        raise ValueError(f"Unsupported video format: {ext}. Use .mp4 or .avi")

    # Create video writer
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height),
        isColor=True,
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Writing video...", total=total_frames)

            for frame_rgb in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                progress.advance(task)
    finally:
        writer.release()


# =============================================================================
# Data Loading
# =============================================================================


def load_data(data_dir: Path) -> np.ndarray:
    """
    Load neural data from parquet file.

    Args:
        data_dir: Directory containing track2_data.parquet

    Returns:
        Neural data array of shape (n_samples, 1024)
    """
    data_path = data_dir / "track2_data.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    console.print(f"[dim]Loading data from {data_path}...[/dim]")
    data_df = pd.read_parquet(data_path)
    data = data_df.values.astype(np.float32)

    return data


# =============================================================================
# CLI
# =============================================================================


def print_header() -> None:
    """Print welcome header."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Neural Data Video Exporter[/bold cyan]\n"
            "[dim]Bad channel detection, imputation, and video export[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def print_stats(
    stats: dict,
    global_mean: float,
    global_std: float,
    detection_frames: int,
    stats_frames: int,
) -> None:
    """Print detection and normalization statistics."""
    table = Table(title="Bad Channel Detection", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Detection frames used", str(detection_frames))
    table.add_row("Dead channels", str(stats["n_dead"]))
    table.add_row("Saturated channels", str(stats["n_saturated"]))
    table.add_row("Total bad channels", str(stats["n_total_bad"]))
    table.add_row("Variance (min)", f"{stats['variance_stats']['min']:.2e}")
    table.add_row("Variance (median)", f"{stats['variance_stats']['median']:.2e}")
    table.add_row("Variance (max)", f"{stats['variance_stats']['max']:.2e}")

    console.print(table)
    console.print()

    if stats["dead_channels"]:
        console.print(f"[dim]Dead channels: {stats['dead_channels']}[/dim]")
    if stats["saturated_channels"]:
        console.print(f"[dim]Saturated channels: {stats['saturated_channels']}[/dim]")

    console.print()
    console.print(f"[dim]Normalization: mean={global_mean:.4f}, std={global_std:.4f}[/dim]")
    console.print()


@app.command()
def main(
    difficulty: str = typer.Option(
        "hard",
        "--difficulty",
        "-d",
        help="Dataset difficulty: super_easy, easy, medium, hard",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output video path (e.g., /store1/candy/bci/output.mp4)",
    ),
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir",
        help="Directory containing downloaded datasets",
    ),
    fps: float = typer.Option(
        30.0,
        "--fps",
        help="Output video frame rate",
    ),
    samples_per_frame: int = typer.Option(
        None,
        "--samples-per-frame",
        help="Number of samples to average per frame (default: fs/fps = 16 at 500Hz/30fps)",
    ),
    frame_size: int = typer.Option(
        512,
        "--frame-size",
        help="Output frame size in pixels (square)",
    ),
    variance_threshold: float = typer.Option(
        1e-6,
        "--variance-threshold",
        help="Variance threshold for bad channel detection",
    ),
    detection_frames: int = typer.Option(
        1000,
        "--detection-frames",
        help="Number of frames to use for bad channel detection",
    ),
    connectivity: int = typer.Option(
        8,
        "--connectivity",
        help="Neighbor connectivity for imputation (4 or 8)",
    ),
    n_std: float = typer.Option(
        3.0,
        "--n-std",
        help="Number of standard deviations for colormap range",
    ),
    stats_frames: int = typer.Option(
        1000,
        "--stats-frames",
        help="Number of frames to use for normalization statistics",
    ),
    mip_window: int = typer.Option(
        500,
        "--mip-window",
        help="Mean intensity projection window size (number of frames)",
    ),
    mip_stride: int = typer.Option(
        250,
        "--mip-stride",
        help="Mean intensity projection stride (half-overlap by default)",
    ),
    skip_imputation: bool = typer.Option(
        False,
        "--skip-imputation",
        help="Skip bad channel imputation",
    ),
    skip_artifact_removal: bool = typer.Option(
        False,
        "--skip-artifact-removal",
        help="Skip 60Hz and white noise artifact removal",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print detailed statistics",
    ),
    no_smoothing: bool = typer.Option(
        False,
        "--no-smoothing",
        help="Disable all temporal smoothing for full resolution (sets samples-per-frame=1, mip-window=1, mip-stride=1)",
    ),
) -> None:
    """
    Export neural data to video file with bad channel imputation.

    Example:
        python scripts/export_video.py -d hard -o /store1/candy/bci/hard.mp4
    """
    print_header()

    # Handle --no-smoothing flag
    if no_smoothing:
        samples_per_frame = 1
        mip_window = 1
        mip_stride = 1
        console.print("[yellow]No smoothing mode: full temporal resolution[/yellow]")
        console.print()

    # Validate inputs
    dataset_path = data_dir / difficulty
    if not dataset_path.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {dataset_path}")
        console.print()
        console.print("Download it with:")
        console.print(f"  [cyan]uv run python -m scripts.download {difficulty}[/cyan]")
        raise typer.Exit(code=1)

    if connectivity not in (4, 8):
        console.print("[red]Error:[/red] connectivity must be 4 or 8")
        raise typer.Exit(code=1)

    output_path = Path(output)
    if output_path.suffix.lower() not in (".mp4", ".avi"):
        console.print("[red]Error:[/red] Output must be .mp4 or .avi")
        raise typer.Exit(code=1)

    # Load data
    try:
        data = load_data(dataset_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    n_samples, n_channels = data.shape
    duration_s = n_samples / SAMPLING_RATE
    console.print(
        f"[green]✓[/green] Loaded {n_samples:,} samples "
        f"({duration_s:.1f}s) with {n_channels} channels"
    )
    console.print()

    # Detect bad channels
    console.print(f"[dim]Detecting bad channels using first {detection_frames} frames...[/dim]")
    bad_mask, detection_stats, should_skip_imputation = detect_bad_channels(
        data,
        variance_threshold=variance_threshold,
        fs=SAMPLING_RATE,
        target_fps=fps,
        detection_frames=detection_frames,
    )

    # Check if stuck value detection triggered skip
    if should_skip_imputation:
        console.print(
            f"[yellow]Skipping imputation: {detection_stats.get('skip_reason', 'stuck values detected')}[/yellow]"
        )
    # Impute bad channels
    elif not skip_imputation and detection_stats["n_total_bad"] > 0:
        console.print(
            f"[dim]Imputing {detection_stats['n_total_bad']} bad channels "
            f"using {connectivity}-connected neighbors...[/dim]"
        )
        data = impute_bad_channels(data, bad_mask, connectivity)
        console.print("[green]✓[/green] Bad channels imputed")
    elif skip_imputation:
        console.print("[yellow]Skipping imputation (user requested)[/yellow]")
    else:
        console.print("[green]✓[/green] No bad channels detected")

    console.print()

    # Artifact removal
    if not skip_artifact_removal:
        console.print("[dim]Detecting and removing artifacts...[/dim]")
        data, artifact_stats = remove_artifacts(data, fs=SAMPLING_RATE, verbose=verbose)
        if artifact_stats["line_noise_60hz"]["removed"] or artifact_stats["white_noise"]["removed"]:
            console.print("[green]✓[/green] Artifacts removed")
        else:
            console.print("[green]✓[/green] No significant artifacts detected")
    else:
        console.print("[yellow]Skipping artifact removal[/yellow]")
        artifact_stats = None

    console.print()

    # Frequency band decomposition
    console.print("[dim]Decomposing into frequency bands...[/dim]")
    bands_data = decompose_frequency_bands(data, fs=SAMPLING_RATE, verbose=verbose)
    console.print("[green]✓[/green] Frequency bands extracted")
    console.print()

    # Determine samples per frame (default: fs/fps)
    if samples_per_frame is None:
        samples_per_frame = int(SAMPLING_RATE / fps)

    console.print(f"[dim]Using {samples_per_frame} samples per frame ({samples_per_frame/SAMPLING_RATE*1000:.1f}ms)[/dim]")

    # Compute per-band normalization statistics
    console.print(f"[dim]Computing per-band normalization stats from first {stats_frames} frames...[/dim]")
    bands_stats = compute_normalization_stats_per_band(
        bands_data, fs=SAMPLING_RATE, target_fps=fps, n_frames=stats_frames,
        samples_per_frame=samples_per_frame
    )

    if verbose:
        for band_name, (median_val, std_above) in bands_stats.items():
            console.print(f"[dim]  {BAND_LABELS[band_name]}: median={median_val:.4f}, std_above={std_above:.4f}[/dim]")
        console.print()

        # Print detection stats
        print_stats(detection_stats, 0.0, 1.0, detection_frames, stats_frames)

    # Calculate frame counts
    raw_frames = n_samples // samples_per_frame

    # Calculate MIP output frames
    if raw_frames >= mip_window:
        mip_frames_count = ((raw_frames - mip_window) // mip_stride) + 1
        remaining = (raw_frames - mip_window) % mip_stride
        if remaining > 0:
            mip_frames_count += 1
    else:
        mip_frames_count = 1

    video_duration_s = mip_frames_count / fps

    # Calculate video dimensions for 2x2 panel layout + colorbar
    panel_size = frame_size // 2
    colorbar_width = 60
    video_width = (panel_size * 2) + colorbar_width
    video_height = panel_size * 2

    console.print(
        f"[dim]Generating {raw_frames:,} raw frames -> "
        f"{mip_frames_count:,} MIP frames ({video_duration_s:.1f}s video at {fps} FPS)...[/dim]"
    )
    console.print(f"[dim]Output resolution: {video_width}x{video_height} (4 bands + colorbar)[/dim]")
    console.print()

    # Generate pipeline: per-band frames -> per-band MIP -> multi-panel render
    raw_frame_iter = generate_frames_per_band(
        bands_data,
        bands_stats,
        fs=SAMPLING_RATE,
        target_fps=fps,
        n_std=n_std,
        samples_per_frame=samples_per_frame,
    )

    mip_frame_iter = apply_mean_intensity_projection_per_band(
        raw_frame_iter,
        window_size=mip_window,
        stride=mip_stride,
    )

    rendered_frame_iter = render_frames_multipanel(
        mip_frame_iter,
        n_std=n_std,
        panel_size=panel_size,
        colorbar_width=colorbar_width,
    )

    export_video(
        rendered_frame_iter,
        output_path=output_path,
        fps=fps,
        frame_width=video_width,
        frame_height=video_height,
        total_frames=mip_frames_count,
    )

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Video saved to:[/bold green]\n{output_path}",
            border_style="green",
        )
    )
    console.print()


if __name__ == "__main__":
    app()
