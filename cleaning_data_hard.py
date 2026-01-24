import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter

def _sos_zi_for_channels(sos, n_ch, dtype=np.float32):
    zi_base = signal.sosfilt_zi(sos).astype(dtype)        # (n_sections, 2)
    zi = np.repeat(zi_base[:, :, None], n_ch, axis=2)     # (n_sections, 2, n_ch)
    return zi

def build_sos_chain(fs, notch_freqs=(60,120), notch_Q=30.0, band=(70,150), bp_order=4, dtype=np.float32):
    sos_chain = []
    for f0 in notch_freqs:
        if f0 is None:
            continue
        if 0 < f0 < fs/2:
            b, a = signal.iirnotch(w0=f0, Q=notch_Q, fs=fs)
            sos_chain.append(signal.tf2sos(b, a).astype(dtype))

    lo, hi = band
    sos_bp = signal.butter(bp_order, [lo, hi], btype="band", fs=fs, output="sos").astype(dtype)
    sos_chain.append(sos_bp)
    return sos_chain

def extract_logpower_streaming_decimate(
    raw, *,
    fs=500,
    notch_freqs=(60,120),
    notch_Q=30.0,
    band=(70,150),
    bp_order=4,
    ema_tau_ms=100.0,
    downsample=10,
    chunk_len=20000,
    log_eps=1e-12,
    dtype=np.float32,
    do_car=True,
    car_use_median=True
):
    """
    raw: (T, C) float32
    Returns:
      feat_ds: (T_ds, C) float32 log-power features
      ds_idx:  (T_ds,) indices into original time axis
    """
    raw = np.asarray(raw)
    T, C = raw.shape
    ds = int(max(1, downsample))

    sos_chain = build_sos_chain(fs, notch_freqs, notch_Q, band, bp_order, dtype=dtype)
    zi_list = [_sos_zi_for_channels(sos, C, dtype=dtype) for sos in sos_chain]

    # EMA smoother on power envelope
    tau_s = float(ema_tau_ms) / 1000.0
    alpha = np.exp(-1.0 / (fs * tau_s)).astype(np.float32)
    b_ema = np.array([1.0 - alpha], dtype=np.float32)
    a_ema = np.array([1.0, -alpha], dtype=np.float32)
    zi_ema = np.zeros((1, C), dtype=np.float32)

    # output prealloc (upper bound)
    T_ds = (T + ds - 1) // ds
    feat_out = np.empty((T_ds, C), dtype=np.float32)
    idx_out  = np.empty((T_ds,), dtype=np.int64)
    out_ptr = 0

    for start in range(0, T, chunk_len):
        end = min(T, start + chunk_len)
        chunk = raw[start:end].astype(dtype, copy=False)

        # robust common-average reference (background/shared mode removal)
        if do_car:
            if car_use_median:
                ref = np.median(chunk, axis=1, keepdims=True).astype(dtype)
            else:
                ref = np.mean(chunk, axis=1, keepdims=True).astype(dtype)
            chunk = chunk - ref

        # notch + bandpass
        y = chunk
        for i, sos in enumerate(sos_chain):
            y, zi_list[i] = signal.sosfilt(sos, y, axis=0, zi=zi_list[i])

        # power + EMA smooth
        p = y * y
        p_smooth, zi_ema = signal.lfilter(b_ema, a_ema, p, axis=0, zi=zi_ema)

        # log-power
        lp = np.log(p_smooth + log_eps).astype(np.float32)

        # exact decimation indices
        offset = (-start) % ds
        keep = lp[offset::ds]
        n_keep = keep.shape[0]
        if n_keep:
            feat_out[out_ptr:out_ptr+n_keep] = keep
            idx_out[out_ptr:out_ptr+n_keep] = np.arange(start + offset, start + offset + n_keep*ds, ds, dtype=np.int64)
            out_ptr += n_keep

    return feat_out[:out_ptr], idx_out[:out_ptr]

def ema_baseline_subtract(X, fs_feat, tau_s=10.0):
    """
    Remove slow background per channel using an EMA baseline.
    X: (N,C)
    """
    alpha = np.exp(-1.0 / (fs_feat * tau_s)).astype(np.float32)
    b = np.array([1.0 - alpha], dtype=np.float32)
    a = np.array([1.0, -alpha], dtype=np.float32)
    z0 = np.zeros((1, X.shape[1]), dtype=np.float32)
    baseline, _ = signal.lfilter(b, a, X.astype(np.float32), axis=0, zi=z0)
    return X - baseline

def spatial_denoise(frames, sigma_bg=3.0, sigma_smooth=0.8):
    """
    frames: (N,32,32)
    - subtract broad spatial background blur
    - keep positive evidence
    - light smooth
    """
    bg = gaussian_filter(frames, sigma=(0, sigma_bg, sigma_bg))
    hp = frames - bg
    hp = np.maximum(hp, 0.0)
    hp = gaussian_filter(hp, sigma=(0, sigma_smooth, sigma_smooth))
    return hp


HARD_PATH = "data/hard/track2_data.parquet"   # change if needed

# Stronger settings for hard
BASELINE_TAU_S = 30.0   # was 10.0 for medium
SIGMA_BG = 5.0          # was 3.0 for medium
SIGMA_SMOOTH = 0.9      # slight smoothing
EMA_TAU_MS=150
df = pd.read_parquet(HARD_PATH)
t = df.index.values.astype(np.float64)
raw = df.to_numpy(dtype=np.float32)  # (T,1024)

print("Loaded HARD:", raw.shape)
assert raw.shape[1] == 1024

# crude bad-channel mask (helps if hard has dead/noisy channels)
var = raw.var(axis=0)
bad = (var < 1e-8) | (var > np.quantile(var, 0.999))  # you can make this stricter if needed
print("bad channels:", bad.sum())

# If you want to zero out bad channels before CAR, do it here:
raw2 = raw.copy()
raw2[:, bad] = 0.0

feat, ds_idx = extract_logpower_streaming_decimate(
    raw2,
    fs=FS,
    notch_freqs=NOTCH_FREQS,   # 60/120
    notch_Q=NOTCH_Q,
    band=BAND,                 # 70-150
    bp_order=BP_ORDER,
    ema_tau_ms=EMA_TAU_MS,
    downsample=DOWNSAMPLE,
    chunk_len=20000,
    do_car=True,
    car_use_median=True
)

t_ds = t[ds_idx]
fs_feat = FS / DOWNSAMPLE
print("Feature shape:", feat.shape, "feature fps:", fs_feat)

# z-score per channel
mu = feat.mean(axis=0, keepdims=True)
sd = feat.std(axis=0, keepdims=True) + 1e-6
feat_z = (feat - mu) / sd

# stronger slow baseline subtraction
feat_bg = ema_baseline_subtract(feat_z, fs_feat=fs_feat, tau_s=BASELINE_TAU_S)

# reshape + spatial denoise
frames_dn = feat_bg.reshape(-1, 32, 32).astype(np.float32)
frames_dn = spatial_denoise(frames_dn, sigma_bg=SIGMA_BG, sigma_smooth=SIGMA_SMOOTH)


from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np

display_stride = 2   # 50 Hz -> 25 Hz display
fps = 25

vid = frames_dn[::display_stride]
t_vid = t_ds[::display_stride]
Tv = vid.shape[0]

vmin, vmax = np.percentile(vid, [1, 99.7])

fig, ax = plt.subplots(figsize=(4,4))
im = ax.imshow(vid[0], vmin=vmin, vmax=vmax, origin="upper")
ax.axis("off")
title = ax.set_title(f"hard denoised | t={t_vid[0]:.3f}s")

def update(i):
    im.set_data(vid[i])
    title.set_text(f"hard denoised | t={t_vid[i]:.3f}s | frame {i}/{Tv-1}")
    return (im, title)

ani = animation.FuncAnimation(fig, update, frames=Tv, interval=1000/fps, blit=True)
plt.close(fig)

HTML(ani.to_jshtml())