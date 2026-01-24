"""
Neural-only “Compass” (no ground truth):
- Take 1024 channels -> 32x32 heatmap per frame
- Track hotspot (centroid of strongest blob)
- Aggregate last 20 frames for stability
- Draw arrow from array center (15.5, 15.5) toward hotspot
- Export a GIF/MP4
"""

import os
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter, label, find_objects

# =========================
# SETTINGS
# =========================
NEURAL_PARQUET = "brainstorm-track2-public/data/medium/track2_data.parquet"
OUT_FILE = "compass_neural_only.gif" 

FS = 500                 
H, W = 32, 32
N_CH = H * W

WIN_MS  = 100            
STEP_MS = 100            
FEATURE = "rms"          

# Detection Params
SPATIAL_SMOOTH_SIGMA = 0.9
TOP_PCT = 98.5           
MIN_AREA = 3             
MAX_BLOBS = 5            

# Temporal aggregation
AGG_FRAMES = 20          

# UI
FPS = int(1000/STEP_MS)
FIGSIZE = (7.5, 7.5)
CMAP = "inferno"
LOCK_DIST = 2.5          

def direction_word(dx, dy):
    ang = np.degrees(np.arctan2(dy, dx))
    if -22.5 <= ang < 22.5: return "RIGHT"
    if 22.5 <= ang < 67.5: return "UP-RIGHT"
    if 67.5 <= ang < 112.5: return "UP"
    if 112.5 <= ang < 157.5: return "UP-LEFT"
    if ang >= 157.5 or ang < -157.5: return "LEFT"
    if -157.5 <= ang < -112.5: return "DOWN-LEFT"
    if -112.5 <= ang < -67.5: return "DOWN"
    return "DOWN-RIGHT"

def main():
    # =========================
    # 1) DATA LOADING
    # =========================
    if not os.path.exists(NEURAL_PARQUET):
        print(f"Error: {NEURAL_PARQUET} not found.")
        return

    print(f"Loading {NEURAL_PARQUET}...")
    dfX = pd.read_parquet(NEURAL_PARQUET)

    num_cols = dfX.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > N_CH:
        electrode_cols = [c for c in num_cols if 'channel' in c.lower() or c.isdigit()]
        if not electrode_cols: electrode_cols = num_cols[:N_CH]
    else:
        electrode_cols = num_cols

    X = dfX[electrode_cols[:N_CH]].to_numpy(dtype=np.float32)
    T, C = X.shape
    Xg = X.reshape(T, H, W)

    # =========================
    # 2) BUILD FRAMES
    # =========================
    win, step = int(FS * WIN_MS / 1000), int(FS * STEP_MS / 1000)
    frames = []
    LIMIT_T = min(T, 5000) 
    for start in range(0, LIMIT_T - win + 1, step):
        chunk = Xg[start:start+win]
        frm = np.sqrt(np.mean(chunk*chunk, axis=0)) if FEATURE == "rms" else np.mean(np.abs(chunk), axis=0)
        frames.append(frm.astype(np.float32))

    frames = np.stack(frames, axis=0)
    N = len(frames)
    lo, hi = np.percentile(frames, [2, 98])
    frames01 = np.clip((frames - lo) / (hi - lo + 1e-6), 0, 1)

    # =========================
    # 3) PROCESSING LOGIC
    # =========================
    center_x, center_y = 15.5, 15.5
    buf = deque(maxlen=AGG_FRAMES)
    agg_maps, hotspots, instructions = [], [], []

    for i in range(N):
        buf.append(frames01[i])
        agg = np.mean(buf, axis=0)
        hm = gaussian_filter(agg, sigma=SPATIAL_SMOOTH_SIGMA)
        mask = hm >= np.percentile(hm, TOP_PCT)
        lab, nlab = label(mask)
        
        cx, cy, instr = center_x, center_y, "STABLE"
        if nlab > 0:
            sizes = [np.sum(lab == (j+1)) for j in range(nlab)]
            big_idx = np.argmax(sizes)
            yy, xx = np.where(lab == (big_idx + 1))
            w = hm[yy, xx]
            cx, cy = (xx * w).sum() / w.sum(), (yy * w).sum() / w.sum()
            dx, dy = (cx - center_x), (cy - center_y)
            dist = np.sqrt(dx**2 + dy**2)
            instr = "LOCKED" if dist < LOCK_DIST else f"MOVE {direction_word(dx, dy)}"

        agg_maps.append(agg)
        hotspots.append((cx, cy))
        instructions.append(instr)

    # =========================
    # 4) ANIMATION & SAVING
    # =========================
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(agg_maps[0], origin="lower", cmap=CMAP, vmin=0, vmax=1, extent=[-0.5, 31.5, -0.5, 31.5])
    ax.set_title("Neural Compass (20-Frame Smooth)")
    ax.plot([center_x], [center_y], 'wo', alpha=0.5)
    hot_plot, = ax.plot([], [], 'co', markersize=10)
    arrow = ax.quiver([center_x], [center_y], [0], [0], angles='xy', scale_units='xy', scale=1, color='cyan')
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white", weight="bold", bbox=dict(facecolor="black", alpha=0.7))

    def update(i):
        im.set_data(agg_maps[i])
        cx, cy = hotspots[i]
        hot_plot.set_data([cx], [cy])
        arrow.set_UVC([cx - center_x], [cy - center_y])
        txt.set_text(f"t = {i*STEP_MS/1000:.1f}s\nGUIDANCE: {instructions[i]}")
        return [im, hot_plot, arrow, txt]

    anim = FuncAnimation(fig, update, frames=N, interval=1000/FPS, blit=True)

    try:
        from matplotlib.animation import FFMpegWriter
        anim.save("compass.mp4", writer=FFMpegWriter(fps=FPS))
        print("✓ Saved as compass.mp4")
    except Exception:
        anim.save(OUT_FILE, writer=PillowWriter(fps=FPS))
        print(f"✓ Saved as {OUT_FILE}")

    plt.close()

if __name__ == "__main__":
    main()
