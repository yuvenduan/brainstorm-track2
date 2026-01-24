Load the parquet file → raw (float32) and t from index

Detect bad channels (dead/noisy) via per-channel variance thresholding

Zero out bad channels (so they don’t poison referencing/filters)

Streaming filter + feature extraction (chunked):

optional common-average referencing (median or mean across channels per time point)

notch at 60/120 Hz (if under Nyquist)

bandpass in 70–150 Hz

compute power and apply EMA smoothing to the power envelope

take log(power)

decimate by an integer factor while keeping exact sample indices

Per-channel z-score of extracted features

Remove slow drift per channel via an EMA baseline subtraction

Reshape (N, 1024) → (N, 32, 32) frames

Spatial denoise frames:

subtract broad blurred background

clip negatives to 0

lightly blur to smooth