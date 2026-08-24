"""
Shared sliding-window inference for the mammography model -- used by both
app.py (predicting on a user's upload) and train_mammography_model.py
(evaluating whole-image performance after training), so the two can't
drift apart and disagree about what the deployed app actually does.

The model is trained on lesion-centered crops, not whole mammograms (see
train_mammography_model.py's module docstring for why), so a whole
uploaded image is searched in overlapping windows at several scales rather
than resized and classified directly.

Aggregation is top-k mean, not a plain max. With this many overlapping
windows per image (several hundred, across scales), even a good per-window
classifier will produce at least one spuriously high score somewhere on a
genuinely normal image just by chance -- a classic multiple-comparisons
problem. A real lesion, by contrast, triggers elevated scores across
several overlapping neighboring windows, not a single isolated one.
Averaging the top few windows keeps sensitivity to a real, spatially
consistent finding while damping a one-off noisy spike.
"""

import cv2
import numpy as np

IMG_SIZE = 128
SCALES = (0.15, 0.25, 0.4, 0.6)
TOP_K = 5


def _windows(gray: np.ndarray):
    h, w = gray.shape
    short_side = min(h, w)
    for scale in SCALES:
        size = max(int(short_side * scale), 16)
        stride = max(size // 2, 8)
        for y0 in range(0, max(h - size, 0) + 1, stride):
            for x0 in range(0, max(w - size, 0) + 1, stride):
                yield x0, y0, size


def sliding_window_predict(gray: np.ndarray, model) -> tuple[float, np.ndarray]:
    """gray: HxW float32 array in [0, 1]. Returns (probability, best_crop)
    where best_crop is the single highest-scoring window as a 0-255 uint8
    HxW array, for display and Grad-CAM."""
    boxes = list(_windows(gray))
    patches = np.stack([
        cv2.resize(gray[y0:y0 + size, x0:x0 + size], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        for x0, y0, size in boxes
    ])
    probs = model.predict(patches[..., np.newaxis], verbose=0)[:, 0]

    k = min(TOP_K, len(probs))
    top_idx = np.argpartition(probs, -k)[-k:]
    aggregate_prob = float(probs[top_idx].mean())

    best = int(top_idx[np.argmax(probs[top_idx])])
    x0, y0, size = boxes[best]
    best_crop = np.uint8(gray[y0:y0 + size, x0:x0 + size] * 255)
    return aggregate_prob, best_crop
