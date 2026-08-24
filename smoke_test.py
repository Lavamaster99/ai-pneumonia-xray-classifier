"""
End-to-end sanity check: runs the exact same preprocess -> predict -> Grad-CAM
path that app.py uses, on known-label sample images for all three models, and
writes the overlay images to examples/. Useful to re-run after any change to
app.py, gradcam.py, or mammo_window.py to confirm nothing broke before
touching the Streamlit UI.

Usage:
    python smoke_test.py
"""

import json
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from app import preprocess, MODES
from gradcam import find_last_conv_layer, make_gradcam_heatmap, overlay_heatmap
from mammo_window import sliding_window_predict

CASES = [
    ("pneumonia", "examples/sample_normal.png", "Normal"),
    ("pneumonia", "examples/sample_pneumonia.png", "Pneumonia"),
    ("breast", "examples/sample_benign.png", "Normal / Benign"),
    ("breast", "examples/sample_malignant.png", "Malignant"),
    ("mammography", "examples/sample_mammo_normal.png", "Normal / Benign"),
    ("mammography", "examples/sample_mammo_malignant.png", "Malignant"),
]


def main():
    models = {}
    for mode_key in MODES:
        model = tf.keras.models.load_model(MODES[mode_key]["model_path"])
        models[mode_key] = (model, find_last_conv_layer(model))

    for mode_key, path, expected in CASES:
        mode = MODES[mode_key]
        model, last_conv = models[mode_key]
        threshold = 0.5
        if os.path.exists(mode["metrics_path"]):
            with open(mode["metrics_path"]) as f:
                threshold = json.load(f).get("decision_threshold", 0.5)

        img = Image.open(path)

        if mode_key == "mammography":
            gray = np.asarray(img.convert("L")).astype("float32") / 255.0
            prob, best_crop = sliding_window_predict(gray, model)
            arr = preprocess(Image.fromarray(best_crop))
        else:
            arr = preprocess(img)
            prob = float(model.predict(arr, verbose=0)[0, 0])

        predicted = mode["positive_label"] if prob >= threshold else mode["negative_label"]

        heatmap = make_gradcam_heatmap(arr, model, last_conv)
        assert heatmap.max() > 0.9, f"Grad-CAM heatmap did not normalize correctly (max={heatmap.max():.4f})"

        display_img = np.uint8(arr[0, :, :, 0] * 255)
        overlay = overlay_heatmap(display_img, heatmap)
        out_path = path.replace("sample_", "overlay_")
        cv2.imwrite(out_path, overlay)

        status = "OK" if predicted == expected else "MISMATCH (may just be a model error, check metrics.json FPR/FNR)"
        print(f"[{mode_key}] {path}: expected={expected} predicted={predicted} prob={prob:.4f} (threshold={threshold:.3f}) -> {status}")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
