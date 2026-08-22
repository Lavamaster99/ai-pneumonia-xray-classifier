"""
End-to-end sanity check: runs the exact same preprocess -> predict -> Grad-CAM
path that app.py uses, on two known-label sample images, and writes the
overlay images to examples/. Useful to re-run after any change to app.py or
gradcam.py to confirm nothing broke before touching the Streamlit UI.

Usage:
    python smoke_test.py
"""

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from app import preprocess, MODEL_PATH
from gradcam import find_last_conv_layer, make_gradcam_heatmap, overlay_heatmap

SAMPLES = [
    ("examples/sample_normal.png", "Normal"),
    ("examples/sample_pneumonia.png", "Pneumonia"),
]


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    last_conv = find_last_conv_layer(model)

    for path, expected in SAMPLES:
        img = Image.open(path)
        arr = preprocess(img)

        prob = float(model.predict(arr, verbose=0)[0, 0])
        predicted = "Pneumonia" if prob >= 0.5 else "Normal"

        heatmap = make_gradcam_heatmap(arr, model, last_conv)
        assert heatmap.max() > 0.9, f"Grad-CAM heatmap did not normalize correctly (max={heatmap.max():.4f})"

        display_img = np.uint8(arr[0, :, :, 0] * 255)
        overlay = overlay_heatmap(display_img, heatmap)
        out_path = path.replace("sample_", "overlay_")
        cv2.imwrite(out_path, overlay)

        status = "OK" if predicted == expected else "MISMATCH (may just be a model error, check metrics.json FPR/FNR)"
        print(f"{path}: expected={expected} predicted={predicted} prob={prob:.4f} -> {status}")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
