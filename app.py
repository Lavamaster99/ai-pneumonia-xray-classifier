"""
Streamlit dashboard for the pneumonia chest-X-ray CNN.

Run with:
    streamlit run app.py
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from gradcam import make_gradcam_heatmap, overlay_heatmap, find_last_conv_layer

IMG_SIZE = 128
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "pneumonia_cnn.keras")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "reports", "metrics.json")

st.set_page_config(page_title="Chest X-Ray Pneumonia Screener", page_icon="🫁", layout="wide")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def preprocess(pil_img: Image.Image) -> np.ndarray:
    gray = pil_img.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(gray).astype("float32") / 255.0
    arr = arr[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
    return arr


st.title("🫁 Chest X-Ray Pneumonia Screening Tool")
st.caption(
    "A convolutional neural network trained on the PneumoniaMNIST benchmark "
    "(pediatric chest X-rays, Kermany et al. / MedMNIST v2) with Grad-CAM "
    "explainability. Built as an educational biomedical-engineering project."
)

st.warning(
    "⚠️ **Not a medical device.** This is a student research/engineering "
    "demonstration, not a clinically validated diagnostic tool. Do not use it "
    "to make real medical decisions.",
    icon="⚠️",
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "No trained model found yet. Run `python train_model.py` first to "
        "train and save `model/pneumonia_cnn.keras`."
    )
    st.stop()

model = load_model()
metrics = load_metrics()
last_conv = find_last_conv_layer(model)

with st.sidebar:
    st.header("Model performance")
    if metrics:
        st.metric("Test accuracy", f"{metrics['accuracy']*100:.1f}%")
        st.metric("Sensitivity (recall)", f"{metrics['recall_sensitivity']*100:.1f}%")
        st.metric("AUC", f"{metrics['auc']:.3f}")
        st.metric("False-negative rate", f"{metrics['false_negative_rate']*100:.1f}%",
                   help="Fraction of true pneumonia cases the model missed -- the "
                        "clinically important error to minimize.")
        st.caption(f"Evaluated on {metrics['test_set_size']} held-out test images.")
        with st.expander("Full metrics JSON"):
            st.json(metrics)
    else:
        st.info("Run train_model.py to generate reports/metrics.json")

    st.divider()
    st.header("How it works")
    st.markdown(
        "1. Upload a chest X-ray image (JPG/PNG).\n"
        "2. The CNN outputs a pneumonia probability.\n"
        "3. Grad-CAM highlights the image regions that most influenced the "
        "prediction, so the output isn't a black box."
    )

uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_img = Image.open(uploaded)
    img_array = preprocess(pil_img)

    prob = float(model.predict(img_array, verbose=0)[0, 0])
    label = "Pneumonia" if prob >= 0.5 else "Normal"
    confidence = prob if prob >= 0.5 else 1 - prob

    heatmap = make_gradcam_heatmap(img_array, model, last_conv)
    display_img = np.uint8(img_array[0, :, :, 0] * 255)
    overlay = overlay_heatmap(display_img, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded X-ray")
        st.image(pil_img, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM: what the model looked at")
        st.image(overlay, channels="BGR", use_container_width=True)

    st.divider()
    if label == "Pneumonia":
        st.error(f"### Prediction: {label}  ({confidence*100:.1f}% confidence)")
    else:
        st.success(f"### Prediction: {label}  ({confidence*100:.1f}% confidence)")

    st.progress(prob, text=f"Pneumonia probability: {prob*100:.1f}%")
else:
    st.info("Upload a chest X-ray image above to run a prediction.")
    st.markdown(
        "Don't have one handy? The [PneumoniaMNIST test set on Zenodo]"
        "(https://zenodo.org/records/10519652) or any public chest X-ray "
        "sample image will work."
    )
