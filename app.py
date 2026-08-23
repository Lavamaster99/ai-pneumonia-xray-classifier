"""
Streamlit dashboard for the pneumonia chest-X-ray CNN.

Run with:
    streamlit run app.py
"""

import base64
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

# ---------------------------------------------------------------------------
# Icons -- simple stroke-based line icons (hand-drawn, 24x24), not emoji.
# ---------------------------------------------------------------------------
_ICONS = {
    "lungs": '<path d="M9 3v7.5c0 1-.5 1.8-1.3 2.4L5 15c-1.2.9-2 2.4-2 4v1.5A1.5 1.5 0 0 0 4.5 22c1.8 0 3.5-1 3.5-3.5V13m4-10v7.5c0 1 .5 1.8 1.3 2.4L19 15c1.2.9 2 2.4 2 4v1.5a1.5 1.5 0 0 1-1.5 1.5c-1.8 0-3.5-1-3.5-3.5V13M9 3h6"/>',
    "shield": '<path d="M12 22c5-1.5 8-5.5 8-11V5l-8-3-8 3v6c0 5.5 3 9.5 8 11Z"/><path d="m9 12 2 2 4-4"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "alert": '<path d="M12 2 1 21h22L12 2Z"/><path d="M12 9v5"/><path d="M12 17.5h.01"/>',
    "upload": '<path d="M12 16V4"/><path d="m6 10 6-6 6 6"/><path d="M4 18v1a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-1"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "flask": '<path d="M9 2v6L4 18a2 2 0 0 0 1.8 3h12.4a2 2 0 0 0 1.8-3L15 8V2"/><path d="M9 2h6"/><path d="M8.5 13h7"/>',
    "arrow": '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
    "github": '<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.9c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.2 4.2 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12.3 12.3 0 0 0-6.2 0C7.1 3.3 6 3.6 6 3.6a4.2 4.2 0 0 0-.1 3.2A4.6 4.6 0 0 0 4.6 10c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V22"/>',
}


def icon(name: str, size: int = 18, color: str = "#5B6472", stroke: float = 2) -> str:
    # Streamlit's HTML sanitizer strips raw inline <svg> elements (a
    # deliberate security measure -- SVG can carry scripts/foreignObject
    # content). Base64-encoding it into an <img src="data:..."> sidesteps
    # that entirely, since it's just an opaque image reference at that
    # point, not embedded markup. The tradeoff: no more `currentColor` --
    # each call site has to pass its actual intended color explicitly,
    # since a data-URI image can't inherit CSS `color` from its context.
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>'
    )
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{size}" height="{size}" style="display:inline-block;vertical-align:middle">'


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


# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------
st.html(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
      :root {
        --ink: #151A21;
        --ink-soft: #5B6472;
        --ink-faint: #99A1AC;
        --accent: #3B4CC0;
        --accent-ink: #29358A;
        --accent-soft: #EAECFA;
        --good: #1B8F5D;
        --good-soft: #E3F5EC;
        --bad: #D6455A;
        --bad-soft: #FBE7EA;
        --line: #E6E8EC;
        --panel: #FFFFFF;
        --bg: #FAFAFB;
        --shadow: 0 1px 2px rgba(15,20,30,.04), 0 10px 28px -16px rgba(15,20,30,.14);
        --radius: 16px;
      }

      html, body, [class*="css"], .stApp { font-family: 'Manrope', -apple-system, sans-serif !important; }
      .stApp { background: var(--bg); }

      /* Hide default Streamlit chrome for a cleaner, product-like frame */
      #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
      .block-container { padding-top: 1.6rem; max-width: 1180px; }

      h1, h2, h3 { color: var(--ink) !important; font-weight: 700 !important; letter-spacing: -0.01em; }

      /* ---- App header ---- */
      .app-header { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 6px; flex-wrap: wrap; }
      .app-header .id { display: flex; align-items: center; gap: 12px; }
      .app-header .mark {
        width: 42px; height: 42px; border-radius: 12px; background: var(--ink); color: #fff;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
      }
      .app-header .name { font-weight: 800; font-size: 17px; color: var(--ink); letter-spacing: -0.01em; }
      .app-header .tag { font-size: 12.5px; color: var(--ink-faint); }
      .gh-link {
        display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px;
        border-radius: 999px; background: var(--panel); border: 1px solid var(--line);
        box-shadow: var(--shadow); font-size: 13px; font-weight: 700; color: var(--ink-soft) !important;
      }
      .lede { color: var(--ink-soft); font-size: 14.5px; max-width: 68ch; margin: 10px 0 20px 0; line-height: 1.6; }

      /* ---- Warning banner ---- */
      .banner-warn {
        display: flex; gap: 12px; align-items: flex-start; padding: 14px 18px; border-radius: var(--radius);
        background: var(--bad-soft); border: 1px solid #F3C6CD; margin-bottom: 26px;
      }
      .banner-warn .ic { color: var(--bad); flex-shrink: 0; margin-top: 1px; }
      .banner-warn b { color: var(--ink); } .banner-warn { color: var(--ink-soft); font-size: 13.5px; line-height: 1.55; }

      /* ---- Sidebar ---- */
      [data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
      .sb-title { display: flex; align-items: center; gap: 9px; font-weight: 800; font-size: 13.5px; color: var(--ink); text-transform: uppercase; letter-spacing: .04em; margin-bottom: 4px; }
      .sb-title svg { color: var(--accent); }
      .sb-sub { font-size: 12px; color: var(--ink-faint); margin-bottom: 18px; }

      .mcard { display: flex; align-items: center; gap: 12px; padding: 13px 14px; border-radius: 13px; border: 1px solid var(--line); background: var(--bg); margin-bottom: 10px; }
      .mcard .ic { width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; background: var(--accent-soft); color: var(--accent-ink); }
      .mcard .val { font-weight: 800; font-size: 17px; color: var(--ink); line-height: 1.1; }
      .mcard .lbl { font-size: 11.5px; color: var(--ink-faint); margin-top: 1px; }

      .sb-note { font-size: 12px; color: var(--ink-faint); margin-top: 10px; }
      .sb-steps { font-size: 13px; color: var(--ink-soft); line-height: 1.8; padding-left: 2px; }

      /* ---- File uploader ---- */
      [data-testid="stFileUploaderDropzone"] {
        background: var(--panel) !important; border: 1.5px dashed var(--line-strong, #D3D8E0) !important;
        border-radius: var(--radius) !important;
      }
      [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }

      /* ---- Buttons ---- */
      .stButton > button, [data-testid="stDownloadButton"] button {
        border-radius: 999px !important; border: 1px solid var(--line) !important;
        font-weight: 700 !important; box-shadow: var(--shadow) !important;
      }

      /* ---- Result card ---- */
      .result-card { border-radius: var(--radius); border: 1px solid var(--line); background: var(--panel); box-shadow: var(--shadow); padding: 18px; margin-top: 4px; }
      .result-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
      .verdict { display: flex; align-items: center; gap: 12px; }
      .verdict .ic { width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
      .verdict.pos .ic { background: var(--bad-soft); color: var(--bad); }
      .verdict.neg .ic { background: var(--good-soft); color: var(--good); }
      .verdict .t { font-weight: 800; font-size: 19px; color: var(--ink); }
      .verdict .s { font-size: 12.5px; color: var(--ink-faint); }
      .conf-chip { font-weight: 800; font-size: 13px; padding: 8px 16px; border-radius: 999px; }
      .verdict.pos ~ .conf-chip, .conf-chip.pos { background: var(--bad-soft); color: var(--bad); }
      .conf-chip.neg { background: var(--good-soft); color: var(--good); }

      .bar-track { height: 8px; border-radius: 999px; background: var(--line); margin-top: 16px; overflow: hidden; }
      .bar-fill { height: 100%; border-radius: 999px; background: var(--accent); }
      .bar-caption { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--ink-faint); margin-top: 6px; }

      .img-card { border-radius: 14px; overflow: hidden; border: 1px solid var(--line); background: var(--panel); }
      .img-card .cap { padding: 10px 14px; font-size: 12.5px; font-weight: 700; color: var(--ink-soft); border-top: 1px solid var(--line); display: flex; align-items: center; gap: 8px; }
      .img-card .cap svg { color: var(--accent); }
    </style>
    """
)

st.html(
    f"""
    <div class="app-header">
      <div class="id">
        <div class="mark">{icon('lungs', 22, '#FFFFFF')}</div>
        <div>
          <div class="name">Chest X-Ray Pneumonia Screener</div>
          <div class="tag">Computational medicine &middot; CNN + Grad-CAM</div>
        </div>
      </div>
      <a class="gh-link" href="https://github.com/Lavamaster99/ai-pneumonia-xray-classifier" target="_blank">
        {icon('github', 16, '#5B6472')} View source
      </a>
    </div>
    <p class="lede">
      A convolutional neural network trained on the PneumoniaMNIST benchmark (pediatric chest X-rays,
      Kermany et al. / MedMNIST v2) with Grad-CAM explainability, so every prediction shows its work.
    </p>
    <div class="banner-warn">
      <span class="ic">{icon('alert', 18, '#D6455A')}</span>
      <span><b>Not a medical device.</b> This is a student research/engineering demonstration, not a
      clinically validated diagnostic tool. Do not use it to make real medical decisions.</span>
    </div>
    """
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
    st.html(
        f"""<div class="sb-title">{icon('activity', 15, '#3B4CC0')}Model performance</div>
        <div class="sb-sub">Measured on held-out test data, not estimated.</div>"""
    )
    if metrics:
        rows = [
            ("target", f"{metrics['accuracy']*100:.1f}%", "Test accuracy"),
            ("shield", f"{metrics['recall_sensitivity']*100:.1f}%", "Sensitivity (recall)"),
            ("activity", f"{metrics['auc']:.3f}", "AUC"),
            ("alert", f"{metrics['false_negative_rate']*100:.1f}%", "False-negative rate"),
        ]
        for ic, val, lbl in rows:
            st.html(
                f"""<div class="mcard"><div class="ic">{icon(ic, 16, '#29358A')}</div>
                <div><div class="val">{val}</div><div class="lbl">{lbl}</div></div></div>"""
            )
        st.html(
            f'<div class="sb-note">Evaluated on {metrics["test_set_size"]} held-out test images.</div>'
        )
        with st.expander("Full metrics JSON"):
            st.json(metrics)
    else:
        st.info("Run train_model.py to generate reports/metrics.json")

    st.divider()
    st.html(
        f"""<div class="sb-title">{icon('flask', 15, '#3B4CC0')}How it works</div>"""
    )
    st.html(
        """<div class="sb-steps">
        1. Upload a chest X-ray image<br>
        2. The CNN outputs a pneumonia probability<br>
        3. Grad-CAM shows what it looked at
        </div>"""
    )

uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_img = Image.open(uploaded)
    img_array = preprocess(pil_img)

    prob = float(model.predict(img_array, verbose=0)[0, 0])
    label = "Pneumonia" if prob >= 0.5 else "Normal"
    confidence = prob if prob >= 0.5 else 1 - prob
    is_pos = label == "Pneumonia"

    heatmap = make_gradcam_heatmap(img_array, model, last_conv)
    display_img = np.uint8(img_array[0, :, :, 0] * 255)
    overlay = overlay_heatmap(display_img, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, use_container_width=True)
        st.html(
            f'<div class="img-card"><div class="cap">{icon("upload", 14, "#3B4CC0")}Uploaded X-ray</div></div>'
        )
    with col2:
        st.image(overlay, channels="BGR", use_container_width=True)
        st.html(
            f'<div class="img-card"><div class="cap">{icon("target", 14, "#3B4CC0")}Grad-CAM &mdash; what the model looked at</div></div>'
        )

    verdict_class = "pos" if is_pos else "neg"
    chip_class = "pos" if is_pos else "neg"
    verdict_icon = "alert" if is_pos else "check"
    verdict_color = "#D6455A" if is_pos else "#1B8F5D"

    st.html(
        f"""
        <div class="result-card">
          <div class="result-head">
            <div class="verdict {verdict_class}">
              <div class="ic">{icon(verdict_icon, 22, verdict_color)}</div>
              <div><div class="t">{label}</div><div class="s">Model prediction</div></div>
            </div>
            <div class="conf-chip {chip_class}">{confidence*100:.1f}% confidence</div>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:{prob*100:.1f}%"></div></div>
          <div class="bar-caption"><span>Normal</span><span>Pneumonia probability: {prob*100:.1f}%</span><span>Pneumonia</span></div>
        </div>
        """
    )
else:
    st.info("Upload a chest X-ray image above to run a prediction.")
    st.markdown(
        "Don't have one handy? The [PneumoniaMNIST test set on Zenodo]"
        "(https://zenodo.org/records/10519652) or any public chest X-ray "
        "sample image will work."
    )
