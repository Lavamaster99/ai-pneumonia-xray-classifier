<div align="center">

# Medical Imaging Screening Tool

**Two CNNs, one dashboard: chest X-ray pneumonia and breast ultrasound malignancy screening, each explaining itself with Grad-CAM — not just a training script.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-3DA639)](LICENSE)
[![Not a medical device](https://img.shields.io/badge/not_a_medical_device-B3453A)](#limitations--ethical-disclosure)

</div>

<br>

<table>
<tr>
<td width="50%" align="center"><b>Chest X-ray input</b><br><img src="examples/sample_pneumonia.png" width="220"></td>
<td width="50%" align="center"><b>Grad-CAM — where it looked</b><br><img src="examples/overlay_pneumonia.png" width="220"></td>
</tr>
<tr>
<td width="50%" align="center"><b>Breast ultrasound input</b><br><img src="examples/sample_malignant.png" width="220"></td>
<td width="50%" align="center"><b>Grad-CAM — where it looked</b><br><img src="examples/overlay_malignant.png" width="220"></td>
</tr>
</table>

<p align="center"><i>Real output from this repo's own trained models — pneumonia predicted <b>Pneumonia</b> at 100% confidence; the ultrasound sample predicted <b>Malignant</b>.</i></p>

## Table of contents

- [Quick start](#quick-start)
- [What it does](#what-it-does)
- [Results](#results)
- [Datasets](#datasets)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions-worth-noting)
- [Limitations / ethical disclosure](#limitations--ethical-disclosure)
- [Attribution](#attribution)

## Quick start

Both trained models ship in this repo — no training, no waiting, just run it.

### One-click install

1. Download **[`Medical-Imaging-Screening-Tool.zip`](../../releases/latest)** from the latest release — this is the whole app, everything's inside.
2. Extract it, then run the setup script for your OS inside — **just this once.**

| OS | Run this |
|---|---|
| Windows | Double-click **`setup_and_run.bat`** |
| macOS | Double-click **`setup_and_run.sh`** in Terminal, or run `./setup_and_run.sh` (first time: `chmod +x setup_and_run.sh` if it won't execute) |
| Linux | Run `./setup_and_run.sh` in a terminal (`chmod +x setup_and_run.sh` first if needed) |

It checks for Python (installing it via `winget` on Windows if missing; on macOS/Linux it points you at Homebrew or your package manager instead), installs the app under your user profile, opens it in its own window, and adds a launcher (Desktop shortcut on Windows/macOS, applications-menu entry on Linux). Once it's done, delete the extracted folder — the real install lives elsewhere from here on. Every time after that, just use the launcher: no terminal, no re-running setup.

### Any OS: manual setup

```bash
git clone https://github.com/Lavamaster99/ai-pneumonia-xray-classifier.git
cd ai-pneumonia-xray-classifier

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

No git? Click **`<> Code`** → **Download ZIP** at the top of this page instead.

Once it's running, use the switch at the top to pick a mode, then try the matching sample: `examples/sample_pneumonia.png` / `sample_normal.png` for chest X-ray, `examples/sample_benign.png` / `sample_malignant.png` for breast ultrasound.

<details>
<summary><b>Stuck?</b> Common issues</summary>
<br>

- **"python is not recognized"** (Windows) — Python wasn't added to PATH during install; reinstall and tick that box, or use `py` instead of `python` above.
- **macOS says the script is from an "unidentified developer"** — right-click the script/launcher → Open → confirm. Only needed the first time; it's an unsigned script, not a bug.
- **Linux desktop icon won't launch on first click** — most desktop environments require right-click → "Allow Launching" (or similar) the first time a new `.desktop` file runs. Standard Linux behavior, not specific to this app.
- **Port 8501 already in use** — another Streamlit app is running; close it, or run `streamlit run app.py --server.port 8502`.
- **Install is slow** — normal on first run; TensorFlow alone is ~500MB.

</details>

## What it does

| | |
|---|---|
| **Two CNN classifiers, one dashboard** | A mode switch at the top swaps between a chest X-ray pneumonia model and a breast ultrasound malignancy model — each with its own network, metrics, and upload flow, sharing one interface. |
| **Grad-CAM heatmaps** | Every prediction, in either mode, shows *which pixels* the network actually attended to ([Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)) — the difference between a bare number and an inspectable result. |
| **Web dashboard** | Upload an image in `app.py` and see the prediction, confidence, heatmap, and the active model's own validated stats side by side. |

## Results

### Chest X-ray — pneumonia

Test set, 624 held-out images (`reports/metrics.json`):

| Metric | Value |
|---|---|
| Accuracy | **93.3%** |
| Precision | 91.4% |
| Sensitivity (recall) | **98.5%** |
| AUC | 0.980 |
| False-negative rate | **1.5%** (6 of 390 pneumonia cases missed) |
| False-positive rate | 15.4% (36 of 234 normal cases flagged) |

<table>
<tr>
<td align="center"><img src="reports/confusion_matrix.png" width="240"></td>
<td align="center"><img src="reports/roc_curve.png" width="260"></td>
</tr>
</table>

The false-negative rate is reported on its own, separately from accuracy, because in a screening context a missed pneumonia case is a qualitatively worse error than a false alarm — accuracy alone would hide that.

**A real bug caught during verification, not hidden:** the first Grad-CAM pass returned nearly blank heatmaps on confident predictions. Root cause — gradients were computed against the post-sigmoid probability, which saturates near 0/1 and kills the gradient for exactly the most confident cases, compounded by a normalization epsilon larger than the signal itself. Fixed by differentiating against the recovered pre-sigmoid logit and switching to `divide_no_nan`. See `gradcam.py` for the fix, with the reasoning left in as a comment.

### Breast ultrasound — malignancy

Test set, 156 held-out images (`reports/breast_metrics.json`):

| Metric | Value |
|---|---|
| Accuracy | 68.6% |
| Precision | 45.5% |
| Sensitivity (recall) | 83.3% |
| AUC | 0.849 |
| False-negative rate | 16.7% (7 of 42 malignant cases missed) |
| False-positive rate | 36.8% (42 of 114 normal/benign cases flagged) |

<table>
<tr>
<td align="center"><img src="reports/breast_confusion_matrix.png" width="240"></td>
<td align="center"><img src="reports/breast_roc_curve.png" width="260"></td>
</tr>
</table>

Reported as-is, not rounded up: BreastMNIST has 780 images total against PneumoniaMNIST's 5,856, and it shows. This model is deliberately kept in the same app as the stronger pneumonia one to be honest about that gap side by side, rather than only shipping the result that looks good. See [Limitations](#limitations--ethical-disclosure).

**A real bug caught during verification, not hidden:** `medmnist.INFO` was checked directly rather than assumed to share the pneumonia project's label convention — good thing, since BreastMNIST's native labels are the *opposite* (`0` = malignant, `1` = normal/benign). Flipped in `train_breast_model.py` so `1` = malignant consistently across this codebase, matching the pneumonia model's "positive = the thing you don't want to miss" convention.

## Datasets

Both from [MedMNIST v2](https://medmnist.com/), a standardized benchmark suite — chosen so results are reproducible against a known baseline rather than a private train/test split.

| | Pneumonia | Breast ultrasound |
|---|---|---|
| Source | Kermany et al., pediatric chest X-rays (Guangzhou Women and Children's Medical Center) | Al-Dhabyani et al., breast ultrasound (Baheya Hospital, Cairo) |
| Size | 5,856 images | 780 images |
| Split | 4,708 / 524 / 624 (train/val/test) | 546 / 78 / 156 |
| License | CC BY 4.0 | CC BY 4.0 |
| Class balance | ~3:1 pneumonia:normal, handled with class weighting | ~2.4:1 benign:malignant, handled with class weighting |

Both download automatically on first training run — no Kaggle account or manual wrangling.

## Project structure

```
setup_and_run.bat      # first-time Windows setup: installs everything, creates a Desktop shortcut
setup_and_run.sh       # first-time macOS/Linux setup: same, via a Desktop .command / .desktop launcher
run.bat / run.sh       # lightweight launchers used by that shortcut for every run after the first
train_model.py         # builds, trains, and evaluates the pneumonia CNN; writes model + reports
train_breast_model.py  # same, for the breast ultrasound CNN
gradcam.py              # Grad-CAM heatmap implementation, shared by both models
app.py                 # Streamlit dashboard: mode switch -> upload -> prediction -> heatmap
smoke_test.py           # end-to-end pipeline sanity check
requirements.txt
model/                 # both trained models -- already committed, no training needed
reports/               # metrics.json + breast_metrics.json + PNG plots -- already committed
examples/              # sample images (both modes) + their Grad-CAM outputs, for instant testing
data/                  # cached dataset downloads (only appears if you retrain)
```

<details>
<summary><b>Retraining from scratch</b> (optional — not needed to use the app)</summary>
<br>

```bash
python train_model.py          # pneumonia -- ~10-20 min on a laptop CPU
python train_breast_model.py   # breast ultrasound -- faster, much smaller dataset
```

Both download their dataset automatically and overwrite the matching files in `model/` and `reports/`. Results will vary slightly from the numbers above since training isn't perfectly deterministic.

</details>

## Design decisions worth noting

- **One dashboard, not two apps**: the mode switch keeps both screening tasks under one interface, one design system, one install — closer to how a real clinical screening tool would be packaged than two disconnected demos.
- **128×128 resolution**, not the raw 28×28 MedMNIST default: large enough for the Grad-CAM overlay to be visually meaningful on real anatomy, small enough to train in minutes on a CPU.
- **Class weighting over oversampling**: avoids duplicating minority-class images, which would inflate apparent performance without adding information.
- **Global average pooling instead of a large Dense head**: keeps the parameter count low to reduce overfitting on both (comparatively small) training sets, and keeps the final conv feature maps spatial — required for Grad-CAM to localize anything.
- **Explicit false-negative rate reporting**: a missed case has a much higher real-world cost than a false alarm in a screening context — the kind of metric choice clinical ML systems are actually evaluated on.
- **The breast model shipped despite weaker numbers**: cutting it would have hidden a real, honest constraint (dataset size) instead of surfacing it. The app's own UI reports both models' numbers plainly, with the false-negative rate front and center either way.

## Limitations / ethical disclosure

This is an educational/portfolio project, **not a validated clinical tool**, for either model. Both `app.py` and the results above say so explicitly rather than only in fine print.

- **Pneumonia model:** PneumoniaMNIST is a pediatric-only, single-institution, single-imaging-protocol dataset; it will not generalize reliably to adult patients, different X-ray machines, or different patient populations without further validation.
- **Breast ultrasound model:** trained on only 780 images total — a fraction of what a clinically validated system would require. Its 68.6% accuracy and 45.5% precision reflect that directly; treat any single prediction from it as illustrative, not diagnostic.

No output from this tool should ever inform a real medical decision. If you or someone you know has a health concern, see a doctor — this tool cannot and should not replace that.

## Attribution

- **Pneumonia dataset:** Yang et al., *MedMNIST v2* (2023); original imagery from Kermany et al., *Cell* (2018), Guangzhou Women and Children's Medical Center. CC BY 4.0.
- **Breast ultrasound dataset:** Yang et al., *MedMNIST v2* (2023); original imagery from Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief (2020), Baheya Hospital, Cairo. CC BY 4.0.
- **Method:** Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- **License:** [MIT](LICENSE)
