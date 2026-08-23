# Chest X-Ray Pneumonia Screening Tool

A convolutional neural network that classifies pediatric chest X-rays as
**Normal** or **Pneumonia**, wrapped in a Streamlit web dashboard with
Grad-CAM explainability so predictions aren't a black box.

Built as a self-directed biomedical engineering project: dataset → model →
evaluation → deployable interface, the full pipeline of an applied
computational medicine tool, not just a training script.

**⚠️ Not a medical device.** This is a student engineering/research
project. It must never be used to make a real medical decision — see
[Limitations / ethical disclosure](#limitations--ethical-disclosure).

## Download & use it (no coding experience required)

The trained model is already included in this repo, so you do **not** need
to train anything or wait for anything — just download, install three
things, and run one command.

**1. Get the code onto your computer.** Either:
- Click the green **`<> Code`** button at the top of this page → **Download
  ZIP** → unzip it anywhere, *or*
- If you use git: `git clone https://github.com/Lavamaster99/ai-pneumonia-xray-classifier.git`

**2. Install Python**, if you don't already have it: download from
[python.org/downloads](https://www.python.org/downloads/) (any version 3.10
or newer). On the Windows installer, tick **"Add python.exe to PATH"**
before clicking Install.

**3. Open a terminal in the project folder** (Windows: open the unzipped
folder, click the address bar, type `cmd`, press Enter) and run:

```bash
python -m venv venv

venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

The first `pip install` takes a few minutes (TensorFlow is a large
download). Once it finishes, Streamlit will automatically open
`http://localhost:8501` in your browser — if it doesn't, open that address
yourself.

**4. Use it.** Click **Browse files**, and either upload your own chest
X-ray image or use one of the real test images already included in
`examples/` (`sample_pneumonia.png` and `sample_normal.png`) to see it work
immediately. You'll get a prediction, a confidence score, and a Grad-CAM
heatmap showing what the model looked at.

**Stuck?** Common issues:
- *"python is not recognized"* (Windows) — Python wasn't added to PATH
  during install; reinstall and tick that box, or use `py` instead of
  `python` in the commands above.
- *Port 8501 already in use* — another Streamlit app is already running;
  either close it or run `streamlit run app.py --server.port 8502`.
- *Install is slow* — normal on the first run; TensorFlow alone is ~500MB.

## What it does

1. **CNN classifier.** An 8-conv-layer network (with batch norm, dropout, and
   light data augmentation) trained on 128×128 grayscale chest X-rays to
   output a pneumonia probability.
2. **Grad-CAM heatmaps.** For every prediction, the app computes which pixels
   of the X-ray the network actually attended to (Selvaraju et al., 2017),
   overlaid on the image — the "engineering" layer that turns a bare
   number into an inspectable, semi-interpretable result.
3. **Web dashboard.** A Streamlit app (`app.py`) where a user uploads an
   image and sees the prediction, confidence, heatmap, and the model's own
   validated performance stats side by side.

## Dataset

[PneumoniaMNIST](https://medmnist.com/) (part of the MedMNIST v2 benchmark
suite), itself derived from Kermany et al.'s pediatric chest X-ray dataset
collected at Guangzhou Women and Children's Medical Center. 5,856 images,
CC BY 4.0 licensed. Downloads automatically on first run — no Kaggle account
or manual dataset wrangling required. This is the same underlying dataset
referenced in the original pneumonia-detection literature, just packaged in
a clean, standardized benchmark form.

Split: 4,708 train / 524 validation / 624 test. The training set is
class-imbalanced (~3:1 pneumonia:normal), handled with class weighting
rather than naive resampling.

## Results (test set, 624 held-out images, `reports/metrics.json`)

| Metric | Value |
|---|---|
| Accuracy | 93.3% |
| Precision | 91.4% |
| Sensitivity (recall) | 98.5% |
| AUC | 0.980 |
| False-negative rate | 1.5% (6 of 390 pneumonia cases missed) |
| False-positive rate | 15.4% (36 of 234 normal cases flagged) |

See `reports/confusion_matrix.png`, `reports/roc_curve.png`, and
`reports/training_history.png` for the visualizations.

The false-negative rate is reported explicitly and separately from overall
accuracy because in a screening context, missing a true pneumonia case is a
qualitatively worse error than a false alarm — accuracy alone can hide that.
`reports/confusion_matrix.png` and `reports/roc_curve.png` visualize the
same tradeoff.

## Project structure

```
train_model.py    # builds, trains, and evaluates the CNN; writes model + reports
gradcam.py        # Grad-CAM heatmap implementation (framework-agnostic logic)
app.py            # Streamlit dashboard: upload -> prediction -> heatmap
requirements.txt
model/            # saved trained model -- already committed, no training needed to use the app
reports/          # metrics.json + PNG plots -- already committed
data/             # cached dataset download (only appears if you retrain)
```

## Retraining from scratch (optional)

The steps in **Download & use it** above are all you need to run the app.
Retraining is only necessary if you change the model architecture or want
to reproduce the training run yourself:

```bash
python train_model.py          # ~10-20 min on a laptop CPU, downloads PneumoniaMNIST automatically
```

This overwrites `model/pneumonia_cnn.keras` and `reports/metrics.json` with
a fresh run -- results will vary slightly from the Results numbers above
since training isn't perfectly deterministic.

## Design decisions worth noting (for write-ups / interviews)

- **128×128 resolution**, not the raw 28×28 MedMNIST default: large enough
  that the Grad-CAM overlay is visually meaningful on real X-ray anatomy,
  small enough to train in minutes on a CPU with no GPU.
- **Class weighting over oversampling**: avoids duplicating minority-class
  images, which would inflate apparent performance without adding real
  information.
- **Global average pooling instead of a large Dense head**: keeps the
  parameter count low (~300K) to reduce overfitting risk on a dataset with
  only ~4.7K training images, and keeps the final conv feature maps spatial
  (necessary for Grad-CAM to produce a meaningful localization map).
- **Explicit false-negative rate reporting**: chosen deliberately over
  reporting accuracy alone, since a missed-pneumonia case has a much higher
  real-world cost than a false alarm — this is the kind of metric choice
  clinical ML systems are actually evaluated on.

## Limitations / ethical disclosure

This is an educational/portfolio project, not a validated clinical tool.
PneumoniaMNIST is a pediatric-only, single-institution, single-imaging-
protocol dataset; a model trained on it will not generalize reliably to
adult patients, different X-ray machines, or different patient populations
without further validation. The Streamlit app includes an explicit
"not a medical device" warning for this reason, and no output from this
tool should ever inform an actual medical decision.

## Attribution

- Dataset: Yang et al., *MedMNIST v2* (2023); original source imagery from
  Kermany et al., *Cell* (2018), Guangzhou Women and Children's Medical
  Center. CC BY 4.0.
- Method: Selvaraju et al., *Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization*, ICCV 2017.
