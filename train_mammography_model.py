"""
Train a CNN to classify screening mammograms as Malignant vs. Normal/Benign.
Third mode alongside the pneumonia and breast ultrasound models -- this app
screens all three from one dashboard, each with its own model.

Dataset: the MIAS database (Mammographic Image Analysis Society, v1.21),
322 digitized film mammograms (161 patients, one image per breast) from a UK
screening programme in the early 1990s (Suckling et al., 1994). Downloads
automatically from the University of Cambridge's institutional repository
(https://doi.org/10.17863/CAM.105113), CC BY 2.0 UK.

IMPORTANT: this is the smallest and oldest of the three datasets (322 images,
digitized film from ~1994, vs. 5,856 modern digital chest X-rays for the
pneumonia model). Expect correspondingly modest, more variable metrics.
Report whatever the real numbers are; do not round up.

Whole-image resizing (the approach the other two models use) does not work
here: MIAS images are full film scans up to ~5200x4000px, so a lesion that's
only a few hundred pixels across becomes a handful of pixels after resizing
to 128x128 -- far too little for a CNN to learn from with just ~50 positive
examples. Instead, each image is reduced to one lesion-centered patch (sized
proportionally to the lesion's radius, from the ground truth below) before
resizing to 128x128; images with no localized abnormality are cropped
around their detected breast-tissue region instead. app.py mirrors this at
inference time with a sliding-window scan over the uploaded image rather
than a single whole-image resize.

Ground truth: MIAS ships its per-image abnormality/severity table only inside
MIASDBv1.21/00README.pdf (not as a separate machine-readable file), so it's
transcribed once below as GROUND_TRUTH, straight from that PDF's table --
same fields (refnum, background tissue, abnormality class, severity, x, y,
radius), same reference: J Suckling et al (1994) "The Mammographic Image
Analysis Society Digital Mammogram Database", Exerpta Medica. International
Congress Series 1069 pp375-378.

Usage:
    python train_mammography_model.py
Outputs (into ./model and ./reports):
    model/mammography_cnn.keras          -- trained model
    reports/mammography_metrics.json     -- accuracy / precision / recall / AUC / false-negative rate
    reports/mammography_confusion_matrix.png
    reports/mammography_roc_curve.png
    reports/mammography_training_history.png
"""

import io
import json
import os
import urllib.request
import zipfile
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from mammo_window import sliding_window_predict

IMG_SIZE = 128
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "mias")
PGM_DIR = os.path.join(DATA_DIR, "pgm")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MIAS_ZIP_URL = "https://www.repository.cam.ac.uk/bitstreams/5960ab2b-5ea2-4db1-96ac-15b3605e7485/download"

# Transcribed from MIASDBv1.21/00README.pdf's ground-truth table. Columns:
# refnum, background tissue (F/G/D), abnormality class, [severity B/M, x, y,
# radius] if not NORM. A few images have more than one abnormality -- those
# appear as a second line repeating the refnum (the PDF indents a
# continuation line instead; flattened here for a simple line-based parse).
# Three entries (mdb216, mdb233, mdb245) omit coordinates in the source
# ("*** see note 2" -- calcifications too widely distributed for a single
# center/radius); severity alone is all this script needs.
GROUND_TRUTH = """
mdb001lm G CIRC B 1815 1116 790
mdb002rl G CIRC B 3091 1262 277
mdb003ll D NORM
mdb004rl D NORM
mdb005ll F CIRC B 647 1163 122
mdb005ll CIRC B 786 1255 107
mdb006rl F NORM
mdb007ll G NORM
mdb008rl G NORM
mdb009ll F NORM
mdb010rm F CIRC B 2509 975 135
mdb011ll F NORM
mdb012rl F CIRC B 2378 1467 162
mdb013ll G MISC B 1574 1923 127
mdb014rl G NORM
mdb015lm G CIRC B 3571 1359 275
mdb016rm G NORM
mdb017ls G CIRC B 2407 943 192
mdb018rs G NORM
mdb019ll G CIRC B 2021 1864 197
mdb020rl G NORM
mdb021ll G CIRC B 612 1224 197
mdb022rm G NORM
mdb023ll G CIRC M 2837 1405 117
mdb024rl G NORM
mdb025ll F CIRC B 1886 1948 318
mdb026rl F NORM
mdb027ll F NORM
mdb028rl F CIRC M 2953 1999 224
mdb029ll G NORM
mdb030rm G MISC B 1505 1785 174
mdb031ll G NORM
mdb032rl G MISC B 1243 1798 267
mdb033ls D NORM
mdb034rs D NORM
mdb035ls D NORM
mdb036rs D NORM
mdb037ls D NORM
mdb038rs D NORM
mdb039ls D NORM
mdb040rs D NORM
mdb041ll G NORM
mdb042rl G NORM
mdb043ls G NORM
mdb044rs G NORM
mdb045lm G NORM
mdb046rm G NORM
mdb047lm G NORM
mdb048rm G NORM
mdb049ll G NORM
mdb050rl G NORM
mdb051ll G NORM
mdb052rm G NORM
mdb053ls D NORM
mdb054rs D NORM
mdb055lm G NORM
mdb056rm G NORM
mdb057ll D NORM
mdb058rl D MISC M 2774 2079 110
mdb059ls F CIRC B
mdb060rs F NORM
mdb061ls D NORM
mdb062rs D NORM
mdb063lm D MISC B 1967 1163 133
mdb064rm D NORM
mdb065lm D NORM
mdb066rm D NORM
mdb067ll D NORM
mdb068rl D NORM
mdb069ll F CIRC B 1739 1101 177
mdb070rl F NORM
mdb071lm G NORM
mdb072rm G ASYM M 2140 2011 115
mdb073ls G NORM
mdb074rs G NORM
mdb075lm F ASYM M 2982 850 92
mdb076rm F NORM
mdb077ll F NORM
mdb078rl F NORM
mdb079lm F NORM
mdb080rm F CIRC B 3615 1344 81
mdb081ll G ASYM B 2007 1220 525
mdb082rl G NORM
mdb083ll G ASYM B 891 1428 152
mdb084rl G NORM
mdb085lm G NORM
mdb086rm G NORM
mdb087lm F NORM
mdb088rm F NORM
mdb089lm G NORM
mdb090rm G ASYM M 2021 1035 198
mdb091lm F CIRC B 2090 1696 82
mdb092rm F ASYM M 1562 1382 175
mdb093lm G NORM
mdb094rm G NORM
mdb095ll F ASYM M 2181 1118 116
mdb096rl F NORM
mdb097ll F ASYM B 1302 1702 137
mdb098rl F NORM
mdb099lm D ASYM B 1473 1834 93
mdb100rm D NORM
mdb101lm D NORM
mdb102rm D ASYM M 2369 1412 152
mdb103lm D NORM
mdb104rm D ASYM B 2751 1645 203
mdb105ll D ASYM M 1229 1318 392
mdb106rl D NORM
mdb107ll D ASYM B 2597 1653 446
mdb108rl D NORM
mdb109ll D NORM
mdb110rl D ASYM M 2502 2590 205
mdb111ll D ASYM M 2414 1275 428
mdb112rl D NORM
mdb113ls G NORM
mdb114rs G NORM
mdb115ll G ARCH M 2240 1096 468
mdb116rl G NORM
mdb117ll G ARCH M 2417 1175 337
mdb118rl G NORM
mdb119ll G NORM
mdb120rl G ARCH M 3162 1659 319
mdb121ll G ARCH B 1849 1221 348
mdb122rl G NORM
mdb123lm G NORM
mdb124rm G ARCH M 1729 1609 135
mdb125ll D ARCH M 2322 2054 242
mdb126rl D ARCH B 2015 2585 93
mdb127lm G ARCH B 2317 1069 194
mdb128rm G NORM
mdb129ll D NORM
mdb130rl D ARCH M 2002 2469 112
mdb131lx F NORM
mdb132rx F CIRC B 1499 3043 211
mdb132rx CIRC B 1587 2709 73
mdb133lx F NORM
mdb134rx F MISC M 1736 2173 199
mdb135lx F NORM
mdb136rx F NORM
mdb137ll D NORM
mdb138rl D NORM
mdb139lx F NORM
mdb140rx F NORM
mdb141lx F CIRC M 3591 1832 117
mdb142rx F CIRC B 2104 2662 104
mdb143lx F NORM
mdb144rx F MISC B 674 3117 119
mdb144rx MISC M 2491 2799 108
mdb145lx D SPIC B 2726 2631 197
mdb146rx D NORM
mdb147lx F NORM
mdb148rx F SPIC M 2220 2745 699
mdb149lx F NORM
mdb150rx F ARCH B 2005 2647 249
mdb151lx F NORM
mdb152rx F ARCH B 2704 1349 195
mdb153lx F NORM
mdb154rx F NORM
mdb155ll F ARCH M 2032 1046 380
mdb156rl F NORM
mdb157lm F NORM
mdb158rm F ARCH M 1951 915 353
mdb159ll F NORM
mdb160rl F ARCH B 2133 1206 245
mdb161lm D NORM
mdb162rm D NORM
mdb163ll D ARCH B 1574 817 202
mdb164rl D NORM
mdb165ls D ARCH B 2073 903 168
mdb166rs D NORM
mdb167ll F ARCH B 2740 1550 141
mdb168rl F NORM
mdb169lm D NORM
mdb170rm D ARCH M 2288 1118 331
mdb171ll D ARCH M 2622 1102 248
mdb172rl D NORM
mdb173ll F NORM
mdb174rl F NORM
mdb175lm G SPIC B 2795 1344 132
mdb176rm G NORM
mdb177ls G NORM
mdb178rs G SPIC M 1810 880 280
mdb179ls D SPIC M 2168 1152 268
mdb180rs D NORM
mdb181lm G SPIC M 1563 1052 217
mdb182rm G NORM
mdb183ll F NORM
mdb184rl F SPIC M 1712 1943 458
mdb185ls G NORM
mdb186rs G SPIC M 2114 1237 191
mdb187lm G NORM
mdb188rm G SPIC B 1741 1448 247
mdb189ll G NORM
mdb190rl G SPIC B 1724 1302 127
mdb191ls G SPIC B 2177 1128 165
mdb192rs G NORM
mdb193ll D SPIC B 2364 850 528
mdb194rl D NORM
mdb195ll F SPIC B 631 2155 107
mdb196rl F NORM
mdb197lm D NORM
mdb198rm D SPIC B 1761 800 373
mdb199lm D SPIC B 820 1543 125
mdb200rm D NORM
mdb201ll D NORM
mdb202rl D SPIC M 1122 1123 149
mdb203ll F NORM
mdb204rl F SPIC B 2614 2005 84
mdb205ll F NORM
mdb206rl F SPIC M 3410 1876 71
mdb207lm D SPIC B 2370 1262 76
mdb208rm D NORM
mdb209ll G CALC M 2126 1842 348
mdb210rl G NORM
mdb211lm G CALC M 1423 1698 53
mdb212rm G CALC B
mdb213ls G CALC M 2193 940 183
mdb214rs G CALC B
mdb215ll D NORM
mdb216rl D CALC M
mdb217ll G NORM
mdb218rl G CALC B 1694 1275 35
mdb219ll G CALC B 3136 1439 119
mdb220rl G NORM
mdb221lm D NORM
mdb222rm D CALC B 2502 1482 70
mdb223ls D CALC B 2043 846 116
mdb223ls CALC B 2231 1116 27
mdb224rs D NORM
mdb225lm D NORM
mdb226rm D CALC B 1770 1927 31
mdb226rm CALC B 2011 1757 102
mdb226rm CALC B 1325 951 33
mdb227lm G CALC B 1981 993 36
mdb228rm G NORM
mdb229ll F NORM
mdb230rl F NORM
mdb231ll F CALC M 2265 1665 179
mdb232rl F NORM
mdb233lm G CALC M
mdb234rm G NORM
mdb235ll D NORM
mdb236rl D CALC B 912 2247 58
mdb237lm F NORM
mdb238rm F CALC M 1998 986 70
mdb239ll D CALC M 3133 1833 160
mdb239ll CALC M 3347 1523 103
mdb240rl D CALC B 1752 776 95
mdb241ls D CALC M 2827 565 155
mdb242rs D NORM
mdb243lm D NORM
mdb244rm D CIRC B 1940 1209 209
mdb245ls F CALC M
mdb246rs F NORM
mdb247ll F NORM
mdb248rl F CALC B 1805 1836 42
mdb249lm D CALC M 2146 1154 194
mdb249lm CALC M 2671 1276 256
mdb250rm D NORM
mdb251lm F NORM
mdb252rm F CALC B 2743 1318 94
mdb253ll D CALC M 2368 2185 112
mdb254rl D NORM
mdb255ll F NORM
mdb256rl F CALC M 2272 1750 149
mdb257ll D NORM
mdb258rl D NORM
mdb259ll D NORM
mdb260rl D NORM
mdb261ls D NORM
mdb262rs D NORM
mdb263lm G NORM
mdb264rm G MISC M 2487 691 147
mdb265lm G MISC M 2104 1351 242
mdb266rm G NORM
mdb267ll F MISC M 2036 2427 227
mdb268rl F NORM
mdb269lm G NORM
mdb270rm G CIRC M 430 1649 291
mdb271ll F MISC M 1193 2391 274
mdb272rl F NORM
mdb273ll F NORM
mdb274rx F MISC M 2630 3542 495
mdb275ll G NORM
mdb276rl G NORM
mdb277lm G NORM
mdb278rm G NORM
mdb279ll G NORM
mdb280rx G NORM
mdb281lm D NORM
mdb282rm D NORM
mdb283lm D NORM
mdb284rm D NORM
mdb285lm D NORM
mdb286rm D NORM
mdb287ls D NORM
mdb288rs D NORM
mdb289ls D NORM
mdb290rs D CIRC B 2799 1502 181
mdb291ll G NORM
mdb292rl G NORM
mdb293ll F NORM
mdb294rl F NORM
mdb295ll D NORM
mdb296rl D NORM
mdb297ll F NORM
mdb298rl F NORM
mdb299ll F NORM
mdb300rl F NORM
mdb301lm F NORM
mdb302rm F NORM
mdb303lm F NORM
mdb304rm F NORM
mdb305lm F NORM
mdb306rm F NORM
mdb307ll F NORM
mdb308rl F NORM
mdb309ll F NORM
mdb310rl F NORM
mdb311ll F NORM
mdb312rl F MISC B 3158 2389 81
mdb313ll F NORM
mdb314rl F MISC B 3447 1277 158
mdb315ll D CIRC B 1900 1317 372
mdb316rl D NORM
mdb317ls D NORM
mdb318rs D NORM
mdb319ll D NORM
mdb320rl D NORM
mdb321lm D NORM
mdb322rm D NORM
""".strip()


def ensure_data():
    if os.path.isdir(PGM_DIR) and len(os.listdir(PGM_DIR)) >= 322:
        return
    print("Downloading MIAS database (~1.5GB, one-time)...")
    os.makedirs(PGM_DIR, exist_ok=True)
    with urllib.request.urlopen(MIAS_ZIP_URL) as resp:
        data = resp.read()
    print("Extracting...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.endswith(".pgm"):
                out_path = os.path.join(PGM_DIR, os.path.basename(name))
                with zf.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
    print(f"Saved {len(os.listdir(PGM_DIR))} images to {PGM_DIR}")


def parse_records() -> dict:
    """refnum -> {"severity": "M"/"B"/None, "lesions": [(x, y, radius), ...]}.

    "severity" is the worst of any abnormality line for that image (M beats
    B beats None/NORM). "lesions" only includes lines with coordinates --
    three malignant cases (mdb216, mdb233, mdb245) list severity but omit
    coordinates ("calcifications ... widely distributed", per the source
    PDF), so they carry a severity but an empty lesion list.
    """
    records = defaultdict(lambda: {"severity": None, "lesions": []})
    severities = defaultdict(set)
    for line in GROUND_TRUTH.splitlines():
        parts = line.split()
        if not parts:
            continue
        refnum = parts[0]
        # Tokens after refnum (and, on an image's first line, the
        # background-tissue letter F/G/D) are: CLASS [SEVERITY X Y RADIUS]
        rest = parts[1:]
        if rest and rest[0] in ("F", "G", "D"):
            rest = rest[1:]
        if not rest or rest[0] == "NORM":
            continue
        if len(rest) >= 2 and rest[1] in ("B", "M"):
            severities[refnum].add(rest[1])
            if len(rest) >= 5:
                x, y, radius = int(rest[2]), int(rest[3]), int(rest[4])
                records[refnum]["lesions"].append((x, y, radius))

    for refnum in severities.keys() | {f.replace(".pgm", "") for f in os.listdir(PGM_DIR)}:
        sevs = severities.get(refnum, set())
        records[refnum]["severity"] = "M" if "M" in sevs else ("B" if "B" in sevs else None)
    return dict(records)


def breast_centroid(arr: np.ndarray) -> tuple[int, int]:
    """Center of the largest bright connected region (the breast tissue),
    used as the crop center for images with no known lesion location.
    Falls back to the image center if segmentation finds nothing usable --
    a blank/degenerate mask is the only failure mode here, not a wrong
    answer, so a crude fallback is fine."""
    h, w = arr.shape
    blurred = cv2.GaussianBlur(arr, (25, 25), 0)
    mask = (blurred > 15).astype("uint8")
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return w // 2, h // 2
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, _ = stats[idx]
    return x + bw // 2, y + bh // 2


def extract_patch(arr: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    """Square crop of `size` centered at (cx, cy), shifted (not padded) to
    stay inside the image, then resized to IMG_SIZE."""
    h, w = arr.shape
    size = min(size, h, w)
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    crop = arr[y0:y0 + size, x0:x0 + size]
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


# Default patch size for images with no known lesion location (NORM cases):
# the median real lesion crop (median radius 177 * 3 =~ 530), so "what
# normal tissue looks like at roughly the same zoom level" as a lesion crop.
DEFAULT_PATCH_SIZE = 530


def load_dataset():
    records = parse_records()
    xs, ys, refs = [], [], []
    skipped_no_coords = 0
    for refnum, rec in sorted(records.items()):
        path = os.path.join(PGM_DIR, f"{refnum}.pgm")
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("L")
        arr = np.asarray(img).astype("float32") / 255.0

        if rec["severity"] is not None and not rec["lesions"]:
            # Malignant/benign but no coordinates given (calcifications too
            # spread out to localize) -- can't build a lesion-centered
            # patch for these, so they're excluded rather than guessed at.
            skipped_no_coords += 1
            continue

        if rec["lesions"]:
            # Largest-radius lesion when an image has more than one.
            x, y, radius = max(rec["lesions"], key=lambda t: t[2])
            size = int(np.clip(radius * 3, 300, 1200))
            label = 1.0 if rec["severity"] == "M" else 0.0
        else:
            cx, cy = breast_centroid(arr)
            x, y = cx, cy
            size = DEFAULT_PATCH_SIZE
            label = 0.0

        patch = extract_patch(arr, x, y, size)
        xs.append(patch)
        ys.append(label)
        refs.append(refnum)

    print(f"Skipped {skipped_no_coords} malignant image(s) with no lesion coordinates.")
    x = np.expand_dims(np.stack(xs), axis=-1)
    y = np.array(ys, dtype="float32")
    return x, y, refs


def build_model() -> tf.keras.Model:
    # A from-scratch CNN (the architecture used for the other two models)
    # never learned usable signal here -- 225 training images is too little
    # to learn discriminative features from raw pixels, especially with
    # each image resized down from several-thousand-pixel-wide film scans.
    # Transfer learning from ImageNet is the standard fix for a dataset this
    # small: freeze a pretrained backbone as a fixed feature extractor and
    # train only a small head. Input stays (128, 128, 1) at [0, 1] -- same
    # contract as the pneumonia/breast models -- with the channel-replicate
    # and MobileNetV2 rescaling folded into the graph so app.py needs no
    # per-model special-casing.
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.03)(x)
    x = tf.keras.layers.RandomZoom(0.08)(x)

    x = tf.keras.layers.Concatenate()([x, x, x])
    x = tf.keras.layers.Rescaling(255.0)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base_model.trainable = False

    # Head kept deliberately tiny (no hidden dense layer) -- with 322 images
    # total, extra head capacity is extra ways to overfit noise rather than
    # signal.
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="mammography_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def class_weight_for(y_subset: np.ndarray) -> dict:
    n_pos = y_subset.sum()
    n_neg = len(y_subset) - n_pos
    return {0: len(y_subset) / (2 * n_neg), 1: len(y_subset) / (2 * n_pos)}


def train_one_model(x_tr, y_tr, x_val, y_val):
    model = build_model()
    # val_loss, not val_auc: AUC computed over a few dozen images (of which
    # only a handful are positive) jumps around too much between epochs to
    # trust as a checkpoint-selection signal. Loss is continuous and
    # smoother even at this sample size.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=6
        ),
    ]
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=80,
        batch_size=16,
        class_weight=class_weight_for(y_tr),
        callbacks=callbacks,
        verbose=2,
    )
    return model, history


def whole_image_probs(model, refs_subset) -> np.ndarray:
    probs = []
    for refnum in refs_subset:
        img = Image.open(os.path.join(PGM_DIR, f"{refnum}.pgm")).convert("L")
        gray = np.asarray(img).astype("float32") / 255.0
        prob, _ = sliding_window_predict(gray, model)
        probs.append(prob)
    return np.array(probs)


def calibrate_threshold(y_val: np.ndarray, val_probs: np.ndarray, target_sensitivity: float = 0.8):
    """Lowest threshold that still catches >= target_sensitivity of
    validation's positives, rather than Youden's J -- tried first, it
    picked a single "optimal" point (0.987) that chased a handful of
    validation positives (~8) and caught zero malignant test cases. A
    screening tool should target sensitivity directly and accept whatever
    specificity that costs, since a missed malignant case is worse than a
    false alarm here."""
    fpr_v, tpr_v, thresholds_v = roc_curve(y_val, val_probs)
    meets = np.where(tpr_v >= target_sensitivity)[0]
    # thresholds_v is sorted highest-to-lowest; the last index meeting the
    # target is the highest (most specific) threshold that still does.
    pick = meets[-1] if len(meets) else int(np.argmax(tpr_v))
    return float(thresholds_v[pick]), float(tpr_v[pick]), float(1 - fpr_v[pick])


def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    ensure_data()
    print("Loading MIAS lesion-centered patches (128x128)...")
    x, y, refs = load_dataset()
    refs = np.array(refs)
    print(f"total={x.shape}, positive(malignant)={int(y.sum())} of {len(y)}")

    # A single train/val/test split was tried first and turned out to be
    # unusably noisy: with only ~52 malignant images in the whole dataset,
    # a test fold has on the order of 10-14 positives, and whole-image
    # sliding-window AUC swung between 0.55 and 0.82 across otherwise-
    # identical runs depending on exactly which images landed in which
    # split. That's not "weak but real" like the breast ultrasound model --
    # it's not reproducible enough to report a single number honestly.
    # 5-fold cross-validation instead trains 5 independent models, each
    # tested on a different fifth of the data never seen during that
    # fold's training, so every image contributes to the test-side numbers
    # exactly once and the reported mean +/- std reflects real run-to-run
    # variance instead of hiding it behind one lucky/unlucky split.
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    pooled_y, pooled_probs = [], []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        print(f"\n=== Fold {fold_i}/{N_FOLDS} ===")
        idx_tr, idx_val = train_test_split(
            train_idx, test_size=0.15, stratify=y[train_idx], random_state=42
        )
        print(f"train={len(idx_tr)}, val={len(idx_val)}, test={len(test_idx)}")

        model, _ = train_one_model(x[idx_tr], y[idx_tr], x[idx_val], y[idx_val])

        val_probs = whole_image_probs(model, refs[idx_val])
        threshold, val_sens, val_spec = calibrate_threshold(y[idx_val], val_probs)
        print(f"fold {fold_i} threshold={threshold:.3f} (val sensitivity={val_sens:.2f}, specificity={val_spec:.2f})")

        test_probs = whole_image_probs(model, refs[test_idx])
        result = evaluate(y[test_idx], test_probs, threshold)
        result["auc"] = float(roc_auc_score(y[test_idx], test_probs)) if len(set(y[test_idx])) > 1 else float("nan")
        print(f"fold {fold_i} test:", json.dumps(result, indent=2))
        fold_results.append(result)
        pooled_y.extend(y[test_idx].tolist())
        pooled_probs.extend(test_probs.tolist())

    pooled_y = np.array(pooled_y)
    pooled_probs = np.array(pooled_probs)

    def agg(key):
        vals = [r[key] for r in fold_results if not np.isnan(r[key])]
        return float(np.mean(vals)), float(np.std(vals))

    cv_summary = {k: agg(k) for k in
                  ("accuracy", "precision", "recall_sensitivity", "auc",
                   "false_negative_rate", "false_positive_rate")}
    print(f"\n=== {N_FOLDS}-fold cross-validated whole-image performance (mean +/- std) ===")
    for k, (m, s) in cv_summary.items():
        print(f"  {k}: {m:.3f} +/- {s:.3f}")

    # Train one final model on (almost) all the data for shipping -- the CV
    # loop above exists to measure honest performance, not to produce the
    # deployed model itself, which should see as much training data as
    # possible.
    print("\n=== Training final model on all data for deployment ===")
    idx_all = np.arange(len(y))
    idx_tr_final, idx_val_final = train_test_split(idx_all, test_size=0.15, stratify=y, random_state=123)
    final_model, history = train_one_model(x[idx_tr_final], y[idx_tr_final], x[idx_val_final], y[idx_val_final])
    final_model.save(os.path.join(MODEL_DIR, "mammography_cnn.keras"))
    print("Saved model to", os.path.join(MODEL_DIR, "mammography_cnn.keras"))

    final_val_probs = whole_image_probs(final_model, refs[idx_val_final])
    final_threshold, final_sens, final_spec = calibrate_threshold(y[idx_val_final], final_val_probs)
    print(f"Final model threshold={final_threshold:.4f} (val sensitivity={final_sens:.2f}, specificity={final_spec:.2f})")

    # Headline accuracy/precision/etc are the cross-validated means (the
    # honest, reproducible estimate); the confusion matrix and ROC curve
    # use the pooled out-of-fold predictions (every one of the usable
    # images, each scored by a model that never trained on it) since a
    # single confusion matrix needs one coherent set of predictions, not
    # five separate small ones. The final model's own threshold is used to
    # binarize the pooled predictions since that's the threshold that
    # actually ships in app.py.
    pooled_eval = evaluate(pooled_y, pooled_probs, final_threshold)

    metrics = {
        "dataset": f"MIAS database v1.21, whole-image sliding-window evaluation, {N_FOLDS}-fold cross-validated, {len(y)} of 322 images used",
        "test_set_size": int(len(y)),
        "positive_class": "malignant",
        "decision_threshold": final_threshold,
        "cross_validation_folds": N_FOLDS,
        "accuracy": cv_summary["accuracy"][0],
        "accuracy_std": cv_summary["accuracy"][1],
        "precision": cv_summary["precision"][0],
        "precision_std": cv_summary["precision"][1],
        "recall_sensitivity": cv_summary["recall_sensitivity"][0],
        "recall_sensitivity_std": cv_summary["recall_sensitivity"][1],
        "auc": cv_summary["auc"][0],
        "auc_std": cv_summary["auc"][1],
        "false_negative_rate": cv_summary["false_negative_rate"][0],
        "false_negative_rate_std": cv_summary["false_negative_rate"][1],
        "false_positive_rate": cv_summary["false_positive_rate"][0],
        "false_positive_rate_std": cv_summary["false_positive_rate"][1],
        "confusion_matrix": pooled_eval["confusion_matrix"],
    }
    with open(os.path.join(REPORT_DIR, "mammography_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    cm = np.array([
        [pooled_eval["confusion_matrix"]["tn"], pooled_eval["confusion_matrix"]["fp"]],
        [pooled_eval["confusion_matrix"]["fn"], pooled_eval["confusion_matrix"]["tp"]],
    ])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal/Benign", "Malignant"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal/Benign", "Malignant"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion Matrix ({N_FOLDS}-fold pooled out-of-fold)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "mammography_confusion_matrix.png"), dpi=150)

    fpr, tpr, _ = roc_curve(pooled_y, pooled_probs)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"pooled AUC = {roc_auc_score(pooled_y, pooled_probs):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({N_FOLDS}-fold pooled out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "mammography_roc_curve.png"), dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss (final deployed model)"); axes[0].legend()
    axes[1].plot(history.history["auc"], label="train")
    axes[1].plot(history.history["val_auc"], label="val")
    axes[1].set_title("AUC (final deployed model)"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "mammography_training_history.png"), dpi=150)

    print("Reports written to", REPORT_DIR)


if __name__ == "__main__":
    main()
