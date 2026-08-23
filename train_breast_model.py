"""
Train a CNN to classify breast ultrasound images as Malignant vs.
Normal/Benign. Companion to train_model.py (chest X-ray / pneumonia) --
this app screens both from one dashboard, each with its own model.

Dataset: BreastMNIST (MedMNIST v2 benchmark), sourced from Al-Dhabyani et
al.'s breast ultrasound dataset (780 images, Baheya Hospital, Cairo).
CC BY 4.0 licensed, downloads automatically -- no manual dataset wrangling.

IMPORTANT: this is a much smaller dataset than the pneumonia one (780
images total vs. 5,856) -- expect correspondingly more modest, more
variable metrics. Report whatever the real numbers are; do not round up.

Usage:
    python train_breast_model.py
Outputs (into ./model and ./reports):
    model/breast_cnn.keras              -- trained model
    reports/breast_metrics.json         -- accuracy / precision / recall / AUC / false-negative rate
    reports/breast_confusion_matrix.png
    reports/breast_roc_curve.png
    reports/breast_training_history.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import medmnist
from medmnist import INFO

IMG_SIZE = 128
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_split(split: str):
    info = INFO["breastmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split=split, download=True, size=IMG_SIZE, root=DATA_ROOT)
    x = ds.imgs.astype("float32") / 255.0
    x = np.expand_dims(x, axis=-1)
    # BreastMNIST's native label convention is 0=malignant, 1=normal/benign
    # (confirmed via medmnist.INFO -- the opposite of the pneumonia project's
    # 0=normal/1=disease convention). Flipped here so label 1 = malignant
    # throughout this codebase, matching app.py and the README, and so
    # "positive class" consistently means "the thing we don't want to miss."
    y = 1.0 - ds.labels.astype("float32").reshape(-1)
    return x, y


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.04)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        return x

    # One fewer block than the pneumonia model: with only ~550 training
    # images, a deeper/wider network overfits fast rather than learning more.
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="breast_cnn")
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


def main():
    print("Loading BreastMNIST (128x128)...")
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")
    print(f"train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: (len(y_train) / (2 * n_neg)), 1: (len(y_train) / (2 * n_pos))}
    print("class_weight:", class_weight, "positive(malignant) count:", int(n_pos), "of", len(y_train))

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=4
        ),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=60,
        batch_size=16,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(os.path.join(MODEL_DIR, "breast_cnn.keras"))
    print("Saved model to", os.path.join(MODEL_DIR, "breast_cnn.keras"))

    y_prob = model.predict(x_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    metrics = {
        "dataset": "BreastMNIST (MedMNIST v2), 128x128, 780 images total",
        "test_set_size": int(len(y_test)),
        "positive_class": "malignant",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "false_negative_rate": float(false_negative_rate),
        "false_positive_rate": float(false_positive_rate),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    with open(os.path.join(REPORT_DIR, "breast_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_test, y_pred)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal/Benign", "Malignant"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal/Benign", "Malignant"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "breast_confusion_matrix.png"), dpi=150)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "breast_roc_curve.png"), dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history.history["auc"], label="train")
    axes[1].plot(history.history["val_auc"], label="val")
    axes[1].set_title("AUC"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "breast_training_history.png"), dpi=150)

    print("Reports written to", REPORT_DIR)


if __name__ == "__main__":
    main()
