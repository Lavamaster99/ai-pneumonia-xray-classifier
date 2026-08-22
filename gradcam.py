"""Grad-CAM: shows which pixels of a chest X-ray drove the model's prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).
"""

import numpy as np
import tensorflow as tf
import cv2


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str | None = None) -> np.ndarray:
    """img_array: shape (1, H, W, C), already preprocessed the same way as training."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        # Differentiate w.r.t. the recovered pre-sigmoid logit, not the
        # probability itself: for confident predictions (p near 0 or 1) the
        # sigmoid derivative p*(1-p) vanishes, which would otherwise zero out
        # the Grad-CAM gradient for exactly the most confident cases.
        prob = tf.clip_by_value(predictions[:, 0], 1e-7, 1 - 1e-7)
        loss = tf.math.log(prob) - tf.math.log(1 - prob)

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    # divide_no_nan (not "+ eps") because the raw Grad-CAM magnitude here is
    # tiny (~1e-9): a fixed epsilon would swamp the true max and flatten the
    # whole map toward zero instead of normalizing it to [0, 1].
    heatmap = tf.math.divide_no_nan(heatmap, tf.math.reduce_max(heatmap))
    return heatmap.numpy()


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """original_img: HxW or HxWx3 uint8 image (0-255). Returns HxWx3 uint8 overlay."""
    if original_img.ndim == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap_color, alpha, original_img, 1 - alpha, 0)
    return overlay
