"""Grad-CAM heatmaps showing which image regions drive the ResNet50 prediction.

Usage:  python -m src.gradcam      -> results/gradcam.png (two test images per class)
"""
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import src.models  # noqa: F401
from src.config import CLASS_NAMES, MODELS_DIR, RESULTS_DIR, SEED
from src.data import load_split
from src.models import BackbonePreprocess


def gradcam(model, image, last_conv="conv5_block3_out"):
    base = model.get_layer("resnet50")
    pre = next(l for l in model.layers if isinstance(l, BackbonePreprocess))
    head = model.layers[model.layers.index(base) + 1:]
    conv_model = keras.Model(base.inputs, [base.get_layer(last_conv).output, base.output])
    x = pre(tf.convert_to_tensor(image[None].astype("float32")))
    with tf.GradientTape() as tape:
        conv, feats = conv_model(x, training=False)
        out = feats
        for layer in head:
            out = layer(out, training=False)
        cls = int(tf.argmax(out[0]))
        score = out[:, cls]
    grads = tape.gradient(score, conv)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., None], image.shape[:2]).numpy()[..., 0]
    return cls, float(out[0, cls]), cam


def main():
    model = keras.models.load_model(MODELS_DIR / "resnet50.keras")
    x, y, _ = load_split("test")
    rng = np.random.default_rng(SEED)
    picks = [i for c in range(len(CLASS_NAMES)) for i in rng.choice(np.where(y == c)[0], 2, replace=False)]
    fig, axes = plt.subplots(2, len(picks), figsize=(2.2 * len(picks), 5.6))
    for k, i in enumerate(picks):
        cls, prob, cam = gradcam(model, x[i])
        short = lambda c: CLASS_NAMES[c].replace("_Apple", "")
        axes[0, k].imshow(x[i])
        axes[0, k].set_title(f"true: {short(y[i])}", fontsize=8)
        axes[1, k].imshow(x[i])
        axes[1, k].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, k].set_title(f"pred: {short(cls)} ({prob:.0%})", fontsize=8,
                             color="green" if cls == y[i] else "red")
        for a in axes[:, k]:
            a.axis("off")
    fig.suptitle("ResNet50 Grad-CAM on test images", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "gradcam.png", dpi=130)


if __name__ == "__main__":
    main()
