"""Shared evaluation: accuracy with a 95% CI, per-class P/R/F1, confusion matrix."""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src.config import CLASS_NAMES, RESULTS_DIR

SHORT = [c.replace("_Apple", "") for c in CLASS_NAMES]


def wilson_ci(correct, n, z=1.96):
    """95% Wilson score interval for a proportion (honest error bars for a small test set)."""
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def evaluate(name, y_true, y_pred, split="test", extra=None):
    acc = accuracy_score(y_true, y_pred)
    lo, hi = wilson_ci(int((y_true == y_pred).sum()), len(y_true))
    report = classification_report(y_true, y_pred, labels=range(len(CLASS_NAMES)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    result = {
        "model": name, "split": split, "n": int(len(y_true)),
        "accuracy": acc, "accuracy_ci95": [lo, hi],
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
        **(extra or {}),
    }
    out = RESULTS_DIR / "per_model"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_{split}.json").write_text(json.dumps(result, indent=2))
    plot_confusion(cm, f"{name} ({split}, n={len(y_true)}, acc={acc:.1%})",
                   out / f"{name}_{split}_confusion.png")
    print(f"[{split}] {name}: acc={acc:.3f} (95% CI {lo:.3f}-{hi:.3f}) macroF1={result['macro_f1']:.3f}")
    return result


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(SHORT)), SHORT)
    ax.set_yticks(range(len(SHORT)), SHORT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
