"""Final, one-time evaluation of every trained model on the held-out test split.

Run this once, after all tuning on the val split is finished. It writes
results/summary.csv, results/summary.md and results/confusion_matrices.png.

Usage:  python -m src.evaluate_test
"""
import json
import pickle

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import src.models  # noqa: F401  (registers the BackbonePreprocess layer for loading)
from src.config import MODELS_DIR, RESULTS_DIR
from src.data import load_split
from src.metrics import SHORT, evaluate

# (file stem, display name, original 2023 reported accuracy)
MODELS = [
    ("resnet50", "ResNet50 (transfer learning)", 0.80),
    ("inceptionv3", "InceptionV3 (transfer learning)", 0.55),
    ("ensemble", "Soft-voting ensemble (DT+RF+KNN+SVC)", 0.63),
    ("random_forest", "Random Forest", 0.69),
    ("knn", "K-Nearest Neighbors", 0.59),
    ("svc", "Support Vector Classifier", 0.55),
    ("decision_tree", "Decision Tree", 0.53),
]
ABLATIONS = [
    ("resnet50_nopreprocess", "ResNet50, no preprocess_input (2023 setup)"),
    ("inceptionv3_nopreprocess", "InceptionV3, no preprocess_input (2023 bug)"),
]


def predict(stem, x):
    keras_path = MODELS_DIR / f"{stem}.keras"
    if keras_path.exists():
        return keras.models.load_model(keras_path).predict(x.astype("float32"), verbose=0).argmax(1)
    with open(MODELS_DIR / f"{stem}.pkl", "rb") as f:
        return pickle.load(f).predict(x.reshape(len(x), -1).astype("float32") / 255.0)


def main():
    x_te, y_te, _ = load_split("test")
    rows, results = [], {}
    for stem, display, *orig in MODELS + ABLATIONS:
        r = evaluate(stem, y_te, predict(stem, x_te), split="test")
        val = json.loads((RESULTS_DIR / "per_model" / f"{stem}_val.json").read_text())
        results[stem] = r
        rows.append({
            "model": display,
            "kind": "ablation" if not orig else "main",
            "reported_2023": orig[0] if orig else None,
            "val_accuracy": val["accuracy"],
            "test_accuracy": r["accuracy"],
            "test_ci95_low": r["accuracy_ci95"][0],
            "test_ci95_high": r["accuracy_ci95"][1],
            "test_macro_f1": r["macro_f1"],
            **{f"f1_{s}": r["per_class"][c]["f1-score"] for s, c in zip(SHORT, r["per_class"])},
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    def pct(v):
        return "–" if pd.isna(v) else f"{v:.1%}"
    lines = ["| Model | 2023 reported | Val acc | **Test acc** (95% CI) | Test macro-F1 | "
             + " | ".join(f"F1 {s}" for s in SHORT) + " |",
             "|---|---|---|---|---|" + "---|" * len(SHORT)]
    for row in rows:
        lines.append(
            f"| {row['model']} | {pct(row['reported_2023'])} | {pct(row['val_accuracy'])} | "
            f"**{pct(row['test_accuracy'])}** ({row['test_ci95_low']:.0%}–{row['test_ci95_high']:.0%}) | "
            f"{row['test_macro_f1']:.3f} | " + " | ".join(f"{row[f'f1_{s}']:.2f}" for s in SHORT) + " |")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Combined confusion-matrix figure for the README (main models only).
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (stem, display, _) in zip(axes.flat, MODELS):
        cm = results[stem]["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4), SHORT)
        ax.set_yticks(range(4), SHORT)
        ax.set_title(f"{display.split(' (')[0]}\n{results[stem]['accuracy']:.1%}", fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, cm[i][j], ha="center", va="center",
                        color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
    axes.flat[-1].axis("off")
    fig.supxlabel("Predicted")
    fig.supylabel("True")
    fig.suptitle(f"Test-set confusion matrices (n={len(y_te)})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=130)


if __name__ == "__main__":
    main()
