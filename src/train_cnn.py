"""Train a transfer-learning CNN on the train split, early-stopping on the val split.

The test split is NOT touched here; see src/evaluate_test.py.

Usage:
  python -m src.train_cnn --backbone resnet50
  python -m src.train_cnn --backbone inceptionv3
  python -m src.train_cnn --backbone inceptionv3 --no-preprocess   # ablation: 2023 bug
"""
import argparse
import json

import keras
import numpy as np

from src.config import MODELS_DIR, RESULTS_DIR, SEED
from src.data import load_split
from src.metrics import evaluate
from src.models import build_cnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["resnet50", "inceptionv3"], required=True)
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    name = args.backbone + ("_nopreprocess" if args.no_preprocess else "")
    x_tr, y_tr, _ = load_split("train")
    x_va, y_va, _ = load_split("val")

    model = build_cnn(args.backbone, preprocess=not args.no_preprocess)
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_tr.astype("float32"), y_tr,
        validation_data=(x_va.astype("float32"), y_va),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                                 restore_best_weights=True)],
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / f"{name}.keras")
    hist_dir = RESULTS_DIR / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    (hist_dir / f"{name}.json").write_text(json.dumps(
        {k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=1))

    y_pred = model.predict(x_va.astype("float32"), verbose=0).argmax(1)
    evaluate(name, y_va, y_pred, split="val",
             extra={"best_epoch": best_epoch, "hyperparameters": vars(args)})


if __name__ == "__main__":
    main()
