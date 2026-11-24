"""Classical baselines on flattened pixels: Decision Tree, Random Forest, KNN, SVC, soft-voting Ensemble.

Hyperparameters are chosen on the val split (PredefinedSplit), and each final
model is fitted on the train split only, the same data the CNNs see. The test
split is NOT touched here; see src/evaluate_test.py.

2023 bug fixed here: the original VotingClassifier(voting="soft") used SVC()
without probability=True, so every GridSearchCV fit failed (all scores NaN) and
the "best" hyperparameters were just the first grid entry.

Usage:  python -m src.train_classical
"""
import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import MODELS_DIR, SEED
from src.data import load_split
from src.metrics import evaluate

GRIDS = {
    "decision_tree": (DecisionTreeClassifier(random_state=SEED),
                      {"max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 4]}),
    "random_forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                      {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20],
                       "min_samples_leaf": [1, 2, 4]}),
    "knn": (KNeighborsClassifier(), {"n_neighbors": [1, 3, 5, 7, 9], "weights": ["uniform", "distance"]}),
    "svc": (SVC(probability=True, random_state=SEED),
            {"C": [0.1, 1, 10, 100], "gamma": ["scale"]}),
}


def flatten(x):
    return x.reshape(len(x), -1).astype("float32") / 255.0


def main():
    x_tr, y_tr, _ = load_split("train")
    x_va, y_va, _ = load_split("val")
    x_tr, x_va = flatten(x_tr), flatten(x_va)
    x_all = np.concatenate([x_tr, x_va])
    y_all = np.concatenate([y_tr, y_va])
    cv = PredefinedSplit(np.r_[np.full(len(x_tr), -1), np.zeros(len(x_va))])

    MODELS_DIR.mkdir(exist_ok=True)
    tuned = {}
    for name, (est, grid) in GRIDS.items():
        search = GridSearchCV(est, grid, cv=cv, scoring="accuracy", refit=False, n_jobs=1,
                              error_score="raise")
        search.fit(x_all, y_all)
        model = est.set_params(**search.best_params_).fit(x_tr, y_tr)  # fit on train only
        tuned[name] = model
        evaluate(name, y_va, model.predict(x_va), split="val",
                 extra={"best_params": search.best_params_})
        with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    ensemble = VotingClassifier([(n, m) for n, m in tuned.items()], voting="soft")
    ensemble.fit(x_tr, y_tr)
    evaluate("ensemble", y_va, ensemble.predict(x_va), split="val",
             extra={"members": list(tuned), "voting": "soft"})
    with open(MODELS_DIR / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)


if __name__ == "__main__":
    main()
