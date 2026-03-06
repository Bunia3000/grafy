from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, roc_auc_score
)


def list_npz(root: Path, cls: str):
    return sorted((root / cls).glob("*.npz"))


def apply_preprocess(A: np.ndarray, pp: dict) -> np.ndarray:
    mode = pp["mode_base"]
    eps = float(pp.get("eps", 1e-6))
    X = A.astype(np.float32)

    if "clip" in mode:
        X = np.clip(X, float(pp["clip_lo"]), float(pp["clip_hi"]))
    if "log" in mode:
        X = np.sign(X) * np.log1p(np.abs(X))
    if "zscore" in mode:
        X = (X - float(pp["mean"])) / (float(pp["std"]) + eps)

    return X


def features_from_A(A: np.ndarray) -> np.ndarray:
    feats = []
    for loop in A:
        feats.extend([
            float(np.mean(loop)),
            float(np.std(loop)),
            float(np.median(loop)),
            float(np.min(loop)),
            float(np.max(loop)),
            float(np.percentile(loop, 5)),
            float(np.percentile(loop, 95)),
            float(np.sum(loop**2)),
            float(np.sum(np.abs(loop))),
            float(np.mean(np.abs(np.diff(loop)))),
        ])

    # korelacje między pętlami (mogą być NaN jeśli std=0 => zabezpieczmy)
    def safe_corr(x, y):
        sx = float(np.std(x))
        sy = float(np.std(y))
        if sx < 1e-12 or sy < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    feats.extend([
        safe_corr(A[0], A[1]),
        safe_corr(A[0], A[2]),
        safe_corr(A[1], A[2]),
    ])

    return np.array(feats, dtype=np.float32)


def build_xy(files, variant: str, pp: dict, class_to_id: dict):
    X, y = [], []
    for f in files:
        data = np.load(f, allow_pickle=False)
        A = data[variant]
        A = apply_preprocess(A, pp)
        X.append(features_from_A(A))
        y.append(class_to_id[f.parent.name])
    return np.stack(X), np.array(y, dtype=np.int64)


def split_per_class(files1, files2, seed: int, train_frac=0.7, val_frac=0.15):
    rng = np.random.default_rng(seed)

    files1 = np.array(files1, dtype=object)
    files2 = np.array(files2, dtype=object)

    n = min(len(files1), len(files2))
    files1 = files1[:n]
    files2 = files2[:n]

    rng.shuffle(files1)
    rng.shuffle(files2)

    def split_one(arr):
        n = len(arr)
        n_tr = int(train_frac * n)
        n_va = int(val_frac * n)
        tr = arr[:n_tr]
        va = arr[n_tr:n_tr + n_va]
        te = arr[n_tr + n_va:]
        return tr.tolist(), va.tolist(), te.tolist()

    tr1, va1, te1 = split_one(files1)
    tr2, va2, te2 = split_one(files2)

    train = tr1 + tr2
    test = te1 + te2

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--variant", default="StA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preprocess_json", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=20000)
    args = ap.parse_args()

    root = Path(args.dataset)
    c1, c2 = args.classes
    class_to_id = {c1: 0, c2: 1}

    pp = json.load(open(args.preprocess_json, "r", encoding="utf-8"))

    f1 = list_npz(root, c1)
    f2 = list_npz(root, c2)

    train_files, test_files = split_per_class(f1, f2, seed=args.seed)

    X_train, y_train = build_xy(train_files, args.variant, pp, class_to_id)
    X_test, y_test = build_xy(test_files, args.variant, pp, class_to_id)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",
            C=args.C,
            max_iter=args.max_iter,
            n_jobs=None
        ))
    ])

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    bal = balanced_accuracy_score(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    cm = confusion_matrix(y_test, pred)

    print("\n===== FEATURE BASELINE (scaled LR) =====\n")
    print(f"n_train={len(y_train)}  n_test={len(y_test)}  (per class equalized)")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bal:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")
    print("\nConfusion matrix [true rows, pred cols]:")
    print(cm)


if __name__ == "__main__":
    main()