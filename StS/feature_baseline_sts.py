# feature_baseline_sts.py
# Simple feature baseline for StS classification:
# - loads StS arrays from dataset/StS/<class>/*.npz
# - equalizes classes
# - train/test split by seed
# - extracts handcrafted features from each (3, L, L)
# - trains scaled Logistic Regression
# - reports acc / bal_acc / mcc / auc / confusion matrix
#
# Example:
# python feature_baseline_sts.py ^
#   --dataset "C:\Users\danil\.vscode\grafy\dataset\StS" ^
#   --classes 3_1 4_1 ^
#   --variant StS ^
#   --seed 0

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def list_class_files(root: Path, class_label: str) -> List[Path]:
    class_dir = root / class_label
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")
    files = sorted(class_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in: {class_dir}")
    return files


def equalized_train_test_split(
    dataset_root: Path,
    class_a: str,
    class_b: str,
    seed: int,
    test_frac: float = 0.15,
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    rng = np.random.default_rng(seed)

    fa = list_class_files(dataset_root, class_a)
    fb = list_class_files(dataset_root, class_b)

    n = min(len(fa), len(fb))
    fa = list(fa[:n])
    fb = list(fb[:n])

    rng.shuffle(fa)
    rng.shuffle(fb)

    n_test = max(1, int(round(n * test_frac)))
    n_train = n - n_test

    train_files = fa[:n_train] + fb[:n_train]
    train_y = [0] * n_train + [1] * n_train

    test_files = fa[n_train:] + fb[n_train:]
    test_y = [0] * n_test + [1] * n_test

    train_perm = rng.permutation(len(train_files))
    test_perm = rng.permutation(len(test_files))

    train_files = [train_files[i] for i in train_perm]
    train_y = [train_y[i] for i in train_perm]

    test_files = [test_files[i] for i in test_perm]
    test_y = [test_y[i] for i in test_perm]

    return train_files, train_y, test_files, test_y


def neighbor_mask(L: int, k: int) -> np.ndarray:
    """
    True outside |i-j| <= k band.
    """
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    return np.abs(i - j) > k


def extract_loop_features(M: np.ndarray, band_k: int = 1) -> List[float]:
    """
    Features for one loop matrix M of shape (L, L).
    """
    M = np.asarray(M, dtype=np.float32)
    L = M.shape[0]

    tri_u = np.triu_indices(L, k=1)
    vals_u = M[tri_u]

    diag = np.diag(M)
    off_mask = ~np.eye(L, dtype=bool)
    off_vals = M[off_mask]

    band_out_mask = neighbor_mask(L, band_k)
    band_out_vals = M[band_out_mask]

    antisym = M + M.T
    antisym_u = antisym[tri_u]

    absM = np.abs(M)

    feats = []

    # global stats
    feats += [
        float(M.mean()),
        float(M.std()),
        float(M.min()),
        float(M.max()),
        float(absM.max()),
        float(np.quantile(M, 0.01)),
        float(np.quantile(M, 0.05)),
        float(np.quantile(M, 0.50)),
        float(np.quantile(M, 0.95)),
        float(np.quantile(M, 0.99)),
    ]

    # off-diagonal / upper triangle
    feats += [
        float(vals_u.mean()),
        float(vals_u.std()),
        float(np.mean(np.abs(vals_u))),
        float(np.quantile(vals_u, 0.01)),
        float(np.quantile(vals_u, 0.99)),
    ]

    feats += [
        float(off_vals.mean()),
        float(off_vals.std()),
        float(np.mean(np.abs(off_vals))),
    ]

    # diagonal
    feats += [
        float(diag.mean()),
        float(diag.std()),
        float(np.max(np.abs(diag))),
    ]

    # outside neighbor band
    feats += [
        float(band_out_vals.mean()),
        float(band_out_vals.std()),
        float(np.mean(np.abs(band_out_vals))),
        float(np.max(np.abs(band_out_vals))),
    ]

    # antisymmetry
    feats += [
        float(np.mean(np.abs(antisym_u))),
        float(np.max(np.abs(antisym_u))),
        float(np.std(antisym_u)),
    ]

    # sparsity / sign
    feats += [
        float(np.mean(M == 0.0)),
        float(np.mean(M > 0.0)),
        float(np.mean(M < 0.0)),
    ]

    # norms
    feats += [
        float(np.linalg.norm(M, ord="fro")),
        float(np.sum(np.abs(M))),
        float(np.mean(np.abs(M))),
    ]

    # row/col summaries
    row_abs_sum = np.sum(np.abs(M), axis=1)
    col_abs_sum = np.sum(np.abs(M), axis=0)
    feats += [
        float(row_abs_sum.mean()),
        float(row_abs_sum.std()),
        float(row_abs_sum.max()),
        float(col_abs_sum.mean()),
        float(col_abs_sum.std()),
        float(col_abs_sum.max()),
    ]

    return feats


def extract_features_sts(A: np.ndarray, band_k: int = 1) -> np.ndarray:
    """
    A shape: (3, L, L)
    Returns 1D feature vector.
    """
    if A.ndim != 3 or A.shape[0] != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(f"Expected (3,L,L), got {A.shape}")

    feats = []
    loop_feature_vectors = []

    for loop_id in range(3):
        loop_feats = extract_loop_features(A[loop_id], band_k=band_k)
        loop_feature_vectors.append(loop_feats)
        feats.extend(loop_feats)

    lf = np.asarray(loop_feature_vectors, dtype=np.float32)  # (3, F)

    # aggregate across loops: mean/std/min/max for each feature position
    feats.extend(np.mean(lf, axis=0).tolist())
    feats.extend(np.std(lf, axis=0).tolist())
    feats.extend(np.min(lf, axis=0).tolist())
    feats.extend(np.max(lf, axis=0).tolist())

    # pairwise loop distances / similarities
    for i in range(3):
        for j in range(i + 1, 3):
            diff = A[i] - A[j]
            feats += [
                float(np.mean(np.abs(diff))),
                float(np.max(np.abs(diff))),
                float(np.linalg.norm(diff, ord="fro")),
                float(np.mean(np.abs((A[i] + A[i].T) - (A[j] + A[j].T)))),
            ]

    return np.asarray(feats, dtype=np.float32)


def load_Xy(files: List[Path], y: List[int], variant: str, band_k: int) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    for p in files:
        d = np.load(str(p), allow_pickle=False)
        if variant not in d:
            raise KeyError(f"Key '{variant}' not found in {p}")
        A = d[variant].astype(np.float32)
        feats = extract_features_sts(A, band_k=band_k)
        X.append(feats)
    return np.vstack(X), np.asarray(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Root folder: dataset/StS/<class>/*.npz")
    ap.add_argument("--classes", nargs=2, default=["3_1", "4_1"])
    ap.add_argument("--variant", default="StS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--neighbor_exclusion", type=int, default=1)
    ap.add_argument("--max_iter", type=int, default=5000)
    args = ap.parse_args()

    root = Path(args.dataset)
    c0, c1 = args.classes

    train_files, y_train, test_files, y_test = equalized_train_test_split(
        dataset_root=root,
        class_a=c0,
        class_b=c1,
        seed=args.seed,
        test_frac=args.test_frac,
    )

    X_train, y_train = load_Xy(train_files, y_train, args.variant, args.neighbor_exclusion)
    X_test, y_test = load_Xy(test_files, y_test, args.variant, args.neighbor_exclusion)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=args.max_iter,
                    solver="lbfgs",
                    random_state=args.seed,
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== FEATURE BASELINE StS (scaled LR) =====\n")
    print(f"n_train={len(y_train)}  n_test={len(y_test)}  (per class equalized)")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")
    print("\nConfusion matrix [true rows, pred cols]:")
    print(cm)


if __name__ == "__main__":
    main()