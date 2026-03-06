from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ------------------------ metrics helpers ------------------------

def cm_from_summary(summary: dict) -> np.ndarray:
    cm = np.array(summary["results"]["confusion_matrix"], dtype=np.int64)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got {cm.shape}")
    return cm


def accuracy_from_cm(cm: np.ndarray) -> float:
    total = cm.sum()
    return float((cm[0, 0] + cm[1, 1]) / total) if total else 0.0


def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    # mean recall across classes
    recalls = []
    for k in range(2):
        denom = cm[k, :].sum()
        recalls.append(float(cm[k, k] / denom) if denom else 0.0)
    return float(sum(recalls) / 2.0)


def mcc_from_cm(cm: np.ndarray) -> float:
    # Matthews correlation coefficient for binary
    tp = float(cm[1, 1])
    tn = float(cm[0, 0])
    fp = float(cm[0, 1])
    fn = float(cm[1, 0])
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def fmt_mean_std(xs: List[float]) -> str:
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{mean(xs):.4f} ± {pstdev(xs):.4f}"


# ------------------------ optional: feature baseline ------------------------

def try_feature_baseline(
    dataset_root: Path,
    classes: Tuple[str, str],
    variant: str,
    n_per_class: int,
    expected_L: int,
    seed: int,
    preprocess_json: Optional[Path],
    out_dir: Path,
) -> Optional[dict]:
    """
    Feature engineering baseline using sklearn LogisticRegression if available.
    Uses the SAME split as train_sta.py (seed-driven shuffle) by reading train/val/test_files.txt from out_dir.

    If sklearn isn't installed -> returns None.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix
    except Exception:
        print("[feature-baseline] sklearn not available -> skipping feature baseline")
        return None

    # Load split lists created by train_sta.py
    train_list = (out_dir / "train_files.txt")
    val_list = (out_dir / "val_files.txt")
    test_list = (out_dir / "test_files.txt")
    if not (train_list.exists() and val_list.exists() and test_list.exists()):
        print("[feature-baseline] split files not found -> skipping")
        return None

    def read_paths(p: Path) -> List[Path]:
        return [Path(line.strip()) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    train_files = read_paths(train_list)
    test_files = read_paths(test_list)

    # load preprocess params if present (recommended)
    pp = None
    if preprocess_json and preprocess_json.exists():
        pp = json.loads(preprocess_json.read_text(encoding="utf-8"))

    c1, c2 = classes
    class_to_id = {c1: 0, c2: 1}

    def apply_preprocess(A: np.ndarray) -> np.ndarray:
        # If no pp -> identity
        if pp is None:
            return A.astype(np.float32, copy=False)

        mode = pp["mode_base"]
        eps = float(pp.get("eps", 1e-6))
        X = A.astype(np.float32, copy=False)

        if mode in ("clip_zscore", "clip_log_zscore"):
            lo = float(pp["clip_lo"])
            hi = float(pp["clip_hi"])
            X = np.clip(X, lo, hi)

        if mode in ("log_zscore", "clip_log_zscore"):
            X = np.sign(X) * np.log1p(np.abs(X))

        if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
            m = float(pp["mean"])
            s = float(pp["std"])
            X = (X - m) / (s + eps)

        return X.astype(np.float32, copy=False)

    def feats_from_A(A: np.ndarray) -> np.ndarray:
        # A shape (3, L)
        # Per-loop stats + cross-loop correlations
        feats = []
        for i in range(3):
            x = A[i]
            feats.extend([
                float(np.mean(x)),
                float(np.std(x)),
                float(np.min(x)),
                float(np.max(x)),
                float(np.quantile(x, 0.05)),
                float(np.quantile(x, 0.50)),
                float(np.quantile(x, 0.95)),
                float(np.sum(x * x)),             # energy
                float(np.sum(np.abs(x))),         # L1
                float(np.mean(np.abs(np.diff(x)))),# smoothness
                float(np.sum(np.sign(x[1:]) != np.sign(x[:-1]))),  # sign changes
            ])
        # correlations
        def corr(a, b) -> float:
            sa = float(np.std(a))
            sb = float(np.std(b))
            if sa == 0.0 or sb == 0.0:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        feats.extend([
            corr(A[0], A[1]),
            corr(A[0], A[2]),
            corr(A[1], A[2]),
        ])
        # energy diffs
        E = [float(np.sum(A[i] * A[i])) for i in range(3)]
        feats.extend([E[0] - E[1], E[0] - E[2], E[1] - E[2]])

        return np.array(feats, dtype=np.float32)

    def load_xy(files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for fp in files:
            d = np.load(str(fp), allow_pickle=False)
            A = d[variant].astype(np.float32)
            if A.shape != (3, expected_L):
                raise ValueError(f"Unexpected shape {A.shape} in {fp}")
            A = apply_preprocess(A)
            Xs.append(feats_from_A(A))
            ys.append(class_to_id[fp.parent.name])
        return np.stack(Xs, axis=0), np.array(ys, dtype=np.int64)

    Xtr, ytr = load_xy(train_files)
    Xte, yte = load_xy(test_files)

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    cm = confusion_matrix(yte, ypred, labels=[0, 1]).astype(np.int64)

    out = {
        "feature_baseline": {
            "cm": cm.tolist(),
            "acc": accuracy_from_cm(cm),
            "bal_acc": balanced_accuracy_from_cm(cm),
            "mcc": mcc_from_cm(cm),
            "n_features": int(Xtr.shape[1]),
        }
    }

    (out_dir / "feature_baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ------------------------ sweep runner ------------------------

@dataclass
class RunResult:
    run_dir: Path
    seed: int
    model: str
    preprocess: str
    test_acc: float
    test_loss: float
    bal_acc: float
    mcc: float
    cm: np.ndarray


def run_one(
    python_exe: str,
    train_script: Path,
    dataset: Path,
    classes: Tuple[str, str],
    n_per_class: int,
    variant: str,
    expected_L: int,
    model: str,
    preprocess: str,
    clip_lo: float,
    clip_hi: float,
    seed: int,
    out_dir: Path,
    extra_args: List[str],
) -> RunResult:
    cmd = [
        python_exe,
        str(train_script),
        "--dataset", str(dataset),
        "--classes", classes[0], classes[1],
        "--n_per_class", str(n_per_class),
        "--variant", variant,
        "--model", model,
        "--expected_L", str(expected_L),
        "--seed", str(seed),
        "--preprocess", preprocess,
        "--clip_lo", str(clip_lo),
        "--clip_hi", str(clip_hi),
        "--out", str(out_dir),
        "--aug", "none",
    ] + extra_args

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"train_sta.py failed (seed={seed}, model={model}, preprocess={preprocess}). "
                           f"See {out_dir/'stderr.txt'}")

    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run_summary.json in {out_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cm = cm_from_summary(summary)
    
    results = summary.get("results", {})

    def _pick_float(keys: List[str]) -> float:
        for k in keys:
            if k in results and results[k] is not None:
                return float(results[k])
        # fallback: czasem ktoś zapisuje wprost na top-level
        for k in keys:
            if k in summary and summary[k] is not None:
                return float(summary[k])
        raise KeyError(f"None of keys found in summary['results']: {keys}. "
                       f"Available results keys: {sorted(list(results.keys()))}")

    test_acc = _pick_float(["test_accuracy", "eval_accuracy", "eval_acc", "accuracy"])
    test_loss = _pick_float(["test_loss", "eval_loss", "loss"])

    bal_acc = balanced_accuracy_from_cm(cm)
    mcc = mcc_from_cm(cm)

    return RunResult(
        run_dir=out_dir,
        seed=seed,
        model=model,
        preprocess=preprocess,
        test_acc=test_acc,
        test_loss=test_loss,
        bal_acc=bal_acc,
        mcc=mcc,
        cm=cm,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--train_script", required=True, help="Path to train_sta.py")
    ap.add_argument("--dataset", required=True, help="Root folder: dataset/StA")
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--n_per_class", type=int, default=695)
    ap.add_argument("--variant", default="StA")
    ap.add_argument("--expected_L", type=int, default=201)

    ap.add_argument("--out_base", required=True, help="Base folder for sweep runs")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")

    ap.add_argument(
        "--models",
        default="baseline_lr,cnn_shared",
        help="Comma-separated: baseline_lr,cnn_shared",
    )
    ap.add_argument(
        "--preprocess_list",
        default="clip_zscore_train,clip_log_zscore_train",
        help="Comma-separated preprocess modes",
    )
    ap.add_argument("--clip_lo", type=float, default=0.001)
    ap.add_argument("--clip_hi", type=float, default=0.999)

    # You can pass any extra args to train_sta.py here (e.g. CNN hyperparams)
    ap.add_argument("--extra", default="", help="Extra args appended to train_sta.py command line")

    ap.add_argument("--do_feature_baseline", action="store_true", help="Also run feature-engineering baseline (sklearn)")
    args = ap.parse_args()

    python_exe = args.python
    train_script = Path(args.train_script)
    dataset = Path(args.dataset)
    out_base = Path(args.out_base)

    classes = (args.classes[0], args.classes[1])
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    preprocess_list = [p.strip() for p in args.preprocess_list.split(",") if p.strip()]

    extra_args = []
    if args.extra.strip():
        # split on whitespace but keep quoted chunks if user provides them
        # easiest: just naive split, user can avoid spaces inside quotes by using ^ in CMD
        extra_args = args.extra.strip().split()

    out_base.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = []

    for preprocess in preprocess_list:
        for model in models:
            for seed in seeds:
                run_name = f"{classes[0]}_vs_{classes[1]}__{model}__{preprocess}__s{seed}"
                run_dir = out_base / run_name

                rr = run_one(
                    python_exe=python_exe,
                    train_script=train_script,
                    dataset=dataset,
                    classes=classes,
                    n_per_class=int(args.n_per_class),
                    variant=args.variant,
                    expected_L=int(args.expected_L),
                    model=model,
                    preprocess=preprocess,
                    clip_lo=float(args.clip_lo),
                    clip_hi=float(args.clip_hi),
                    seed=int(seed),
                    out_dir=run_dir,
                    extra_args=extra_args,
                )

                # Optional: feature baseline using the SAME split lists produced by train_sta.py
                if args.do_feature_baseline:
                    pp_json = run_dir / "preprocess.json"
                    try_feature_baseline(
                        dataset_root=dataset,
                        classes=classes,
                        variant=args.variant,
                        n_per_class=int(args.n_per_class),
                        expected_L=int(args.expected_L),
                        seed=int(seed),
                        preprocess_json=pp_json if pp_json.exists() else None,
                        out_dir=run_dir,
                    )

                results.append(rr)

    # Save per-run CSV
    sweep_csv = out_base / "sweep_results.csv"
    with sweep_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_dir", "seed", "model", "preprocess",
            "test_acc", "test_loss", "balanced_acc", "mcc",
            "cm_00", "cm_01", "cm_10", "cm_11",
        ])
        for r in results:
            w.writerow([
                str(r.run_dir), r.seed, r.model, r.preprocess,
                f"{r.test_acc:.6f}", f"{r.test_loss:.6f}", f"{r.bal_acc:.6f}", f"{r.mcc:.6f}",
                int(r.cm[0, 0]), int(r.cm[0, 1]), int(r.cm[1, 0]), int(r.cm[1, 1]),
            ])

    # Aggregate summary
    summary_lines = []
    summary_lines.append(f"Classes: {classes[0]} vs {classes[1]}")
    summary_lines.append(f"Seeds: {seeds}")
    summary_lines.append(f"Models: {models}")
    summary_lines.append(f"Preprocess: {preprocess_list}")
    summary_lines.append("")

    # group by (model, preprocess)
    for preprocess in preprocess_list:
        for model in models:
            grp = [r for r in results if r.model == model and r.preprocess == preprocess]
            accs = [r.test_acc for r in grp]
            bals = [r.bal_acc for r in grp]
            mccs = [r.mcc for r in grp]
            losses = [r.test_loss for r in grp]
            summary_lines.append(f"{model} | {preprocess}")
            summary_lines.append(f"  test_acc:     {fmt_mean_std(accs)}")
            summary_lines.append(f"  balanced_acc: {fmt_mean_std(bals)}")
            summary_lines.append(f"  mcc:          {fmt_mean_std(mccs)}")
            summary_lines.append(f"  test_loss:    {fmt_mean_std(losses)}")
            summary_lines.append("")

    (out_base / "sweep_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nSaved:")
    print(" -", sweep_csv)
    print(" -", out_base / "sweep_summary.txt")


if __name__ == "__main__":
    main()