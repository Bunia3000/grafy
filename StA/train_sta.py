# train_sta.py
# One unified runner for StA experiments (3 loops per graph).
# Supports:
#   --model baseline_lr | cnn_shared
#   --variant <npz_key>  (e.g. StA, StA_raw, StA_smooth)
#   --aug none | perm
#   --classes 3_1 4_1
#   --n_per_class 695  (subsample bigger class to match smaller)
#
# Outputs (in --out dir):
#   splits: train_files.txt, val_files.txt, test_files.txt
#   metrics: confusion_matrix.csv, classification_report.txt, predictions.csv
#   models: best_model/, final_model/
#   run_summary.json
#   preprocess.json (when *_train preprocessing is used)
#   results.csv  (append one line per run)
#
# Debug:
#   --debug_overfit_k K
#   --debug_overfit_epochs E
#   -> trains only on 2K samples (K per class) taken from TRAIN split and evaluates on the same set

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf


# ------------------------ utils: files & splits ------------------------

def list_class_files(root: Path, class_label: str) -> List[Path]:
    class_dir = root / class_label
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")
    files = sorted(class_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in: {class_dir}")
    return files


def stratified_split_two_classes_equal_n(
    root: Path,
    c1: str,
    c2: str,
    n_per_class: int,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[List[Path], List[Path], List[Path], int]:
    rng = random.Random(seed)

    f1 = list_class_files(root, c1)
    f2 = list_class_files(root, c2)
    rng.shuffle(f1)
    rng.shuffle(f2)

    effective_n = min(n_per_class, len(f1), len(f2))
    f1 = f1[:effective_n]
    f2 = f2[:effective_n]

    def _split_one(lst: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        n = len(lst)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train = lst[:n_train]
        val = lst[n_train:n_train + n_val]
        test = lst[n_train + n_val:]
        return train, val, test

    tr1, va1, te1 = _split_one(f1)
    tr2, va2, te2 = _split_one(f2)

    train = tr1 + tr2
    val = va1 + va2
    test = te1 + te2

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test, effective_n


def save_file_list(paths: List[Path], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")


# ------------------------ metrics ------------------------

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_from_cm(cm: np.ndarray) -> dict:
    n_classes = cm.shape[0]
    per_class = []
    supports = cm.sum(axis=1)

    for k in range(n_classes):
        tp = int(cm[k, k])
        fp = int(cm[:, k].sum() - tp)
        fn = int(cm[k, :].sum() - tp)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class.append({
            "class": int(k),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(supports[k]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    macro_p = float(np.mean([d["precision"] for d in per_class]))
    macro_r = float(np.mean([d["recall"] for d in per_class]))
    macro_f1 = float(np.mean([d["f1"] for d in per_class]))

    total = float(np.sum(supports)) if np.sum(supports) > 0 else 1.0
    weighted_p = float(np.sum([d["precision"] * d["support"] for d in per_class]) / total)
    weighted_r = float(np.sum([d["recall"] * d["support"] for d in per_class]) / total)
    weighted_f1 = float(np.sum([d["f1"] * d["support"] for d in per_class]) / total)

    return {
        "per_class": per_class,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
    }


def evaluate_and_save_metrics(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    out_dir: Path,
    id_to_class: Dict[int, str],
) -> dict:
    y_true_all = []
    y_pred_all = []

    for Xb, yb in test_ds:
        probs = model.predict(Xb, verbose=0)
        pred = np.argmax(probs, axis=1).astype(np.int64)
        y_true_all.append(yb.numpy().astype(np.int64))
        y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    n_classes = len(id_to_class)
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=n_classes)
    stats = precision_recall_f1_from_cm(cm)

    cm_path = out_dir / "confusion_matrix.csv"
    with cm_path.open("w", encoding="utf-8") as f:
        f.write("true\\pred," + ",".join(id_to_class[i] for i in range(n_classes)) + "\n")
        for i in range(n_classes):
            row = ",".join(str(int(x)) for x in cm[i])
            f.write(f"{id_to_class[i]},{row}\n")

    report_path = out_dir / "classification_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Per-class metrics:\n")
        for d in stats["per_class"]:
            name = id_to_class[d["class"]]
            f.write(
                f"- {name:>6}  precision={d['precision']:.4f}  recall={d['recall']:.4f}  "
                f"f1={d['f1']:.4f}  support={d['support']}  tp={d['tp']} fp={d['fp']} fn={d['fn']}\n"
            )
        f.write("\nMacro avg:\n")
        f.write(
            f"precision={stats['macro']['precision']:.4f}  recall={stats['macro']['recall']:.4f}  "
            f"f1={stats['macro']['f1']:.4f}\n"
        )
        f.write("\nWeighted avg:\n")
        f.write(
            f"precision={stats['weighted']['precision']:.4f}  recall={stats['weighted']['recall']:.4f}  "
            f"f1={stats['weighted']['f1']:.4f}\n"
        )

    pred_path = out_dir / "predictions.csv"
    with pred_path.open("w", encoding="utf-8") as f:
        f.write("y_true_id,y_true_label,y_pred_id,y_pred_label\n")
        for t, p in zip(y_true, y_pred):
            f.write(f"{int(t)},{id_to_class[int(t)]},{int(p)},{id_to_class[int(p)]}\n")

    return {"confusion_matrix": cm.tolist(), "metrics": stats}


# ------------------------ preprocessing ------------------------

def preprocess_sta_per_sample(A: np.ndarray, mode: str, clip_lo: float, clip_hi: float, eps: float) -> np.ndarray:
    """
    PER-SAMPLE preprocessing (debug only).
    A: (3,L)
    clip_lo/clip_hi are quantiles (0..1).
    """
    X = A.astype(np.float32, copy=False)

    if mode in ("clip_zscore", "clip_log_zscore"):
        lo = np.quantile(X, clip_lo)
        hi = np.quantile(X, clip_hi)
        X = np.clip(X, lo, hi)

    if mode in ("log_zscore", "clip_log_zscore"):
        X = np.sign(X) * np.log1p(np.abs(X))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        m = float(X.mean())
        s = float(X.std())
        X = (X - m) / (s + eps)

    return X.astype(np.float32, copy=False)


def fit_preprocess_params_train(
    train_files: List[Path],
    variant_key: str,
    mode: str,
    clip_qlo: float,
    clip_qhi: float,
    eps: float,
) -> dict:
    """
    Fit preprocessing parameters on TRAIN ONLY.
    mode in ["zscore", "clip_zscore", "log_zscore", "clip_log_zscore"]
    """
    vals = []
    for p in train_files:
        d = np.load(str(p), allow_pickle=False)
        A = d[variant_key].astype(np.float32)  # (3,L)
        vals.append(A.reshape(-1))
    Xall = np.concatenate(vals, axis=0).astype(np.float32, copy=False)

    params = {
        "mode_base": mode,
        "clip_qlo": float(clip_qlo),
        "clip_qhi": float(clip_qhi),
        "clip_lo": None,
        "clip_hi": None,
        "mean": None,
        "std": None,
        "eps": float(eps),
    }

    if mode in ("clip_zscore", "clip_log_zscore"):
        lo = float(np.quantile(Xall, clip_qlo))
        hi = float(np.quantile(Xall, clip_qhi))
        params["clip_lo"] = lo
        params["clip_hi"] = hi
        Xall = np.clip(Xall, lo, hi)

    if mode in ("log_zscore", "clip_log_zscore"):
        Xall = np.sign(Xall) * np.log1p(np.abs(Xall))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        params["mean"] = float(Xall.mean())
        params["std"] = float(Xall.std())

    return params


def apply_preprocess_sta_global(A: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply train-fitted preprocessing to one sample.
    A: (3,L)
    """
    X = A.astype(np.float32, copy=False)
    mode = params["mode_base"]
    eps = float(params.get("eps", 1e-6))

    if mode in ("clip_zscore", "clip_log_zscore"):
        lo = float(params["clip_lo"])
        hi = float(params["clip_hi"])
        X = np.clip(X, lo, hi)

    if mode in ("log_zscore", "clip_log_zscore"):
        X = np.sign(X) * np.log1p(np.abs(X))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        m = float(params["mean"])
        s = float(params["std"])
        X = (X - m) / (s + eps)

    return X.astype(np.float32, copy=False)


# ------------------------ loader (StA) ------------------------

def _np_load_sta_with_aug(
    path_bytes: bytes,
    class_to_id,
    expected_L,
    seed,
    variant_key,
    aug,
    preprocess_mode: str,
    # NOTE: for *_train these are absolute values (clip bounds + mean/std), for per-sample they are quantiles
    clip_lo: float,
    clip_hi: float,
    pp_mean: float,
    pp_std: float,
    norm_eps: float,
):
    path = Path(path_bytes.decode("utf-8"))
    class_label = path.parent.name

    d = np.load(str(path), allow_pickle=False)
    A = d[variant_key].astype(np.float32)  # (3, L)

    # preprocess (before aug)
    if preprocess_mode.endswith("_train"):
        base = preprocess_mode.replace("_train", "")
        params = {
            "mode_base": base,
            "clip_lo": float(clip_lo),
            "clip_hi": float(clip_hi),
            "mean": float(pp_mean),
            "std": float(pp_std),
            "eps": float(norm_eps),
        }
        A = apply_preprocess_sta_global(A, params)
    else:
        if preprocess_mode == "none":
            pass
        else:
            # per-sample (debug)
            A = preprocess_sta_per_sample(A, mode=preprocess_mode, clip_lo=clip_lo, clip_hi=clip_hi, eps=norm_eps)

    # expected_L check
    L = A.shape[1]
    if expected_L is not None and L != expected_L:
        raise ValueError(f"Unexpected L={L} in {path}; expected {expected_L}")

    # deterministic RNG per file (for reproducible permutation)
    h = abs(hash(str(path))) % (2**31 - 1)
    rng = np.random.default_rng(seed + h)

    # aug: only permute loops (NO reverse)
    if aug == "perm":
        perm = rng.permutation(3)
        A = A[perm, :]

    y = np.int64(class_to_id[class_label])
    return A.astype(np.float32), y


def make_tf_dataset(
    files,
    class_to_id,
    expected_L,
    bs,
    shuffle,
    seed,
    variant_key,
    aug,
    preprocess_mode: str,
    clip_lo: float,
    clip_hi: float,
    pp_mean: float,
    pp_std: float,
    norm_eps: float,
):
    paths = tf.constant([str(p) for p in files])
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(files), seed=seed, reshuffle_each_iteration=True)

    def _map_fn(p):
        X, y = tf.numpy_function(
            func=lambda pb: _np_load_sta_with_aug(
                pb,
                class_to_id,
                expected_L,
                seed,
                variant_key,
                aug,
                preprocess_mode,
                clip_lo,
                clip_hi,
                pp_mean,
                pp_std,
                norm_eps,
            ),
            inp=[p],
            Tout=[tf.float32, tf.int64],
        )

        X = tf.expand_dims(X, axis=-1)  # (3, L, 1)
        if expected_L is not None:
            X.set_shape((3, expected_L, 1))
        else:
            X.set_shape((3, None, 1))
        y.set_shape(())
        return X, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(bs).prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------ models ------------------------

@dataclass
class CnnCfg:
    expected_L: int
    filters: int
    blocks: int
    kernel: int
    dropout: float
    dense_units: int
    pool: str       # avg | max | avgmax
    layernorm: bool


def build_shared_cnn_sta(cfg: CnnCfg, n_classes: int, lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(3, cfg.expected_L, 1), name="StA")

    loops = [
        tf.keras.layers.Lambda(lambda t, i=i: t[:, i, :, :], name=f"loop_{i}")(inp)
        for i in range(3)
    ]

    if cfg.layernorm:
        ln = tf.keras.layers.LayerNormalization(axis=[1, 2])  # normalize over (L,C)
        loops = [ln(l) for l in loops]

    convs, bns, drops = [], [], []
    for _ in range(cfg.blocks):
        convs.append(tf.keras.layers.Conv1D(cfg.filters, cfg.kernel, padding="same", activation=None))
        bns.append(tf.keras.layers.Lambda(lambda x: x))  # no BN
        drops.append(tf.keras.layers.Dropout(cfg.dropout) if cfg.dropout > 0 else tf.keras.layers.Lambda(lambda x: x))

    gap = tf.keras.layers.GlobalAveragePooling1D()
    gmp = tf.keras.layers.GlobalMaxPooling1D()

    def encode(x):
        for bi in range(cfg.blocks):
            x = convs[bi](x)
            x = bns[bi](x)
            x = tf.keras.layers.Activation("relu")(x)
            x = drops[bi](x)
        if cfg.pool == "avg":
            return gap(x)
        if cfg.pool == "max":
            return gmp(x)
        if cfg.pool == "avgmax":
            return tf.keras.layers.Concatenate()([gap(x), gmp(x)])
        raise ValueError(cfg.pool)

    emb = [encode(l) for l in loops]
    x = tf.keras.layers.Concatenate(name="concat_loops")(emb)

    if cfg.dense_units > 0:
        x = tf.keras.layers.Dense(cfg.dense_units, activation="relu")(x)
        if cfg.dropout > 0:
            x = tf.keras.layers.Dropout(cfg.dropout)(x)

    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="y")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@dataclass
class BaselineCfg:
    expected_L: int
    dense_units: int
    dropout: float
    layernorm: bool


def build_baseline_lr_sta(cfg: BaselineCfg, n_classes: int, lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(3, cfg.expected_L, 1), name="StA")
    x = tf.keras.layers.Reshape((3 * cfg.expected_L,))(inp)

    if cfg.layernorm:
        x = tf.keras.layers.LayerNormalization(axis=-1)(x)

    if cfg.dense_units > 0:
        x = tf.keras.layers.Dense(cfg.dense_units, activation="relu")(x)
        if cfg.dropout > 0:
            x = tf.keras.layers.Dropout(cfg.dropout)(x)

    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="y")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------ results.csv appender ------------------------

def append_results_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# ------------------------ debug: overfit subset ------------------------

def pick_overfit_subset(train_files: List[Path], c1: str, c2: str, k_per_class: int, seed: int) -> List[Path]:
    """
    Pick K files per class from TRAIN split only, deterministic by seed.
    """
    rng = random.Random(seed + 12345)
    f1 = [p for p in train_files if p.parent.name == c1]
    f2 = [p for p in train_files if p.parent.name == c2]
    rng.shuffle(f1)
    rng.shuffle(f2)
    if len(f1) < k_per_class or len(f2) < k_per_class:
        raise ValueError(f"Not enough train samples for overfit: {len(f1)} vs {len(f2)}, need {k_per_class} each.")
    subset = f1[:k_per_class] + f2[:k_per_class]
    rng.shuffle(subset)
    return subset


# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True, help="Root folder: dataset/StA/<class>/*.npz")
    ap.add_argument("--classes", nargs=2, default=["3_1", "4_1"])
    ap.add_argument("--n_per_class", type=int, default=695)
    ap.add_argument("--variant", default="StA", help="npz key: StA | StA_raw | StA_smooth etc.")
    ap.add_argument("--model", default="cnn_shared", choices=["baseline_lr", "cnn_shared"])
    ap.add_argument("--aug", default="none", choices=["none", "perm"])

    ap.add_argument("--expected_L", type=int, default=201)
    ap.add_argument("--split_train", type=float, default=0.8)
    ap.add_argument("--split_val", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)

    # shared CNN params
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--dense_units", type=int, default=64)
    ap.add_argument("--pool", default="avgmax", choices=["avg", "max", "avgmax"])
    ap.add_argument("--layernorm", action="store_true")

    ap.add_argument("--out", required=True, help="Run output dir (one run = one folder)")
    ap.add_argument("--results_csv", default=None, help="If set, append a line per run")

    # preprocessing
    ap.add_argument(
        "--preprocess",
        default="none",
        choices=[
            "none",
            # per-sample debug (NOT recommended for real runs)
            "zscore", "clip_zscore", "log_zscore", "clip_log_zscore",
            # train-fitted (recommended)
            "zscore_train", "clip_zscore_train", "log_zscore_train", "clip_log_zscore_train",
        ],
        help="Preprocessing for StA before augmentation/model. Use *_train for train-fitted preprocessing.",
    )
    ap.add_argument("--clip_lo", type=float, default=0.005, help="Lower quantile for clipping (used in clip_*).")
    ap.add_argument("--clip_hi", type=float, default=0.995, help="Upper quantile for clipping (used in clip_*).")
    ap.add_argument("--norm_eps", type=float, default=1e-6, help="Epsilon for normalization.")

    # debug overfit
    ap.add_argument("--debug_overfit_k", type=int, default=0, help="If >0: overfit on K samples per class from TRAIN only.")
    ap.add_argument("--debug_overfit_epochs", type=int, default=300, help="Epochs used for overfit mode.")

    args = ap.parse_args()

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    root = Path(args.dataset)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    c1, c2 = args.classes
    class_to_id = {c1: 0, c2: 1}
    id_to_class = {0: c1, 1: c2}

    train_files, val_files, test_files, effective_n = stratified_split_two_classes_equal_n(
        root=root,
        c1=c1,
        c2=c2,
        n_per_class=args.n_per_class,
        seed=args.seed,
        train_frac=args.split_train,
        val_frac=args.split_val,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_file_list(train_files, out_dir / "train_files.txt")
    save_file_list(val_files, out_dir / "val_files.txt")
    save_file_list(test_files, out_dir / "test_files.txt")

    # ---------- preprocessing params (train-fitted) ----------
    pp_params = None
    ds_clip_lo = float(args.clip_lo)
    ds_clip_hi = float(args.clip_hi)
    ds_mean = 0.0
    ds_std = 1.0

    if args.preprocess.endswith("_train"):
        base = args.preprocess.replace("_train", "")
        pp_params = fit_preprocess_params_train(
            train_files=train_files,
            variant_key=args.variant,
            mode=base,
            clip_qlo=float(args.clip_lo),
            clip_qhi=float(args.clip_hi),
            eps=float(args.norm_eps),
        )
        if pp_params["clip_lo"] is not None:
            ds_clip_lo = float(pp_params["clip_lo"])
            ds_clip_hi = float(pp_params["clip_hi"])
        if pp_params["mean"] is not None:
            ds_mean = float(pp_params["mean"])
            ds_std = float(pp_params["std"])

        with (out_dir / "preprocess.json").open("w", encoding="utf-8") as f:
            json.dump(pp_params, f, ensure_ascii=False, indent=2)

    # ---------- debug overfit subset ----------
    debug_subset: Optional[List[Path]] = None
    if args.debug_overfit_k and args.debug_overfit_k > 0:
        debug_subset = pick_overfit_subset(train_files, c1=c1, c2=c2, k_per_class=int(args.debug_overfit_k), seed=int(args.seed))
        save_file_list(debug_subset, out_dir / "debug_overfit_files.txt")
        print(f"[DEBUG] Overfit mode ON: {len(debug_subset)} files (k_per_class={args.debug_overfit_k}).")
        # In overfit mode we train AND evaluate on the same subset.
        train_for_fit = debug_subset
        val_for_fit = debug_subset
        test_for_eval = debug_subset
        epochs = int(args.debug_overfit_epochs)
    else:
        train_for_fit = train_files
        val_for_fit = val_files
        test_for_eval = test_files
        epochs = int(args.epochs)

    # Train aug only on train; in debug overfit we still typically set aug=none to not confuse the test
    train_aug = args.aug if (debug_subset is None) else "none"

    train_ds = make_tf_dataset(
        train_for_fit, class_to_id, args.expected_L, args.bs,
        shuffle=True, seed=args.seed,
        variant_key=args.variant,
        aug=train_aug,
        preprocess_mode=args.preprocess,
        clip_lo=ds_clip_lo,
        clip_hi=ds_clip_hi,
        pp_mean=ds_mean,
        pp_std=ds_std,
        norm_eps=args.norm_eps,
    )

    val_ds = make_tf_dataset(
        val_for_fit, class_to_id, args.expected_L, args.bs,
        shuffle=False, seed=args.seed,
        variant_key=args.variant,
        aug="none",
        preprocess_mode=args.preprocess,
        clip_lo=ds_clip_lo,
        clip_hi=ds_clip_hi,
        pp_mean=ds_mean,
        pp_std=ds_std,
        norm_eps=args.norm_eps,
    )

    test_ds = make_tf_dataset(
        test_for_eval, class_to_id, args.expected_L, args.bs,
        shuffle=False, seed=args.seed,
        variant_key=args.variant,
        aug="none",
        preprocess_mode=args.preprocess,
        clip_lo=ds_clip_lo,
        clip_hi=ds_clip_hi,
        pp_mean=ds_mean,
        pp_std=ds_std,
        norm_eps=args.norm_eps,
    )

    # Build model
    if args.model == "cnn_shared":
        # In debug overfit: turn off dropout by force (so we can see if it can memorize)
        dropout_eff = 0.0 if (debug_subset is not None) else float(args.dropout)
        cfg = CnnCfg(
            expected_L=int(args.expected_L),
            filters=int(args.filters),
            blocks=int(args.blocks),
            kernel=int(args.kernel),
            dropout=float(dropout_eff),
            dense_units=int(args.dense_units),
            pool=str(args.pool),
            layernorm=bool(args.layernorm),
        )
        model = build_shared_cnn_sta(cfg, n_classes=2, lr=args.lr)
        model_cfg_blob = asdict(cfg)

    elif args.model == "baseline_lr":
        dropout_eff = 0.0 if (debug_subset is not None) else float(args.dropout)
        cfg = BaselineCfg(
            expected_L=int(args.expected_L),
            dense_units=int(args.dense_units),
            dropout=float(dropout_eff),
            layernorm=bool(args.layernorm),
        )
        model = build_baseline_lr_sta(cfg, n_classes=2, lr=args.lr)
        model_cfg_blob = asdict(cfg)

    else:
        raise NotImplementedError(f"{args.model} not supported.")

    ckpt_path = str(out_dir / "best_model")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, min_delta=1e-4),
        tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_best_only=True, monitor="val_loss"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    eval_out = model.evaluate(test_ds, verbose=0)
    test_loss = float(eval_out[0]) if isinstance(eval_out, (list, tuple)) else float(eval_out)
    test_acc = float(eval_out[1]) if isinstance(eval_out, (list, tuple)) and len(eval_out) > 1 else float("nan")

    metrics_blob = evaluate_and_save_metrics(model, test_ds, out_dir, id_to_class)

    final_path = out_dir / "final_model"
    model.save(str(final_path))

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(root),
        "classes": [c1, c2],
        "n_per_class_requested": int(args.n_per_class),
        "n_per_class_effective": int(effective_n),
        "splits": {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
        "debug_overfit": {
            "enabled": bool(debug_subset is not None),
            "k_per_class": int(args.debug_overfit_k),
            "epochs": int(epochs),
            "files_list": str(out_dir / "debug_overfit_files.txt") if debug_subset is not None else None,
        },
        "input": {"L": int(args.expected_L), "loops": 3, "representation": "StA", "variant_key": args.variant},
        "train_config": {
            "epochs": int(epochs),
            "batch_size": int(args.bs),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "split": [float(args.split_train), float(args.split_val), float(1.0 - args.split_train - args.split_val)],
        },
        "preprocess": {
            "mode": args.preprocess,
            "clip_lo": float(ds_clip_lo),
            "clip_hi": float(ds_clip_hi),
            "mean": float(ds_mean),
            "std": float(ds_std),
            "norm_eps": float(args.norm_eps),
            "train_fitted_params_path": str(out_dir / "preprocess.json") if args.preprocess.endswith("_train") else None,
        },
        "augmentations": {"aug": train_aug},
        "model": {"name": args.model, "config": model_cfg_blob},
        "results": {
            "eval_loss": test_loss,
            "eval_accuracy": test_acc,
            "confusion_matrix": metrics_blob["confusion_matrix"],
            "metrics": metrics_blob["metrics"],
        },
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "artifacts": {
            "final_model": str(final_path),
            "best_model": str(out_dir / "best_model"),
            "confusion_matrix_csv": str(out_dir / "confusion_matrix.csv"),
            "classification_report": str(out_dir / "classification_report.txt"),
            "predictions_csv": str(out_dir / "predictions.csv"),
            "train_files": str(out_dir / "train_files.txt"),
            "val_files": str(out_dir / "val_files.txt"),
            "test_files": str(out_dir / "test_files.txt"),
        },
    }

    summary_path = out_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.results_csv:
        row = {
            "timestamp": summary["timestamp"],
            "classes": f"{c1}_vs_{c2}",
            "n_per_class": effective_n,
            "variant": args.variant,
            "model": args.model,
            "preprocess": args.preprocess,
            "aug": train_aug,
            "seed": args.seed,
            "expected_L": args.expected_L,
            "epochs": epochs,
            "bs": args.bs,
            "lr": args.lr,
            "eval_accuracy": test_acc,
            "eval_loss": test_loss,
            "debug_overfit_k": int(args.debug_overfit_k),
            "out_dir": str(out_dir),
        }
        append_results_csv(Path(args.results_csv), row)

    print(f"\nDone. Eval acc={test_acc:.4f} loss={test_loss:.4f}")
    print(f"Saved: {summary_path}")

    train_eval = model.evaluate(train_ds, verbose=0)
    print(f"Train-eval acc={train_eval[1]:.4f} loss={train_eval[0]:.4f}")


if __name__ == "__main__":
    main()