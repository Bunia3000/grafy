# train_sta_sts_fusion.py
# Fusion model for StA + StS.
#
# Assumes matching files:
#   <sta_dataset>/<class>/<graph_id>.npz
#   <sts_dataset>/<class>/<graph_id>.npz
#
# Inputs:
#   StA: (3, L)      -> 1D shared CNN branch
#   StS: (3, L, L)   -> shared 2D CNN branch
#
# Output:
#   binary class prediction for classes like 3_1 vs 4_1

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


# ------------------------ file helpers ------------------------

def list_class_files(root: Path, class_label: str) -> List[Path]:
    class_dir = root / class_label
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")
    files = sorted(class_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in: {class_dir}")
    return files


def matched_class_files(sta_root: Path, sts_root: Path, class_label: str) -> List[Tuple[Path, Path]]:
    sta_files = list_class_files(sta_root, class_label)
    sts_files = list_class_files(sts_root, class_label)

    sta_map = {p.stem: p for p in sta_files}
    sts_map = {p.stem: p for p in sts_files}

    common = sorted(set(sta_map.keys()) & set(sts_map.keys()))
    if not common:
        raise RuntimeError(f"No matched StA/StS files for class {class_label}")

    pairs = [(sta_map[k], sts_map[k]) for k in common]
    return pairs


def stratified_split_two_classes_equal_n_pairs(
    sta_root: Path,
    sts_root: Path,
    c1: str,
    c2: str,
    n_per_class: int,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]], int]:
    rng = random.Random(seed)

    p1 = matched_class_files(sta_root, sts_root, c1)
    p2 = matched_class_files(sta_root, sts_root, c2)

    rng.shuffle(p1)
    rng.shuffle(p2)

    effective_n = min(n_per_class, len(p1), len(p2))
    p1 = p1[:effective_n]
    p2 = p2[:effective_n]

    def _split_one(lst):
        n = len(lst)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train = lst[:n_train]
        val = lst[n_train:n_train + n_val]
        test = lst[n_train + n_val:]
        return train, val, test

    tr1, va1, te1 = _split_one(p1)
    tr2, va2, te2 = _split_one(p2)

    train = tr1 + tr2
    val = va1 + va2
    test = te1 + te2

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test, effective_n


def save_pair_file_list(pairs: List[Tuple[Path, Path]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for sta_p, sts_p in pairs:
            f.write(f"{sta_p}\t{sts_p}\n")


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


def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    recalls = []
    for k in range(cm.shape[0]):
        tp = float(cm[k, k])
        fn = float(cm[k, :].sum() - cm[k, k])
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(recalls))


def matthews_corrcoef_binary_from_cm(cm: np.ndarray) -> float:
    if cm.shape != (2, 2):
        return float("nan")
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    num = (tp * tn) - (fp * fn)
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float(num / den) if den > 0 else 0.0


def evaluate_and_save_metrics(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    out_dir: Path,
    id_to_class: Dict[int, str],
) -> dict:
    y_true_all = []
    y_pred_all = []

    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0)
        pred = np.argmax(probs, axis=1).astype(np.int64)
        y_true_all.append(yb.numpy().astype(np.int64))
        y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    n_classes = len(id_to_class)
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=n_classes)
    stats = precision_recall_f1_from_cm(cm)
    bal_acc = balanced_accuracy_from_cm(cm)
    mcc = matthews_corrcoef_binary_from_cm(cm) if n_classes == 2 else float("nan")

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
        f.write(f"\nBalanced accuracy={bal_acc:.4f}\n")
        f.write(f"MCC={mcc:.4f}\n")

    pred_path = out_dir / "predictions.csv"
    with pred_path.open("w", encoding="utf-8") as f:
        f.write("y_true_id,y_true_label,y_pred_id,y_pred_label\n")
        for t, p in zip(y_true, y_pred):
            f.write(f"{int(t)},{id_to_class[int(t)]},{int(p)},{id_to_class[int(p)]}\n")

    return {
        "confusion_matrix": cm.tolist(),
        "metrics": stats,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
    }


# ------------------------ preprocessing: StA ------------------------

def fit_preprocess_params_train_sta(
    train_pairs: List[Tuple[Path, Path]],
    variant_key: str,
    mode: str,
    clip_qlo: float,
    clip_qhi: float,
    eps: float,
) -> dict:
    vals = []
    for sta_p, _ in train_pairs:
        d = np.load(str(sta_p), allow_pickle=False)
        X = d[variant_key].astype(np.float32)  # (3,L)
        vals.append(X.reshape(-1))
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


def apply_preprocess_sta_global(X: np.ndarray, params: dict) -> np.ndarray:
    A = X.astype(np.float32, copy=False)
    mode = params["mode_base"]
    eps = float(params.get("eps", 1e-6))

    if mode in ("clip_zscore", "clip_log_zscore"):
        A = np.clip(A, float(params["clip_lo"]), float(params["clip_hi"]))

    if mode in ("log_zscore", "clip_log_zscore"):
        A = np.sign(A) * np.log1p(np.abs(A))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        A = (A - float(params["mean"])) / (float(params["std"]) + eps)

    return A.astype(np.float32, copy=False)


# ------------------------ preprocessing: StS ------------------------

def fit_preprocess_params_train_sts(
    train_pairs: List[Tuple[Path, Path]],
    variant_key: str,
    mode: str,
    clip_qlo: float,
    clip_qhi: float,
    eps: float,
) -> dict:
    vals = []
    for _, sts_p in train_pairs:
        d = np.load(str(sts_p), allow_pickle=False)
        X = d[variant_key].astype(np.float32)  # (3,L,L)
        vals.append(X.reshape(-1))
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


def apply_preprocess_sts_global(X: np.ndarray, params: dict) -> np.ndarray:
    A = X.astype(np.float32, copy=False)
    mode = params["mode_base"]
    eps = float(params.get("eps", 1e-6))

    if mode in ("clip_zscore", "clip_log_zscore"):
        A = np.clip(A, float(params["clip_lo"]), float(params["clip_hi"]))

    if mode in ("log_zscore", "clip_log_zscore"):
        A = np.sign(A) * np.log1p(np.abs(A))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        A = (A - float(params["mean"])) / (float(params["std"]) + eps)

    return A.astype(np.float32, copy=False)


# ------------------------ loader ------------------------

def _np_load_pair(
    sta_path_bytes: bytes,
    sts_path_bytes: bytes,
    class_to_id,
    expected_L,
    seed,
    sta_variant,
    sts_variant,
    aug,
    sta_pp: dict,
    sts_pp: dict,
):
    sta_path = Path(sta_path_bytes.decode("utf-8"))
    sts_path = Path(sts_path_bytes.decode("utf-8"))

    class_label = sta_path.parent.name
    if class_label not in class_to_id:
        raise ValueError(f"Unknown class_label={class_label}")

    d_sta = np.load(str(sta_path), allow_pickle=False)
    d_sts = np.load(str(sts_path), allow_pickle=False)

    A = d_sta[sta_variant].astype(np.float32)     # (3,L)
    S = d_sts[sts_variant].astype(np.float32)     # (3,L,L)

    if A.ndim != 2 or A.shape[0] != 3:
        raise ValueError(f"Bad StA shape {A.shape} in {sta_path}")
    if S.ndim != 3 or S.shape[0] != 3 or S.shape[1] != S.shape[2]:
        raise ValueError(f"Bad StS shape {S.shape} in {sts_path}")

    A = apply_preprocess_sta_global(A, sta_pp)
    S = apply_preprocess_sts_global(S, sts_pp)

    L_a = A.shape[1]
    L_s = S.shape[1]
    if expected_L is not None and (L_a != expected_L or L_s != expected_L):
        raise ValueError(f"Unexpected L in pair {sta_path.stem}: StA={L_a}, StS={L_s}, expected={expected_L}")

    # deterministic augmentation
    h = abs(hash(str(sta_path))) % (2**31 - 1)
    rng = np.random.default_rng(seed + h)

    if aug == "permute":
        perm = rng.permutation(3)
        A = A[perm, :]
        S = S[perm, :, :]

    # expand channels:
    # A: (3,L) -> (3,L,1)
    A = A[..., None].astype(np.float32)

    # S stays (3,L,L,1) as 3 separate maps for shared branch
    S = S[..., None].astype(np.float32)

    y = np.int64(class_to_id[class_label])
    return A, S, y


def make_tf_dataset(
    pairs,
    class_to_id,
    expected_L,
    bs,
    shuffle,
    seed,
    sta_variant,
    sts_variant,
    aug,
    sta_pp,
    sts_pp,
):
    sta_paths = tf.constant([str(p[0]) for p in pairs])
    sts_paths = tf.constant([str(p[1]) for p in pairs])

    ds = tf.data.Dataset.from_tensor_slices((sta_paths, sts_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), seed=seed, reshuffle_each_iteration=True)

    def _map_fn(sta_p, sts_p):
        A, S, y = tf.numpy_function(
            func=lambda a, s: _np_load_pair(
                a, s,
                class_to_id,
                expected_L,
                seed,
                sta_variant,
                sts_variant,
                aug,
                sta_pp,
                sts_pp,
            ),
            inp=[sta_p, sts_p],
            Tout=[tf.float32, tf.float32, tf.int64],
        )

        if expected_L is not None:
            A.set_shape((3, expected_L, 1))
            S.set_shape((3, expected_L, expected_L, 1))
        else:
            A.set_shape((3, None, 1))
            S.set_shape((3, None, None, 1))
        y.set_shape(())

        return {"sta": A, "sts": S}, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(bs).prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------ model ------------------------

@dataclass
class FusionCfg:
    expected_L: int

    sta_filters: int
    sta_blocks: int
    sta_kernel: int

    sts_filters: int
    sts_blocks: int
    sts_kernel: int

    dropout: float
    dense_units: int
    pool: str
    layernorm: bool


def build_sta_sts_fusion_model(cfg: FusionCfg, n_classes: int, lr: float) -> tf.keras.Model:
    # ---------- inputs ----------
    inp_sta = tf.keras.Input(shape=(3, cfg.expected_L, 1), name="sta")
    inp_sts = tf.keras.Input(shape=(3, cfg.expected_L, cfg.expected_L, 1), name="sts")

    # ---------- split loops ----------
    sta_loops = [
        tf.keras.layers.Lambda(lambda t, i=i: t[:, i, :, :], name=f"sta_loop_{i}")(inp_sta)
        for i in range(3)
    ]  # each: (batch,L,1)

    sts_loops = [
        tf.keras.layers.Lambda(lambda t, i=i: t[:, i, :, :, :], name=f"sts_loop_{i}")(inp_sts)
        for i in range(3)
    ]  # each: (batch,L,L,1)

    if cfg.layernorm:
        ln1 = tf.keras.layers.LayerNormalization(axis=[1, 2])
        sta_loops = [ln1(x) for x in sta_loops]

        ln2 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])
        sts_loops = [ln2(x) for x in sts_loops]

    # ---------- shared StA branch ----------
    sta_convs = []
    sta_drops = []
    for _ in range(cfg.sta_blocks):
        sta_convs.append(tf.keras.layers.Conv1D(cfg.sta_filters, cfg.sta_kernel, padding="same", activation="relu"))
        sta_drops.append(tf.keras.layers.Dropout(cfg.dropout) if cfg.dropout > 0 else tf.keras.layers.Lambda(lambda x: x))

    sta_gap = tf.keras.layers.GlobalAveragePooling1D()
    sta_gmp = tf.keras.layers.GlobalMaxPooling1D()

    def encode_sta(x):
        for i in range(cfg.sta_blocks):
            x = sta_convs[i](x)
            x = sta_drops[i](x)
        if cfg.pool == "avg":
            return sta_gap(x)
        if cfg.pool == "max":
            return sta_gmp(x)
        if cfg.pool == "avgmax":
            return tf.keras.layers.Concatenate()([sta_gap(x), sta_gmp(x)])
        raise ValueError(cfg.pool)

    sta_emb = [encode_sta(x) for x in sta_loops]
    z_sta = tf.keras.layers.Concatenate(name="concat_sta")(sta_emb)

    # ---------- shared StS branch ----------
    sts_convs = []
    sts_drops = []
    for _ in range(cfg.sts_blocks):
        sts_convs.append(tf.keras.layers.Conv2D(cfg.sts_filters, cfg.sts_kernel, padding="same", activation="relu"))
        sts_drops.append(tf.keras.layers.Dropout(cfg.dropout) if cfg.dropout > 0 else tf.keras.layers.Lambda(lambda x: x))

    sts_gap = tf.keras.layers.GlobalAveragePooling2D()
    sts_gmp = tf.keras.layers.GlobalMaxPooling2D()

    def encode_sts(x):
        for i in range(cfg.sts_blocks):
            x = sts_convs[i](x)
            x = sts_drops[i](x)
        if cfg.pool == "avg":
            return sts_gap(x)
        if cfg.pool == "max":
            return sts_gmp(x)
        if cfg.pool == "avgmax":
            return tf.keras.layers.Concatenate()([sts_gap(x), sts_gmp(x)])
        raise ValueError(cfg.pool)

    sts_emb = [encode_sts(x) for x in sts_loops]
    z_sts = tf.keras.layers.Concatenate(name="concat_sts")(sts_emb)

    # ---------- fusion ----------
    x = tf.keras.layers.Concatenate(name="fusion_concat")([z_sta, z_sts])

    if cfg.dense_units > 0:
        x = tf.keras.layers.Dense(cfg.dense_units, activation="relu")(x)
        if cfg.dropout > 0:
            x = tf.keras.layers.Dropout(cfg.dropout)(x)

    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="y")(x)

    model = tf.keras.Model(inputs={"sta": inp_sta, "sts": inp_sts}, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------ results csv ------------------------

def append_results_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--sta_dataset", required=True, help="Root folder: dataset/StA/<class>/*.npz")
    ap.add_argument("--sts_dataset", required=True, help="Root folder: dataset/StS/<class>/*.npz")

    ap.add_argument("--classes", nargs=2, default=["3_1", "4_1"])
    ap.add_argument("--n_per_class", type=int, default=694)

    ap.add_argument("--sta_variant", default="StA")
    ap.add_argument("--sts_variant", default="StS")

    ap.add_argument("--expected_L", type=int, default=201)
    ap.add_argument("--split_train", type=float, default=0.8)
    ap.add_argument("--split_val", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--sta_filters", type=int, default=32)
    ap.add_argument("--sta_blocks", type=int, default=2)
    ap.add_argument("--sta_kernel", type=int, default=5)

    ap.add_argument("--sts_filters", type=int, default=16)
    ap.add_argument("--sts_blocks", type=int, default=2)
    ap.add_argument("--sts_kernel", type=int, default=5)

    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--dense_units", type=int, default=64)
    ap.add_argument("--pool", default="avgmax", choices=["avg", "max", "avgmax"])
    ap.add_argument("--layernorm", action="store_true")

    ap.add_argument("--aug", default="none", choices=["none", "permute"])

    ap.add_argument("--sta_preprocess", default="clip_zscore")
    ap.add_argument("--sts_preprocess", default="clip_log_zscore")

    ap.add_argument("--sta_clip_lo", type=float, default=0.001)
    ap.add_argument("--sta_clip_hi", type=float, default=0.999)
    ap.add_argument("--sts_clip_lo", type=float, default=0.001)
    ap.add_argument("--sts_clip_hi", type=float, default=0.999)

    ap.add_argument("--norm_eps", type=float, default=1e-6)

    ap.add_argument("--out", required=True)
    ap.add_argument("--results_csv", default=None)

    args = ap.parse_args()

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    sta_root = Path(args.sta_dataset)
    sts_root = Path(args.sts_dataset)

    if not sta_root.exists():
        raise FileNotFoundError(f"StA dataset root not found: {sta_root}")
    if not sts_root.exists():
        raise FileNotFoundError(f"StS dataset root not found: {sts_root}")

    c1, c2 = args.classes
    class_to_id = {c1: 0, c2: 1}
    id_to_class = {0: c1, 1: c2}

    train_pairs, val_pairs, test_pairs, effective_n = stratified_split_two_classes_equal_n_pairs(
        sta_root=sta_root,
        sts_root=sts_root,
        c1=c1,
        c2=c2,
        n_per_class=args.n_per_class,
        seed=args.seed,
        train_frac=args.split_train,
        val_frac=args.split_val,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_pair_file_list(train_pairs, out_dir / "train_pairs.txt")
    save_pair_file_list(val_pairs, out_dir / "val_pairs.txt")
    save_pair_file_list(test_pairs, out_dir / "test_pairs.txt")

    # fit preprocessing on TRAIN only
    sta_pp = fit_preprocess_params_train_sta(
        train_pairs=train_pairs,
        variant_key=args.sta_variant,
        mode=args.sta_preprocess,
        clip_qlo=args.sta_clip_lo,
        clip_qhi=args.sta_clip_hi,
        eps=args.norm_eps,
    )
    sts_pp = fit_preprocess_params_train_sts(
        train_pairs=train_pairs,
        variant_key=args.sts_variant,
        mode=args.sts_preprocess,
        clip_qlo=args.sts_clip_lo,
        clip_qhi=args.sts_clip_hi,
        eps=args.norm_eps,
    )

    with (out_dir / "sta_preprocess.json").open("w", encoding="utf-8") as f:
        json.dump(sta_pp, f, ensure_ascii=False, indent=2)
    with (out_dir / "sts_preprocess.json").open("w", encoding="utf-8") as f:
        json.dump(sts_pp, f, ensure_ascii=False, indent=2)

    train_ds = make_tf_dataset(
        pairs=train_pairs,
        class_to_id=class_to_id,
        expected_L=args.expected_L,
        bs=args.bs,
        shuffle=True,
        seed=args.seed,
        sta_variant=args.sta_variant,
        sts_variant=args.sts_variant,
        aug=args.aug,
        sta_pp=sta_pp,
        sts_pp=sts_pp,
    )
    val_ds = make_tf_dataset(
        pairs=val_pairs,
        class_to_id=class_to_id,
        expected_L=args.expected_L,
        bs=args.bs,
        shuffle=False,
        seed=args.seed,
        sta_variant=args.sta_variant,
        sts_variant=args.sts_variant,
        aug="none",
        sta_pp=sta_pp,
        sts_pp=sts_pp,
    )
    test_ds = make_tf_dataset(
        pairs=test_pairs,
        class_to_id=class_to_id,
        expected_L=args.expected_L,
        bs=args.bs,
        shuffle=False,
        seed=args.seed,
        sta_variant=args.sta_variant,
        sts_variant=args.sts_variant,
        aug="none",
        sta_pp=sta_pp,
        sts_pp=sts_pp,
    )

    cfg = FusionCfg(
        expected_L=int(args.expected_L),

        sta_filters=int(args.sta_filters),
        sta_blocks=int(args.sta_blocks),
        sta_kernel=int(args.sta_kernel),

        sts_filters=int(args.sts_filters),
        sts_blocks=int(args.sts_blocks),
        sts_kernel=int(args.sts_kernel),

        dropout=float(args.dropout),
        dense_units=int(args.dense_units),
        pool=str(args.pool),
        layernorm=bool(args.layernorm),
    )

    model = build_sta_sts_fusion_model(cfg, n_classes=2, lr=args.lr)

    ckpt_path = str(out_dir / "best_model")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=1e-3,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=True,
            monitor="val_loss",
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    eval_out = model.evaluate(test_ds, verbose=0)
    test_loss = float(eval_out[0]) if isinstance(eval_out, (list, tuple)) else float(eval_out)
    test_acc = float(eval_out[1]) if isinstance(eval_out, (list, tuple)) and len(eval_out) > 1 else float("nan")

    metrics_blob = evaluate_and_save_metrics(model, test_ds, out_dir, id_to_class)

    final_path = out_dir / "final_model"
    model.save(str(final_path))

    train_eval = model.evaluate(train_ds, verbose=0)
    train_loss = float(train_eval[0]) if isinstance(train_eval, (list, tuple)) else float(train_eval)
    train_acc = float(train_eval[1]) if isinstance(train_eval, (list, tuple)) and len(train_eval) > 1 else float("nan")

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sta_dataset_root": str(sta_root),
        "sts_dataset_root": str(sts_root),
        "classes": [c1, c2],
        "n_per_class_requested": int(args.n_per_class),
        "n_per_class_effective": int(effective_n),
        "splits": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
        "input": {
            "L": int(args.expected_L),
            "representation": "StA+StS_fusion",
            "sta_variant": args.sta_variant,
            "sts_variant": args.sts_variant,
        },
        "train_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.bs),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "split": [float(args.split_train), float(args.split_val), float(1.0 - args.split_train - args.split_val)],
        },
        "preprocess": {
            "sta": sta_pp,
            "sts": sts_pp,
        },
        "augmentations": {
            "aug": args.aug,
        },
        "model": {
            "name": "sta_sts_fusion",
            "config": asdict(cfg),
        },
        "results": {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "train_loss_eval": train_loss,
            "train_accuracy_eval": train_acc,
            "confusion_matrix": metrics_blob["confusion_matrix"],
            "metrics": metrics_blob["metrics"],
            "balanced_accuracy": metrics_blob["balanced_accuracy"],
            "mcc": metrics_blob["mcc"],
        },
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "artifacts": {
            "final_model": str(final_path),
            "best_model": str(out_dir / "best_model"),
            "confusion_matrix_csv": str(out_dir / "confusion_matrix.csv"),
            "classification_report": str(out_dir / "classification_report.txt"),
            "predictions_csv": str(out_dir / "predictions.csv"),
            "train_pairs": str(out_dir / "train_pairs.txt"),
            "val_pairs": str(out_dir / "val_pairs.txt"),
            "test_pairs": str(out_dir / "test_pairs.txt"),
            "sta_preprocess_json": str(out_dir / "sta_preprocess.json"),
            "sts_preprocess_json": str(out_dir / "sts_preprocess.json"),
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
            "model": "sta_sts_fusion",
            "sta_variant": args.sta_variant,
            "sts_variant": args.sts_variant,
            "sta_preprocess": args.sta_preprocess,
            "sts_preprocess": args.sts_preprocess,
            "aug": args.aug,
            "seed": args.seed,
            "expected_L": args.expected_L,
            "epochs": args.epochs,
            "bs": args.bs,
            "lr": args.lr,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "balanced_accuracy": metrics_blob["balanced_accuracy"],
            "mcc": metrics_blob["mcc"],
            "out_dir": str(out_dir),
        }
        append_results_csv(Path(args.results_csv), row)

    print(f"\nDone. Test acc={test_acc:.4f} loss={test_loss:.4f}")
    print(f"Train-eval acc={train_acc:.4f} loss={train_loss:.4f}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()