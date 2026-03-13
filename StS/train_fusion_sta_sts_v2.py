from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


# ============================================================
# utils: pairing / splits
# ============================================================

def list_class_files(root: Path, class_label: str) -> List[Path]:
    class_dir = root / class_label
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")
    files = sorted(class_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in: {class_dir}")
    return files


def build_paired_files(
    sta_root: Path,
    sts_root: Path,
    c1: str,
    c2: str,
    n_per_class: int,
    seed: int,
) -> Tuple[List[Tuple[Path, Path, str]], int]:
    rng = random.Random(seed)

    def make_pairs_for_class(cls: str) -> List[Tuple[Path, Path, str]]:
        sta_files = list_class_files(sta_root, cls)
        sts_files = list_class_files(sts_root, cls)

        sta_map = {p.stem: p for p in sta_files}
        sts_map = {p.stem: p for p in sts_files}

        common_ids = sorted(set(sta_map.keys()) & set(sts_map.keys()))
        if not common_ids:
            raise RuntimeError(f"No common StA/StS ids for class {cls}")

        rng.shuffle(common_ids)
        use_ids = common_ids[:min(n_per_class, len(common_ids))]

        return [(sta_map[i], sts_map[i], cls) for i in use_ids]

    pairs_c1 = make_pairs_for_class(c1)
    pairs_c2 = make_pairs_for_class(c2)

    effective_n = min(len(pairs_c1), len(pairs_c2))
    pairs_c1 = pairs_c1[:effective_n]
    pairs_c2 = pairs_c2[:effective_n]

    all_pairs = pairs_c1 + pairs_c2
    return all_pairs, effective_n


def stratified_split_paired_equal_n(
    sta_root: Path,
    sts_root: Path,
    c1: str,
    c2: str,
    n_per_class: int,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[List[Tuple[Path, Path, str]], List[Tuple[Path, Path, str]], List[Tuple[Path, Path, str]], int]:
    rng = random.Random(seed)

    all_pairs, effective_n = build_paired_files(
        sta_root=sta_root,
        sts_root=sts_root,
        c1=c1,
        c2=c2,
        n_per_class=n_per_class,
        seed=seed,
    )

    pairs_c1 = [p for p in all_pairs if p[2] == c1]
    pairs_c2 = [p for p in all_pairs if p[2] == c2]

    def split_one(lst):
        n = len(lst)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train = lst[:n_train]
        val = lst[n_train:n_train + n_val]
        test = lst[n_train + n_val:]
        return train, val, test

    tr1, va1, te1 = split_one(pairs_c1)
    tr2, va2, te2 = split_one(pairs_c2)

    train_pairs = tr1 + tr2
    val_pairs = va1 + va2
    test_pairs = te1 + te2

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    rng.shuffle(test_pairs)

    return train_pairs, val_pairs, test_pairs, effective_n


def save_pair_list(pairs: List[Tuple[Path, Path, str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for sta_p, sts_p, cls in pairs:
            f.write(f"{cls}\t{sta_p}\t{sts_p}\n")


# ============================================================
# metrics
# ============================================================

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


# ============================================================
# preprocessing
# ============================================================

def fit_preprocess_params_train_sta(
    train_pairs: List[Tuple[Path, Path, str]],
    variant_key: str,
    mode: str,
    clip_qlo: float,
    clip_qhi: float,
    eps: float,
) -> dict:
    vals = []
    for sta_path, _, _ in train_pairs:
        d = np.load(str(sta_path), allow_pickle=False)
        x = d[variant_key].astype(np.float32)  # (3,L)
        vals.append(x.reshape(-1))
    xall = np.concatenate(vals, axis=0).astype(np.float32, copy=False)

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
        lo = float(np.quantile(xall, clip_qlo))
        hi = float(np.quantile(xall, clip_qhi))
        params["clip_lo"] = lo
        params["clip_hi"] = hi
        xall = np.clip(xall, lo, hi)

    if mode in ("log_zscore", "clip_log_zscore"):
        xall = np.sign(xall) * np.log1p(np.abs(xall))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        params["mean"] = float(xall.mean())
        params["std"] = float(xall.std())

    return params


def apply_preprocess_sta_global(x: np.ndarray, params: dict) -> np.ndarray:
    a = x.astype(np.float32, copy=False)
    mode = params["mode_base"]
    eps = float(params.get("eps", 1e-6))

    if mode in ("clip_zscore", "clip_log_zscore"):
        a = np.clip(a, float(params["clip_lo"]), float(params["clip_hi"]))

    if mode in ("log_zscore", "clip_log_zscore"):
        a = np.sign(a) * np.log1p(np.abs(a))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        a = (a - float(params["mean"])) / (float(params["std"]) + eps)

    return a.astype(np.float32, copy=False)


def fit_preprocess_params_train_sts(
    train_pairs: List[Tuple[Path, Path, str]],
    variant_key: str,
    mode: str,
    clip_qlo: float,
    clip_qhi: float,
    eps: float,
) -> dict:
    vals = []
    for _, sts_path, _ in train_pairs:
        d = np.load(str(sts_path), allow_pickle=False)
        x = d[variant_key].astype(np.float32)  # (3,L,L)
        vals.append(x.reshape(-1))
    xall = np.concatenate(vals, axis=0).astype(np.float32, copy=False)

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
        lo = float(np.quantile(xall, clip_qlo))
        hi = float(np.quantile(xall, clip_qhi))
        params["clip_lo"] = lo
        params["clip_hi"] = hi
        xall = np.clip(xall, lo, hi)

    if mode in ("log_zscore", "clip_log_zscore"):
        xall = np.sign(xall) * np.log1p(np.abs(xall))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        params["mean"] = float(xall.mean())
        params["std"] = float(xall.std())

    return params


def apply_preprocess_sts_global(x: np.ndarray, params: dict) -> np.ndarray:
    a = x.astype(np.float32, copy=False)
    mode = params["mode_base"]
    eps = float(params.get("eps", 1e-6))

    if mode in ("clip_zscore", "clip_log_zscore"):
        a = np.clip(a, float(params["clip_lo"]), float(params["clip_hi"]))

    if mode in ("log_zscore", "clip_log_zscore"):
        a = np.sign(a) * np.log1p(np.abs(a))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        a = (a - float(params["mean"])) / (float(params["std"]) + eps)

    return a.astype(np.float32, copy=False)


# ============================================================
# dataset loader
# ============================================================

def pad_sts_to_patch_multiple(x: np.ndarray, patch: int) -> np.ndarray:
    # x: (3,L,L)
    L = x.shape[1]
    target = int(math.ceil(L / patch) * patch)
    if target == L:
        return x
    pad = target - L
    return np.pad(x, ((0, 0), (0, pad), (0, pad)), mode="constant")


def _np_load_fusion_pair(
    sta_path_bytes: bytes,
    sts_path_bytes: bytes,
    class_label_bytes: bytes,
    class_to_id,
    expected_L,
    sts_patch,
    sta_variant_key,
    sts_variant_key,
    sta_pp_params,
    sts_pp_params,
):
    sta_path = Path(sta_path_bytes.decode("utf-8"))
    sts_path = Path(sts_path_bytes.decode("utf-8"))
    class_label = class_label_bytes.decode("utf-8")

    d_sta = np.load(str(sta_path), allow_pickle=False)
    d_sts = np.load(str(sts_path), allow_pickle=False)

    x_sta = d_sta[sta_variant_key].astype(np.float32)   # (3,L)
    x_sts = d_sts[sts_variant_key].astype(np.float32)   # (3,L,L)

    x_sta = apply_preprocess_sta_global(x_sta, sta_pp_params)
    x_sts = apply_preprocess_sts_global(x_sts, sts_pp_params)

    if expected_L is not None:
        if x_sta.shape != (3, expected_L):
            raise ValueError(f"Bad StA shape {x_sta.shape} in {sta_path}; expected (3,{expected_L})")
        if x_sts.shape != (3, expected_L, expected_L):
            raise ValueError(f"Bad StS shape {x_sts.shape} in {sts_path}; expected (3,{expected_L},{expected_L})")

    x_sta = x_sta[..., None]  # (3,L,1)

    x_sts = pad_sts_to_patch_multiple(x_sts, sts_patch)
    x_sts = x_sts[..., None]  # (3,P,P,1)

    y = np.int64(class_to_id[class_label])

    return x_sta.astype(np.float32), x_sts.astype(np.float32), y


def make_fusion_tf_dataset(
    pairs: List[Tuple[Path, Path, str]],
    class_to_id,
    expected_L: int,
    sts_patch: int,
    bs: int,
    shuffle: bool,
    seed: int,
    sta_variant_key: str,
    sts_variant_key: str,
    sta_pp_params: dict,
    sts_pp_params: dict,
):
    sta_paths = tf.constant([str(p[0]) for p in pairs])
    sts_paths = tf.constant([str(p[1]) for p in pairs])
    labels = tf.constant([p[2] for p in pairs])

    ds = tf.data.Dataset.from_tensor_slices((sta_paths, sts_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), seed=seed, reshuffle_each_iteration=True)

    padded_size = int(math.ceil(expected_L / sts_patch) * sts_patch)

    def _map_fn(sta_p, sts_p, cls):
        x_sta, x_sts, y = tf.numpy_function(
            func=lambda a, b, c: _np_load_fusion_pair(
                a, b, c,
                class_to_id=class_to_id,
                expected_L=expected_L,
                sts_patch=sts_patch,
                sta_variant_key=sta_variant_key,
                sts_variant_key=sts_variant_key,
                sta_pp_params=sta_pp_params,
                sts_pp_params=sts_pp_params,
            ),
            inp=[sta_p, sts_p, cls],
            Tout=[tf.float32, tf.float32, tf.int64],
        )

        x_sta.set_shape((3, expected_L, 1))
        x_sts.set_shape((3, padded_size, padded_size, 1))
        y.set_shape(())

        return {"sta": x_sta, "sts": x_sts}, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(bs).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# model blocks
# ============================================================

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size: int, embed_dim: int, num_patches: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.proj = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )

    def call(self, x):
        x = self.proj(x)
        x = tf.reshape(x, (-1, self.num_patches, self.embed_dim))
        return x


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos_embedding",
            shape=(1, self.num_patches, self.embed_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, embed_dim // num_heads),
            dropout=dropout,
        )
        self.drop1 = tf.keras.layers.Dropout(dropout)

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(dropout),
        ])

    def call(self, x, training=None):
        y = self.norm1(x)
        y = self.attn(y, y, training=training)
        y = self.drop1(y, training=training)
        x = x + y

        y = self.norm2(x)
        y = self.mlp(y, training=training)
        x = x + y
        return x


# ============================================================
# models
# ============================================================

@dataclass
class StaBranchCfg:
    expected_L: int
    filters: int
    blocks: int
    kernel: int
    dropout: float
    dense_units: int
    pool: str


@dataclass
class StsBranchCfg:
    expected_L: int
    patch: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_dim: int
    dropout: float


@dataclass
class FusionHeadCfg:
    dense_units: int
    dropout: float


def build_sta_branch(cfg: StaBranchCfg) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(3, cfg.expected_L, 1), name="sta")

    loops = [
        tf.keras.layers.Lambda(lambda t, i=i: t[:, i, :, :], name=f"sta_loop_{i}")(inp)
        for i in range(3)
    ]

    convs = []
    drops = []
    for _ in range(cfg.blocks):
        convs.append(tf.keras.layers.Conv1D(cfg.filters, cfg.kernel, padding="same", activation="relu"))
        drops.append(tf.keras.layers.Dropout(cfg.dropout) if cfg.dropout > 0 else tf.keras.layers.Lambda(lambda x: x))

    gap = tf.keras.layers.GlobalAveragePooling1D()
    gmp = tf.keras.layers.GlobalMaxPooling1D()

    def encode_one(x):
        for bi in range(cfg.blocks):
            x = convs[bi](x)
            x = drops[bi](x)
        if cfg.pool == "avg":
            return gap(x)
        if cfg.pool == "max":
            return gmp(x)
        if cfg.pool == "avgmax":
            return tf.keras.layers.Concatenate()([gap(x), gmp(x)])
        raise ValueError(cfg.pool)

    emb = [encode_one(x) for x in loops]
    x = tf.keras.layers.Concatenate(name="sta_concat_loops")(emb)

    if cfg.dense_units > 0:
        x = tf.keras.layers.Dense(cfg.dense_units, activation="relu")(x)

    return tf.keras.Model(inp, x, name="sta_branch")


def build_sts_branch(cfg: StsBranchCfg) -> tf.keras.Model:
    padded_size = int(math.ceil(cfg.expected_L / cfg.patch) * cfg.patch)
    num_patches_1d = padded_size // cfg.patch
    num_patches = num_patches_1d * num_patches_1d

    inp = tf.keras.Input(shape=(3, padded_size, padded_size, 1), name="sts")

    loops = [
        tf.keras.layers.Lambda(lambda t, i=i: t[:, i, :, :, :], name=f"sts_loop_{i}")(inp)
        for i in range(3)
    ]

    patch_embed = PatchEmbedding(cfg.patch, cfg.embed_dim, num_patches)
    pos_embed = PositionalEmbedding(num_patches, cfg.embed_dim)
    blocks = [
        TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_dim, cfg.dropout)
        for _ in range(cfg.depth)
    ]
    final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def encode_one(x):
        x = patch_embed(x)
        x = pos_embed(x)
        for blk in blocks:
            x = blk(x)
        x = final_norm(x)
        x = tf.reduce_mean(x, axis=1)
        return x

    emb = [encode_one(x) for x in loops]
    x = tf.keras.layers.Concatenate(name="sts_concat_loops")(emb)

    return tf.keras.Model(inp, x, name="sts_branch")


def build_fusion_model(
    sta_cfg: StaBranchCfg,
    sts_cfg: StsBranchCfg,
    fusion_cfg: FusionHeadCfg,
    n_classes: int,
    lr: float,
    fusion_mode: str = "fusion",
    sta_weight: float = 1.0,
    sts_weight: float = 1.0,
    gate_hidden_units: int = 64,
) -> tf.keras.Model:
    sta_branch = build_sta_branch(sta_cfg)
    sts_branch = build_sts_branch(sts_cfg)

    inp_sta = tf.keras.Input(shape=(3, sta_cfg.expected_L, 1), name="sta")
    padded_size = int(math.ceil(sts_cfg.expected_L / sts_cfg.patch) * sts_cfg.patch)
    inp_sts = tf.keras.Input(shape=(3, padded_size, padded_size, 1), name="sts")

    emb_sta = sta_branch(inp_sta)
    emb_sts = sts_branch(inp_sts)

    if fusion_mode == "fusion":
        emb_sta_w = tf.keras.layers.Lambda(
            lambda t: t * tf.cast(sta_weight, t.dtype),
            name="sta_weighted"
        )(emb_sta)
        emb_sts_w = tf.keras.layers.Lambda(
            lambda t: t * tf.cast(sts_weight, t.dtype),
            name="sts_weighted"
        )(emb_sts)
        x = tf.keras.layers.Concatenate(name="fusion_concat")([emb_sta_w, emb_sts_w])

    elif fusion_mode == "sts_only_control":
        emb_sta_zero = tf.keras.layers.Lambda(
            lambda t: tf.stop_gradient(tf.zeros_like(t)),
            name="sta_zero_branch"
        )(emb_sta)
        emb_sts_w = tf.keras.layers.Lambda(
            lambda t: t * tf.cast(sts_weight, t.dtype),
            name="sts_weighted"
        )(emb_sts)
        x = tf.keras.layers.Concatenate(name="fusion_concat")([emb_sta_zero, emb_sts_w])

    elif fusion_mode == "sta_only_control":
        emb_sta_w = tf.keras.layers.Lambda(
            lambda t: t * tf.cast(sta_weight, t.dtype),
            name="sta_weighted"
        )(emb_sta)
        emb_sts_zero = tf.keras.layers.Lambda(
            lambda t: tf.stop_gradient(tf.zeros_like(t)),
            name="sts_zero_branch"
        )(emb_sts)
        x = tf.keras.layers.Concatenate(name="fusion_concat")([emb_sta_w, emb_sts_zero])

    elif fusion_mode == "gated_fusion":
        gate_input = tf.keras.layers.Concatenate(name="gate_input")([emb_sta, emb_sts])

        gate_hidden = tf.keras.layers.Dense(
            gate_hidden_units,
            activation="relu",
            name="gate_hidden"
        )(gate_input)

        sta_dim = emb_sta.shape[-1]
        sts_dim = emb_sts.shape[-1]

        g_sta = tf.keras.layers.Dense(
            sta_dim,
            activation="sigmoid",
            name="gate_sta"
        )(gate_hidden)

        g_sts = tf.keras.layers.Dense(
            sts_dim,
            activation="sigmoid",
            name="gate_sts"
        )(gate_hidden)

        emb_sta_g = tf.keras.layers.Multiply(name="sta_gated")([emb_sta, g_sta])
        emb_sts_g = tf.keras.layers.Multiply(name="sts_gated")([emb_sts, g_sts])

        x = tf.keras.layers.Concatenate(name="fusion_concat")([emb_sta_g, emb_sts_g])

    else:
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    if fusion_cfg.dense_units > 0:
        x = tf.keras.layers.Dense(fusion_cfg.dense_units, activation="relu")(x)
        if fusion_cfg.dropout > 0:
            x = tf.keras.layers.Dropout(fusion_cfg.dropout)(x)

    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="y")(x)

    model = tf.keras.Model(inputs={"sta": inp_sta, "sts": inp_sts}, outputs=out, name="fusion_sta_sts_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# results csv
# ============================================================

def append_results_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_root", required=True, help=r"Root containing dataset\StA and dataset\StS")
    ap.add_argument("--classes", nargs=2, default=["3_1", "4_1"])
    ap.add_argument("--n_per_class", type=int, default=694)

    ap.add_argument("--sta_variant", default="StA")
    ap.add_argument("--sts_variant", default="StS")
    ap.add_argument("--expected_L", type=int, default=201)

    ap.add_argument("--split_train", type=float, default=0.8)
    ap.add_argument("--split_val", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--seed", type=int, default=42)

    # StA branch
    ap.add_argument("--sta_filters", type=int, default=32)
    ap.add_argument("--sta_blocks", type=int, default=2)
    ap.add_argument("--sta_kernel", type=int, default=5)
    ap.add_argument("--sta_dropout", type=float, default=0.1)
    ap.add_argument("--sta_dense_units", type=int, default=64)
    ap.add_argument("--sta_pool", default="avgmax", choices=["avg", "max", "avgmax"])

    # StS branch
    ap.add_argument("--sts_patch", type=int, default=15)
    ap.add_argument("--sts_embed_dim", type=int, default=128)
    ap.add_argument("--sts_depth", type=int, default=4)
    ap.add_argument("--sts_num_heads", type=int, default=8)
    ap.add_argument("--sts_mlp_dim", type=int, default=256)
    ap.add_argument("--sts_dropout", type=float, default=0.1)

    # Fusion head
    ap.add_argument("--fusion_dense_units", type=int, default=64)
    ap.add_argument("--fusion_dropout", type=float, default=0.2)

    # preprocessing
    ap.add_argument("--sta_preprocess", default="clip_zscore", choices=["zscore", "clip_zscore", "log_zscore", "clip_log_zscore"])
    ap.add_argument("--sts_preprocess", default="clip_log_zscore", choices=["zscore", "clip_zscore", "log_zscore", "clip_log_zscore"])
    ap.add_argument("--sta_clip_lo", type=float, default=0.001)
    ap.add_argument("--sta_clip_hi", type=float, default=0.999)
    ap.add_argument("--sts_clip_lo", type=float, default=0.001)
    ap.add_argument("--sts_clip_hi", type=float, default=0.999)
    ap.add_argument("--norm_eps", type=float, default=1e-6)

    ap.add_argument("--out", required=True)
    ap.add_argument("--results_csv", default=None)

    ap.add_argument(
        "--fusion_mode",
        default="fusion",
        choices=["fusion", "sts_only_control", "sta_only_control", "gated_fusion"],
        help=(
            "fusion = normalne StA+StS, "
            "sts_only_control = zerowanie branchu StA przed fuzją, "
            "sta_only_control = zerowanie branchu StS przed fuzją, "
            "gated_fusion = uczona bramka dla obu branchy"
        ),
    )   

    ap.add_argument("--sta_weight", type=float, default=1.0)
    ap.add_argument("--sts_weight", type=float, default=1.0)

    ap.add_argument("--gate_hidden_units", type=int, default=64)

    args = ap.parse_args()

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    root = Path(args.dataset_root)
    sta_root = root / "StA"
    sts_root = root / "StS"

    if not sta_root.exists():
        raise FileNotFoundError(f"StA root not found: {sta_root}")
    if not sts_root.exists():
        raise FileNotFoundError(f"StS root not found: {sts_root}")

    c1, c2 = args.classes
    class_to_id = {c1: 0, c2: 1}
    id_to_class = {0: c1, 1: c2}

    train_pairs, val_pairs, test_pairs, effective_n = stratified_split_paired_equal_n(
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

    save_pair_list(train_pairs, out_dir / "train_pairs.txt")
    save_pair_list(val_pairs, out_dir / "val_pairs.txt")
    save_pair_list(test_pairs, out_dir / "test_pairs.txt")

    sta_pp_params = fit_preprocess_params_train_sta(
        train_pairs=train_pairs,
        variant_key=args.sta_variant,
        mode=args.sta_preprocess,
        clip_qlo=args.sta_clip_lo,
        clip_qhi=args.sta_clip_hi,
        eps=args.norm_eps,
    )
    with (out_dir / "preprocess_sta.json").open("w", encoding="utf-8") as f:
        json.dump(sta_pp_params, f, ensure_ascii=False, indent=2)

    sts_pp_params = fit_preprocess_params_train_sts(
        train_pairs=train_pairs,
        variant_key=args.sts_variant,
        mode=args.sts_preprocess,
        clip_qlo=args.sts_clip_lo,
        clip_qhi=args.sts_clip_hi,
        eps=args.norm_eps,
    )
    with (out_dir / "preprocess_sts.json").open("w", encoding="utf-8") as f:
        json.dump(sts_pp_params, f, ensure_ascii=False, indent=2)

    train_ds = make_fusion_tf_dataset(
        train_pairs,
        class_to_id=class_to_id,
        expected_L=args.expected_L,
        sts_patch=args.sts_patch,
        bs=args.bs,
        shuffle=True,
        seed=args.seed,
        sta_variant_key=args.sta_variant,
        sts_variant_key=args.sts_variant,
        sta_pp_params=sta_pp_params,
        sts_pp_params=sts_pp_params,
    )

    val_ds = make_fusion_tf_dataset(
        val_pairs,
        class_to_id=class_to_id,
        expected_L=args.expected_L,
        sts_patch=args.sts_patch,
        bs=args.bs,
        shuffle=False,
        seed=args.seed,
        sta_variant_key=args.sta_variant,
        sts_variant_key=args.sts_variant,
        sta_pp_params=sta_pp_params,
        sts_pp_params=sts_pp_params,
    )

    test_ds = make_fusion_tf_dataset(
        test_pairs,
        class_to_id=class_to_id,
        expected_L=args.expected_L,
        sts_patch=args.sts_patch,
        bs=args.bs,
        shuffle=False,
        seed=args.seed,
        sta_variant_key=args.sta_variant,
        sts_variant_key=args.sts_variant,
        sta_pp_params=sta_pp_params,
        sts_pp_params=sts_pp_params,
    )

    sta_cfg = StaBranchCfg(
        expected_L=args.expected_L,
        filters=args.sta_filters,
        blocks=args.sta_blocks,
        kernel=args.sta_kernel,
        dropout=args.sta_dropout,
        dense_units=args.sta_dense_units,
        pool=args.sta_pool,
    )

    sts_cfg = StsBranchCfg(
        expected_L=args.expected_L,
        patch=args.sts_patch,
        embed_dim=args.sts_embed_dim,
        depth=args.sts_depth,
        num_heads=args.sts_num_heads,
        mlp_dim=args.sts_mlp_dim,
        dropout=args.sts_dropout,
    )

    fusion_cfg = FusionHeadCfg(
        dense_units=args.fusion_dense_units,
        dropout=args.fusion_dropout,
    )

    model = build_fusion_model(
        sta_cfg=sta_cfg,
        sts_cfg=sts_cfg,
        fusion_cfg=fusion_cfg,
        n_classes=2,
        lr=args.lr,
        fusion_mode=args.fusion_mode,
        sta_weight=args.sta_weight,
        sts_weight=args.sts_weight,
        gate_hidden_units=args.gate_hidden_units,
    )

    ckpt_path = str(out_dir / "best_model")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
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

    train_eval = model.evaluate(train_ds, verbose=0)
    train_loss = float(train_eval[0]) if isinstance(train_eval, (list, tuple)) else float(train_eval)
    train_acc = float(train_eval[1]) if isinstance(train_eval, (list, tuple)) and len(train_eval) > 1 else float("nan")

    metrics_blob = evaluate_and_save_metrics(model, test_ds, out_dir, id_to_class)

    final_path = out_dir / "final_model"
    model.save(str(final_path))

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(root),
        "classes": [c1, c2],
        "n_per_class_requested": int(args.n_per_class),
        "n_per_class_effective": int(effective_n),
        "splits": {
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(test_pairs),
        },
        "input": {
            "L": int(args.expected_L),
            "representation": "fusion_StA_StS",
            "sta_variant_key": args.sta_variant,
            "sts_variant_key": args.sts_variant,
            "sts_patch": int(args.sts_patch),
            "sts_padded_size": int(math.ceil(args.expected_L / args.sts_patch) * args.sts_patch),
        },
        "train_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.bs),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "split": [float(args.split_train), float(args.split_val), float(1.0 - args.split_train - args.split_val)],
        },
        "preprocess_sta": sta_pp_params,
        "preprocess_sts": sts_pp_params,
        "model": {
            "name": "fusion_sta_sts_v2",
            "fusion_mode": args.fusion_mode,
            "sta_weight": float(args.sta_weight),
            "sts_weight": float(args.sts_weight),
            "gate_hidden_units": int(args.gate_hidden_units),
            "sta_branch": asdict(sta_cfg),
            "sts_branch": asdict(sts_cfg),
            "fusion_head": asdict(fusion_cfg),
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
            "preprocess_sta_json": str(out_dir / "preprocess_sta.json"),
            "preprocess_sts_json": str(out_dir / "preprocess_sts.json"),
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
            "model": "fusion_sta_sts_v2",
            "sta_variant": args.sta_variant,
            "sts_variant": args.sts_variant,
            "sta_preprocess": args.sta_preprocess,
            "sts_preprocess": args.sts_preprocess,
            "seed": args.seed,
            "expected_L": args.expected_L,
            "epochs": args.epochs,
            "bs": args.bs,
            "lr": args.lr,
            "sta_filters": args.sta_filters,
            "sta_blocks": args.sta_blocks,
            "sta_kernel": args.sta_kernel,
            "sts_patch": args.sts_patch,
            "sts_embed_dim": args.sts_embed_dim,
            "sts_depth": args.sts_depth,
            "sts_num_heads": args.sts_num_heads,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "balanced_accuracy": metrics_blob["balanced_accuracy"],
            "mcc": metrics_blob["mcc"],
            "out_dir": str(out_dir),
            "fusion_mode": args.fusion_mode,
            "sta_weight": args.sta_weight,
            "sts_weight": args.sts_weight,
            "gate_hidden_units": args.gate_hidden_units,
        }
        append_results_csv(Path(args.results_csv), row)

    print(f"\nDone. Test acc={test_acc:.4f} loss={test_loss:.4f}")
    print(f"Train-eval acc={train_acc:.4f} loss={train_loss:.4f}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":  
    main()