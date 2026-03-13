# train_sts_transformer.py
# Transformer / ViT-like training for StS.
#
# Representation:
#   StS: (3, L, L)
# Each loop is processed separately by the SAME transformer encoder.
# Then 3 loop embeddings are concatenated and classified.
#
# Recommended first run:
# python train_sts_transformer.py ^
#   --dataset "C:\Users\danil\.vscode\grafy\dataset\StS" ^
#   --classes 3_1 4_1 ^
#   --n_per_class 694 ^
#   --variant StS ^
#   --expected_L 201 ^
#   --epochs 80 ^
#   --bs 4 ^
#   --lr 0.0003 ^
#   --seed 42 ^
#   --patch 15 ^
#   --embed_dim 64 ^
#   --depth 3 ^
#   --num_heads 4 ^
#   --mlp_dim 128 ^
#   --dropout 0.1 ^
#   --dense_units 64 ^
#   --preprocess clip_log_zscore_train ^
#   --clip_lo 0.001 --clip_hi 0.999 ^
#   --out "C:\Users\danil\.vscode\grafy\experiments\sts_3_1_vs_4_1\run08_transformer_cliplogz_s42" ^
#   --results_csv "C:\Users\danil\.vscode\grafy\experiments\sts_3_1_vs_4_1\results_transformer.csv"

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


# ------------------------ preprocessing ------------------------

def fit_preprocess_params_train(
    train_files: List[Path],
    variant_key: str,
    mode: str,
    clip_qlo: float,
    clip_qhi: float,
    eps: float,
) -> dict:
    vals = []
    for p in train_files:
        d = np.load(str(p), allow_pickle=False)
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
        lo = float(params["clip_lo"])
        hi = float(params["clip_hi"])
        A = np.clip(A, lo, hi)

    if mode in ("log_zscore", "clip_log_zscore"):
        A = np.sign(A) * np.log1p(np.abs(A))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        A = (A - float(params["mean"])) / (float(params["std"]) + eps)

    return A.astype(np.float32, copy=False)


# ------------------------ loader ------------------------

def pad_to_patch_multiple(X: np.ndarray, patch: int) -> np.ndarray:
    # X: (3,L,L)
    L = X.shape[1]
    target = int(math.ceil(L / patch) * patch)
    if target == L:
        return X
    pad = target - L
    return np.pad(X, ((0, 0), (0, pad), (0, pad)), mode="constant")


def _np_load_sts_transformer(
    path_bytes: bytes,
    class_to_id,
    expected_L,
    patch,
    seed,
    variant_key,
    aug,
    pp_params,
):
    path = Path(path_bytes.decode("utf-8"))
    class_label = path.parent.name

    d = np.load(str(path), allow_pickle=False)
    X = d[variant_key].astype(np.float32)  # (3,L,L)

    if X.ndim != 3 or X.shape[0] != 3 or X.shape[1] != X.shape[2]:
        raise ValueError(f"Bad StS shape {X.shape} in {path}; expected (3,L,L)")

    X = apply_preprocess_sts_global(X, pp_params)

    L = X.shape[1]
    if expected_L is not None and L != expected_L:
        raise ValueError(f"Unexpected L={L} in {path}; expected {expected_L}")

    h = abs(hash(str(path))) % (2**31 - 1)
    rng = np.random.default_rng(seed + h)

    # only loop permutation
    if aug == "permute":
        perm = rng.permutation(3)
        X = X[perm, :, :]

    X = pad_to_patch_multiple(X, patch)   # (3,P,P)
    X = X[..., None].astype(np.float32)   # (3,P,P,1)

    y = np.int64(class_to_id[class_label])
    return X, y


def make_tf_dataset(
    files,
    class_to_id,
    expected_L,
    patch,
    bs,
    shuffle,
    seed,
    variant_key,
    aug,
    pp_params,
):
    paths = tf.constant([str(p) for p in files])
    ds = tf.data.Dataset.from_tensor_slices(paths)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(files), seed=seed, reshuffle_each_iteration=True)

    padded_size = int(math.ceil(expected_L / patch) * patch)

    def _map_fn(p):
        X, y = tf.numpy_function(
            func=lambda pb: _np_load_sts_transformer(
                pb,
                class_to_id,
                expected_L,
                patch,
                seed,
                variant_key,
                aug,
                pp_params,
            ),
            inp=[p],
            Tout=[tf.float32, tf.int64],
        )
        X.set_shape((3, padded_size, padded_size, 1))
        y.set_shape(())
        return X, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(bs).prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------ model blocks ------------------------

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
        # x: (B,H,W,1)
        x = self.proj(x)  # (B, H/p, W/p, D)
        x = tf.reshape(x, (-1, self.num_patches, self.embed_dim))  # (B, N, D)
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
            key_dim=embed_dim // num_heads,
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


# ------------------------ model ------------------------

@dataclass
class TransformerCfg:
    expected_L: int
    patch: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_dim: int
    dropout: float
    dense_units: int


def build_sts_transformer_model(cfg: TransformerCfg, n_classes: int, lr: float) -> tf.keras.Model:
    padded_size = int(math.ceil(cfg.expected_L / cfg.patch) * cfg.patch)
    num_patches_1d = padded_size // cfg.patch
    num_patches = num_patches_1d * num_patches_1d

    inp = tf.keras.Input(shape=(3, padded_size, padded_size, 1), name="StS")

    loops = [
        tf.keras.layers.Lambda(lambda t, i=i: t[:, i, :, :, :], name=f"loop_{i}")(inp)
        for i in range(3)
    ]

    patch_embed = PatchEmbedding(
        patch_size=cfg.patch,
        embed_dim=cfg.embed_dim,
        num_patches=num_patches,
    )
    pos_embed = PositionalEmbedding(
        num_patches=num_patches,
        embed_dim=cfg.embed_dim,
    )

    blocks = [
        TransformerBlock(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
        )
        for _ in range(cfg.depth)
    ]

    final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def encode_one(x):
        x = patch_embed(x)
        x = pos_embed(x)
        for blk in blocks:
            x = blk(x)
        x = final_norm(x)
        x = tf.reduce_mean(x, axis=1)  # token average pooling
        return x

    emb = [encode_one(x) for x in loops]
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

    ap.add_argument("--dataset", required=True, help="Root folder: dataset/StS/<class>/*.npz")
    ap.add_argument("--classes", nargs=2, default=["3_1", "4_1"])
    ap.add_argument("--n_per_class", type=int, default=694)
    ap.add_argument("--variant", default="StS")

    ap.add_argument("--expected_L", type=int, default=201)
    ap.add_argument("--split_train", type=float, default=0.8)
    ap.add_argument("--split_val", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--patch", type=int, default=15)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--mlp_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--dense_units", type=int, default=64)

    ap.add_argument("--aug", default="none", choices=["none", "permute"])

    ap.add_argument(
        "--preprocess",
        default="clip_log_zscore_train",
        choices=[
            "zscore",
            "clip_zscore",
            "log_zscore",
            "clip_log_zscore",
        ],
        help="Train-fitted preprocessing base mode for StS.",
    )
    ap.add_argument("--clip_lo", type=float, default=0.001)
    ap.add_argument("--clip_hi", type=float, default=0.999)
    ap.add_argument("--norm_eps", type=float, default=1e-6)

    ap.add_argument("--out", required=True)
    ap.add_argument("--results_csv", default=None)

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

    pp_params = fit_preprocess_params_train(
        train_files=train_files,
        variant_key=args.variant,
        mode=args.preprocess,
        clip_qlo=float(args.clip_lo),
        clip_qhi=float(args.clip_hi),
        eps=float(args.norm_eps),
    )

    with (out_dir / "preprocess.json").open("w", encoding="utf-8") as f:
        json.dump(pp_params, f, ensure_ascii=False, indent=2)

    train_ds = make_tf_dataset(
        train_files, class_to_id, args.expected_L, args.patch, args.bs,
        shuffle=True, seed=args.seed,
        variant_key=args.variant,
        aug=args.aug,
        pp_params=pp_params,
    )

    val_ds = make_tf_dataset(
        val_files, class_to_id, args.expected_L, args.patch, args.bs,
        shuffle=False, seed=args.seed,
        variant_key=args.variant,
        aug="none",
        pp_params=pp_params,
    )

    test_ds = make_tf_dataset(
        test_files, class_to_id, args.expected_L, args.patch, args.bs,
        shuffle=False, seed=args.seed,
        variant_key=args.variant,
        aug="none",
        pp_params=pp_params,
    )

    cfg = TransformerCfg(
        expected_L=int(args.expected_L),
        patch=int(args.patch),
        embed_dim=int(args.embed_dim),
        depth=int(args.depth),
        num_heads=int(args.num_heads),
        mlp_dim=int(args.mlp_dim),
        dropout=float(args.dropout),
        dense_units=int(args.dense_units),
    )

    model = build_sts_transformer_model(cfg, n_classes=2, lr=args.lr)

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

    metrics_blob = evaluate_and_save_metrics(model, test_ds, out_dir, id_to_class)

    final_path = out_dir / "final_model"
    model.save(str(final_path))

    train_eval = model.evaluate(train_ds, verbose=0)
    train_loss = float(train_eval[0]) if isinstance(train_eval, (list, tuple)) else float(train_eval)
    train_acc = float(train_eval[1]) if isinstance(train_eval, (list, tuple)) and len(train_eval) > 1 else float("nan")

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(root),
        "classes": [c1, c2],
        "n_per_class_requested": int(args.n_per_class),
        "n_per_class_effective": int(effective_n),
        "splits": {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
        "input": {
            "L": int(args.expected_L),
            "loops": 3,
            "representation": "StS_transformer",
            "variant_key": args.variant,
            "patch": int(args.patch),
            "padded_size": int(math.ceil(args.expected_L / args.patch) * args.patch),
        },
        "train_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.bs),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "split": [float(args.split_train), float(args.split_val), float(1.0 - args.split_train - args.split_val)],
        },
        "preprocess": pp_params,
        "augmentations": {
            "aug": args.aug,
        },
        "model": {
            "name": "sts_transformer",
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
            "train_files": str(out_dir / "train_files.txt"),
            "val_files": str(out_dir / "val_files.txt"),
            "test_files": str(out_dir / "test_files.txt"),
            "preprocess_json": str(out_dir / "preprocess.json"),
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
            "model": "sts_transformer",
            "preprocess": args.preprocess,
            "aug": args.aug,
            "seed": args.seed,
            "expected_L": args.expected_L,
            "patch": args.patch,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "mlp_dim": args.mlp_dim,
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