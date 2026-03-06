# train_sts.py
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


# ============================================================
# file helpers / splits
# ============================================================

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


# ============================================================
# preprocessing
# ============================================================

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
    y = x.astype(np.float32, copy=False)
    mode = params["mode_base"]
    eps = float(params.get("eps", 1e-6))

    if mode in ("clip_zscore", "clip_log_zscore"):
        lo = float(params["clip_lo"])
        hi = float(params["clip_hi"])
        y = np.clip(y, lo, hi)

    if mode in ("log_zscore", "clip_log_zscore"):
        y = np.sign(y) * np.log1p(np.abs(y))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        m = float(params["mean"])
        s = float(params["std"])
        y = (y - m) / (s + eps)

    return y.astype(np.float32, copy=False)


# ============================================================
# loader
# ============================================================

def _np_load_sts(
    path_bytes: bytes,
    class_to_id,
    expected_L,
    variant_key,
    preprocess_mode,
    clip_lo,
    clip_hi,
    pp_mean,
    pp_std,
    norm_eps,
):
    path = Path(path_bytes.decode("utf-8"))
    class_label = path.parent.name

    d = np.load(str(path), allow_pickle=False)
    x = d[variant_key].astype(np.float32)  # (3,L,L)

    if x.ndim != 3 or x.shape[0] != 3 or x.shape[1] != x.shape[2]:
        raise ValueError(f"Bad shape in {path}: got {x.shape}, expected (3,L,L)")

    if preprocess_mode == "none":
        pass
    else:
        params = {
            "mode_base": preprocess_mode.replace("_train", ""),
            "clip_lo": float(clip_lo),
            "clip_hi": float(clip_hi),
            "mean": float(pp_mean),
            "std": float(pp_std),
            "eps": float(norm_eps),
        }
        x = apply_preprocess_sts_global(x, params)

    L = x.shape[1]
    if expected_L is not None and L != expected_L:
        raise ValueError(f"Unexpected L={L} in {path}; expected {expected_L}")

    y = np.int64(class_to_id[class_label])
    return x.astype(np.float32), y


def make_tf_dataset(
    files,
    class_to_id,
    expected_L,
    bs,
    shuffle,
    seed,
    variant_key,
    preprocess_mode,
    clip_lo,
    clip_hi,
    pp_mean,
    pp_std,
    norm_eps,
):
    paths = tf.constant([str(p) for p in files])
    ds = tf.data.Dataset.from_tensor_slices(paths)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(files), seed=seed, reshuffle_each_iteration=True)

    def _map_fn(p):
        x, y = tf.numpy_function(
            func=lambda pb: _np_load_sts(
                pb,
                class_to_id,
                expected_L,
                variant_key,
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

        x = tf.expand_dims(x, axis=-1)  # (3,L,L,1)

        if expected_L is not None:
            x.set_shape((3, expected_L, expected_L, 1))
        else:
            x.set_shape((3, None, None, 1))

        y.set_shape(())
        return x, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(bs).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# models
# ============================================================

@dataclass
class BaselineCfg:
    expected_L: int
    dense_units: int
    dropout: float
    layernorm: bool


@dataclass
class Cnn2DCfg:
    expected_L: int
    filters: int
    blocks: int
    kernel: int
    dropout: float
    dense_units: int
    pool: str
    layernorm: bool


def build_baseline_lr_sts(cfg: BaselineCfg, n_classes: int, lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(3, cfg.expected_L, cfg.expected_L, 1), name="StS")
    x = tf.keras.layers.Reshape((3 * cfg.expected_L * cfg.expected_L,))(inp)

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


def build_shared_cnn_sts(cfg: Cnn2DCfg, n_classes: int, lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(3, cfg.expected_L, cfg.expected_L, 1), name="StS")

    loops = [
        tf.keras.layers.Lambda(lambda t, i=i: t[:, i, :, :, :], name=f"loop_{i}")(inp)
        for i in range(3)
    ]

    if cfg.layernorm:
        ln = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])
        loops = [ln(l) for l in loops]

    convs, drops = [], []
    for bi in range(cfg.blocks):
        f = cfg.filters * (2 ** bi)
        convs.append(tf.keras.layers.Conv2D(f, cfg.kernel, padding="same", activation="relu"))
        drops.append(tf.keras.layers.Dropout(cfg.dropout) if cfg.dropout > 0 else tf.keras.layers.Lambda(lambda z: z))

    gap = tf.keras.layers.GlobalAveragePooling2D()
    gmp = tf.keras.layers.GlobalMaxPooling2D()

    def encode(x):
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


# ============================================================
# csv appender
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

    ap.add_argument("--dataset", required=True, help="Root folder: dataset/StS/<class>/*.npz")
    ap.add_argument("--classes", nargs=2, default=["3_1", "4_1"])
    ap.add_argument("--n_per_class", type=int, default=694)
    ap.add_argument("--variant", default="StS", help="npz key, e.g. StS")
    ap.add_argument("--model", default="cnn_shared", choices=["baseline_lr", "cnn_shared"])

    ap.add_argument("--expected_L", type=int, default=201)
    ap.add_argument("--split_train", type=float, default=0.8)
    ap.add_argument("--split_val", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--filters", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--dense_units", type=int, default=64)
    ap.add_argument("--pool", default="avgmax", choices=["avg", "max", "avgmax"])
    ap.add_argument("--layernorm", action="store_true")

    ap.add_argument(
        "--preprocess",
        default="none",
        choices=[
            "none",
            "zscore_train",
            "clip_zscore_train",
            "log_zscore_train",
            "clip_log_zscore_train",
        ],
    )
    ap.add_argument("--clip_lo", type=float, default=0.001)
    ap.add_argument("--clip_hi", type=float, default=0.999)
    ap.add_argument("--norm_eps", type=float, default=1e-6)

    ap.add_argument("--out", required=True)
    ap.add_argument("--results_csv", default=None)

    ap.add_argument("--debug_overfit_k", type=int, default=0)
    ap.add_argument("--debug_overfit_epochs", type=int, default=300)

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

    # optional debug overfit
    if args.debug_overfit_k and args.debug_overfit_k > 0:
        k = int(args.debug_overfit_k)
        tr1 = [p for p in train_files if p.parent.name == c1][:k]
        tr2 = [p for p in train_files if p.parent.name == c2][:k]
        train_files = tr1 + tr2
        val_files = tr1 + tr2
        test_files = tr1 + tr2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_file_list(train_files, out_dir / "train_files.txt")
    save_file_list(val_files, out_dir / "val_files.txt")
    save_file_list(test_files, out_dir / "test_files.txt")

    # preprocess params fit on train only
    pp_params = None
    ds_clip_lo = float(args.clip_lo)
    ds_clip_hi = float(args.clip_hi)
    ds_mean = 0.0
    ds_std = 1.0

    if args.preprocess != "none":
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

    train_ds = make_tf_dataset(
        train_files, class_to_id, args.expected_L, args.bs,
        shuffle=True, seed=args.seed,
        variant_key=args.variant,
        preprocess_mode=args.preprocess,
        clip_lo=ds_clip_lo,
        clip_hi=ds_clip_hi,
        pp_mean=ds_mean,
        pp_std=ds_std,
        norm_eps=args.norm_eps,
    )

    val_ds = make_tf_dataset(
        val_files, class_to_id, args.expected_L, args.bs,
        shuffle=False, seed=args.seed,
        variant_key=args.variant,
        preprocess_mode=args.preprocess,
        clip_lo=ds_clip_lo,
        clip_hi=ds_clip_hi,
        pp_mean=ds_mean,
        pp_std=ds_std,
        norm_eps=args.norm_eps,
    )

    test_ds = make_tf_dataset(
        test_files, class_to_id, args.expected_L, args.bs,
        shuffle=False, seed=args.seed,
        variant_key=args.variant,
        preprocess_mode=args.preprocess,
        clip_lo=ds_clip_lo,
        clip_hi=ds_clip_hi,
        pp_mean=ds_mean,
        pp_std=ds_std,
        norm_eps=args.norm_eps,
    )

    if args.model == "baseline_lr":
        cfg = BaselineCfg(
            expected_L=int(args.expected_L),
            dense_units=int(args.dense_units),
            dropout=float(args.dropout),
            layernorm=bool(args.layernorm),
        )
        model = build_baseline_lr_sts(cfg, n_classes=2, lr=args.lr)
        model_cfg_blob = asdict(cfg)

    elif args.model == "cnn_shared":
        cfg = Cnn2DCfg(
            expected_L=int(args.expected_L),
            filters=int(args.filters),
            blocks=int(args.blocks),
            kernel=int(args.kernel),
            dropout=float(args.dropout),
            dense_units=int(args.dense_units),
            pool=str(args.pool),
            layernorm=bool(args.layernorm),
        )
        model = build_shared_cnn_sts(cfg, n_classes=2, lr=args.lr)
        model_cfg_blob = asdict(cfg)

    else:
        raise ValueError(args.model)

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

    fit_epochs = int(args.debug_overfit_epochs) if args.debug_overfit_k > 0 else int(args.epochs)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fit_epochs,
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
    train_loss = float(train_eval[0])
    train_acc = float(train_eval[1])

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(root),
        "classes": [c1, c2],
        "n_per_class_requested": int(args.n_per_class),
        "n_per_class_effective": int(effective_n),
        "splits": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "input": {
            "L": int(args.expected_L),
            "loops": 3,
            "representation": "StS",
            "variant_key": args.variant,
        },
        "train_config": {
            "epochs": int(fit_epochs),
            "batch_size": int(args.bs),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "split": [
                float(args.split_train),
                float(args.split_val),
                float(1.0 - args.split_train - args.split_val),
            ],
            "debug_overfit_k": int(args.debug_overfit_k),
        },
        "preprocess": {
            "mode": args.preprocess,
            "clip_lo": float(ds_clip_lo),
            "clip_hi": float(ds_clip_hi),
            "mean": float(ds_mean),
            "std": float(ds_std),
            "norm_eps": float(args.norm_eps),
            "train_fitted_params_path": str(out_dir / "preprocess.json") if args.preprocess != "none" else None,
        },
        "model": {
            "name": args.model,
            "config": model_cfg_blob,
        },
        "results": {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "train_loss_eval": train_loss,
            "train_accuracy_eval": train_acc,
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
            "seed": args.seed,
            "expected_L": args.expected_L,
            "epochs": fit_epochs,
            "bs": args.bs,
            "lr": args.lr,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "train_accuracy_eval": train_acc,
            "train_loss_eval": train_loss,
            "out_dir": str(out_dir),
        }
        append_results_csv(Path(args.results_csv), row)

    print(f"\nDone. Test acc={test_acc:.4f} loss={test_loss:.4f}")
    print(f"Saved: {summary_path}")
    print(f"Train-eval acc={train_acc:.4f} loss={train_loss:.4f}")


if __name__ == "__main__":
    main()