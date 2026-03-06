# train_loops_shared_lstm_xyz_v2.py
# XYZ-only v2:
# - shared encoder per loop (3 loops)
# - optional permutation augmentation of loops (order invariance)
# - optional reverse augmentation per loop (direction invariance)  <-- still XYZ
# - LayerNorm on each loop
# - return_sequences=True + temporal pooling (avg/max/last)
# - metrics + run_summary.json + split file lists
#
# Recommended run:
#   python ...\train_loops_shared_lstm_xyz_v2.py ^
#     --dataset "C:\Users\danil\.vscode\grafy\dataset\X-loops" ^
#     --classes 3_1 4_1 ^
#     --cap 695 ^
#     --epochs 25 ^
#     --bs 64 ^
#     --lr 0.001 ^
#     --expected_L 201 ^
#     --permute_loops ^
#     --reverse_aug --reverse_p 0.5 ^
#     --bidirectional ^
#     --pool avgmax ^
#     --out "C:\Users\danil\.vscode\grafy\results\sharedLSTM_XYZv2_3_1_vs_4_1"

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


# --- Make sure we can import your project modules from src/ (optional; not required here) ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # ...\grafy
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ------------------------ file helpers ------------------------

def list_class_files(npz_root: Path, class_label: str) -> List[Path]:
    class_dir = npz_root / class_label
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")
    files = sorted(class_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in: {class_dir}")
    return files


def stratified_split_two_classes(
    npz_root: Path,
    c1: str,
    c2: str,
    cap: int,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[List[Path], List[Path], List[Path], int]:
    rng = random.Random(seed)

    f1 = list_class_files(npz_root, c1)
    f2 = list_class_files(npz_root, c2)
    rng.shuffle(f1)
    rng.shuffle(f2)

    effective_cap = min(cap, len(f1), len(f2))
    f1 = f1[:effective_cap]
    f2 = f2[:effective_cap]

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

    return train, val, test, effective_cap


def save_file_list(paths: List[Path], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")


# ------------------------ metrics helpers ------------------------

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

    print(f"\nSaved metrics:")
    print(f"- {cm_path}")
    print(f"- {report_path}")
    print(f"- {pred_path}")
    print("\nConfusion matrix:\n", cm)

    return {
        "confusion_matrix": cm.tolist(),
        "metrics": stats,
    }


# ------------------------ loader (XYZ only) ------------------------

def _np_load_loops_raw(
    path_bytes: bytes,
    class_to_id: Dict[str, int],
    expected_L: int | None,
    permute_loops: bool,
    reverse_aug: bool,
    reverse_p: float,
    seed: int,
) -> Tuple[np.ndarray, np.int64]:
    """
    Loads X (3, L, 3) as float32 and applies optional augmentations:
      - permute loops order (3!)
      - reverse direction per loop with prob reverse_p  (XYZ only; just time reversal)
    Deterministic per-file given seed (so runs are reproducible).
    """
    path = Path(path_bytes.decode("utf-8"))
    class_label = path.parent.name
    if class_label not in class_to_id:
        raise ValueError(f"Unknown class_label={class_label} for file={path}")

    d = np.load(str(path), allow_pickle=True)
    if "X" not in d:
        raise ValueError(f"Missing 'X' in {path}")

    X = d["X"].astype(np.float32)
    if X.ndim != 3 or X.shape[0] != 3 or X.shape[2] != 3:
        raise ValueError(f"Bad X shape {X.shape} in {path}; expected (3,L,3)")

    L = int(d["L"]) if "L" in d else int(X.shape[1])
    if expected_L is not None and L != expected_L:
        raise ValueError(f"Unexpected L={L} in {path}; expected {expected_L}")

    # Deterministic RNG per file
    h = abs(hash(str(path))) % (2**31 - 1)
    rng = np.random.default_rng(seed + h)

    if permute_loops:
        perm = rng.permutation(3)
        X = X[perm, :, :]

    if reverse_aug:
        # independently decide reversal for each loop
        for i in range(3):
            if float(rng.random()) < reverse_p:
                X[i] = X[i, ::-1, :]

    y = np.int64(class_to_id[class_label])
    return X, y


def make_tf_dataset(
    files: List[Path],
    class_to_id: Dict[str, int],
    expected_L: int | None,
    bs: int,
    shuffle: bool,
    seed: int,
    permute_loops: bool,
    reverse_aug: bool,
    reverse_p: float,
) -> tf.data.Dataset:
    paths = tf.constant([str(p) for p in files])
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(files), seed=seed, reshuffle_each_iteration=True)

    def _map_fn(p):
        X, y = tf.numpy_function(
            func=lambda pb: _np_load_loops_raw(pb, class_to_id, expected_L, permute_loops, reverse_aug, reverse_p, seed),
            inp=[p],
            Tout=[tf.float32, tf.int64],
        )
        if expected_L is not None:
            X.set_shape((3, expected_L, 3))
        else:
            X.set_shape((3, None, 3))
        y.set_shape(())
        return X, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(bs).prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------ model (XYZ-only) ------------------------

@dataclass
class ModelConfig:
    expected_L: int
    rnn_type: str
    rnn_units: int
    rnn_layers: int
    bidirectional: bool
    dropout: float
    dense_units: int
    pool: str          # "last" | "avg" | "max" | "avgmax"
    layernorm: bool    # apply LayerNorm to loop inputs


def build_shared_loop_model_xyz(
    cfg: ModelConfig,
    n_classes: int,
    lr: float,
) -> tf.keras.Model:
    """
    Input: (3, L, 3)
    Shared encoder over loops.
    Encoder returns sequences, then we pool over time.
    """
    inp = tf.keras.Input(shape=(3, cfg.expected_L, 3), name="X")  # (batch,3,L,3)

    # Split loops: (batch,L,3)
    loops = [
        tf.keras.layers.Lambda(lambda t, i=i: t[:, i, :, :], name=f"loop_{i}")(inp)
        for i in range(3)
    ]

    # Shared LayerNorm (optional)
    if cfg.layernorm:
        ln = tf.keras.layers.LayerNormalization(axis=-1)
        loops = [ln(l) for l in loops]

    def rnn_layer(units: int, return_sequences: bool):
        if cfg.rnn_type == "GRU":
            base = tf.keras.layers.GRU(units, return_sequences=return_sequences)
        else:
            base = tf.keras.layers.LSTM(units, return_sequences=return_sequences)
        return tf.keras.layers.Bidirectional(base) if cfg.bidirectional else base

    # Shared stack: same layer objects reused across all loops
    shared_layers: List[tf.keras.layers.Layer] = []
    for _ in range(cfg.rnn_layers):
        # Keep sequences for pooling
        shared_layers.append(rnn_layer(cfg.rnn_units, return_sequences=True))
        if cfg.dropout > 0:
            shared_layers.append(tf.keras.layers.Dropout(cfg.dropout))

    # Shared pooling layers (no explicit names -> no collisions)
    gap = tf.keras.layers.GlobalAveragePooling1D()
    gmp = tf.keras.layers.GlobalMaxPooling1D()

    def pool_seq(seq, loop_index: int):
        # seq: (batch, L, feat)
        if cfg.pool == "last":
            # Use Lambda with unique name per loop
            return tf.keras.layers.Lambda(lambda x: x[:, -1, :], name=f"last_t_loop{loop_index}")(seq)
        if cfg.pool == "avg":
            return gap(seq)
        if cfg.pool == "max":
            return gmp(seq)
        if cfg.pool == "avgmax":
            # No fixed name -> Keras auto-unique
            return tf.keras.layers.Concatenate()([gap(seq), gmp(seq)])
        raise ValueError(f"Unknown pool={cfg.pool}")

    def encode_loop(x, loop_index: int):
        for layer in shared_layers:
            x = layer(x)
        return pool_seq(x, loop_index)

    emb = [encode_loop(loops[i], i) for i in range(3)]
    x = tf.keras.layers.Concatenate(name="concat_loops")(emb)

    if cfg.dense_units > 0:
        x = tf.keras.layers.Dense(cfg.dense_units, activation="relu")(x)
        if cfg.dropout > 0:
            x = tf.keras.layers.Dropout(cfg.dropout)(x)

    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="y")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Root folder containing npz/ (e.g. X-loops)")
    ap.add_argument("--classes", nargs=2, default=["3_1", "4_1"])
    ap.add_argument("--cap", type=int, default=695)

    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expected_L", type=int, default=201)

    # aug
    ap.add_argument("--permute_loops", action="store_true", help="Augment by permuting loop order")
    ap.add_argument("--reverse_aug", action="store_true", help="Augment by reversing loop direction (XYZ-only)")
    ap.add_argument("--reverse_p", type=float, default=0.5, help="Prob of reversing each loop (default 0.5)")

    # model
    ap.add_argument("--rnn_type", default="LSTM", choices=["LSTM", "GRU"])
    ap.add_argument("--rnn_units", type=int, default=64)
    ap.add_argument("--rnn_layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--dense_units", type=int, default=64)
    ap.add_argument("--pool", default="avgmax", choices=["last", "avg", "max", "avgmax"])
    ap.add_argument("--layernorm", action="store_true", help="Apply LayerNorm on loop inputs (recommended)")

    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "shared_lstm_xyz_v2_2class"))
    args = ap.parse_args()

    # CPU-friendly threading (tune if needed)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    root = Path(args.dataset)
    npz_root = root / "npz"
    if not npz_root.exists():
        raise FileNotFoundError(f"Cannot find npz/ at: {npz_root}")

    c1, c2 = args.classes[0], args.classes[1]
    class_to_id = {c1: 0, c2: 1}
    id_to_class = {0: c1, 1: c2}

    train_files, val_files, test_files, effective_cap = stratified_split_two_classes(
        npz_root=npz_root,
        c1=c1,
        c2=c2,
        cap=args.cap,
        seed=args.seed,
        train_frac=0.8,
        val_frac=0.1,
    )

    print(f"\nDataset: {root}")
    print(f"Classes: {[c1, c2]} mapped={class_to_id}")
    print(f"Cap requested={args.cap} effective={effective_cap}")
    print(f"Splits: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print(f"Input: (3, L, 3) with L={args.expected_L}")
    print(f"Train config: epochs={args.epochs} bs={args.bs} lr={args.lr} seed={args.seed}")
    print(f"Aug: permute_loops={args.permute_loops} reverse_aug={args.reverse_aug} reverse_p={args.reverse_p}")
    print(f"Model: {args.rnn_type} units={args.rnn_units} layers={args.rnn_layers} "
          f"bidi={args.bidirectional} dropout={args.dropout} dense_units={args.dense_units} "
          f"pool={args.pool} layernorm={args.layernorm}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_file_list(train_files, out_dir / "train_files.txt")
    save_file_list(val_files, out_dir / "val_files.txt")
    save_file_list(test_files, out_dir / "test_files.txt")

    train_ds = make_tf_dataset(
        train_files, class_to_id, args.expected_L, args.bs,
        shuffle=True, seed=args.seed,
        permute_loops=args.permute_loops,
        reverse_aug=args.reverse_aug,
        reverse_p=args.reverse_p,
    )
    val_ds = make_tf_dataset(
        val_files, class_to_id, args.expected_L, args.bs,
        shuffle=False, seed=args.seed,
        permute_loops=False,
        reverse_aug=False,
        reverse_p=0.0,
    )
    test_ds = make_tf_dataset(
        test_files, class_to_id, args.expected_L, args.bs,
        shuffle=False, seed=args.seed,
        permute_loops=False,
        reverse_aug=False,
        reverse_p=0.0,
    )

    cfg = ModelConfig(
        expected_L=int(args.expected_L),
        rnn_type=args.rnn_type,
        rnn_units=int(args.rnn_units),
        rnn_layers=int(args.rnn_layers),
        bidirectional=bool(args.bidirectional),
        dropout=float(args.dropout),
        dense_units=int(args.dense_units),
        pool=str(args.pool),
        layernorm=bool(args.layernorm),
    )

    model = build_shared_loop_model_xyz(cfg, n_classes=2, lr=args.lr)
    print("\nModel summary:")
    model.summary()

    ckpt_path = str(out_dir / "best_model")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, min_delta=1e-3),
        tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_best_only=True, monitor="val_loss"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating on test set:")
    eval_out = model.evaluate(test_ds, verbose=1)
    test_loss = float(eval_out[0]) if isinstance(eval_out, (list, tuple)) else float(eval_out)
    test_acc = float(eval_out[1]) if isinstance(eval_out, (list, tuple)) and len(eval_out) > 1 else float("nan")

    metrics_blob = evaluate_and_save_metrics(model, test_ds, out_dir, id_to_class)

    final_path = out_dir / "final_model"
    model.save(str(final_path))
    print(f"\nSaved final model to: {final_path}")

    map_path = out_dir / "class_to_id.txt"
    with map_path.open("w", encoding="utf-8") as f:
        f.write(f"0\t{c1}\n")
        f.write(f"1\t{c2}\n")
    print(f"Saved class_to_id map to: {map_path}")

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(root),
        "npz_root": str(npz_root),
        "classes": [c1, c2],
        "cap_requested": int(args.cap),
        "cap_effective": int(effective_cap),
        "splits": {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
        "input": {"L": int(args.expected_L), "loops": 3, "coords": 3, "representation": "XYZ"},
        "train_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.bs),
            "lr": float(args.lr),
            "seed": int(args.seed),
        },
        "augmentations": {
            "permute_loops": bool(args.permute_loops),
            "reverse_aug": bool(args.reverse_aug),
            "reverse_p": float(args.reverse_p),
        },
        "model_config": asdict(cfg),
        "results": {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "confusion_matrix": metrics_blob["confusion_matrix"],
            "metrics": metrics_blob["metrics"],
        },
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "artifacts": {
            "final_model": str(final_path),
            "best_model": str(out_dir / "best_model"),
            "class_to_id": str(map_path),
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
    print(f"Saved run summary to: {summary_path}")


if __name__ == "__main__":
    main()