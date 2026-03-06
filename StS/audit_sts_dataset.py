# audit_sts_dataset.py
# Audit dataset of StS tensors stored as:
#   <dataset_root>/<class_label>/*.npz
#
# Expected content per file:
#   key "StS" with shape (3, L, L), usually (3, 201, 201)
#
# What it does:
# - validates files and shapes
# - checks NaN / inf
# - checks diagonal and excluded band near diagonal
# - measures antisymmetry error: |A + A^T|
# - computes per-file and per-loop statistics
# - saves:
#     sts_per_file_stats.csv
#     sts_top_outliers.csv
#     sts_class_summary.csv
#     sts_audit_summary.json
#
# Example:
# python audit_sts_dataset.py ^
#   --dataset "C:\Users\danil\.vscode\grafy\dataset\StS" ^
#   --classes 3_1 4_1 ^
#   --expected_L 201 ^
#   --neighbor_exclusion 1 ^
#   --top_k 50 ^
#   --out_dir "C:\Users\danil\.vscode\grafy\experiments\sts_audit"

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


# ----------------------------- helpers -----------------------------

def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def list_class_files(root: Path, class_label: str) -> list[Path]:
    class_dir = root / class_label
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")
    files = sorted(class_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in: {class_dir}")
    return files


def circular_band_mask(L: int, k: int) -> np.ndarray:
    """
    Mask for circular distance <= k around diagonal.
    """
    idx = np.arange(L)
    d1 = (idx[:, None] - idx[None, :]) % L
    d2 = (idx[None, :] - idx[:, None]) % L
    circ_dist = np.minimum(d1, d2)
    return circ_dist <= k


def stats_from_array(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "max_abs": np.nan,
            "q001": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q50": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "q999": np.nan,
        }

    return {
        "mean": safe_float(np.mean(x)),
        "std": safe_float(np.std(x)),
        "min": safe_float(np.min(x)),
        "max": safe_float(np.max(x)),
        "max_abs": safe_float(np.max(np.abs(x))),
        "q001": safe_float(np.quantile(x, 0.001)),
        "q01": safe_float(np.quantile(x, 0.01)),
        "q05": safe_float(np.quantile(x, 0.05)),
        "q50": safe_float(np.quantile(x, 0.50)),
        "q95": safe_float(np.quantile(x, 0.95)),
        "q99": safe_float(np.quantile(x, 0.99)),
        "q999": safe_float(np.quantile(x, 0.999)),
    }


def aggregate_numeric(records: list[dict], key: str) -> dict[str, float]:
    vals = [safe_float(r[key]) for r in records if key in r and np.isfinite(safe_float(r[key]))]
    if not vals:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": safe_float(np.mean(arr)),
        "std": safe_float(np.std(arr)),
        "min": safe_float(np.min(arr)),
        "max": safe_float(np.max(arr)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ----------------------------- audit core -----------------------------

@dataclass
class FileAuditResult:
    status: str
    class_label: str
    file: str
    graph_id: str
    shape_0: int | None
    shape_1: int | None
    shape_2: int | None
    expected_L: int | None
    has_nan: bool
    has_inf: bool
    finite_frac: float | None
    zero_frac: float | None
    diag_abs_max: float | None
    excl_band_abs_max: float | None
    antisym_abs_mean: float | None
    antisym_abs_max: float | None
    fro_norm: float | None
    sum_abs: float | None
    pos_frac: float | None
    neg_frac: float | None
    err_msg: str | None = None


def audit_file(
    npz_path: Path,
    key: str,
    expected_L: int | None,
    neighbor_exclusion: int,
) -> tuple[dict, list[dict]]:
    """
    Returns:
      file_row: one row per file
      loop_rows: three rows per loop
    """
    class_label = npz_path.parent.name
    graph_id = npz_path.stem

    try:
        d = np.load(str(npz_path), allow_pickle=False)
        if key not in d:
            raise KeyError(f"Missing key '{key}'. Keys={list(d.keys())}")

        A = np.asarray(d[key])
        if A.ndim != 3:
            raise ValueError(f"Expected ndim=3, got shape={A.shape}")
        if A.shape[0] != 3:
            raise ValueError(f"Expected first dim=3, got shape={A.shape}")
        if A.shape[1] != A.shape[2]:
            raise ValueError(f"Expected square matrices, got shape={A.shape}")

        L = int(A.shape[1])
        if expected_L is not None and L != expected_L:
            raise ValueError(f"Unexpected L={L}, expected {expected_L}")

        A = A.astype(np.float32, copy=False)

        has_nan = bool(np.isnan(A).any())
        has_inf = bool(np.isinf(A).any())
        finite_mask = np.isfinite(A)
        finite_frac = safe_float(np.mean(finite_mask))

        if has_nan or has_inf:
            # dalej policzmy ile się da na finite
            A_work = np.where(finite_mask, A, 0.0).astype(np.float32)
        else:
            A_work = A

        zero_frac = safe_float(np.mean(A_work == 0.0))

        band_mask = circular_band_mask(L, neighbor_exclusion)

        diag_vals = []
        excl_vals = []
        anti_abs_mean_all = []
        anti_abs_max_all = []
        loop_rows: list[dict] = []

        fro_sq_total = 0.0
        sum_abs_total = 0.0
        pos_count = 0
        neg_count = 0
        total_count = int(A_work.size)

        global_values = A_work.reshape(-1)
        global_stats = stats_from_array(global_values)

        for loop_idx in range(3):
            M = A_work[loop_idx].astype(np.float64, copy=False)

            diag = np.diag(M)
            diag_abs_max = safe_float(np.max(np.abs(diag)))

            excl_abs_max = safe_float(np.max(np.abs(M[band_mask])))

            anti = M + M.T
            anti_abs = np.abs(anti)
            anti_abs_mean = safe_float(np.mean(anti_abs))
            anti_abs_max = safe_float(np.max(anti_abs))

            vals = M.reshape(-1)
            st = stats_from_array(vals)

            fro_norm = safe_float(np.linalg.norm(M, ord="fro"))
            sum_abs = safe_float(np.sum(np.abs(M)))
            pos_frac = safe_float(np.mean(M > 0))
            neg_frac = safe_float(np.mean(M < 0))
            zero_frac_loop = safe_float(np.mean(M == 0.0))

            fro_sq_total += float(fro_norm ** 2)
            sum_abs_total += float(sum_abs)
            pos_count += int(np.sum(M > 0))
            neg_count += int(np.sum(M < 0))

            diag_vals.append(diag_abs_max)
            excl_vals.append(excl_abs_max)
            anti_abs_mean_all.append(anti_abs_mean)
            anti_abs_max_all.append(anti_abs_max)

            loop_rows.append({
                "status": "OK",
                "class_label": class_label,
                "file": str(npz_path),
                "graph_id": graph_id,
                "loop_idx": loop_idx,
                "L": L,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "finite_frac": finite_frac,
                "zero_frac": zero_frac_loop,
                "diag_abs_max": diag_abs_max,
                "excl_band_abs_max": excl_abs_max,
                "antisym_abs_mean": anti_abs_mean,
                "antisym_abs_max": anti_abs_max,
                "fro_norm": fro_norm,
                "sum_abs": sum_abs,
                "pos_frac": pos_frac,
                "neg_frac": neg_frac,
                **st,
            })

        file_row = asdict(FileAuditResult(
            status="OK",
            class_label=class_label,
            file=str(npz_path),
            graph_id=graph_id,
            shape_0=int(A.shape[0]),
            shape_1=int(A.shape[1]),
            shape_2=int(A.shape[2]),
            expected_L=expected_L,
            has_nan=has_nan,
            has_inf=has_inf,
            finite_frac=finite_frac,
            zero_frac=zero_frac,
            diag_abs_max=safe_float(np.max(diag_vals)),
            excl_band_abs_max=safe_float(np.max(excl_vals)),
            antisym_abs_mean=safe_float(np.mean(anti_abs_mean_all)),
            antisym_abs_max=safe_float(np.max(anti_abs_max_all)),
            fro_norm=safe_float(np.sqrt(fro_sq_total)),
            sum_abs=safe_float(sum_abs_total),
            pos_frac=safe_float(pos_count / total_count),
            neg_frac=safe_float(neg_count / total_count),
            err_msg=None,
        ))
        file_row.update(global_stats)

        return file_row, loop_rows

    except Exception as e:
        bad_row = asdict(FileAuditResult(
            status="ERR",
            class_label=class_label,
            file=str(npz_path),
            graph_id=graph_id,
            shape_0=None,
            shape_1=None,
            shape_2=None,
            expected_L=expected_L,
            has_nan=False,
            has_inf=False,
            finite_frac=None,
            zero_frac=None,
            diag_abs_max=None,
            excl_band_abs_max=None,
            antisym_abs_mean=None,
            antisym_abs_max=None,
            fro_norm=None,
            sum_abs=None,
            pos_frac=None,
            neg_frac=None,
            err_msg=str(e),
        ))
        return bad_row, []


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Root folder: dataset/StS/<class>/*.npz")
    ap.add_argument("--classes", nargs="*", default=None, help="Optional subset of classes, e.g. 3_1 4_1")
    ap.add_argument("--key", default="StS", help="Key in npz, default: StS")
    ap.add_argument("--expected_L", type=int, default=201, help="Expected L dimension")
    ap.add_argument("--neighbor_exclusion", type=int, default=1, help="Width of excluded circular band")
    ap.add_argument("--top_k", type=int, default=50, help="Top outliers to save")
    ap.add_argument("--out_dir", required=True, help="Where to save audit outputs")
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if args.classes:
        classes = list(args.classes)
    else:
        classes = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])

    all_files: list[Path] = []
    for cls in classes:
        try:
            all_files.extend(list_class_files(dataset_root, cls))
        except Exception as e:
            print(f"[WARN] Skipping class {cls}: {e}")

    if not all_files:
        raise RuntimeError("No files found to audit.")

    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {classes}")
    print(f"Files to audit: {len(all_files)}")
    print(f"Key: {args.key}")
    print(f"Expected L: {args.expected_L}")
    print(f"Neighbor exclusion: {args.neighbor_exclusion}")
    print()

    per_file_rows: list[dict] = []
    per_loop_rows: list[dict] = []

    ok = 0
    err = 0

    for idx, p in enumerate(all_files, start=1):
        file_row, loop_rows = audit_file(
            npz_path=p,
            key=args.key,
            expected_L=args.expected_L,
            neighbor_exclusion=args.neighbor_exclusion,
        )
        per_file_rows.append(file_row)
        per_loop_rows.extend(loop_rows)

        if file_row["status"] == "OK":
            ok += 1
        else:
            err += 1

        if idx % 100 == 0 or idx == len(all_files):
            print(f"[{idx}/{len(all_files)}] OK={ok} ERR={err} last={p.name}")

    # per-file csv
    per_file_csv = out_dir / "sts_per_file_stats.csv"
    write_csv(per_file_csv, per_file_rows)

    # top outliers by max_abs, only OK files
    ok_rows = [r for r in per_file_rows if r["status"] == "OK"]
    top_outliers = sorted(
        ok_rows,
        key=lambda r: safe_float(r.get("max_abs", np.nan)),
        reverse=True
    )[:args.top_k]

    top_csv = out_dir / "sts_top_outliers.csv"
    write_csv(top_csv, top_outliers)

    # class summary
    class_summary_rows: list[dict] = []
    for cls in classes:
        rows = [r for r in ok_rows if r["class_label"] == cls]
        if not rows:
            class_summary_rows.append({
                "class_label": cls,
                "n_ok": 0,
                "n_err": len([r for r in per_file_rows if r["class_label"] == cls and r["status"] != "OK"]),
            })
            continue

        class_summary_rows.append({
            "class_label": cls,
            "n_ok": len(rows),
            "n_err": len([r for r in per_file_rows if r["class_label"] == cls and r["status"] != "OK"]),
            "max_abs_mean": aggregate_numeric(rows, "max_abs")["mean"],
            "max_abs_std": aggregate_numeric(rows, "max_abs")["std"],
            "std_mean": aggregate_numeric(rows, "std")["mean"],
            "std_std": aggregate_numeric(rows, "std")["std"],
            "diag_abs_max_mean": aggregate_numeric(rows, "diag_abs_max")["mean"],
            "diag_abs_max_max": aggregate_numeric(rows, "diag_abs_max")["max"],
            "excl_band_abs_max_mean": aggregate_numeric(rows, "excl_band_abs_max")["mean"],
            "excl_band_abs_max_max": aggregate_numeric(rows, "excl_band_abs_max")["max"],
            "antisym_abs_mean_mean": aggregate_numeric(rows, "antisym_abs_mean")["mean"],
            "antisym_abs_mean_std": aggregate_numeric(rows, "antisym_abs_mean")["std"],
            "antisym_abs_max_mean": aggregate_numeric(rows, "antisym_abs_max")["mean"],
            "antisym_abs_max_max": aggregate_numeric(rows, "antisym_abs_max")["max"],
            "fro_norm_mean": aggregate_numeric(rows, "fro_norm")["mean"],
            "fro_norm_std": aggregate_numeric(rows, "fro_norm")["std"],
            "sum_abs_mean": aggregate_numeric(rows, "sum_abs")["mean"],
            "sum_abs_std": aggregate_numeric(rows, "sum_abs")["std"],
            "zero_frac_mean": aggregate_numeric(rows, "zero_frac")["mean"],
            "q001_mean": aggregate_numeric(rows, "q001")["mean"],
            "q01_mean": aggregate_numeric(rows, "q01")["mean"],
            "q05_mean": aggregate_numeric(rows, "q05")["mean"],
            "q50_mean": aggregate_numeric(rows, "q50")["mean"],
            "q95_mean": aggregate_numeric(rows, "q95")["mean"],
            "q99_mean": aggregate_numeric(rows, "q99")["mean"],
            "q999_mean": aggregate_numeric(rows, "q999")["mean"],
        })

    class_csv = out_dir / "sts_class_summary.csv"
    write_csv(class_csv, class_summary_rows)

    # global json summary
    summary = {
        "dataset_root": str(dataset_root),
        "out_dir": str(out_dir),
        "classes": classes,
        "key": args.key,
        "expected_L": args.expected_L,
        "neighbor_exclusion": args.neighbor_exclusion,
        "n_files_total": len(all_files),
        "n_ok": ok,
        "n_err": err,
        "artifacts": {
            "per_file_stats_csv": str(per_file_csv),
            "top_outliers_csv": str(top_csv),
            "class_summary_csv": str(class_csv),
        },
        "global_ok_stats": {
            "max_abs": aggregate_numeric(ok_rows, "max_abs"),
            "std": aggregate_numeric(ok_rows, "std"),
            "diag_abs_max": aggregate_numeric(ok_rows, "diag_abs_max"),
            "excl_band_abs_max": aggregate_numeric(ok_rows, "excl_band_abs_max"),
            "antisym_abs_mean": aggregate_numeric(ok_rows, "antisym_abs_mean"),
            "antisym_abs_max": aggregate_numeric(ok_rows, "antisym_abs_max"),
            "fro_norm": aggregate_numeric(ok_rows, "fro_norm"),
            "sum_abs": aggregate_numeric(ok_rows, "sum_abs"),
            "zero_frac": aggregate_numeric(ok_rows, "zero_frac"),
        },
        "worst_examples": {
            "by_max_abs": top_outliers[:10],
        },
    }

    summary_json = out_dir / "sts_audit_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"OK files:  {ok}")
    print(f"ERR files: {err}")
    print(f"Saved: {per_file_csv}")
    print(f"Saved: {top_csv}")
    print(f"Saved: {class_csv}")
    print(f"Saved: {summary_json}")

    if top_outliers:
        print("\nTop 10 by max_abs:")
        for row in top_outliers[:10]:
            print(
                f"{safe_float(row['max_abs']):12.6f}  "
                f"diag={safe_float(row['diag_abs_max']):10.6f}  "
                f"band={safe_float(row['excl_band_abs_max']):10.6f}  "
                f"anti_mean={safe_float(row['antisym_abs_mean']):10.6f}  "
                f"{row['class_label']:>6}  {Path(row['file']).name}"
            )


if __name__ == "__main__":
    main()