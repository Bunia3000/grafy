# visualize_sts_samples.py
# Visualize StS tensors stored as:
#   <dataset_root>/<class_label>/*.npz
# or from rows listed in:
#   sts_top_outliers.csv / sts_per_file_stats.csv
#
# Expected per file:
#   key "StS" with shape (3, L, L)
#
# What it does:
# - loads selected files
# - plots 3 heatmaps (loop 0, 1, 2)
# - optionally uses symmetric color scale around 0
# - optionally clips display range to robust quantiles
# - saves PNG figures
#
# Example:
# python visualize_sts_samples.py ^
#   --input_csv "C:\Users\danil\.vscode\grafy\experiments\sts_audit_3_1_vs_4_1\sts_top_outliers.csv" ^
#   --key StS ^
#   --top_k 20 ^
#   --out_dir "C:\Users\danil\.vscode\grafy\experiments\sts_audit_3_1_vs_4_1\png_top20"
#
# Or direct file list:
# python visualize_sts_samples.py ^
#   --files "C:\...\3_1-000001.npz" "C:\...\4_1-000010.npz" ^
#   --out_dir "C:\...\png_manual"

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


def read_rows_from_csv(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_file_list_from_csv(
    csv_path: Path,
    top_k: int | None,
    only_ok: bool,
) -> list[Path]:
    rows = read_rows_from_csv(csv_path)

    if only_ok and rows and "status" in rows[0]:
        rows = [r for r in rows if r.get("status", "") == "OK"]

    if top_k is not None:
        rows = rows[:top_k]

    files = []
    for r in rows:
        p = r.get("file", "")
        if p:
            files.append(Path(p))

    return files


def robust_limits(
    arr: np.ndarray,
    qlo: float,
    qhi: float,
    symmetric: bool,
) -> tuple[float, float]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    lo = float(np.quantile(flat, qlo))
    hi = float(np.quantile(flat, qhi))

    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def draw_one_sample(
    npz_path: Path,
    out_dir: Path,
    key: str,
    symmetric: bool,
    use_robust: bool,
    qlo: float,
    qhi: float,
    dpi: int,
    cmap: str,
) -> Path:
    d = np.load(str(npz_path), allow_pickle=False)
    if key not in d:
        raise KeyError(f"Missing key '{key}' in {npz_path}. Keys={list(d.keys())}")

    X = np.asarray(d[key], dtype=np.float32)
    if X.ndim != 3 or X.shape[0] != 3 or X.shape[1] != X.shape[2]:
        raise ValueError(f"Expected shape (3,L,L), got {X.shape} in {npz_path}")

    class_label = npz_path.parent.name
    graph_id = npz_path.stem
    L = X.shape[1]

    if use_robust:
        vmin, vmax = robust_limits(X, qlo=qlo, qhi=qhi, symmetric=symmetric)
    else:
        vmin = float(np.min(X))
        vmax = float(np.max(X))
        if symmetric:
            m = max(abs(vmin), abs(vmax))
            vmin, vmax = -m, m

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    loop_titles = ["loop 0", "loop 1", "loop 2"]
    ims = []

    for i in range(3):
        ax = axes[i]
        im = ax.imshow(
            X[i],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

        arr = X[i]
        diag_abs_max = float(np.max(np.abs(np.diag(arr))))
        anti_mean = float(np.mean(np.abs(arr + arr.T)))
        max_abs = float(np.max(np.abs(arr)))

        ax.set_title(
            f"{loop_titles[i]}\n"
            f"max|x|={max_abs:.3g}, diag={diag_abs_max:.3g}, anti_mean={anti_mean:.3g}",
            fontsize=10,
        )
        ax.set_xlabel("j")
        if i == 0:
            ax.set_ylabel("i")

    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(key)

    fig.suptitle(
        f"{class_label} / {graph_id}   shape=(3,{L},{L})   "
        f"range=[{vmin:.3g}, {vmax:.3g}]",
        fontsize=12,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "sym" if symmetric else "raw"
    suffix2 = "robust" if use_robust else "full"
    out_path = out_dir / f"{class_label}__{graph_id}__{suffix}__{suffix2}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def unique_existing_paths(paths: Iterable[Path]) -> list[Path]:
    out = []
    seen = set()
    for p in paths:
        p = Path(p)
        if str(p) in seen:
            continue
        seen.add(str(p))
        if p.exists():
            out.append(p)
        else:
            print(f"[WARN] Missing file: {p}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_csv", help="CSV with column 'file', e.g. sts_top_outliers.csv")
    src.add_argument("--files", nargs="*", help="Direct list of npz files")

    ap.add_argument("--key", default="StS", help="Key in npz, default: StS")
    ap.add_argument("--top_k", type=int, default=20, help="Use first K rows from CSV")
    ap.add_argument("--only_ok", action="store_true", help="When CSV has status column, keep only status=OK")
    ap.add_argument("--out_dir", required=True, help="Output folder for PNG files")

    ap.add_argument("--symmetric", action="store_true", help="Use symmetric color range around 0")
    ap.add_argument("--full_range", action="store_true", help="Use full min/max instead of robust quantiles")
    ap.add_argument("--qlo", type=float, default=0.01, help="Lower quantile for robust display range")
    ap.add_argument("--qhi", type=float, default=0.99, help="Upper quantile for robust display range")
    ap.add_argument("--dpi", type=int, default=160, help="PNG dpi")
    ap.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap")

    args = ap.parse_args()

    if args.input_csv:
        files = parse_file_list_from_csv(
            csv_path=Path(args.input_csv),
            top_k=args.top_k,
            only_ok=args.only_ok,
        )
    else:
        files = [Path(x) for x in (args.files or [])]

    files = unique_existing_paths(files)

    if not files:
        raise RuntimeError("No valid files to visualize.")

    out_dir = Path(args.out_dir)
    use_robust = not args.full_range

    print(f"Files to draw: {len(files)}")
    print(f"Output dir:    {out_dir}")
    print(f"Key:           {args.key}")
    print(f"Symmetric:     {args.symmetric}")
    print(f"Robust range:  {use_robust}")
    if use_robust:
        print(f"Quantiles:     [{args.qlo}, {args.qhi}]")
    print()

    saved = []
    for idx, p in enumerate(files, start=1):
        try:
            out_path = draw_one_sample(
                npz_path=p,
                out_dir=out_dir,
                key=args.key,
                symmetric=args.symmetric,
                use_robust=use_robust,
                qlo=args.qlo,
                qhi=args.qhi,
                dpi=args.dpi,
                cmap=args.cmap,
            )
            saved.append(out_path)
            print(f"[{idx}/{len(files)}] OK  {p.name} -> {out_path.name}")
        except Exception as e:
            print(f"[{idx}/{len(files)}] ERR {p.name}: {e}")

    print("\nDone.")
    print(f"Saved images: {len(saved)}")
    if saved:
        print(f"First output: {saved[0]}")


if __name__ == "__main__":
    main()