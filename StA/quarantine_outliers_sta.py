from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np


def list_npz(root: Path, cls: str) -> List[Path]:
    d = root / cls
    if not d.exists():
        raise FileNotFoundError(d)
    return sorted(d.glob("*.npz"))


def max_abs_in_file(fp: Path, variant: str) -> float:
    d = np.load(str(fp), allow_pickle=False)
    A = d[variant].astype(np.float32)
    return float(np.max(np.abs(A)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset/StA root")
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--variant", default="StA")
    ap.add_argument("--threshold", type=float, required=True, help="Mark outliers with max_abs > threshold")
    ap.add_argument("--out_csv", required=True, help="Output CSV with outliers")
    ap.add_argument("--quarantine_dir", default="", help="If set and --move, move files here")
    ap.add_argument("--move", action="store_true", help="Actually move outliers to quarantine_dir")
    args = ap.parse_args()

    root = Path(args.dataset)
    c1, c2 = args.classes
    thr = float(args.threshold)
    out_csv = Path(args.out_csv)

    outliers: List[Tuple[str, float, Path]] = []
    for cls in (c1, c2):
        for fp in list_npz(root, cls):
            m = max_abs_in_file(fp, args.variant)
            if m > thr:
                outliers.append((cls, m, fp))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "max_abs", "path"])
        for cls, m, fp in sorted(outliers, key=lambda t: -t[1]):
            w.writerow([cls, f"{m:.6f}", str(fp)])

    print(f"Found {len(outliers)} outliers with max_abs > {thr}")
    print(f"Saved list: {out_csv}")

    if args.move:
        if not args.quarantine_dir:
            raise ValueError("--move requires --quarantine_dir")
        qroot = Path(args.quarantine_dir)
        for cls, m, fp in outliers:
            rel = fp.relative_to(root)  # e.g. 3_1\file.npz
            dest = qroot / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"Moving: {fp}  ->  {dest}")
            shutil.move(str(fp), str(dest))
        print(f"Moved {len(outliers)} files to: {qroot}")


if __name__ == "__main__":
    main()