# rewrite_npz_to_loops.py
# Converts grafs-dataset/*.npz (X shape: (3,L,3) edges) into X-loops/*.npz
# where X becomes 3 loops:
#   Loop12 = concat(E1, reverse(E2)[1:])
#   Loop23 = concat(E2, reverse(E3)[1:])
#   Loop13 = concat(E1, reverse(E3)[1:])
#
# Output keeps the same folder structure: <out_root>/npz/<class_label>/<same_filename>.npz
# By default, it overwrites key "X" with loops and updates "L" to (2*L-1).
# It also stores original_L for traceability.
#
# Usage (CMD/PowerShell):
#   python rewrite_npz_to_loops.py ^
#     --input  "C:\Users\danil\.vscode\grafy\dataset\grafs-dataset" ^
#     --output "C:\Users\danil\.vscode\grafy\dataset\X-loops"

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np


def make_loop(Ea: np.ndarray, Eb: np.ndarray) -> np.ndarray:
    """
    Ea, Eb: (L,3) both oriented v1->v2.
    Loop = Ea (v1->v2) + reverse(Eb)[1:] (v2->v1), dropping duplicate v2.
    Returns: (2L-1,3)
    """
    Eb_rev = Eb[::-1]
    return np.concatenate([Ea, Eb_rev[1:]], axis=0)


def edges_to_loops(X_edges: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    X_edges: (3, L, 3)
    Returns:
      X_loops: (3, 2L-1, 3) with loops [12,23,13]
      L_loop:  2L-1
    """
    if X_edges.ndim != 3 or X_edges.shape[0] != 3 or X_edges.shape[2] != 3:
        raise ValueError(f"Expected X shape (3,L,3), got {X_edges.shape}")

    L = int(X_edges.shape[1])
    E1, E2, E3 = X_edges[0], X_edges[1], X_edges[2]

    L12 = make_loop(E1, E2)
    L23 = make_loop(E2, E3)
    L13 = make_loop(E1, E3)

    X_loops = np.stack([L12, L23, L13], axis=0)  # (3, 2L-1, 3)
    return X_loops, X_loops.shape[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to grafs-dataset root (contains npz/)")
    ap.add_argument("--output", required=True, help="Path to output root (will create npz/...)")
    ap.add_argument("--dry-run", action="store_true", help="Validate only; do not write files")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)

    in_npz = in_root / "npz"
    if not in_npz.exists():
        raise FileNotFoundError(f"Cannot find input npz/ at: {in_npz}")

    out_npz = out_root / "npz"
    ensure_dir(out_npz)

    files = sorted(in_npz.rglob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found under: {in_npz}")

    ok = 0
    bad = 0

    for src in files:
        rel = src.relative_to(in_npz)  # <class_label>/<file>.npz
        dst = out_npz / rel
        ensure_dir(dst.parent)

        if dst.exists() and not args.overwrite:
            # Skip to avoid accidental overwrite unless explicitly allowed
            print(f"SKIP exists: {dst}")
            continue

        try:
            d = np.load(str(src), allow_pickle=True)
            if "X" not in d:
                raise ValueError("Missing key 'X' in npz")

            X_edges = d["X"]
            X_loops, L_loop = edges_to_loops(X_edges)

            # Keep metadata (graph_id, class_label, source_file) if present
            graph_id = d["graph_id"] if "graph_id" in d else np.array(src.stem)
            class_label = d["class_label"] if "class_label" in d else np.array(src.parent.name)
            source_file = d["source_file"] if "source_file" in d else np.array(str(src))
            original_L = d["L"] if "L" in d else np.array(int(X_edges.shape[1]), dtype=np.int32)

            if not args.dry_run:
                np.savez(
                    str(dst),
                    X=X_loops.astype(np.float32),
                    graph_id=graph_id,
                    class_label=class_label,
                    source_file=source_file,
                    L=np.array(L_loop, dtype=np.int32),
                    original_L=np.array(int(original_L), dtype=np.int32),
                )

            ok += 1

        except Exception as e:
            bad += 1
            print(f"BAD {src} -> {e}")

    print("\nDone.")
    print(f"Input:  {in_root}")
    print(f"Output: {out_root}")
    print(f"Files scanned: {len(files)}")
    print(f"OK: {ok}  BAD: {bad}")
    if args.dry_run:
        print("Dry run mode: no files were written.")


if __name__ == "__main__":
    main()