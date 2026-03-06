from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to grafs-dataset root (contains npz/)")
    parser.add_argument("--k", type=int, default=10, help="Number of random samples to test")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--expected_L", type=int, default=101, help="Expected number of points per edge")
    args = parser.parse_args()

    root = Path(args.dataset)
    npz_root = root / "npz"
    if not npz_root.exists():
        raise FileNotFoundError(f"Cannot find npz/ at: {npz_root}")

    files = sorted(npz_root.rglob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found under: {npz_root}")

    random.seed(args.seed)
    sample_files = random.sample(files, k=min(args.k, len(files)))

    print(f"Found {len(files)} npz files under {npz_root}")
    print(f"Testing {len(sample_files)} samples...\n")

    ok = 0
    bad = 0

    for p in sample_files:
        try:
            d = np.load(str(p), allow_pickle=True)
            X = d["X"]
            graph_id = str(d["graph_id"])
            class_label = str(d["class_label"])
            L = int(d["L"]) if "L" in d else X.shape[1]

            # Shape checks
            if X.ndim != 3 or X.shape[0] != 3 or X.shape[2] != 3:
                raise ValueError(f"Bad X shape: {X.shape} (expected (3,L,3))")
            if L != args.expected_L:
                raise ValueError(f"Unexpected L: {L} (expected {args.expected_L})")

            # Canonical checks: v1->v2 aligned with +Z and length ~ 1
            v1 = X[0, 0, :]
            v2 = X[0, L - 1, :]
            vec = v2 - v1
            dist = float(np.linalg.norm(vec))
            if not np.isfinite(dist) or dist <= 0:
                raise ValueError(f"Non-positive/invalid vertex distance: {dist}")

            zhat = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            cos = float(np.dot(vec / dist, zhat))

            # We expect dist ~ 1 after scaling; allow small tolerance
            if abs(dist - 1.0) > 1e-3:
                raise ValueError(f"Vertex distance not ~1 after canonicalization: {dist}")

            # We expect strong alignment with +Z
            if cos < 1.0 - 1e-6:
                raise ValueError(f"Axis not aligned with +Z enough: cos={cos}")

            ok += 1
            print(f"OK  {graph_id:>12}  class={class_label:<6}  L={L}  dist={dist:.6f}  cos={cos:.9f}  file={p.name}")

        except Exception as e:
            bad += 1
            print(f"BAD {p} -> {e}")

    print("\nSummary:")
    print(f"OK:  {ok}")
    print(f"BAD: {bad}")
    if bad == 0:
        print("All tested samples look good.")


if __name__ == "__main__":
    main()
