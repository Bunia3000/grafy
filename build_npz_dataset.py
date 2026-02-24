# python dataset/build_npz_dataset.py --input "C:\Users\danil\.vscode\grafy\dataset\grafyzexela" --output "C:\Users\danil\.vscode\grafy\dataset\grafs-dataset" --L 101
#
#
# build_npz_dataset.py
# Converts your .xyz graph files into per-graph .npz files with canonical XYZ preprocessing.
#
# Input structure (example):
#   C:\Users\danil\.vscode\praca\grafyzexela\
#     0_1\1.xyz
#     0_1\2.xyz
#     ...
#     7_64\1.xyz
#
# Output structure (example):
#   C:\Users\danil\.vscode\praca\grafs-dataset\
#     npz\
#       0_1\
#         0_1-000001.npz
#         0_1-000002.npz
#       7_64\
#         7_64-000001.npz
#
# Each .npz contains:
#   - X: float32 array of shape (3, 100, 3) after canonicalization
#   - graph_id: string, e.g. "0_1-000002"
#   - class_label: string, e.g. "0_1"
#   - source_file: string, original path
#
# Usage (Windows PowerShell):
#   python build_npz_dataset.py ^
#     --input "C:\Users\danil\.vscode\praca\grafyzexela" ^
#     --output "C:\Users\danil\.vscode\praca\grafs-dataset" ^
#     --L 100
#
# Optional:
#   --tol 1e-6        # tolerance for v1/v2 equality checks
#   --dry-run         # validate only, no writing
#
# Produces a CSV report:
#   <output>\manifests\build_report.csv

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class BuildRow:
    class_label: str
    source_file: str
    graph_id: str
    out_file: str
    ok: bool
    note: str


def parse_numeric_stem(path: Path) -> Optional[int]:
    """
    Tries to parse an integer graph number from filename stem.
    Examples:
      "1" -> 1
      "0002" -> 2
      "graph_12" -> 12
    Returns None if not found.
    """
    stem = path.stem
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", stem)
    if m:
        return int(m.group(1))
    return None


def make_graph_id(class_label: str, number: int) -> str:
    # Safe version requested: dash, 6-digit numbering.
    return f"{class_label}-{number:06d}"


def read_xyz_file(file_path: Path) -> np.ndarray:
    """
    Reads a .xyz file with rows: edge_id i x y z
    Returns raw array of shape (n_rows, 5) float64.
    """
    # Files appear whitespace-separated, no header.
    data = np.loadtxt(str(file_path), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def validate_and_build_X(
    data: np.ndarray, L: int, tol: float
) -> Tuple[np.ndarray, str]:
    """
    Validates raw data and builds X of shape (3, L, 3).
    Returns (X, note). Raises ValueError on hard failure.
    """
    if data.shape[1] != 5:
        raise ValueError(f"Expected 5 columns (edge_id i x y z), got {data.shape[1]}")

    n_expected = 3 * L
    if data.shape[0] != n_expected:
        raise ValueError(f"Expected {n_expected} rows (3*L), got {data.shape[0]}")

    edge_id = data[:, 0].astype(np.int64)
    idx_i = data[:, 1].astype(np.int64)
    coords = data[:, 2:5].astype(np.float64)

    edges_present = set(edge_id.tolist())
    if edges_present != {0, 1, 2}:
        raise ValueError(f"edge_id must be exactly {{0,1,2}}, got {sorted(edges_present)}")

    X = np.zeros((3, L, 3), dtype=np.float64)

    note_parts = []

    for e in (0, 1, 2):
        mask = edge_id == e
        if mask.sum() != L:
            raise ValueError(f"Edge {e} must have exactly L={L} rows, got {mask.sum()}")

        i_vals = idx_i[mask]
        if set(i_vals.tolist()) != set(range(L)):
            missing = sorted(set(range(L)) - set(i_vals.tolist()))
            extra = sorted(set(i_vals.tolist()) - set(range(L)))
            raise ValueError(f"Edge {e}: indices i must cover 0..{L-1}. Missing={missing[:10]} Extra={extra[:10]}")

        # Put points into correct positions by i
        pts = coords[mask]
        # order by i
        order = np.argsort(i_vals)
        i_sorted = i_vals[order]
        pts_sorted = pts[order]

        # double-check consecutive
        if not np.array_equal(i_sorted, np.arange(L)):
            raise ValueError(f"Edge {e}: after sorting, i is not exactly 0..{L-1}")

        X[e, :, :] = pts_sorted

    # Validate v1/v2 consistency across edges (within tol)
    v1s = X[:, 0, :]      # (3,3)
    v2s = X[:, L - 1, :]  # (3,3)
    v1_mean = v1s.mean(axis=0)
    v2_mean = v2s.mean(axis=0)

    v1_max_dev = np.max(np.linalg.norm(v1s - v1_mean, axis=1))
    v2_max_dev = np.max(np.linalg.norm(v2s - v2_mean, axis=1))

    if v1_max_dev > tol:
        raise ValueError(f"v1 mismatch across edges: max deviation {v1_max_dev:.3e} > tol {tol:.3e}")
    if v2_max_dev > tol:
        raise ValueError(f"v2 mismatch across edges: max deviation {v2_max_dev:.3e} > tol {tol:.3e}")

    # Sanity: vertex distance > 0
    d = float(np.linalg.norm(v2_mean - v1_mean))
    if d <= 0 or not np.isfinite(d):
        raise ValueError("Vertex distance ||v2-v1|| must be positive and finite")

    note_parts.append(f"v1_dev={v1_max_dev:.2e}")
    note_parts.append(f"v2_dev={v2_max_dev:.2e}")
    note_parts.append(f"vertex_dist={d:.6g}")

    return X, "; ".join(note_parts)


def rotation_matrix_map_u_to_z(u: np.ndarray) -> np.ndarray:
    """
    Returns R (3x3) such that R @ u = z_hat, where u is a unit vector.
    Uses Rodrigues' rotation formula with stable special cases.
    """
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = u.astype(np.float64)

    # Clamp dot to [-1,1] for numeric stability
    c = float(np.clip(np.dot(u, z), -1.0, 1.0))

    # If u ~ z: no rotation
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)

    # If u ~ -z: rotate 180 degrees around x-axis (or any axis orthogonal to z)
    if c < -1.0 + 1e-12:
        # 180° around x maps (0,0,-1) -> (0,0,1)
        return np.array([[1.0, 0.0,  0.0],
                         [0.0, -1.0, 0.0],
                         [0.0, 0.0, -1.0]], dtype=np.float64)

    # General case:
    # axis k = u x z
    k = np.cross(u, z)
    s = float(np.linalg.norm(k))
    k = k / s  # unit axis

    K = np.array([[0.0,    -k[2],  k[1]],
                  [k[2],   0.0,   -k[0]],
                  [-k[1],  k[0],   0.0]], dtype=np.float64)

    # Rodrigues: R = I*cosθ + (1-cosθ)kk^T + sinθ*K
    # Here cosθ = c, sinθ = s
    I = np.eye(3, dtype=np.float64)
    kkT = np.outer(k, k)
    R = I * c + (1.0 - c) * kkT + s * K
    return R


def canonicalize_X(X: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Canonical preprocessing:
      - translate so midpoint of (v1,v2) is at origin
      - scale so ||v2-v1|| = 1
      - rotate so (v2-v1) is aligned with +Z axis
    Returns (X_canonical, meta)
    """
    L = X.shape[1]
    v1 = X[0, 0, :].copy()
    v2 = X[0, L - 1, :].copy()

    c = 0.5 * (v1 + v2)
    X0 = X - c

    a = (v2 - v1)
    d = float(np.linalg.norm(a))
    X1 = X0 / d

    u = a / d  # unit direction in original coordinates
    R = rotation_matrix_map_u_to_z(u)

    # Apply rotation: for each point p, p' = R @ p
    X2 = np.einsum("ij,elj->eli", R, X1)  # (3,L,3)

    meta = {
        "midpoint": c,
        "scale_d": d,
        "u_dir": u,
        "R": R,
    }
    return X2, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input root folder with class subfolders (e.g., grafyzexela)")
    parser.add_argument("--output", required=True, help="Output root folder for grafs-dataset")
    parser.add_argument("--L", type=int, default=100, help="Number of points per edge (default 100)")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for v1/v2 consistency checks")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report only; do not write .npz")
    args = parser.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)

    if not in_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {in_root}")

    npz_root = out_root / "npz"
    manifests_root = out_root / "manifests"
    npz_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    build_rows: list[BuildRow] = []

    # Class folders = directories directly under input root
    class_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])

    if not class_dirs:
        raise RuntimeError(f"No class directories found under: {in_root}")

    total_ok = 0
    total_bad = 0

    for class_dir in class_dirs:
        class_label = class_dir.name
        out_class_dir = npz_root / class_label
        if not args.dry_run:
            out_class_dir.mkdir(parents=True, exist_ok=True)

        xyz_files = sorted(class_dir.glob("*.xyz"))
        if not xyz_files:
            # skip silently, but log
            build_rows.append(BuildRow(
                class_label=class_label,
                source_file="",
                graph_id="",
                out_file="",
                ok=False,
                note="No .xyz files found in class folder"
            ))
            continue

        for f in xyz_files:
            num = parse_numeric_stem(f)
            if num is None:
                graph_id = f"{class_label}-{f.stem}"  # fallback
                ok = False
                note = "Could not parse numeric id from filename; rename file to a number like 1.xyz"
                out_file = ""
                build_rows.append(BuildRow(class_label, str(f), graph_id, out_file, ok, note))
                total_bad += 1
                continue

            graph_id = make_graph_id(class_label, num)
            out_file_path = out_class_dir / f"{graph_id}.npz"

            try:
                raw = read_xyz_file(f)
                X, vnote = validate_and_build_X(raw, L=args.L, tol=args.tol)
                Xc, meta = canonicalize_X(X)

                # Final sanity: v1->v2 should be aligned with +Z after canonicalization
                v1c = Xc[0, 0, :]
                v2c = Xc[0, args.L - 1, :]
                vec = v2c - v1c
                vec_norm = float(np.linalg.norm(vec))
                # vec_norm should be 1 after scaling
                align = float(vec[2] / vec_norm) if vec_norm > 0 else float("nan")
                if not (np.isfinite(align) and align > 1.0 - 1e-6):
                    # Not fatal, but suspicious
                    vnote += f"; warn_align={align:.6g}"

                if not args.dry_run:
                    np.savez(
                        str(out_file_path),
                        X=Xc.astype(np.float32),
                        graph_id=np.array(graph_id),
                        class_label=np.array(class_label),
                        source_file=np.array(str(f)),
                        L=np.array(args.L, dtype=np.int32),
                    )

                build_rows.append(BuildRow(
                    class_label=class_label,
                    source_file=str(f),
                    graph_id=graph_id,
                    out_file=str(out_file_path),
                    ok=True,
                    note=vnote
                ))
                total_ok += 1

            except Exception as e:
                build_rows.append(BuildRow(
                    class_label=class_label,
                    source_file=str(f),
                    graph_id=graph_id,
                    out_file=str(out_file_path),
                    ok=False,
                    note=str(e)
                ))
                total_bad += 1

    report_path = manifests_root / "build_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["class_label", "source_file", "graph_id", "out_file", "ok", "note"])
        for r in build_rows:
            w.writerow([r.class_label, r.source_file, r.graph_id, r.out_file, int(r.ok), r.note])

    print("Done.")
    print(f"Output root: {out_root}")
    print(f"NPZ root:    {npz_root}")
    print(f"Report:      {report_path}")
    print(f"OK: {total_ok}  BAD: {total_bad}")
    if args.dry_run:
        print("Dry run mode: no .npz files were written.")


if __name__ == "__main__":
    main()

