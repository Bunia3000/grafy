# calculate_sta_sts_v2.py
import argparse
from pathlib import Path
import numpy as np


def compute_segments(P: np.ndarray, eps: float = 1e-12):
    """
    P: (L,3) points of a closed loop.
    Returns:
      r: (L,3) segment midpoints
      t: (L,3) unit tangents
      ds: (L,) segment lengths
    """
    P_next = np.roll(P, shift=-1, axis=0)
    d = P_next - P                           # (L,3)
    ds = np.linalg.norm(d, axis=1)           # (L,)
    ds_safe = np.maximum(ds, eps)
    t = d / ds_safe[:, None]                 # (L,3)
    r = 0.5 * (P + P_next)                   # (L,3)
    return r, t, ds


def compute_sts_sta_for_loop(
    P: np.ndarray,
    eps: float = 1e-12,
    neighbor_exclusion: int = 1,
    weighted: bool = True
):
    """
    P: (L,3)
    Returns:
      StS: (L,L)
      StA: (L,)
    """
    r, t, ds = compute_segments(P, eps=eps)
    L = r.shape[0]

    # Pairwise differences r_i - r_j : (L,L,3)
    rij = r[:, None, :] - r[None, :, :]

    # Pairwise norms: (L,L)
    dist = np.linalg.norm(rij, axis=-1)
    dist_safe = np.maximum(dist, eps)

    # cross(t_i, t_j): (L,L,3) via broadcasting
    cross_t = np.cross(t[:, None, :], t[None, :, :])

    # numerator: (cross · r_ij): (L,L)
    num = np.sum(cross_t * rij, axis=-1)

    # denom: |r_ij|^3
    denom = dist_safe ** 3

    StS = num / denom  # (L,L)

    # Optional ds_i ds_j weighting (Gauss integral discretization)
    if weighted:
        StS = StS * (ds[:, None] * ds[None, :])

    # Zero out diagonal + optionally near-diagonal (neighbors) for stability
    np.fill_diagonal(StS, 0.0)

    if neighbor_exclusion is not None and neighbor_exclusion > 0:
        k = int(neighbor_exclusion)
        idx = np.arange(L)
        # circular distance on a ring
        circ_dist = np.minimum((idx[:, None] - idx[None, :]) % L,
                               (idx[None, :] - idx[:, None]) % L)
        StS[circ_dist <= k] = 0.0

    StA = StS.sum(axis=1)  # (L,)
    return StS.astype(np.float32), StA.astype(np.float32)


def compute_sample(
    X: np.ndarray,
    expected_L: int | None,
    eps: float,
    neighbor_exclusion: int,
    weighted: bool
):
    """
    X: (3,L,3) float
    Returns:
      StS_all: (3,L,L)
      StA_all: (3,L)
    """
    if X.ndim != 3 or X.shape[0] != 3 or X.shape[2] != 3:
        raise ValueError(f"Expected X shape (3,L,3), got {X.shape}")

    L = X.shape[1]
    if expected_L is not None and L != expected_L:
        raise ValueError(f"Unexpected L={L}, expected {expected_L}")

    sts_list = []
    sta_list = []
    for k in range(3):
        P = X[k]  # (L,3)
        StS, StA = compute_sts_sta_for_loop(
            P, eps=eps, neighbor_exclusion=neighbor_exclusion, weighted=weighted
        )
        sts_list.append(StS)
        sta_list.append(StA)

    StS_all = np.stack(sts_list, axis=0)  # (3,L,L)
    StA_all = np.stack(sta_list, axis=0)  # (3,L)
    return StS_all, StA_all


# -------------------- smoothing (NEW) --------------------

def smooth_1d_circular_ma(x: np.ndarray, w: int) -> np.ndarray:
    """
    Circular moving-average smoothing for a 1D signal on a closed loop.
    x: (L,)
    w: window size; if w<=1 -> no smoothing. If w even -> will be made odd.
    """
    x = np.asarray(x, dtype=np.float32)
    L = x.shape[0]
    if w is None or w <= 1:
        return x.copy()

    if w % 2 == 0:
        w += 1

    half = w // 2
    xp = np.concatenate([x[-half:], x, x[:half]], axis=0)  # length L + 2*half
    kernel = np.ones(w, dtype=np.float32) / np.float32(w)
    y = np.convolve(xp, kernel, mode="valid")  # length L
    return y.astype(np.float32)


def smooth_sta_circular_ma(sta_raw: np.ndarray, w: int) -> np.ndarray:
    """
    sta_raw: (3, L)
    returns: (3, L)
    """
    sta_raw = np.asarray(sta_raw, dtype=np.float32)
    if sta_raw.ndim != 2 or sta_raw.shape[0] != 3:
        raise ValueError(f"Expected StA shape (3,L), got {sta_raw.shape}")
    return np.stack([smooth_1d_circular_ma(sta_raw[k], w) for k in range(3)], axis=0)


# -------------------- IO helpers --------------------

def save_npz(out_path: Path, **arrays):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help=r'npz root, e.g. C:\Users\danil\.vscode\grafy\dataset\X-loops\npz')
    ap.add_argument("--out_sta", required=True,
                    help=r'output root for StA, e.g. C:\Users\danil\.vscode\grafy\dataset\StA')
    ap.add_argument("--out_sts", required=True,
                    help=r'output root for StS, e.g. C:\Users\danil\.vscode\grafy\dataset\StS')
    ap.add_argument("--expected_L", type=int, default=None,
                    help="if set, enforce loop length L")
    ap.add_argument("--eps", type=float, default=1e-12,
                    help="small epsilon to avoid division by zero")
    ap.add_argument("--neighbor_exclusion", type=int, default=1,
                    help="exclude |i-j| <= k in circular sense (default 1)")
    ap.add_argument("--no_weight", action="store_true",
                    help="disable ds_i*ds_j weighting")
    ap.add_argument("--key", default="X",
                    help="npz key containing XYZ array (default: X)")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: process only first N files")

    # NEW: smoothing
    ap.add_argument("--smooth_w", type=int, default=0,
                    help="StA smoothing window (circular moving average). 0/1 disables. Suggested: 9, 13, 17.")

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_sta = Path(args.out_sta)
    out_sts = Path(args.out_sts)

    weighted = not args.no_weight
    smooth_w = int(args.smooth_w) if args.smooth_w is not None else 0

    files = sorted(input_dir.rglob("*.npz"))
    if args.limit is not None:
        files = files[:args.limit]

    ok = 0
    bad = 0

    for p in files:
        class_label = p.parent.name
        graph_id = p.stem

        try:
            data = np.load(p, allow_pickle=False)
            if args.key not in data:
                raise KeyError(f"Key '{args.key}' not found in {p.name}. Keys={list(data.keys())}")

            X = data[args.key].astype(np.float32)

            StS_all, StA_raw = compute_sample(
                X=X,
                expected_L=args.expected_L,
                eps=args.eps,
                neighbor_exclusion=args.neighbor_exclusion,
                weighted=weighted
            )

            # NEW: smoothed StA (same shape)
            StA_smooth = smooth_sta_circular_ma(StA_raw, smooth_w) if smooth_w and smooth_w > 1 else StA_raw.copy()

            # Save StA with BOTH variants in same file
            # Keep StA key as alias to raw for backward compatibility
            save_npz(
                out_sta / class_label / f"{graph_id}.npz",
                StA=StA_raw,                 # alias (old name)
                StA_raw=StA_raw,
                StA_smooth=StA_smooth,
                smooth_w=np.int32(smooth_w),
                smooth_method=np.array("circular_moving_average"),
                source=str(p),
                class_label=class_label,
                graph_id=graph_id,
                L=np.int32(X.shape[1]),
                weighted=np.bool_(weighted),
                neighbor_exclusion=np.int32(args.neighbor_exclusion),
            )

            # Save StS as before (no smoothing here)
            save_npz(
                out_sts / class_label / f"{graph_id}.npz",
                StS=StS_all,
                source=str(p),
                class_label=class_label,
                graph_id=graph_id,
                L=np.int32(X.shape[1]),
                weighted=np.bool_(weighted),
                neighbor_exclusion=np.int32(args.neighbor_exclusion),
            )

            ok += 1
            if ok % 50 == 0:
                print(f"[OK] {ok} processed... (last: {p.name})")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            bad += 1
            print(f"[ERR] {p.name}: {e}")

    print(f"Done. OK={ok}, ERR={bad}")
    print(f"StA root: {out_sta}")
    print(f"StS root: {out_sts}")
    print(f"StA smoothing: window={smooth_w} method=circular_moving_average (disabled if <=1)")


if __name__ == "__main__":
    main()