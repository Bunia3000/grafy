from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


def read_list(p: Path) -> List[Path]:
    return [Path(x.strip()) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def apply_preprocess_sta_global(A: np.ndarray, params: dict) -> np.ndarray:
    """
    A: (3,L) float32
    params: preprocess.json from train_sta.py (train-fitted)
    """
    X = A.astype(np.float32, copy=False)
    mode = params["mode_base"]
    eps = float(params.get("eps", 1e-6))

    if mode in ("clip_zscore", "clip_log_zscore"):
        lo = float(params["clip_lo"])
        hi = float(params["clip_hi"])
        X = np.clip(X, lo, hi)

    if mode in ("log_zscore", "clip_log_zscore"):
        X = np.sign(X) * np.log1p(np.abs(X))

    if mode in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        m = float(params["mean"])
        s = float(params["std"])
        X = (X - m) / (s + eps)

    return X.astype(np.float32, copy=False)


def fit_preprocess_params_train(
    train_files: List[Path],
    variant_key: str,
    mode_base: str,
    clip_qlo: float,
    clip_qhi: float,
    eps: float,
) -> dict:
    vals = []
    for p in train_files:
        d = np.load(str(p), allow_pickle=False)
        A = d[variant_key].astype(np.float32)
        vals.append(A.reshape(-1))
    Xall = np.concatenate(vals, axis=0).astype(np.float32, copy=False)

    params = {
        "mode_base": mode_base,
        "clip_qlo": float(clip_qlo),
        "clip_qhi": float(clip_qhi),
        "clip_lo": None,
        "clip_hi": None,
        "mean": None,
        "std": None,
        "eps": float(eps),
    }

    if mode_base in ("clip_zscore", "clip_log_zscore"):
        lo = float(np.quantile(Xall, clip_qlo))
        hi = float(np.quantile(Xall, clip_qhi))
        params["clip_lo"] = lo
        params["clip_hi"] = hi
        Xall = np.clip(Xall, lo, hi)

    if mode_base in ("log_zscore", "clip_log_zscore"):
        Xall = np.sign(Xall) * np.log1p(np.abs(Xall))

    if mode_base in ("zscore", "clip_zscore", "log_zscore", "clip_log_zscore"):
        params["mean"] = float(Xall.mean())
        params["std"] = float(Xall.std())

    return params


def summarize_split(
    files: List[Path],
    variant_key: str,
    params: dict,
) -> Dict[str, float]:
    # streaming-ish summary (bez trzymania wszystkiego w RAM)
    n = 0
    sum_ = 0.0
    sum2 = 0.0
    max_abs = 0.0

    clip_lo = params.get("clip_lo", None)
    clip_hi = params.get("clip_hi", None)
    will_clip = params["mode_base"] in ("clip_zscore", "clip_log_zscore")

    clipped_lo = 0
    clipped_hi = 0
    total_vals = 0

    # do percentyli zbieramy próbkę (wystarczy)
    sample = []
    rng = np.random.default_rng(0)

    for p in files:
        d = np.load(str(p), allow_pickle=False)
        A = d[variant_key].astype(np.float32)
        X = apply_preprocess_sta_global(A, params).reshape(-1)

        # clipping stats: liczymy na wartościach PRZED log/zscore czy po?
        # Tu liczymy na PRZED log/zscore nie mamy już dostępu, więc liczymy na ETAPIE clipowania przez detekcję równości
        # (w praktyce OK: po clipowaniu wartości dokładnie równe granicom).
        if will_clip and clip_lo is not None and clip_hi is not None:
            clipped_lo += int(np.sum(X == X))  # placeholder, skorygujemy poniżej

        # stats
        x = X.astype(np.float64, copy=False)
        n += x.size
        sum_ += float(x.sum())
        sum2 += float(np.square(x).sum())
        max_abs = max(max_abs, float(np.max(np.abs(x))))

        # clip-hit count (po preprocessingu: jeśli mode zawiera clip, to wartości równe granicom po CLIP,
        # ale po zscore/log nie będą równe. Dlatego liczmy clip-hit PRZED log/zscore wprost niżej w innej funkcji.
        # Tu to pomijamy.
        total_vals += x.size

        # sample do percentyli
        if x.size > 0:
            take = min(2000, x.size)
            idx = rng.choice(x.size, size=take, replace=False)
            sample.append(x[idx])

    mean = sum_ / max(n, 1)
    var = (sum2 / max(n, 1)) - mean * mean
    std = float(np.sqrt(max(var, 0.0)))

    if sample:
        s = np.concatenate(sample, axis=0)
        pcts = np.quantile(s, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]).tolist()
    else:
        pcts = [float("nan")] * 7

    return {
        "n_values": float(n),
        "mean": float(mean),
        "std": float(std),
        "p0.1%": float(pcts[0]),
        "p1%": float(pcts[1]),
        "p5%": float(pcts[2]),
        "p50%": float(pcts[3]),
        "p95%": float(pcts[4]),
        "p99%": float(pcts[5]),
        "p99.9%": float(pcts[6]),
        "max_abs": float(max_abs),
    }


def count_clip_hits_raw(
    files: List[Path],
    variant_key: str,
    params: dict,
) -> Tuple[int, int, int]:
    """
    Liczy ile elementów zostało uciętych do clip_lo/clip_hi na etapie clippingu (na surowych wartościach, przed log/zscore).
    """
    mode = params["mode_base"]
    if mode not in ("clip_zscore", "clip_log_zscore"):
        return (0, 0, 0)

    lo = float(params["clip_lo"])
    hi = float(params["clip_hi"])

    n_lo = 0
    n_hi = 0
    total = 0

    for p in files:
        d = np.load(str(p), allow_pickle=False)
        A = d[variant_key].astype(np.float32).reshape(-1)
        n_lo += int(np.sum(A < lo))
        n_hi += int(np.sum(A > hi))
        total += int(A.size)

    return n_lo, n_hi, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Folder z train_files.txt/val_files.txt/test_files.txt")
    ap.add_argument("--variant", default="StA")
    ap.add_argument("--preprocess_json", default=None, help="Jeśli None, weź <run_dir>/preprocess.json")
    ap.add_argument("--mode_base", default="clip_zscore", choices=["zscore", "clip_zscore", "log_zscore", "clip_log_zscore"])
    ap.add_argument("--clip_lo", type=float, default=0.001, help="Quantile qlo (tylko gdy liczymy preprocess na train)")
    ap.add_argument("--clip_hi", type=float, default=0.999, help="Quantile qhi (tylko gdy liczymy preprocess na train)")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    tr = read_list(run_dir / "train_files.txt")
    va = read_list(run_dir / "val_files.txt")
    te = read_list(run_dir / "test_files.txt")

    pp_path = Path(args.preprocess_json) if args.preprocess_json else (run_dir / "preprocess.json")

    if pp_path.exists():
        params = json.loads(pp_path.read_text(encoding="utf-8"))
    else:
        # policz na train i zapisz
        params = fit_preprocess_params_train(
            train_files=tr,
            variant_key=args.variant,
            mode_base=args.mode_base,
            clip_qlo=float(args.clip_lo),
            clip_qhi=float(args.clip_hi),
            eps=float(args.eps),
        )
        pp_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== preprocess params ===")
    print(json.dumps(params, ensure_ascii=False, indent=2))

    for name, files in [("TRAIN", tr), ("VAL", va), ("TEST", te)]:
        s = summarize_split(files, args.variant, params)
        lo_hits, hi_hits, total = count_clip_hits_raw(files, args.variant, params)
        frac_lo = lo_hits / total if total else 0.0
        frac_hi = hi_hits / total if total else 0.0
        print(f"\n=== {name} (after preprocess) ===")
        for k, v in s.items():
            print(f"{k:>10}: {v}")
        if params["mode_base"] in ("clip_zscore", "clip_log_zscore"):
            print(f"clip_hits_raw: lo={lo_hits} hi={hi_hits} total={total}  frac_lo={frac_lo:.6f} frac_hi={frac_hi:.6f}")


if __name__ == "__main__":
    main()