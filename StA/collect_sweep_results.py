# collect_sweep_results.py
# Usage:
#   python collect_sweep_results.py --out_base "...\SWEEP_v1"
#
# Produces in out_base:
#   - sweep_results.csv
#   - sweep_summary.txt
#
# Robust to different key names in run_summary.json (test_* vs eval_*).
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def _get(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _pick_first(summary: Dict[str, Any], candidates: List[List[str]]) -> Optional[float]:
    """
    candidates: list of key paths, e.g. [["results","test_accuracy"], ["results","eval_accuracy"], ...]
    returns first available float
    """
    for key_path in candidates:
        v = _get(summary, key_path)
        f = _safe_float(v)
        if f is not None:
            return f
    return None


def _pick_first_any(summary: Dict[str, Any], candidates: List[List[str]]) -> Optional[Any]:
    for key_path in candidates:
        v = _get(summary, key_path)
        if v is not None:
            return v
    return None


def _infer_config_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    # Best-effort: read what train_sta.py typically writes
    cfg = {}
    cfg["timestamp"] = _get(summary, ["timestamp"])
    cfg["classes"] = _pick_first_any(summary, [["classes"], ["input", "classes"]])
    cfg["variant"] = _pick_first_any(summary, [["input", "variant_key"], ["input", "variant"], ["variant"]])
    cfg["model"] = _pick_first_any(summary, [["model", "name"], ["model"]])
    cfg["preprocess"] = _pick_first_any(summary, [["preprocess", "mode"], ["preprocess"], ["input", "preprocess"]])
    cfg["aug"] = _pick_first_any(summary, [["augmentations", "aug"], ["aug"]])
    cfg["reverse_p"] = _pick_first_any(summary, [["augmentations", "reverse_p"], ["reverse_p"]])
    cfg["seed"] = _pick_first_any(summary, [["train_config", "seed"], ["seed"]])
    cfg["epochs"] = _pick_first_any(summary, [["train_config", "epochs"], ["epochs"]])
    cfg["bs"] = _pick_first_any(summary, [["train_config", "batch_size"], ["bs"], ["batch_size"]])
    cfg["lr"] = _pick_first_any(summary, [["train_config", "lr"], ["lr"]])
    cfg["expected_L"] = _pick_first_any(summary, [["input", "L"], ["expected_L"]])
    cfg["n_per_class_effective"] = _pick_first_any(summary, [["n_per_class_effective"], ["n_per_class"]])
    return cfg


def _confusion_matrix(summary: Dict[str, Any]) -> Optional[List[List[int]]]:
    cm = _get(summary, ["results", "confusion_matrix"])
    if isinstance(cm, list) and cm and isinstance(cm[0], list):
        return cm
    # sometimes nested differently
    cm = _get(summary, ["confusion_matrix"])
    if isinstance(cm, list) and cm and isinstance(cm[0], list):
        return cm
    return None


def _balanced_accuracy_from_cm(cm: List[List[int]]) -> Optional[float]:
    # For binary/multiclass: mean recall over classes
    try:
        k = len(cm)
        recalls = []
        for i in range(k):
            row_sum = sum(cm[i])
            if row_sum <= 0:
                return None
            recalls.append(cm[i][i] / row_sum)
        return float(sum(recalls) / len(recalls))
    except Exception:
        return None


def _mcc_binary_from_cm(cm: List[List[int]]) -> Optional[float]:
    # Binary only: cm = [[tn, fp],[fn,tp]] but ours is [[c0->c0, c0->c1],[c1->c0,c1->c1]]
    try:
        if len(cm) != 2 or len(cm[0]) != 2:
            return None
        tn = float(cm[0][0])
        fp = float(cm[0][1])
        fn = float(cm[1][0])
        tp = float(cm[1][1])
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom == 0:
            return None
        return float((tp * tn - fp * fn) / denom)
    except Exception:
        return None


@dataclass
class RunRow:
    run_dir: str
    status: str  # OK / MISSING / BADJSON
    timestamp: Optional[str]
    classes: Optional[str]
    variant: Optional[str]
    model: Optional[str]
    preprocess: Optional[str]
    aug: Optional[str]
    reverse_p: Optional[float]
    seed: Optional[int]
    expected_L: Optional[int]
    n_per_class_effective: Optional[int]
    epochs: Optional[int]
    bs: Optional[int]
    lr: Optional[float]
    test_acc: Optional[float]
    test_loss: Optional[float]
    bal_acc: Optional[float]
    mcc: Optional[float]


def collect_runs(out_base: Path) -> List[RunRow]:
    rows: List[RunRow] = []
    for run_dir in sorted([p for p in out_base.iterdir() if p.is_dir()]):
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            rows.append(RunRow(
                run_dir=str(run_dir),
                status="MISSING",
                timestamp=None, classes=None, variant=None, model=None, preprocess=None,
                aug=None, reverse_p=None, seed=None, expected_L=None, n_per_class_effective=None,
                epochs=None, bs=None, lr=None, test_acc=None, test_loss=None, bal_acc=None, mcc=None
            ))
            continue

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            rows.append(RunRow(
                run_dir=str(run_dir),
                status="BADJSON",
                timestamp=None, classes=None, variant=None, model=None, preprocess=None,
                aug=None, reverse_p=None, seed=None, expected_L=None, n_per_class_effective=None,
                epochs=None, bs=None, lr=None, test_acc=None, test_loss=None, bal_acc=None, mcc=None
            ))
            continue

        cfg = _infer_config_from_summary(summary)

        test_acc = _pick_first(summary, [
            ["results", "test_accuracy"],
            ["results", "eval_accuracy"],
            ["results", "eval_acc"],
            ["results", "accuracy"],
            ["test_accuracy"],
            ["eval_accuracy"],
        ])

        test_loss = _pick_first(summary, [
            ["results", "test_loss"],
            ["results", "eval_loss"],
            ["results", "loss"],
            ["test_loss"],
            ["eval_loss"],
        ])

        cm = _confusion_matrix(summary)
        bal_acc = _balanced_accuracy_from_cm(cm) if cm else None
        mcc = _mcc_binary_from_cm(cm) if cm else None

        # normalize types
        def _as_int(x):
            try:
                return int(x) if x is not None else None
            except Exception:
                return None

        rows.append(RunRow(
            run_dir=str(run_dir),
            status="OK",
            timestamp=cfg.get("timestamp"),
            classes="_".join(cfg["classes"]) if isinstance(cfg.get("classes"), list) else (cfg.get("classes") if isinstance(cfg.get("classes"), str) else None),
            variant=cfg.get("variant"),
            model=cfg.get("model"),
            preprocess=cfg.get("preprocess"),
            aug=cfg.get("aug"),
            reverse_p=_safe_float(cfg.get("reverse_p")),
            seed=_as_int(cfg.get("seed")),
            expected_L=_as_int(cfg.get("expected_L")),
            n_per_class_effective=_as_int(cfg.get("n_per_class_effective")),
            epochs=_as_int(cfg.get("epochs")),
            bs=_as_int(cfg.get("bs")),
            lr=_safe_float(cfg.get("lr")),
            test_acc=test_acc,
            test_loss=test_loss,
            bal_acc=bal_acc,
            mcc=mcc,
        ))

    return rows


def write_results_csv(out_path: Path, rows: List[RunRow]) -> None:
    fieldnames = [f.name for f in RunRow.__dataclass_fields__.values()]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})


def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def write_summary_txt(out_path: Path, rows: List[RunRow]) -> None:
    ok = [r for r in rows if r.status == "OK"]
    missing = [r for r in rows if r.status != "OK"]

    # group by (model, preprocess, aug)
    groups = defaultdict(list)
    for r in ok:
        key = (r.model or "?", r.preprocess or "?", r.aug or "?")
        groups[key].append(r)

    lines: List[str] = []
    lines.append(f"Total run folders: {len(rows)}")
    lines.append(f"OK summaries:       {len(ok)}")
    lines.append(f"Missing/bad:        {len(missing)}")
    if missing:
        lines.append("")
        lines.append("Missing/BADJSON run dirs:")
        for r in missing[:50]:
            lines.append(f"  - {r.status}: {r.run_dir}")
        if len(missing) > 50:
            lines.append(f"  ... ({len(missing)-50} more)")

    lines.append("")
    lines.append("Per-config aggregates (mean±std over seeds):")
    lines.append("key = (model, preprocess, aug)")
    lines.append("")

    # compute best runs overall (by test_acc then by mcc)
    def _sort_key(r: RunRow):
        # prefer test_acc, then mcc, then -loss
        return (
            -1.0 * (r.test_acc if r.test_acc is not None else -1e9),
            -1.0 * (r.mcc if r.mcc is not None else -1e9),
            +1.0 * (r.test_loss if r.test_loss is not None else 1e9),
        )

    best_overall = sorted(ok, key=_sort_key)[:10]

    for key, items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        accs = [r.test_acc for r in items if r.test_acc is not None]
        losses = [r.test_loss for r in items if r.test_loss is not None]
        bals = [r.bal_acc for r in items if r.bal_acc is not None]
        mccs = [r.mcc for r in items if r.mcc is not None]

        acc_m, acc_s = _mean_std([a for a in accs if a is not None])
        loss_m, loss_s = _mean_std([l for l in losses if l is not None])
        bal_m, bal_s = _mean_std([b for b in bals if b is not None])
        mcc_m, mcc_s = _mean_std([m for m in mccs if m is not None])

        best_in_group = sorted(items, key=_sort_key)[0] if items else None

        lines.append(f"{key}: n={len(items)}")
        lines.append(f"  test_acc: {acc_m:.4f} ± {acc_s:.4f}" if acc_m is not None else "  test_acc: (missing)")
        lines.append(f"  bal_acc : {bal_m:.4f} ± {bal_s:.4f}" if bal_m is not None else "  bal_acc : (missing)")
        lines.append(f"  mcc     : {mcc_m:.4f} ± {mcc_s:.4f}" if mcc_m is not None else "  mcc     : (missing)")
        lines.append(f"  test_loss: {loss_m:.4f} ± {loss_s:.4f}" if loss_m is not None else "  test_loss: (missing)")
        if best_in_group:
            lines.append(f"  best_run: acc={best_in_group.test_acc} mcc={best_in_group.mcc} loss={best_in_group.test_loss}")
            lines.append(f"           dir={best_in_group.run_dir}")
        lines.append("")

    lines.append("Top 10 runs overall (sorted by test_acc, then mcc, then loss):")
    for r in best_overall:
        lines.append(f"- acc={r.test_acc} bal_acc={r.bal_acc} mcc={r.mcc} loss={r.test_loss}  dir={r.run_dir}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True, help="Folder with run subfolders containing run_summary.json")
    args = ap.parse_args()

    out_base = Path(args.out_base)
    if not out_base.exists():
        raise FileNotFoundError(out_base)

    rows = collect_runs(out_base)

    results_csv = out_base / "sweep_results.csv"
    summary_txt = out_base / "sweep_summary.txt"

    write_results_csv(results_csv, rows)
    write_summary_txt(summary_txt, rows)

    print(f"Saved: {results_csv}")
    print(f"Saved: {summary_txt}")


if __name__ == "__main__":
    main()