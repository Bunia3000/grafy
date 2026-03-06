from __future__ import annotations
import argparse
import re
from pathlib import Path
import csv
import numpy as np

ACC_RE = re.compile(r"Accuracy:\s*([0-9]*\.[0-9]+)")
BAL_RE = re.compile(r"Balanced accuracy:\s*([0-9]*\.[0-9]+)")
MCC_RE = re.compile(r"MCC:\s*([-0-9]*\.[0-9]+)")
AUC_RE = re.compile(r"ROC-AUC:\s*([0-9]*\.[0-9]+)")

def parse_one(txt: str) -> dict | None:
    def g(rx):
        m = rx.search(txt)
        return float(m.group(1)) if m else None

    acc = g(ACC_RE)
    bal = g(BAL_RE)
    mcc = g(MCC_RE)
    auc = g(AUC_RE)
    if acc is None or bal is None or mcc is None:
        return None
    return {"acc": acc, "bal_acc": bal, "mcc": mcc, "auc": auc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    rows = []
    for p in sorted(log_dir.glob("seed_*.txt")):
        seed = int(p.stem.split("_")[1])
        data = parse_one(p.read_text(encoding="utf-8", errors="ignore"))
        if data is None:
            rows.append({"seed": seed, "status": "BAD", "acc": "", "bal_acc": "", "mcc": "", "auc": "", "log": str(p)})
        else:
            rows.append({"seed": seed, "status": "OK", **data, "log": str(p)})

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed","status","acc","bal_acc","mcc","auc","log"])
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "OK"]
    def arr(k): return np.array([r[k] for r in ok], dtype=float)

    summary_lines = []
    summary_lines.append(f"Total logs: {len(rows)}  OK: {len(ok)}  BAD: {len(rows)-len(ok)}")
    if ok:
        for k in ["acc","bal_acc","mcc","auc"]:
            a = arr(k)
            summary_lines.append(f"{k}: mean={a.mean():.4f}  std={a.std(ddof=1) if len(a)>1 else 0.0:.4f}  "
                                 f"min={a.min():.4f}  max={a.max():.4f}")
        # best by acc then mcc
        best = sorted(ok, key=lambda r: (r["acc"], r["mcc"]), reverse=True)[0]
        summary_lines.append(f"BEST: seed={best['seed']} acc={best['acc']:.4f} mcc={best['mcc']:.4f} auc={best['auc']:.4f}")
        summary_lines.append(f"      log={best['log']}")
    else:
        summary_lines.append("No OK runs parsed.")

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))

if __name__ == "__main__":
    main()