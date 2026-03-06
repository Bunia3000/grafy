import numpy as np
from pathlib import Path

def read_list(p):
    return [Path(line.strip()) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]

def stats_for(paths, variant="StA"):
    vals = []
    maxs = []
    mins = []
    for p in paths:
        A = np.load(str(p), allow_pickle=False)[variant].astype(np.float32)
        vals.append(A.reshape(-1))
        maxs.append(float(A.max()))
        mins.append(float(A.min()))
    x = np.concatenate(vals)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "min": float(np.min(mins)),
        "max": float(np.max(maxs)),
    }

run_dir = Path(r"C:\Users\danil\.vscode\grafy\experiments\sta_3_1_vs_4_1\baseline_lr")
tr = read_list(run_dir/"train_files.txt")
va = read_list(run_dir/"val_files.txt")
te = read_list(run_dir/"test_files.txt")

print("TRAIN:", stats_for(tr))
print("VAL  :", stats_for(va))
print("TEST :", stats_for(te))