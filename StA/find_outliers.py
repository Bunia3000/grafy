from pathlib import Path
import numpy as np

DATASET = Path(r"C:\Users\danil\.vscode\grafy\dataset\StA")
VARIANT = "StA"
TOPK = 30

rows = []
for cls_dir in sorted([p for p in DATASET.iterdir() if p.is_dir()]):
    for p in cls_dir.glob("*.npz"):
        A = np.load(str(p), allow_pickle=False)[VARIANT].astype(np.float32)
        m = float(np.max(np.abs(A)))
        rows.append((m, cls_dir.name, str(p), float(A.min()), float(A.max())))

rows.sort(reverse=True, key=lambda x: x[0])

print(f"Top {TOPK} by max(abs(A)):")
for m, cls, path, mn, mx in rows[:TOPK]:
    print(f"{m:12.3f}  cls={cls:>4}  min={mn:12.3f}  max={mx:12.3f}  {path}")