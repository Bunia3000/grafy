from pathlib import Path
import hashlib

def read_list(p):
    return [line.strip() for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]

def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

run_dir = Path(r"C:\Users\danil\.vscode\grafy\experiments\sta_3_1_vs_4_1\baseline_lr")  # <- zmień

tr = set(read_list(run_dir / "train_files.txt"))
va = set(read_list(run_dir / "val_files.txt"))
te = set(read_list(run_dir / "test_files.txt"))

print("Path overlap train∩val:", len(tr & va))
print("Path overlap train∩test:", len(tr & te))
print("Path overlap val∩test:", len(va & te))

# hash overlap (duplikaty treści pod różnymi nazwami)
def hashes(paths):
    d = {}
    for p in paths:
        d[p] = file_md5(p)
    return d

trh = hashes(tr)
vah = hashes(va)
teh = hashes(te)

tr_hashes = set(trh.values())
va_hashes = set(vah.values())
te_hashes = set(teh.values())

print("Content overlap train∩val:", len(tr_hashes & va_hashes))
print("Content overlap train∩test:", len(tr_hashes & set(teh.values())))
print("Content overlap val∩test:", len(va_hashes & te_hashes))