from pathlib import Path
import argparse
import numpy as np


def get_npz_map(root: Path, cls: str):
    class_dir = root / cls
    if not class_dir.exists():
        raise FileNotFoundError(f"Brak katalogu: {class_dir}")

    files = sorted(class_dir.glob("*.npz"))
    mapping = {}
    for p in files:
        mapping[p.stem] = p
    return mapping


def infer_first_key(npz_path: Path):
    with np.load(npz_path, allow_pickle=False) as d:
        keys = list(d.keys())
    if not keys:
        raise ValueError(f"Brak tablic w pliku: {npz_path}")
    return keys[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sta_root", required=True, help=r"np. C:\Users\danil\.vscode\grafy\dataset\StA")
    ap.add_argument("--sts_root", required=True, help=r"np. C:\Users\danil\.vscode\grafy\dataset\StS")
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--sta_key", default=None, help="np. StA; jeśli nie podasz, weźmie pierwszy key z pliku")
    ap.add_argument("--sts_key", default=None, help="np. StS; jeśli nie podasz, weźmie pierwszy key z pliku")
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    sta_root = Path(args.sta_root)
    sts_root = Path(args.sts_root)

    total_pairs = 0
    total_missing_in_sts = 0
    total_missing_in_sta = 0

    for cls in args.classes:
        print(f"\n=== CLASS {cls} ===")

        sta_map = get_npz_map(sta_root, cls)
        sts_map = get_npz_map(sts_root, cls)

        sta_ids = set(sta_map.keys())
        sts_ids = set(sts_map.keys())

        common_ids = sorted(sta_ids & sts_ids)
        only_sta = sorted(sta_ids - sts_ids)
        only_sts = sorted(sts_ids - sta_ids)

        print(f"StA files: {len(sta_ids)}")
        print(f"StS files: {len(sts_ids)}")
        print(f"Common:    {len(common_ids)}")
        print(f"Only StA:  {len(only_sta)}")
        print(f"Only StS:  {len(only_sts)}")

        total_pairs += len(common_ids)
        total_missing_in_sts += len(only_sta)
        total_missing_in_sta += len(only_sts)

        if only_sta:
            print("\nPrzykłady braków w StS:")
            for x in only_sta[:args.top_n]:
                print(" ", x)

        if only_sts:
            print("\nPrzykłady braków w StA:")
            for x in only_sts[:args.top_n]:
                print(" ", x)

        if not common_ids:
            print("\nBrak wspólnych plików dla tej klasy.")
            continue

        sample_id = common_ids[0]
        sta_sample = sta_map[sample_id]
        sts_sample = sts_map[sample_id]

        sta_key = args.sta_key or infer_first_key(sta_sample)
        sts_key = args.sts_key or infer_first_key(sts_sample)

        with np.load(sta_sample, allow_pickle=False) as d_sta:
            sta = d_sta[sta_key]
        with np.load(sts_sample, allow_pickle=False) as d_sts:
            sts = d_sts[sts_key]

        print(f"\nSample pair id: {sample_id}")
        print(f"StA file: {sta_sample.name}")
        print(f"StS file: {sts_sample.name}")
        print(f"StA key:  {sta_key}, shape={sta.shape}, dtype={sta.dtype}")
        print(f"StS key:  {sts_key}, shape={sts.shape}, dtype={sts.dtype}")

        if sta.ndim != 2 or sta.shape[0] != 3:
            print("UWAGA: StA nie ma oczekiwanego kształtu (3, L)")
        if sts.ndim != 3 or sts.shape[0] != 3 or sts.shape[1] != sts.shape[2]:
            print("UWAGA: StS nie ma oczekiwanego kształtu (3, L, L)")

        if sta.ndim == 2 and sts.ndim == 3:
            l_sta = sta.shape[1]
            l_sts = sts.shape[1]
            print(f"L from StA: {l_sta}")
            print(f"L from StS: {l_sts}")
            if l_sta != l_sts:
                print("UWAGA: L w StA i StS się różni")

    print("\n=== GLOBAL SUMMARY ===")
    print(f"Total common pairs:      {total_pairs}")
    print(f"Total missing in StS:    {total_missing_in_sts}")
    print(f"Total missing in StA:    {total_missing_in_sta}")


if __name__ == "__main__":
    main()