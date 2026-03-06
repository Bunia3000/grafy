import os, subprocess

script = r"C:\Users\danil\.vscode\grafy\StA\feature_baseline_sta.py"
dataset = r"C:\Users\danil\.vscode\grafy\dataset\StA"
pp = r"C:\Users\danil\.vscode\grafy\experiments\sta_3_1_vs_4_1\baseline_lr_clipz_train_q001_999_maxabs2000\preprocess.json"
out = r"C:\Users\danil\.vscode\grafy\experiments\sta_3_1_vs_4_1\FEATURE_BASELINE_v1"
os.makedirs(out, exist_ok=True)

for seed in range(10):
    log = os.path.join(out, f"seed_{seed}.txt")
    cmd = [
        "python", script,
        "--dataset", dataset,
        "--classes", "3_1", "4_1",
        "--variant", "StA",
        "--seed", str(seed),
        "--preprocess_json", pp,
    ]
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"DONE seed={seed} -> {log}")