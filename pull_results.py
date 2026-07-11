"""从服务器 nami 拉取夜间实验结果到本地 results_data/。

只拉取 Fig.1-4 需要的文件；训练未完成的（如 CARLTON N=20/30/50）会 MISS，属正常。
"""
import subprocess
import os

SERVER = "nami"
SRV_DIR = "/mnt/data1/liushiyang/large_freq_plan/results"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_data")
os.makedirs(LOCAL_DIR, exist_ok=True)

files = []
for n in [10, 20, 30, 50]:
    files.append(f"bc_orthogonal_N{n}_stats.json")
    files.append(f"dynamic_bc_bc_orthogonal_N{n}.json")
    files.append(f"dynamic_carlton_N{n}.json")
for n in [10, 20, 30, 50]:
    for s in [0, 10, 20, 30, 40]:
        files.append(f"dist_N{n}_speed{s}.json")

ok, miss = 0, 0
for f in files:
    src = f"{SERVER}:{SRV_DIR}/{f}"
    r = subprocess.run(["scp", src, LOCAL_DIR + "/"], capture_output=True, text=True)
    if r.returncode == 0:
        ok += 1
        print("OK  ", f)
    else:
        miss += 1
        print("MISS", f)
print(f"\npulled={ok} missing={miss}")
