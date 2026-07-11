#!/home/lsy/.conda/envs/uav/bin/python
# 后台等待 IQL + MADDPG 共 8 个动态评估 json 全部生成后，自动汇总成与
# BC / CARLTON 同口径的 (N, speed) P_int 对照表，写到 results_data/baselines_summary.md。
# 只读 json，不修改任何源码；超时(6h)也会用已有数据产出部分结果。
import os, json, time

BASE = "/mnt/data1/liushiyang/large_freq_plan"
DYNAMIC_MARL = os.path.join(BASE, "results", "dynamic_marl")
RESULTS_DATA = os.path.join(BASE, "results_data")
SPEEDS = [0, 10, 20, 30, 40]
NS = [10, 20, 30, 50]
LOG = os.path.join(BASE, "wait_summarize.log")
OUT = os.path.join(RESULTS_DATA, "baselines_summary.md")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_pi(data):
    """返回 {speed(float): (mean, std)}"""
    out = {}
    if not data:
        return out
    for k, v in data.get("results", {}).items():
        sp = v.get("speed")
        if sp is None:
            try:
                sp = float(str(k).split("_")[-1])
            except Exception:
                continue
        m, s = v.get("cum_interf_prob_mean"), v.get("cum_interf_prob_std")
        if m is not None:
            out[float(sp)] = (m, s)
    return out


def find_bc(n):
    p1 = os.path.join(RESULTS_DATA, f"dynamic_bc_bc_dynamic_s10_N{n}.json")
    p2 = os.path.join(DYNAMIC_MARL, f"dynamic_bc_bc_orthogonal_N{n}.json")
    if os.path.exists(p1):
        return load_json(p1), "BC-dynamic(s10)"
    if os.path.exists(p2):
        return load_json(p2), "BC-orthogonal(static)"
    return None, "N/A"


def avg(d):
    vals = [d[s][0] for s in SPEEDS if s in d]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    log("WAIT_SUMMARIZE started; target = IQL+MADDPG x N{10,20,30,50}")
    targets = {}
    for n in NS:
        targets[("IQL", n)] = os.path.join(DYNAMIC_MARL, f"dynamic_iql_iql_N{n}.json")
        targets[("MADDPG", n)] = os.path.join(DYNAMIC_MARL, f"dynamic_maddpg_maddpg_N{n}.json")

    deadline = time.time() + 6 * 3600
    while True:
        missing = [f"{m} N{n}" for (m, n), p in targets.items() if not os.path.exists(p)]
        if not missing:
            log("All 8 target jsons present -> summarizing.")
            break
        if time.time() > deadline:
            log("Deadline reached -> summarizing with available data.")
            break
        log(f"Waiting... missing {len(missing)}: {missing}; sleep 60s")
        time.sleep(60)

    # 载入
    data = {}
    for (m, n), p in targets.items():
        data[(m, n)] = get_pi(load_json(p)) if os.path.exists(p) else {}
    carlton = {n: get_pi(load_json(os.path.join(RESULTS_DATA, f"dynamic_carlton_N{n}.json"))) for n in NS}
    bc, bc_src = {}, {}
    for n in NS:
        d, src = find_bc(n)
        bc[n] = get_pi(d)
        bc_src[n] = src

    def row(name, n, d, src=""):
        cells = []
        for s in SPEEDS:
            if s in d:
                m, sd = d[s]
                cells.append(f"{m:.3f}±{sd:.3f}")
            else:
                cells.append("—")
        return f"| {name} | {n} | " + " | ".join(cells) + f" | {src} |"

    L = []
    L.append("# Baselines Dynamic-Eval P_int Summary\n")
    L.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} | speeds(m/s): {SPEEDS}_\n")
    L.append("\nFormat: mean±std P_int at v=0/10/20/30/40.\n")
    L.append("| Method | N | " + " | ".join(f"v{s}" for s in SPEEDS) + " | BC source |")
    L.append("|" + "---|" * (2 + len(SPEEDS)) + "---|")
    for n in NS:
        L.append(row("BC", n, bc[n], bc_src[n]))
    for n in NS:
        L.append(row("CARLTON", n, carlton[n]))
    for n in NS:
        L.append(row("IQL", n, data[("IQL", n)]))
    for n in NS:
        L.append(row("MADDPG", n, data[("MADDPG", n)]))

    L.append("\n## Avg P_int (mean over 5 speeds) and ratio vs BC\n")
    L.append("| Method | N | AvgP_int | xBC |")
    L.append("|---|---|---|---|")
    for n in NS:
        bca = avg(bc[n])
        for name, key in [("CARLTON", None), ("IQL", ("IQL", n)), ("MADDPG", ("MADDPG", n))]:
            d = carlton[n] if name == "CARLTON" else data[key]
            a = avg(d)
            ratio = (a / bca) if (bca == bca and bca > 0) else float("nan")
            L.append(f"| {name} | {n} | {a:.3f} | {ratio:.1f}x |")
        L.append(f"| BC | {n} | {bca:.3f} | 1.0x |")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"SUMMARY WRITTEN -> {OUT}")


if __name__ == "__main__":
    main()
