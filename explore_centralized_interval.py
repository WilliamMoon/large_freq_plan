"""扫描集中式贪心(periodic greedy)的重规划间隔 k，寻找满足：
- 低速(v=0) P_int < BC(v=0)
- 高速(v=40) P_int > BC(v=40)
的最长间隔 k（间隔越长越能体现'集中式计算慢、无法每时隙重规划'）。
在 N=10 和 N=20 上扫描，速度 0/10/20/30/40。
结果写入 centralized_interval_sweep.json。
"""
import json
import os
import numpy as np
import multiprocessing as mp

from config import ENV_CONFIG, ACTION_CONFIG
from dynamic_env import DynamicMultiAgentEnv
from dynamic_baselines import run_periodic_greedy


def trial(args):
    n, speed, k, num_episodes = args
    probs = []
    times = []
    for ep in range(num_episodes):
        env = DynamicMultiAgentEnv(
            num_uav=n,
            num_slots=50,
            uav_speed=speed,
            observation_radius=ENV_CONFIG["observation_radius"],
            area_size=ENV_CONFIG["area_size"],
            limit_neighbors=ENV_CONFIG["limit_neighbors"],
        )
        sp, t = run_periodic_greedy(env, replan_interval=k, greedy_mode="sequential")
        probs.append(float(np.mean(sp)))
        times.append(t)
    return (n, speed, k, float(np.mean(probs)), float(np.std(probs)), float(np.mean(times)))


def main():
    grid_k = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
    speeds = [0, 10, 20, 30, 40]
    num_episodes = 10
    tasks = []
    for n in [10, 20]:
        for s in speeds:
            for k in grid_k:
                tasks.append((n, s, k, num_episodes))

    print(f"共 {len(tasks)} 个 (N,speed,k) 任务", flush=True)
    with mp.Pool(min(30, len(tasks))) as p:
        res = p.map(trial, tasks)

    out = {}
    for n, s, k, mp_, sp_, t in res:
        out.setdefault(str(n), {}).setdefault(str(s), {})[str(k)] = {
            "p_int": mp_, "std": sp_, "time_s": t,
        }
        print(f"N={n} speed={s} k={k}: P_int={mp_:.4f} time={t:.3f}s", flush=True)

    with open("centralized_interval_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("保存 centralized_interval_sweep.json", flush=True)


if __name__ == "__main__":
    main()
