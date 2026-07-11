"""分布式基线动态对比实验。支持多方法 / 多N / 多速度并行。"""
import argparse
import json
import os
import time
import multiprocessing as mp
import numpy as np
from typing import Dict, List

from config import ENV_CONFIG, ACTION_CONFIG, RESULTS_DIR, LoggerSingleton
from dynamic_env import DynamicMultiAgentEnv
from dynamic_baselines import run_random_dynamic, run_fixed_orthogonal, run_periodic_greedy, run_per_slot_greedy
from distributed_baselines import run_distributed_greedy, run_jar, run_distributed_coloring

logger = LoggerSingleton.get_instance()

ALL_METHODS = ["random", "fixed", "dist_coloring", "jar", "dist_greedy", "greedy_periodic_5", "greedy_per_slot"]


def run_single(env, method, **kwargs):
    start = time.time()
    if method == "random":
        probs = run_random_dynamic(env, seed=kwargs.get("seed", 42))
    elif method == "fixed":
        probs = run_fixed_orthogonal(env)
    elif method == "dist_greedy":
        probs = run_distributed_greedy(env)
    elif method == "jar":
        probs = run_jar(env)
    elif method == "dist_coloring":
        probs = run_distributed_coloring(env)
    elif method == "greedy_periodic_5":
        probs, _ = run_periodic_greedy(env, replan_interval=5)
    elif method == "greedy_per_slot":
        probs, _ = run_per_slot_greedy(env)
    else:
        raise ValueError(method)
    elapsed = time.time() - start
    return probs, elapsed


def run_speed(n, speed, methods, num_episodes, num_slots, area_size, seed, output):
    """对单个 (N, speed) 跑所有 method，写出 dist_N{n}_speed{int(speed)}.json。供多进程调用。"""
    summary = []
    for method in methods:
        episode_probs = []
        episode_times = []
        for ep in range(num_episodes):
            env = DynamicMultiAgentEnv(
                num_uav=n,
                num_slots=num_slots,
                uav_speed=speed,
                observation_radius=ENV_CONFIG["observation_radius"],
                area_size=area_size,
                limit_neighbors=ENV_CONFIG["limit_neighbors"],
            )
            probs, elapsed = run_single(env, method, seed=seed + ep)
            episode_probs.append(float(np.mean(probs)))
            episode_times.append(elapsed)
        avg_prob = float(np.mean(episode_probs))
        std_prob = float(np.std(episode_probs))
        avg_time = float(np.mean(episode_times))
        summary.append({
            "method": method,
            "cum_interf_prob": avg_prob,
            "std": std_prob,
            "avg_time_s": avg_time,
        })
        logger.info(f"[N={n} speed={speed}] {method:20s}: P_int={avg_prob:.4f}±{std_prob:.4f}, time={avg_time:.1f}s")
    out_file = os.path.join(output, f"dist_N{n}_speed{int(speed)}.json")
    with open(out_file, "w") as f:
        json.dump({"speed": speed, "summary": summary}, f, indent=2, ensure_ascii=False)
    logger.info(f"保存 {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_uav", type=int, nargs="+", default=[10],
                        help="要跑的网络规模（可多个，如 10 20 30 50）")
    parser.add_argument("--num_slots", type=int, default=50)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0, 10, 20, 30, 40])
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--area_size", type=float, default=None)
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="要跑的方法，默认全部（ALL_METHODS）")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="并行 worker 数（按 (N,speed) 任务划分），默认用满可用核")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "dist_comparison"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)

    area_size = args.area_size or ENV_CONFIG["area_size"]
    methods = args.methods if args.methods else ALL_METHODS

    tasks = [(n, speed) for n in args.num_uav for speed in args.speeds]
    logger.info(f"共 {len(tasks)} 个 (N,speed) 任务，方法={methods}")

    if args.num_workers and args.num_workers > 1:
        workers = min(args.num_workers, len(tasks))
        logger.info(f"并行模式：{workers} workers")
        with mp.Pool(workers) as pool:
            pool.starmap(
                run_speed,
                [(n, speed, methods, args.num_episodes, args.num_slots, area_size, args.seed, args.output)
                 for (n, speed) in tasks],
            )
    else:
        for (n, speed) in tasks:
            run_speed(n, speed, methods, args.num_episodes, args.num_slots, area_size, args.seed, args.output)

    logger.info(f"\n结果保存至 {args.output}")


if __name__ == "__main__":
    main()
