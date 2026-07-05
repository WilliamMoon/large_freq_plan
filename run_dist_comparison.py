"""分布式基线动态对比实验。"""
import argparse
import json
import os
import time
import numpy as np
from typing import Dict, List

from config import ENV_CONFIG, ACTION_CONFIG, RESULTS_DIR, LoggerSingleton
from dynamic_env import DynamicMultiAgentEnv
from dynamic_baselines import run_random_dynamic, run_fixed_orthogonal, run_periodic_greedy, run_per_slot_greedy
from distributed_baselines import run_distributed_greedy, run_jar, run_distributed_coloring

logger = LoggerSingleton.get_instance()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_uav", type=int, default=10)
    parser.add_argument("--num_slots", type=int, default=50)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0, 10, 20, 30])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "dist_comparison"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)

    methods = ["random", "fixed", "dist_coloring", "jar", "dist_greedy", "greedy_periodic_5", "greedy_per_slot"]

    for speed in args.speeds:
        logger.info(f"\n{'='*60}\nSpeed = {speed} m/s\n{'='*60}")
        summary = []

        for method in methods:
            episode_probs = []
            episode_times = []

            for ep in range(args.num_episodes):
                env = DynamicMultiAgentEnv(
                    num_uav=args.num_uav,
                    num_slots=args.num_slots,
                    uav_speed=speed,
                    observation_radius=ENV_CONFIG["observation_radius"],
                    area_size=ENV_CONFIG["area_size"],
                    limit_neighbors=ENV_CONFIG["limit_neighbors"],
                )
                probs, elapsed = run_single(env, method, seed=args.seed + ep)
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
            logger.info(f"  {method:25s}: P_int={avg_prob:.4f}±{std_prob:.4f}, time={avg_time:.1f}s")

        # 保存
        out_file = os.path.join(args.output, f"dist_N{args.num_uav}_speed{int(speed)}.json")
        with open(out_file, "w") as f:
            json.dump({"speed": speed, "summary": summary}, f, indent=2, ensure_ascii=False)

    logger.info(f"\n结果保存至 {args.output}")


if __name__ == "__main__":
    main()
