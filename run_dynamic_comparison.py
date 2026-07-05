"""动态场景对比实验：验证各方法在动态 UAV 集群下的表现。

用法:
  python run_dynamic_comparison.py --num_uav 10 --num_slots 50 --uav_speed 20
"""
import argparse
import json
import os
import time
import numpy as np
from typing import Dict, List

from config import ENV_CONFIG, ACTION_CONFIG, RESULTS_DIR, LoggerSingleton
from dynamic_env import DynamicMultiAgentEnv
from dynamic_baselines import (
    run_fixed_orthogonal,
    run_random_dynamic,
    run_periodic_greedy,
    run_per_slot_greedy,
)

logger = LoggerSingleton.get_instance()


def run_single_episode(env: DynamicMultiAgentEnv, method: str, **kwargs) -> dict:
    """运行一个动态 episode，返回结果。"""
    start_time = time.time()

    if method == "random":
        slot_probs = run_random_dynamic(env, seed=kwargs.get("seed", 42))
        planning_time = 0.0
    elif method == "fixed":
        slot_probs = run_fixed_orthogonal(env)
        planning_time = 0.0
    elif method == "greedy_periodic":
        interval = kwargs.get("replan_interval", 5)
        slot_probs, planning_time = run_periodic_greedy(env, replan_interval=interval)
    elif method == "greedy_per_slot":
        slot_probs, planning_time = run_per_slot_greedy(env)
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - start_time
    return {
        "method": method,
        "slot_probs": slot_probs,
        "cumulative_interf_prob": float(np.mean(slot_probs)),
        "total_time_s": elapsed,
        "planning_time_s": planning_time,
        "num_slots": len(slot_probs),
    }


def main():
    parser = argparse.ArgumentParser(description="动态场景对比实验")
    parser.add_argument("--num_uav", type=int, default=10)
    parser.add_argument("--num_slots", type=int, default=50)
    parser.add_argument("--uav_speed", type=float, default=20.0, help="UAV移动速度 m/s, 0=静态")
    parser.add_argument("--num_episodes", type=int, default=10, help="评估episode数")
    parser.add_argument("--replan_intervals", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "dynamic_comparison"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)

    methods = ["random", "fixed"] + [f"greedy_periodic_{i}" for i in args.replan_intervals] + ["greedy_per_slot"]

    all_results = {}
    for method in methods:
        all_results[method] = []
        logger.info(f"\n{'='*60}\n运行 {method} (N={args.num_uav}, speed={args.uav_speed}m/s, {args.num_episodes} episodes)\n{'='*60}")

        for ep in range(args.num_episodes):
            env = DynamicMultiAgentEnv(
                num_uav=args.num_uav,
                num_slots=args.num_slots,
                uav_speed=args.uav_speed,
                observation_radius=ENV_CONFIG["observation_radius"],
                area_size=ENV_CONFIG["area_size"],
                limit_neighbors=ENV_CONFIG["limit_neighbors"],
            )

            if "greedy_periodic" in method:
                interval = int(method.split("_")[-1])
                result = run_single_episode(env, "greedy_periodic", replan_interval=interval, seed=args.seed + ep)
            else:
                result = run_single_episode(env, method, seed=args.seed + ep)

            result["episode"] = ep
            all_results[method].append(result)

            if (ep + 1) % 5 == 0:
                logger.info(f"  {method} ep {ep+1}/{args.num_episodes}: "
                           f"cum_P_int={result['cumulative_interf_prob']:.4f}, "
                           f"time={result['total_time_s']:.1f}s")

    # 汇总
    logger.info(f"\n{'='*70}\n动态场景对比汇总 (N={args.num_uav}, speed={args.uav_speed}m/s, slots={args.num_slots})\n{'='*70}")
    summary = []
    for method, results in all_results.items():
        cum_probs = [r["cumulative_interf_prob"] for r in results]
        times = [r["total_time_s"] for r in results]
        plan_times = [r["planning_time_s"] for r in results]
        entry = {
            "method": method,
            "cum_interf_prob_mean": float(np.mean(cum_probs)),
            "cum_interf_prob_std": float(np.std(cum_probs)),
            "avg_total_time_s": float(np.mean(times)),
            "avg_planning_time_s": float(np.mean(plan_times)),
            "num_episodes": len(results),
        }
        summary.append(entry)
        logger.info(f"  {method:25s}: P_int={entry['cum_interf_prob_mean']:.4f}±{entry['cum_interf_prob_std']:.4f}, "
                   f"time={entry['avg_total_time_s']:.1f}s, plan={entry['avg_planning_time_s']:.1f}s")

    # 保存
    output_file = os.path.join(args.output, f"dynamic_N{args.num_uav}_speed{int(args.uav_speed)}.json")
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "num_uav": args.num_uav,
                "num_slots": args.num_slots,
                "uav_speed": args.uav_speed,
                "num_episodes": args.num_episodes,
            },
            "summary": summary,
            "detailed": {m: [{"cumulative_interf_prob": r["cumulative_interf_prob"],
                             "slot_probs": r["slot_probs"],
                             "total_time_s": r["total_time_s"],
                             "planning_time_s": r["planning_time_s"]}
                            for r in results]
                         for m, results in all_results.items()},
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果保存至 {output_file}")


if __name__ == "__main__":
    main()
