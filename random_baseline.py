"""随机基线：每个UAV均匀随机选择功率和频率。"""
import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from config import ENV_CONFIG, EVAL_CONFIG, LAYOUTS_DIR, RESULTS_DIR, LoggerSingleton
from marl_env import MultiAgentEnv

logger = LoggerSingleton.get_instance()


def run_random_baseline(env: MultiAgentEnv, layouts: Dict, seed: int = 42) -> List[Dict]:
    rng = np.random.default_rng(seed)
    results = []
    for name, layout in tqdm(layouts.items(), desc="Random eval", ncols=120):
        env.load_layout(layout)
        # 随机选择功率和频率
        for node in env.base_env.nodes.values():
            tx = node.tx
            tx.power = float(rng.uniform(tx.min_power, tx.max_power))
            freq_lo = env.freq_min + env.bandwidth / 2
            freq_hi = env.freq_max - env.bandwidth / 2
            tx.frequency = float(rng.uniform(freq_lo, freq_hi))
            tx.peer.frequency = tx.frequency
        env.base_env.update_sinr()
        interf_prob = env.base_env.calc_interf_prob()

        sinrs = []
        for rx in env.base_env.receivers:
            if rx.sinr != float('-inf') and rx.sinr != float('inf'):
                sinrs.append(rx.sinr)
        results.append({
            "layout": name,
            "interference_prob": float(interf_prob),
            "avg_sinr": float(np.mean(sinrs)) if sinrs else 0.0,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="随机基线评估")
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--num_layouts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "random_baseline"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)

    summary = []
    for count in args.counts:
        layout_path = os.path.join(LAYOUTS_DIR, f"N{count}_{args.num_layouts}layouts.json")
        if not os.path.exists(layout_path):
            logger.error(f"布局文件不存在: {layout_path}，请先运行 gen_eval_layouts.py")
            continue
        with open(layout_path, "r") as f:
            layouts = json.load(f)

        env = MultiAgentEnv(
            num_uav=count,
            observation_radius=ENV_CONFIG["observation_radius"],
            area_size=ENV_CONFIG["area_size"],
            limit_neighbors=ENV_CONFIG["limit_neighbors"],
        )
        results = run_random_baseline(env, layouts, seed=args.seed)

        probs = [r["interference_prob"] for r in results]
        avg_prob = float(np.mean(probs))
        std_prob = float(np.std(probs))
        avg_sinr = float(np.mean([r["avg_sinr"] for r in results]))
        logger.info(f"Random N={count}: P_int={avg_prob:.4f}±{std_prob:.4f}, avg_SINR={avg_sinr:.2f}dB")
        summary.append({
            "uav_count": count,
            "method": "random",
            "avg_interference_prob": avg_prob,
            "std_interference_prob": std_prob,
            "avg_sinr": avg_sinr,
        })

        # 保存明细
        with open(os.path.join(args.output, f"random_N{count}.json"), "w") as f:
            json.dump(results, f, indent=2)

    with open(os.path.join(args.output, "random_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"随机基线汇总保存至 {args.output}")


if __name__ == "__main__":
    main()
