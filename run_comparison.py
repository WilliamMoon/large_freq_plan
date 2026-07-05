"""统一对比实验：在同一批固定布局上对比 Random / Greedy / MADDPG。

用法:
  # 1. 先生成布局
  python gen_eval_layouts.py --counts 10 25 50

  # 2. 跑对比实验（会自动跑random、greedy，MADDPG需要先训练）
  python run_comparison.py --counts 10 --num_layouts 50

  # 3. 如果MADDPG已训练，指定checkpoint
  python run_comparison.py --counts 10 --maddpg_ckpt checkpoints/maddpg_N10.pt
"""
import argparse
import json
import os
import time
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from config import ENV_CONFIG, ACTION_CONFIG, RESULTS_DIR, LAYOUTS_DIR, CHECKPOINT_DIR, LoggerSingleton
from marl_env import MultiAgentEnv
from maddpg_trainer import MADDPGTrainer
from models import Actor

logger = LoggerSingleton.get_instance()

# 贪心相关从 test_cgreedy 导入
from test_cgreedy import (
    _enumerate_best_frequency_central,
    _enumerate_best_frequency_sequential,
    _enumerate_best_frequency_independent,
    _summarize_episode,
)


def eval_random(env: MultiAgentEnv, layouts: dict, seed: int = 42) -> List[dict]:
    rng = np.random.default_rng(seed)
    results = []
    for name, layout in tqdm(layouts.items(), desc="Random", ncols=120):
        env.load_layout(layout)
        for node in env.base_env.nodes.values():
            tx = node.tx
            tx.power = float(rng.uniform(tx.min_power, tx.max_power))
            freq_lo = env.freq_min + env.bandwidth / 2
            freq_hi = env.freq_max - env.bandwidth / 2
            tx.frequency = float(rng.uniform(freq_lo, freq_hi))
            tx.peer.frequency = tx.frequency
        env.base_env.update_sinr()
        interf_prob = env.base_env.calc_interf_prob()
        sinrs = [rx.sinr for rx in env.base_env.receivers
                 if rx.sinr != float('-inf') and rx.sinr != float('inf')]
        results.append({
            "layout": name,
            "interference_prob": float(interf_prob),
            "avg_sinr": float(np.mean(sinrs)) if sinrs else 0.0,
        })
    return results


def eval_greedy(env: MultiAgentEnv, layouts: dict, mode: str = "central") -> List[dict]:
    # 对齐离散化粒度
    n_freq = ACTION_CONFIG["n_freq"]
    n_power = ACTION_CONFIG["n_power"]
    freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
    freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
    freq_candidates = np.linspace(freq_lo, freq_hi, n_freq)
    power_candidates = np.linspace(0, ENV_CONFIG["max_power"], n_power)
    
    results = []
    for name, layout in tqdm(layouts.items(), desc=f"Greedy-{mode}", ncols=120):
        env.load_layout(layout)
        if mode == "central":
            details = _enumerate_best_frequency_central(env, freq_candidates, power_candidates)
        elif mode == "sequential":
            details = _enumerate_best_frequency_sequential(
                env, freq_candidates, power_candidates,
                shuffle_order=True, neighbor_sample=None,
                freq_sample=None, power_sample=None,
            )
        elif mode == "independent":
            details = _enumerate_best_frequency_independent(env, freq_candidates, power_candidates)
        else:
            raise ValueError(f"Unknown greedy mode: {mode}")
        
        summary = _summarize_episode(env, details, 0, include_details=False)
        results.append({
            "layout": name,
            "interference_prob": summary["interference_prob"],
            "avg_sinr": summary["avg_sinr"],
        })
    return results


def eval_maddpg(env: MultiAgentEnv, layouts: dict, ckpt_path: str, device: str = "cuda:0") -> List[dict]:
    import torch
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    n_power = ckpt["n_power"]
    n_freq = ckpt["n_freq"]
    
    trainer = MADDPGTrainer(
        num_uav=env.num_uav,
        obs_dim=obs_dim,
        n_power=n_power,
        n_freq=n_freq,
        device=device,
    )
    trainer.load(ckpt_path)
    
    results = []
    for name, layout in tqdm(layouts.items(), desc="MADDPG", ncols=120):
        env.load_layout(layout)
        trainer.select_actions(env, temperature=trainer.tau_final, evaluate=True, sequential=True)
        interf_prob = env.get_interference_prob()
        sinrs = [rx.sinr for rx in env.base_env.receivers
                 if rx.sinr != float('-inf') and rx.sinr != float('inf')]
        results.append({
            "layout": name,
            "interference_prob": float(interf_prob),
            "avg_sinr": float(np.mean(sinrs)) if sinrs else 0.0,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="统一对比实验")
    parser.add_argument("--counts", type=int, nargs="+", default=[10])
    parser.add_argument("--num_layouts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maddpg_ckpts", type=str, nargs="+", default=None,
                        help="MADDPG checkpoint路径列表，顺序对应counts。如不提供则跳过MADDPG")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "comparison"))
    parser.add_argument("--skip_random", action="store_true")
    parser.add_argument("--skip_greedy", action="store_true")
    parser.add_argument("--skip_maddpg", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)

    all_summary = []

    for idx, count in enumerate(args.counts):
        layout_path = os.path.join(LAYOUTS_DIR, f"N{count}_{args.num_layouts}layouts.json")
        if not os.path.exists(layout_path):
            logger.error(f"布局文件不存在: {layout_path}，请先运行 gen_eval_layouts.py")
            continue
        with open(layout_path, "r") as f:
            layouts = json.load(f)
        logger.info(f"\n{'='*60}\nN={count}, {len(layouts)} 个评估布局\n{'='*60}")

        env = MultiAgentEnv(
            num_uav=count,
            observation_radius=ENV_CONFIG["observation_radius"],
            area_size=ENV_CONFIG["area_size"],
            limit_neighbors=ENV_CONFIG["limit_neighbors"],
        )

        # Random
        if not args.skip_random:
            results = eval_random(env, layouts, seed=args.seed)
            probs = [r["interference_prob"] for r in results]
            entry = {
                "uav_count": count, "method": "random",
                "avg_interference_prob": float(np.mean(probs)),
                "std_interference_prob": float(np.std(probs)),
                "avg_sinr": float(np.mean([r["avg_sinr"] for r in results])),
            }
            all_summary.append(entry)
            logger.info(f"  Random:        P_int={entry['avg_interference_prob']:.4f}±{entry['std_interference_prob']:.4f}")

        # Greedy
        if not args.skip_greedy:
            for mode in ["independent", "sequential", "central"]:
                results = eval_greedy(env, layouts, mode=mode)
                probs = [r["interference_prob"] for r in results]
                entry = {
                    "uav_count": count, "method": f"greedy-{mode}",
                    "avg_interference_prob": float(np.mean(probs)),
                    "std_interference_prob": float(np.std(probs)),
                    "avg_sinr": float(np.mean([r["avg_sinr"] for r in results])),
                }
                all_summary.append(entry)
                logger.info(f"  Greedy-{mode:10s}: P_int={entry['avg_interference_prob']:.4f}±{entry['std_interference_prob']:.4f}")

        # MADDPG
        if not args.skip_maddpg and args.maddpg_ckpts and idx < len(args.maddpg_ckpts):
            ckpt_path = args.maddpg_ckpts[idx]
            if os.path.exists(ckpt_path):
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                results = eval_maddpg(env, layouts, ckpt_path, device)
                probs = [r["interference_prob"] for r in results]
                entry = {
                    "uav_count": count, "method": "maddpg",
                    "avg_interference_prob": float(np.mean(probs)),
                    "std_interference_prob": float(np.std(probs)),
                    "avg_sinr": float(np.mean([r["avg_sinr"] for r in results])),
                }
                all_summary.append(entry)
                logger.info(f"  MADDPG:        P_int={entry['avg_interference_prob']:.4f}±{entry['std_interference_prob']:.4f}")
            else:
                logger.warning(f"  MADDPG checkpoint 不存在: {ckpt_path}")

    # 保存汇总
    summary_path = os.path.join(args.output, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n汇总保存至 {summary_path}")

    # 打印对比表
    logger.info(f"\n{'='*70}\n{'方法':<20s} {'N=10':>12s} {'N=25':>12s} {'N=50':>12s}\n{'='*70}")
    methods = {}
    for e in all_summary:
        m = e["method"]
        if m not in methods:
            methods[m] = {}
        methods[m][e["uav_count"]] = e
    for m in ["random", "greedy-independent", "greedy-sequential", "greedy-central", "maddpg"]:
        if m not in methods:
            continue
        row = f"{m:<20s}"
        for c in args.counts:
            if c in methods[m]:
                e = methods[m][c]
                row += f" {e['avg_interference_prob']:.4f}±{e['std_interference_prob']:.3f}"
            else:
                row += f" {'N/A':>12s}"
        logger.info(row)


if __name__ == "__main__":
    main()
