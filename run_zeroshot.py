"""Zero-shot 跨规模泛化实验：用N=10训练的BC模型直接部署到N=20/30/50。

验证观测维度与N无关带来的可扩展性。
"""
import argparse
import json
import os
import time
import numpy as np
import torch
from typing import Dict, List

from config import ENV_CONFIG, ACTION_CONFIG, RESULTS_DIR, LAYOUTS_DIR, LoggerSingleton
from dynamic_env import DynamicMultiAgentEnv
from bc_trainer import BCTrainer

logger = LoggerSingleton.get_instance()


def make_policy_fn(trainer):
    # 并行推理口径：真实部署中所有 UAV 同时推理，一轮推理完成时间 = 最慢单个 UAV 的前向耗时。
    holder = {"slot_inference_s": 0.0}

    def policy_fn(env):
        env.reset_commit_state()
        node_ids = list(env.base_env.nodes.keys())
        actions = {}
        agent_inf_times = []
        for exec_idx, agent_id in enumerate(node_ids):
            obs = env.get_sequential_observation(agent_id, exec_idx)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=trainer.device).unsqueeze(0)
            t0 = time.time()
            with torch.no_grad():
                q_values = trainer.q_net(obs_t)
            agent_inf_times.append(time.time() - t0)
            action_idx = int(q_values.argmax(dim=-1).item())
            p_idx = action_idx // trainer.n_freq
            f_idx = action_idx % trainer.n_freq
            power_val = float(trainer.power_levels[p_idx])
            freq_val = float(trainer.freq_levels[f_idx])
            freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
            freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
            power_norm = (power_val / ENV_CONFIG["max_power"]) * 2 - 1
            freq_norm = ((freq_val - freq_lo) / (freq_hi - freq_lo)) * 2 - 1
            normalized = np.array([power_norm, freq_norm], dtype=np.float32)
            env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
            actions[agent_id] = normalized
        # 真实部署中所有 UAV 并行推理：本轮推理用时取最慢单个 agent 的前向耗时。
        holder["slot_inference_s"] = max(agent_inf_times) if agent_inf_times else 0.0
        return actions

    policy_fn.slot_inference_time = holder  # type: ignore[attr-defined]
    return policy_fn


def main():
    parser = argparse.ArgumentParser(description="Zero-shot跨规模泛化实验")
    parser.add_argument("--ckpt", type=str, default="checkpoints/bc_orthogonal_N10.pt",
                        help="N=10训练的BC模型")
    parser.add_argument("--deploy_sizes", type=int, nargs="+", default=[10, 20, 30, 50])
    parser.add_argument("--num_slots", type=int, default=50)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "zeroshot"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    trainer = BCTrainer(num_uav=10, obs_dim=obs_dim, device=device)  # num_uav只影响动作表，不影响推理
    trainer.load(args.ckpt)
    policy_fn = make_policy_fn(trainer)

    results = {}
    for n in args.deploy_sizes:
        logger.info(f"\n=== Zero-shot: N=10 model -> deploy N={n}, speed={args.speed}m/s ===")
        episode_probs = []
        episode_times = []

        for ep in range(args.num_episodes):
            env = DynamicMultiAgentEnv(
                num_uav=n,
                num_slots=args.num_slots,
                uav_speed=args.speed,
                observation_radius=ENV_CONFIG["observation_radius"],
                area_size=ENV_CONFIG["area_size"],
                limit_neighbors=ENV_CONFIG["limit_neighbors"],
            )
            env.reset()

            slot_probs = []
            infer_time = 0.0  # 并行口径：各时隙最慢单个 UAV 前向耗时之和
            done = False
            while not done:
                actions = policy_fn(env)
                infer_time += policy_fn.slot_inference_time["slot_inference_s"]
                obs, rewards, info, done = env.step(actions)
                slot_probs.append(info["interference_prob"])

            cum_prob = float(np.mean(slot_probs))
            episode_probs.append(cum_prob)
            episode_times.append(infer_time)

            if (ep + 1) % 5 == 0:
                logger.info(f"  N={n} ep {ep+1}: cum_P_int={cum_prob:.4f}, parallel_infer={infer_time:.4f}s")

        avg_prob = float(np.mean(episode_probs))
        std_prob = float(np.std(episode_probs))
        avg_time = float(np.mean(episode_times))
        results[f"N{n}"] = {
            "deploy_size": n,
            "cum_interf_prob": avg_prob,
            "std": std_prob,
            "avg_time_s": avg_time,
        }
        logger.info(f"  N={n}: P_int={avg_prob:.4f}±{std_prob:.4f}, time={avg_time:.2f}s")

    # 保存
    output_file = os.path.join(args.output, f"zeroshot_N10_deploy_speed{int(args.speed)}.json")
    with open(output_file, "w") as f:
        json.dump({
            "train_size": 10,
            "deploy_sizes": args.deploy_sizes,
            "speed": args.speed,
            "ckpt": args.ckpt,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果保存至 {output_file}")

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"Zero-shot 泛化汇总 (train N=10, speed={args.speed}m/s)")
    logger.info(f"{'='*60}")
    for key, r in results.items():
        logger.info(f"  Deploy {key}: P_int={r['cum_interf_prob']:.4f}±{r['std']:.4f}, time={r['avg_time_s']:.2f}s")


if __name__ == "__main__":
    main()
