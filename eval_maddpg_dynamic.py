"""用 MADDPG 模型在动态环境下评估（独立脚本，避免改动 eval_dynamic_marl.py）。

协议与 eval_dynamic_marl.py 保持一致：每个时隙顺序决策，策略用 actor 输出的
(power, freq) logits 取 argmax，再归一化到 [-1,1] 喂给 DynamicMultiAgentEnv。
输出 JSON 格式与 eval_dynamic_marl.py 完全对齐，便于 Fig1/Fig2/Table 统一读取。
"""
import argparse
import json
import os
import time
import numpy as np
import torch
from typing import Dict

from config import ENV_CONFIG, RESULTS_DIR, CHECKPOINT_DIR, LoggerSingleton
from dynamic_env import DynamicMultiAgentEnv
from maddpg_trainer import MADDPGTrainer

logger = LoggerSingleton.get_instance()


def make_policy_fn(trainer):
    """MADDPG 的动态评估策略：actor argmax（评估模式），结果归一化。"""
    holder = {"slot_inference_s": 0.0}

    def policy_fn(env: DynamicMultiAgentEnv) -> Dict[str, np.ndarray]:
        env.reset_commit_state()
        node_ids = list(env.base_env.nodes.keys())
        ordered_ids = node_ids  # 评估时固定顺序

        actions = {}
        agent_inf_times = []
        for exec_idx, agent_id in enumerate(ordered_ids):
            obs = env.get_sequential_observation(agent_id, exec_idx)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=trainer.device).unsqueeze(0)

            t0 = time.time()
            with torch.no_grad():
                power_logits, freq_logits = trainer.actor(obs_t)
                power_idx = int(power_logits.argmax(dim=-1).item())
                freq_idx = int(freq_logits.argmax(dim=-1).item())
            agent_inf_times.append(time.time() - t0)

            power_val = float(trainer.power_levels[power_idx])
            freq_val = float(trainer.freq_levels[freq_idx])

            freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
            freq_span = ENV_CONFIG["freq_max"] - ENV_CONFIG["freq_min"] - ENV_CONFIG["bandwidth"]
            power_norm = (power_val / ENV_CONFIG["max_power"]) * 2 - 1
            freq_norm = ((freq_val - freq_lo) / freq_span) * 2 - 1

            normalized = np.array([power_norm, freq_norm], dtype=np.float32)
            env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
            actions[agent_id] = normalized

        holder["slot_inference_s"] = max(agent_inf_times) if agent_inf_times else 0.0
        return actions

    policy_fn.slot_inference_time = holder  # type: ignore[attr-defined]
    return policy_fn


def main():
    parser = argparse.ArgumentParser(description="MADDPG 动态场景评估")
    parser.add_argument("--ckpt", type=str, required=True, help="MADDPG checkpoint 路径")
    parser.add_argument("--num_uav", type=int, default=10)
    parser.add_argument("--num_slots", type=int, default=50)
    parser.add_argument("--uav_speed", type=float, default=20.0)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--area_size", type=float, default=None)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "dynamic_marl"))
    parser.add_argument("--speeds", type=float, nargs="+", default=[0, 10, 20, 30, 40])
    parser.add_argument("--gpu", type=int, default=4, help="GPU 编号 (1-5)")
    args = parser.parse_args()

    assert args.gpu >= 1, "禁止使用 GPU 0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    area_size = args.area_size or ENV_CONFIG["area_size"]
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]

    trainer = MADDPGTrainer(num_uav=args.num_uav, obs_dim=obs_dim, device=device)
    trainer.load(args.ckpt)

    policy_fn = make_policy_fn(trainer)

    speeds = args.speeds
    all_results = {}

    for speed in speeds:
        logger.info(f"\n=== MADDPG speed={speed}m/s ===")
        episode_results = []
        for ep in range(args.num_episodes):
            env = DynamicMultiAgentEnv(
                num_uav=args.num_uav,
                num_slots=args.num_slots,
                uav_speed=speed,
                observation_radius=ENV_CONFIG["observation_radius"],
                area_size=area_size,
                limit_neighbors=ENV_CONFIG["limit_neighbors"],
            )
            env.reset()

            slot_probs = []
            inference_times = []
            done = False
            while not done:
                actions = policy_fn(env)
                inference_times.append(policy_fn.slot_inference_time["slot_inference_s"])
                obs, rewards, info, done = env.step(actions)
                slot_probs.append(info["interference_prob"])

            cum_prob = float(np.mean(slot_probs))
            total_inference = float(np.sum(inference_times))
            episode_results.append({
                "episode": ep,
                "cumulative_interf_prob": cum_prob,
                "slot_probs": slot_probs,
                "total_inference_time_s": total_inference,
                "avg_inference_time_per_slot_s": total_inference / len(slot_probs),
            })

        cum_probs = [r["cumulative_interf_prob"] for r in episode_results]
        infer_times = [r["total_inference_time_s"] for r in episode_results]
        all_results[f"speed_{speed}"] = {
            "speed": speed,
            "cum_interf_prob_mean": float(np.mean(cum_probs)),
            "cum_interf_prob_std": float(np.std(cum_probs)),
            "avg_total_inference_time_s": float(np.mean(infer_times)),
            "episodes": episode_results,
        }
        logger.info(f"  MADDPG speed={speed}: P_int={np.mean(cum_probs):.4f}±{np.std(cum_probs):.4f}")

    ckpt_name = os.path.basename(args.ckpt).replace(".pt", "")
    output_file = os.path.join(args.output, f"dynamic_maddpg_{ckpt_name}.json")
    with open(output_file, "w") as f:
        json.dump({
            "config": {"num_uav": args.num_uav, "num_slots": args.num_slots,
                       "num_episodes": args.num_episodes},
            "ckpt": args.ckpt,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果保存至 {output_file}")

    logger.info(f"\n{'='*60}\nMADDPG 动态场景汇总\n{'='*60}")
    for speed in speeds:
        r = all_results[f"speed_{speed}"]
        logger.info(f"  speed={speed:.0f}m/s: P_int={r['cum_interf_prob_mean']:.4f}"
                    f"±{r['cum_interf_prob_std']:.4f}, infer={r['avg_total_inference_time_s']:.3f}s")


if __name__ == "__main__":
    main()
