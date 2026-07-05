"""用 BC/IQL 模型在动态环境下评估。

将训练好的 BC 模型部署到 DynamicMultiAgentEnv 上，
每个时隙用策略推理出频率/功率选择，不随拓扑变化重新训练。
"""
import argparse
import json
import os
import time
import numpy as np
import torch
from typing import Dict, List

from config import ENV_CONFIG, ACTION_CONFIG, RESULTS_DIR, CHECKPOINT_DIR, LoggerSingleton
from dynamic_env import DynamicMultiAgentEnv
from iql_trainer import IQLTrainer
from bc_trainer import BCTrainer

logger = LoggerSingleton.get_instance()


def make_policy_fn(trainer, method: str):
    """创建动态评估的策略回调函数。"""
    def policy_fn(env: DynamicMultiAgentEnv) -> Dict[str, np.ndarray]:
        """每个时隙调用：用顺序决策为所有 UAV 选择动作。"""
        env.reset_commit_state()
        node_ids = list(env.base_env.nodes.keys())
        # 评估时固定顺序
        ordered_ids = node_ids  # 不打乱

        actions = {}
        for exec_idx, agent_id in enumerate(ordered_ids):
            obs = env.get_sequential_observation(agent_id, exec_idx)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=trainer.device).unsqueeze(0)

            with torch.no_grad():
                q_values = trainer.q_net(obs_t)
            action_idx = int(q_values.argmax(dim=-1).item())

            # 转换为归一化动作
            p_idx = action_idx // trainer.n_freq
            f_idx = action_idx % trainer.n_freq
            power_val = float(trainer.power_levels[p_idx])
            freq_val = float(trainer.freq_levels[f_idx])

            freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
            freq_span = ENV_CONFIG["freq_max"] - ENV_CONFIG["freq_min"] - ENV_CONFIG["bandwidth"]
            power_norm = (power_val / ENV_CONFIG["max_power"]) * 2 - 1
            freq_norm = ((freq_val - freq_lo) / freq_span) * 2 - 1

            normalized = np.array([power_norm, freq_norm], dtype=np.float32)
            env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
            actions[agent_id] = normalized

        return actions
    return policy_fn


def main():
    parser = argparse.ArgumentParser(description="MARL动态场景评估")
    parser.add_argument("--ckpt", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--method", type=str, default="bc", choices=["bc", "iql"])
    parser.add_argument("--num_uav", type=int, default=10)
    parser.add_argument("--num_slots", type=int, default=50)
    parser.add_argument("--uav_speed", type=float, default=20.0)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "dynamic_marl"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 加载模型
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]

    if args.method == "bc":
        trainer = BCTrainer(num_uav=args.num_uav, obs_dim=obs_dim, device=device)
    else:
        trainer = IQLTrainer(num_uav=args.num_uav, obs_dim=obs_dim, device=device)
    trainer.load(args.ckpt)

    policy_fn = make_policy_fn(trainer, args.method)

    # 运行多组速度
    speeds = [0, 10, 20, 30]
    all_results = {}

    for speed in speeds:
        logger.info(f"\n=== {args.method.upper()} speed={speed}m/s ===")
        episode_results = []

        for ep in range(args.num_episodes):
            env = DynamicMultiAgentEnv(
                num_uav=args.num_uav,
                num_slots=args.num_slots,
                uav_speed=speed,
                observation_radius=ENV_CONFIG["observation_radius"],
                area_size=ENV_CONFIG["area_size"],
                limit_neighbors=ENV_CONFIG["limit_neighbors"],
            )
            env.reset()

            slot_probs = []
            inference_times = []
            done = False

            while not done:
                t_start = time.time()
                actions = policy_fn(env)
                inference_times.append(time.time() - t_start)

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

            if (ep + 1) % 5 == 0:
                logger.info(f"  ep {ep+1}: cum_P_int={cum_prob:.4f}, infer_time={total_inference:.3f}s")

        cum_probs = [r["cumulative_interf_prob"] for r in episode_results]
        infer_times = [r["total_inference_time_s"] for r in episode_results]
        all_results[f"speed_{speed}"] = {
            "speed": speed,
            "cum_interf_prob_mean": float(np.mean(cum_probs)),
            "cum_interf_prob_std": float(np.std(cum_probs)),
            "avg_total_inference_time_s": float(np.mean(infer_times)),
            "episodes": episode_results,
        }
        logger.info(f"  {args.method.upper()} speed={speed}: P_int={np.mean(cum_probs):.4f}±{np.std(cum_probs):.4f}, "
                   f"infer={np.mean(infer_times):.3f}s")

    # 保存
    ckpt_name = os.path.basename(args.ckpt).replace(".pt", "")
    output_file = os.path.join(args.output, f"dynamic_{args.method}_{ckpt_name}.json")
    with open(output_file, "w") as f:
        json.dump({
            "config": {"num_uav": args.num_uav, "num_slots": args.num_slots, "num_episodes": args.num_episodes},
            "ckpt": args.ckpt,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果保存至 {output_file}")

    # 打印汇总对比
    logger.info(f"\n{'='*60}\n{args.method.upper()} 动态场景汇总\n{'='*60}")
    for speed in speeds:
        r = all_results[f"speed_{speed}"]
        logger.info(f"  speed={speed:2d}m/s: P_int={r['cum_interf_prob_mean']:.4f}±{r['cum_interf_prob_std']:.4f}, "
                   f"infer={r['avg_total_inference_time_s']:.3f}s")


if __name__ == "__main__":
    main()
