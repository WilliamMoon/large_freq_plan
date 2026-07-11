"""动态场景 BC 训练：在移动的 DynamicMultiAgentEnv 中收集 greedy-sequential 专家轨迹。

与静态 BC（bc_trainer.collect_expert_data，只在冻结拓扑上打一次标签）的唯一区别：
专家数据来自 uav_speed>0 的多时隙 episode，每个时隙都用 greedy-sequential 重新决策并记录
(obs, expert_action)，因此 BC 能见到"移动导致的拓扑漂移"这一分布，缓解静态训练在高速下的 OOD 退化。

用法（服务器，仅用 GPU4）：
    CUDA_VISIBLE_DEVICES=4 python train_bc_dynamic.py \
        --num_uav 10 --train_speed 10 --episodes 120 --num_slots 50 --bc_epochs 50 --gpu 4
评估（多速度）：
    python eval_dynamic_marl.py --method bc --ckpt checkpoints/bc_dynamic_s10_N10.pt \
        --num_uav 10 --speeds 0 10 20 30 40
"""
import argparse
import json
import os
import time
import numpy as np
import torch
from typing import List, Tuple

from config import ENV_CONFIG, CHECKPOINT_DIR, RESULTS_DIR, LAYOUTS_DIR, LoggerSingleton
from marl_env import MultiAgentEnv
from dynamic_env import DynamicMultiAgentEnv
from bc_trainer import BCTrainer
from test_cgreedy import _choose_best_freq_power

logger = LoggerSingleton.get_instance()


def collect_expert_data_dynamic(
    trainer: BCTrainer,
    num_episodes: int,
    train_speeds: List[float],
    num_slots: int,
    area_size: float,
) -> List[Tuple[np.ndarray, int]]:
    """在移动环境中逐时隙收集 greedy-sequential 专家数据。

    多速度域随机化：每个 episode 从 train_speeds 中随机抽一个 UAV 速度重新构造环境，
    使 BC 的(obs, expert_action)数据集覆盖 0~40 m/s 的完整动力学分布，缓解只在单速度
    采集导致的分布外(OOD)退化（典型表现为 N 较大、速度较高时 BC 落后于集中式周期贪心）。

    每个 episode 内：环境按该 episode 的速度移动 num_slots 个时隙；每个时隙对所有 UAV 做
    顺序贪心（每个 agent 决策后立即 update_sinr，后续 agent 可感知），记录 (obs, action)。
    """
    freq_candidates = trainer.freq_levels
    power_candidates = trainer.power_levels

    expert_data: List[Tuple[np.ndarray, int]] = []
    rng = np.random.default_rng()
    t0 = time.time()

    for ep in range(num_episodes):
        uav_speed = float(train_speeds[rng.integers(0, len(train_speeds))])
        env = DynamicMultiAgentEnv(
            num_uav=trainer.num_uav,
            num_slots=num_slots,
            uav_speed=uav_speed,
            observation_radius=ENV_CONFIG["observation_radius"],
            area_size=area_size,
            limit_neighbors=ENV_CONFIG["limit_neighbors"],
        )
        env.reset()
        done = False
        while not done:
            # 每时隙顺序贪心决策
            env.reset_commit_state()
            node_list = list(env.base_env.nodes.values())
            rng.shuffle(node_list)  # 训练时随机顺序，增强泛化

            actions = {}
            for exec_idx, node in enumerate(node_list):
                obs = env.get_sequential_observation(node.node_id, exec_idx)
                _, best_freq, best_power = _choose_best_freq_power(
                    env, node, freq_candidates, power_candidates,
                    use_neighbors=True, neighbor_sample=None,
                )
                action_idx = trainer._values_to_action_idx(best_power, best_freq)
                expert_data.append((obs, action_idx))

                # commit 让后续 agent 能感知（与静态收集一致）
                tx = node.tx
                rx = tx.peer
                tx.frequency = best_freq
                tx.power = best_power
                rx.frequency = best_freq
                env.base_env.update_sinr()

                actions[node.node_id] = trainer._normalized_action(best_power, best_freq)

            # 推进环境到下一时隙（移动 + 重配对）
            _, _, _, done = env.step(actions)

        if (ep + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info(f"  collect ep {ep+1}/{num_episodes}: {len(expert_data)} samples, "
                        f"{elapsed:.1f}s (speed={uav_speed:.0f})")

    speeds_str = ",".join(str(int(s)) for s in train_speeds)
    logger.info(f"动态专家数据收集完成：{len(expert_data)} 条 "
                f"（{num_episodes} eps × {num_slots} slots × {trainer.num_uav} agents, "
                f"speeds=[{speeds_str}]）")
    return expert_data


def main():
    parser = argparse.ArgumentParser(description="动态场景 BC 训练")
    parser.add_argument("--num_uav", type=int, default=10)
    parser.add_argument("--train_speed", type=float, default=10.0, help="单速度模式下的 UAV 速度 (m/s)，被 --train_speeds 覆盖")
    parser.add_argument("--train_speeds", type=float, nargs="+", default=None,
                        help="多速度域随机化：每个 episode 随机抽一个速度收集专家数据；提供则覆盖 --train_speed")
    parser.add_argument("--episodes", type=int, default=120, help="收集专家数据的 episode 数")
    parser.add_argument("--num_slots", type=int, default=50, help="每 episode 时隙数")
    parser.add_argument("--bc_epochs", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=4, help="GPU编号(1-5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--area_size", type=float, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()

    assert args.gpu >= 1, "禁止使用 GPU 0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    area_size = args.area_size or ENV_CONFIG["area_size"]

    # 多速度域随机化：--train_speeds 优先；否则退化为单速度 --train_speed
    if args.train_speeds:
        train_speeds = sorted(set(args.train_speeds))
        is_multispeed = True
    else:
        train_speeds = [args.train_speed]
        is_multispeed = False
    logger.info(f"[Dynamic BC] 专家数据速度集合 = {train_speeds} (multi-speed={is_multispeed})")

    # obs_dim
    probe = MultiAgentEnv(
        num_uav=args.num_uav,
        observation_radius=ENV_CONFIG["observation_radius"],
        area_size=area_size,
        limit_neighbors=ENV_CONFIG["limit_neighbors"],
    )
    probe.reset()
    obs_dim = probe.obs_dim
    logger.info(f"[Dynamic BC] N={args.num_uav}, train_speed={args.train_speed}, "
                f"obs_dim={obs_dim}, device={device}")

    # 评估布局（静态 50-layouts，与静态 BC 相同标准，保证公平对比）
    layout_path = os.path.join(LAYOUTS_DIR, f"N{args.num_uav}_50layouts.json")
    eval_layouts = []
    if os.path.exists(layout_path):
        with open(layout_path, "r") as f:
            eval_layouts = list(json.load(f).values())
        logger.info(f"加载 {len(eval_layouts)} 个评估布局")
    eval_env = MultiAgentEnv(
        num_uav=args.num_uav,
        observation_radius=ENV_CONFIG["observation_radius"],
        area_size=area_size,
        limit_neighbors=ENV_CONFIG["limit_neighbors"],
    )

    save_name = args.save_name or (
        f"bc_dynamic_ms_N{args.num_uav}" if is_multispeed
        else f"bc_dynamic_s{int(args.train_speed)}_N{args.num_uav}"
    )

    trainer = BCTrainer(num_uav=args.num_uav, obs_dim=obs_dim, device=device)

    start = time.time()
    expert_data = collect_expert_data_dynamic(
        trainer, num_episodes=args.episodes, train_speeds=train_speeds,
        num_slots=args.num_slots, area_size=area_size,
    )
    logger.info(f"收集耗时 {time.time()-start:.1f}s")

    history = trainer.train(
        expert_data, num_epochs=args.bc_epochs, batch_size=256,
        eval_env=eval_env, eval_layouts=eval_layouts, eval_interval=5,
    )
    logger.info(f"训练完成，总耗时 {time.time()-start:.1f}s")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{save_name}.pt")
    trainer.save(ckpt_path)
    with open(os.path.join(RESULTS_DIR, f"{save_name}_stats.json"), "w") as f:
        json.dump(history, f, indent=2)

    if eval_layouts:
        probs, sinrs = trainer.evaluate_on_layouts(eval_env, eval_layouts)
        logger.info(f"[Dynamic BC] 静态 50-layout 最终评估 N={args.num_uav}: "
                    f"P_int={np.mean(probs):.4f}±{np.std(probs):.4f}")


if __name__ == "__main__":
    main()
