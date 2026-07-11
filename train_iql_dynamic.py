"""动态场景 IQL 训练：在移动的 DynamicMultiAgentEnv 中、只用奖励（无专家）训练 Q 网络。

作为 BC（同结构 QNetwork、同动态多速度环境、但用专家动作标签）的对照，凸显专家数据的作用。
本脚本刻意与 train_bc_dynamic.py 的 collect_expert_data_dynamic 对称：
  - 同样的 DynamicMultiAgentEnv、同样的多速度域随机化 [0,10,20,30,40]、同样的 num_slots、同样的顺序决策；
  - 唯一区别：BC 把每个 (obs, 贪心动作) 当作监督标签，本脚本把 (obs, 动作, 奖励) 当作 RL 样本。
  - 复用 IQLTrainer.QNetwork（与 BC 完全同构，hidden_dim=256 对齐），argmax 即动作。

用法（服务器，GPU1-5）：
    CUDA_VISIBLE_DEVICES=1 python train_iql_dynamic.py \
        --num_uav 10 --episodes 1500 --num_slots 50 --gpu 1
评估：
    python eval_dynamic_marl.py --method iql --ckpt checkpoints/iql_dynamic_ms_N10.pt \
        --num_uav 10 --speeds 0 10 20 30 40 --output results_data
"""
import argparse
import json
import os
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from config import ENV_CONFIG, CHECKPOINT_DIR, RESULTS_DIR, LAYOUTS_DIR, LoggerSingleton
from marl_env import MultiAgentEnv
from dynamic_env import DynamicMultiAgentEnv
from iql_trainer import IQLTrainer

logger = LoggerSingleton.get_instance()


def main():
    parser = argparse.ArgumentParser(description="动态场景 IQL 训练（无专家，只用奖励）")
    parser.add_argument("--num_uav", type=int, default=10)
    parser.add_argument("--train_speeds", type=float, nargs="+", default=[0, 10, 20, 30, 40],
                        help="多速度域随机化：每个 episode 随机抽一个速度训练，与 BC 多速度一致")
    parser.add_argument("--episodes", type=int, default=1500, help="训练 episode 数")
    parser.add_argument("--num_slots", type=int, default=50, help="每 episode 时隙数")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Q 网络隐藏层维度（对齐 BC）")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_per_slot", type=int, default=4, help="每时隙 Q 更新次数")
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.997)
    parser.add_argument("--warmup_episodes", type=int, default=30, help="前 N 个 episode 纯随机探索")
    parser.add_argument("--gpu", type=int, default=1, help="GPU编号(1-5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--area_size", type=float, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()

    assert 1 <= args.gpu <= 5, "禁止使用 GPU 0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    area_size = args.area_size or ENV_CONFIG["area_size"]
    train_speeds = sorted(set(args.train_speeds))
    logger.info(f"[Dynamic IQL] 速度集合 = {train_speeds} (无专家，只用奖励), device={device}")

    # obs_dim
    probe = MultiAgentEnv(
        num_uav=args.num_uav,
        observation_radius=ENV_CONFIG["observation_radius"],
        area_size=area_size,
        limit_neighbors=ENV_CONFIG["limit_neighbors"],
    )
    probe.reset()
    obs_dim = probe.obs_dim
    logger.info(f"[Dynamic IQL] N={args.num_uav}, obs_dim={obs_dim}, hidden_dim={args.hidden_dim}")

    # 实例化 IQLTrainer（复用与 BC 同构的 QNetwork）
    trainer = IQLTrainer(
        num_uav=args.num_uav, obs_dim=obs_dim, lr=args.lr,
        hidden_dim=args.hidden_dim, device=device,
    )
    q_net = trainer.q_net
    optimizer = trainer.optimizer
    n_freq = trainer.n_freq
    power_levels = trainer.power_levels
    freq_levels = trainer.freq_levels
    n_actions = trainer.n_actions

    buffer = deque(maxlen=args.buffer_size)

    # 静态 50-layouts 评估（与 BC 训练监控一致，保证可比）
    layout_path = os.path.join(LAYOUTS_DIR, f"N{args.num_uav}_50layouts.json")
    eval_layouts = []
    if os.path.exists(layout_path):
        with open(layout_path) as f:
            eval_layouts = list(json.load(f).values())
        logger.info(f"加载 {len(eval_layouts)} 个评估布局")
    eval_env = MultiAgentEnv(
        num_uav=args.num_uav,
        observation_radius=ENV_CONFIG["observation_radius"],
        area_size=area_size,
        limit_neighbors=ENV_CONFIG["limit_neighbors"],
    )

    save_name = args.save_name or f"iql_dynamic_ms_N{args.num_uav}"
    best_eval = float("inf")
    best_state = None

    epsilon = args.epsilon_start
    rng = np.random.default_rng(args.seed)
    freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
    freq_span = ENV_CONFIG["freq_max"] - ENV_CONFIG["freq_min"] - ENV_CONFIG["bandwidth"]

    t0 = time.time()
    for ep in range(args.episodes):
        uav_speed = float(train_speeds[rng.integers(0, len(train_speeds))])
        env = DynamicMultiAgentEnv(
            num_uav=args.num_uav, num_slots=args.num_slots, uav_speed=uav_speed,
            observation_radius=ENV_CONFIG["observation_radius"], area_size=area_size,
            limit_neighbors=ENV_CONFIG["limit_neighbors"],
        )
        env.reset()
        done = False
        slot_losses = []
        while not done:
            env.reset_commit_state()
            node_list = list(env.base_env.nodes.values())
            rng.shuffle(node_list)
            slot_samples = []
            actions = {}
            for exec_idx, node in enumerate(node_list):
                obs = env.get_sequential_observation(node.node_id, exec_idx)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values = q_net(obs_t)
                if ep < args.warmup_episodes or np.random.random() < epsilon:
                    action_idx = int(np.random.randint(n_actions))
                else:
                    action_idx = int(q_values.argmax(dim=-1).item())
                slot_samples.append((obs, action_idx, node.node_id))

                p_idx = action_idx // n_freq
                f_idx = action_idx % n_freq
                power_val = float(power_levels[p_idx])
                freq_val = float(freq_levels[f_idx])

                # commit 让后续 agent 感知（与 BC 收集完全一致）
                tx = node.tx
                rx = tx.peer
                tx.frequency = freq_val
                tx.power = power_val
                rx.frequency = freq_val
                env.base_env.update_sinr()

                power_norm = (power_val / ENV_CONFIG["max_power"]) * 2 - 1
                freq_norm = ((freq_val - freq_lo) / freq_span) * 2 - 1
                actions[node.node_id] = np.array([power_norm, freq_norm], dtype=np.float32)

            _, rewards, info, done = env.step(actions)
            if not isinstance(rewards, dict):
                logger.warning(f"rewards 不是 dict: {type(rewards)}，跳过本时隙样本")
            else:
                for (obs, action_idx, agent_id) in slot_samples:
                    if agent_id in rewards:
                        buffer.append((obs, action_idx, float(rewards[agent_id])))

            # Q 网络更新（单步 target = reward，与 IQLTrainer 静态训练一致）
            if len(buffer) >= args.batch_size:
                for _ in range(args.update_per_slot):
                    batch = random.sample(buffer, args.batch_size)
                    obs_b = np.array([b[0] for b in batch], dtype=np.float32)
                    act_b = np.array([b[1] for b in batch], dtype=np.int64)
                    rew_b = np.array([b[2] for b in batch], dtype=np.float32)
                    obs_t = torch.tensor(obs_b, device=device)
                    act_t = torch.tensor(act_b, device=device)
                    rew_t = torch.tensor(rew_b, device=device)
                    q_sa = q_net(obs_t).gather(1, act_t.unsqueeze(-1)).squeeze(-1)
                    loss = F.mse_loss(q_sa, rew_t)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                    optimizer.step()
                    slot_losses.append(float(loss.item()))

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if (ep + 1) % args.eval_interval == 0:
            if eval_layouts:
                probs, _ = trainer.evaluate_on_layouts(eval_env, eval_layouts)
                avg = float(np.mean(probs))
                if avg < best_eval:
                    best_eval = avg
                    best_state = {k: v.clone() for k, v in q_net.state_dict().items()}
                logger.info(f"  ep {ep+1}/{args.episodes}: speed={uav_speed:.0f} "
                            f"eps={epsilon:.3f} loss={np.mean(slot_losses) if slot_losses else 0:.4f} "
                            f"eval_Pint={avg:.4f} best={best_eval:.4f} ({time.time()-t0:.0f}s)")
            else:
                logger.info(f"  ep {ep+1}: loss={np.mean(slot_losses) if slot_losses else 0:.4f} eps={epsilon:.3f}")

    if best_state is not None:
        q_net.load_state_dict(best_state)
        logger.info(f"恢复最佳模型 (eval P_int={best_eval:.4f})")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{save_name}.pt")
    trainer.save(ckpt_path)
    logger.info(f"保存 {ckpt_path}，总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
