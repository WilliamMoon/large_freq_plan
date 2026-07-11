"""CARLTON 基线：CTDE value-based 多智能体 RL（DeepMellow 风格）。

参考: Cohen et al. 2024, "SINR-Aware Deep Reinforcement Learning for
Distributed Dynamic Channel Allocation in Cognitive Interference Networks"
(IEEE TWC). 该论文提出 CARLTON——基于 CTDE 的 value-based 多智能体 RL，
使用低维 QoS 观测与 DeepMellow（mellowmax 聚合）实现分布式 DCA。

本实现忠实于其核心思想，并复用本项目的顺序决策协议与离散动作空间：
- Decentralized Q-network（执行用，仅看局部 obs，与 IQL/BC 同构 QNetwork）
  → 可零改动接入 eval_dynamic_marl.py 的 make_policy_fn
- Centralized critic Q-network（训练用，看 [局部obs + 全局QoS特征]）
  → 体现 CTDE：集中训练利用全局信息，分散执行只用局部观测
- Mellowmax 算子用于探索与 actor-critic 蒸馏（DeepMellow 特色）
- 单步 episode（顺序决策一次即结束），Q target = r_i（无 bootstrap）
- actor 通过蒸馏 critic 的全局视野 Q 估计来学习，缓解 IQL 的非平稳性

与现有基线的定位：
- 比 IQL（独立学习、无 critic）多一层集中式协调
- 比 MADDPG（actor-critic + Gumbel）是 value-based、更稳
- 作为"纯 RL（无 BC 预训练）CTDE"基线，衬托 BC 预训练的稳定收敛与实时性
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from tqdm import tqdm

from config import ENV_CONFIG, ACTION_CONFIG, CHECKPOINT_DIR, RESULTS_DIR, LoggerSingleton
from marl_env import MultiAgentEnv
from iql_trainer import QNetwork

logger = LoggerSingleton.get_instance()


def mellowmax_dist(q: torch.Tensor, omega: float) -> torch.Tensor:
    """Mellowmax 分布：softmax(omega * q)。

    DeepMellow 用 mellowmax 算子替代 max 做价值聚合与策略，
    提供熵正则化的平滑策略。omega 越大越接近 argmax，越小越接近均匀。
    """
    return torch.softmax(omega * q, dim=-1)


class CentralQNetwork(nn.Module):
    """集中式 critic：输入 [局部 obs; 全局 QoS 特征] -> 各动作 Q 值。

    训练时使用（看到全局信息），执行时不使用。
    """

    def __init__(self, obs_dim: int, global_feat_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = obs_dim + global_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, global_feat], dim=-1)
        return self.net(x)


class TrainingStats:
    def __init__(self):
        self.episode_rewards = []
        self.episode_interf_probs = []
        self.eval_episodes = []
        self.eval_interf_probs = []
        self.eval_avg_sinrs = []
        self.critic_losses = []
        self.actor_losses = []

    def to_dict(self):
        return {
            "episode_rewards": self.episode_rewards,
            "episode_interf_probs": self.episode_interf_probs,
            "eval_episodes": self.eval_episodes,
            "eval_interf_probs": self.eval_interf_probs,
            "eval_avg_sinrs": self.eval_avg_sinrs,
            "critic_losses": self.critic_losses,
            "actor_losses": self.actor_losses,
        }


class CarltonTrainer:
    """CARLTON: CTDE value-based RL with centralized critic + mellowmax.

    执行用 decentralized q_net（与 IQL/BC 同构），训练用 centralized q_central。
    """

    def __init__(
        self,
        num_uav: int,
        obs_dim: int,
        n_power: int = None,
        n_freq: int = None,
        lr: float = 1e-3,
        hidden_dim: int = 128,
        device: str = "cuda",
    ):
        self.num_uav = num_uav
        self.obs_dim = obs_dim
        self.n_power = n_power or ACTION_CONFIG["n_power"]
        self.n_freq = n_freq or ACTION_CONFIG["n_freq"]
        self.n_actions = self.n_power * self.n_freq
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 全局 QoS 特征维度（低维，贴合 CARLTON "low-dimensional QoS-type measure"）
        # [executed_fraction, committed_freq_hist(n_freq), all_freq_hist(n_freq),
        #  global_interf_prob, avg_sinr_norm, num_uav_norm]
        self.global_feat_dim = 1 + self.n_freq + self.n_freq + 1 + 1 + 1

        # decentralized actor（执行用）—— 与 IQL/BC 同构，可被 eval_dynamic_marl 复用
        self.q_net = QNetwork(obs_dim, self.n_power, self.n_freq, hidden_dim).to(self.device)

        # centralized critic（训练用）—— 看全局 QoS 特征
        self.q_central = CentralQNetwork(
            obs_dim, self.global_feat_dim, self.n_actions, hidden_dim
        ).to(self.device)
        self.q_central_target = CentralQNetwork(
            obs_dim, self.global_feat_dim, self.n_actions, hidden_dim
        ).to(self.device)
        self.q_central_target.load_state_dict(self.q_central.state_dict())

        self.actor_optim = optim.Adam(self.q_net.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.q_central.parameters(), lr=lr)

        # 离散动作档位
        self.power_levels = np.linspace(0, ENV_CONFIG["max_power"], self.n_power)
        freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
        freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
        self.freq_levels = np.linspace(freq_lo, freq_hi, self.n_freq)

        self.tau_soft = 0.005
        # mellowmax omega 修正（原实现方向写反）：softmax(omega*q) 中
        #   omega 越小 -> 越接近均匀随机；omega 越大 -> 越接近 argmax（确定）。
        # 训练初期保留探索（omega 适中），后期增大 omega 趋近 argmax，
        # 使 actor 被蒸馏成确定策略而非退化成近随机。
        self.omega_init = 1.5
        self.omega_final = 8.0

    # ------------------------------------------------------------
    # 动作转换工具（与 IQL 一致）
    # ------------------------------------------------------------
    def _action_idx_to_values(self, idx: int) -> Tuple[float, float]:
        p_idx = idx // self.n_freq
        f_idx = idx % self.n_freq
        return float(self.power_levels[p_idx]), float(self.freq_levels[f_idx])

    def _normalized_action(self, power: float, freq: float) -> np.ndarray:
        freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
        freq_span = ENV_CONFIG["freq_max"] - ENV_CONFIG["freq_min"] - ENV_CONFIG["bandwidth"]
        return np.array([
            (power / ENV_CONFIG["max_power"]) * 2 - 1,
            ((freq - freq_lo) / freq_span) * 2 - 1,
        ], dtype=np.float32)

    # ------------------------------------------------------------
    # 全局 QoS 特征（低维，集中式 critic 输入的一部分）
    # ------------------------------------------------------------
    def _build_global_feat(self, env: MultiAgentEnv) -> np.ndarray:
        executed = sum(1 for n in env.base_env.nodes.values() if n.commit_index >= 0)
        g = [executed / max(1, env.num_uav)]
        g.extend(env._compute_freq_histogram().tolist())
        g.extend(env._compute_all_freq_histogram().tolist())
        interf = env.base_env.calc_interf_prob()
        g.append(float(np.clip(interf, 0.0, 1.0)))
        sinrs = [rx.sinr for rx in env.base_env.receivers
                 if rx.sinr != float('-inf') and rx.sinr != float('inf')]
        avg_sinr = float(np.mean(sinrs)) if sinrs else 0.0
        g.append(float(np.tanh(avg_sinr / 10.0)))
        g.append(env.num_uav / 50.0)
        return np.array(g, dtype=np.float32)

    def _omega(self, ep: int, warmup: int, total: int) -> float:
        # warmup 期实际走随机动作（见 train/select_actions），此返回值不生效。
        # 训练期：omega 从 omega_init 升到 omega_final（趋近确定）。
        if ep < warmup:
            return self.omega_init
        prog = (ep - warmup) / max(1, total - warmup)
        return self.omega_init + (self.omega_final - self.omega_init) * (1.0 - np.exp(-3.0 * prog))

    # ------------------------------------------------------------
    # 动作选择（mellowmax 探索；评估 argmax）
    # ------------------------------------------------------------
    def select_actions(self, env: MultiAgentEnv, omega: float = 0.1,
                       evaluate: bool = False, warmup_random: bool = False) -> Dict[str, np.ndarray]:
        actions = {}
        node_ids = list(env.base_env.nodes.keys())
        start_offset = 0 if evaluate else np.random.randint(0, self.num_uav)
        ordered_ids = node_ids[start_offset:] + node_ids[:start_offset]

        for exec_idx, agent_id in enumerate(ordered_ids):
            obs = env.get_sequential_observation(agent_id, exec_idx)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                q_values = self.q_net(obs_t)

            if warmup_random:
                action_idx = np.random.randint(self.n_actions)
            elif evaluate:
                action_idx = int(q_values.argmax(dim=-1).item())
            else:
                p = mellowmax_dist(q_values, omega).squeeze(0)
                action_idx = int(torch.multinomial(p, 1).item())

            power_val, freq_val = self._action_idx_to_values(action_idx)
            normalized = self._normalized_action(power_val, freq_val)
            env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
            actions[agent_id] = np.array([power_val, freq_val], dtype=np.float32)

        return actions

    # ------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------
    def train(self, num_episodes: int, warmup: int = 30,
              eval_env: MultiAgentEnv = None, eval_layouts: list = None,
              eval_interval: int = 50) -> TrainingStats:
        stats = TrainingStats()

        train_env = MultiAgentEnv(
            num_uav=self.num_uav,
            observation_radius=ENV_CONFIG["observation_radius"],
            area_size=ENV_CONFIG["area_size"],
            limit_neighbors=ENV_CONFIG["limit_neighbors"],
        )

        from collections import deque
        import random
        buffer = deque(maxlen=20000)
        batch_size = 64
        update_per_episode = 10

        pbar = tqdm(range(num_episodes), desc=f"CARLTON N={self.num_uav}", ncols=150)
        for ep in pbar:
            omega = self._omega(ep, warmup, num_episodes)

            # ---- 收集 transition（顺序决策，单步 episode）----
            train_env.reset()
            node_ids = list(train_env.base_env.nodes.keys())
            start_offset = np.random.randint(0, self.num_uav)
            ordered_ids = node_ids[start_offset:] + node_ids[:start_offset]

            transitions = []
            for exec_idx, agent_id in enumerate(ordered_ids):
                obs = train_env.get_sequential_observation(agent_id, exec_idx)
                gfeat = self._build_global_feat(train_env)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    q_values = self.q_net(obs_t)

                if ep < warmup:
                    action_idx = np.random.randint(self.n_actions)
                else:
                    p = mellowmax_dist(q_values, omega).squeeze(0)
                    action_idx = int(torch.multinomial(p, 1).item())

                power_val, freq_val = self._action_idx_to_values(action_idx)
                normalized = self._normalized_action(power_val, freq_val)
                train_env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
                transitions.append((obs, gfeat, action_idx, agent_id))

            # 所有 agent 决策完毕，计算最终 reward（基于最终全局互扰/SINR）
            rewards = train_env.calculate_rewards()
            for obs, gfeat, action_idx, agent_id in transitions:
                buffer.append((obs, gfeat, action_idx, rewards[agent_id]))

            total_reward = float(np.mean(list(rewards.values())))
            interf_prob = train_env.get_interference_prob()
            stats.episode_rewards.append(total_reward)
            stats.episode_interf_probs.append(interf_prob)

            # ---- 网络更新 ----
            avg_critic_loss = 0.0
            avg_actor_loss = 0.0
            if ep >= warmup and len(buffer) >= batch_size:
                for _ in range(update_per_episode):
                    batch = random.sample(buffer, batch_size)
                    obs_b = np.array([t[0] for t in batch], dtype=np.float32)
                    gfeat_b = np.array([t[1] for t in batch], dtype=np.float32)
                    act_b = np.array([t[2] for t in batch], dtype=np.int64)
                    rew_b = np.array([t[3] for t in batch], dtype=np.float32)

                    obs_t = torch.tensor(obs_b, device=self.device)
                    gfeat_t = torch.tensor(gfeat_b, device=self.device)
                    act_t = torch.tensor(act_b, device=self.device)
                    rew_t = torch.tensor(rew_b, device=self.device)

                    # === Critic 更新：Q_central([o;g])[a] -> r（单步，target=r）===
                    # 使用原始 reward 作为 Q target（保留绝对尺度，避免批内 z-score
                    # 标准化导致的跨 batch 尺度漂移与 actor 蒸馏目标噪声）。
                    rew_norm = rew_t
                    q_c = self.q_central(obs_t, gfeat_t)  # (B, n_actions)
                    q_sa = q_c.gather(1, act_t.unsqueeze(-1)).squeeze(-1)  # (B,)
                    critic_loss = nn.functional.mse_loss(q_sa, rew_norm)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.q_central.parameters(), max_norm=1.0)
                    self.critic_optim.step()

                    # === Actor 更新：蒸馏 critic 的全局视野 Q（mellowmax soft target）===
                    with torch.no_grad():
                        q_c_detach = self.q_central(obs_t, gfeat_t).detach()
                        soft_target = mellowmax_dist(q_c_detach, omega)  # (B, n_actions) 概率分布
                    q_a = self.q_net(obs_t)  # (B, n_actions) 任意实数 Q 值
                    log_p = torch.log_softmax(q_a, dim=-1)  # (B, n_actions) 数值稳定
                    # cross-entropy：让 actor 的 Q 分布逼近 critic 的 mellowmax 分布
                    actor_loss = -(soft_target * log_p).sum(dim=-1).mean()

                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
                    self.actor_optim.step()

                    avg_critic_loss += float(critic_loss.item())
                    avg_actor_loss += float(actor_loss.item())

                avg_critic_loss /= update_per_episode
                avg_actor_loss /= update_per_episode
                stats.critic_losses.append(avg_critic_loss)
                stats.actor_losses.append(avg_actor_loss)

                # critic target 软更新
                with torch.no_grad():
                    for p, tp in zip(self.q_central.parameters(), self.q_central_target.parameters()):
                        tp.data.mul_(1 - self.tau_soft).add_(self.tau_soft * p.data)

            # ---- 评估 ----
            if (ep + 1) % eval_interval == 0:
                if eval_env is not None and eval_layouts:
                    eval_probs, eval_sinrs = self.evaluate_on_layouts(eval_env, eval_layouts)
                else:
                    eval_probs, eval_sinrs = self.evaluate_on_env(train_env, num_eval=10)
                avg_eval_prob = float(np.mean(eval_probs))
                avg_eval_sinr = float(np.mean(eval_sinrs))
                stats.eval_episodes.append(ep)
                stats.eval_interf_probs.append(avg_eval_prob)
                stats.eval_avg_sinrs.append(avg_eval_sinr)
                pbar.set_postfix({
                    "omg": f"{omega:.2f}",
                    "P_int": f"{interf_prob:.3f}",
                    "eval": f"{avg_eval_prob:.3f}",
                    "cl": f"{avg_critic_loss:.3f}",
                    "al": f"{avg_actor_loss:.3f}",
                })
            else:
                pbar.set_postfix({
                    "omg": f"{omega:.2f}",
                    "P_int": f"{interf_prob:.3f}",
                    "cl": f"{avg_critic_loss:.3f}",
                    "al": f"{avg_actor_loss:.3f}",
                })

            # 周期性存盘，防止长训练中断丢失模型
            if (ep + 1) % 1000 == 0:
                interim = os.path.join(CHECKPOINT_DIR, f"carlton_N{self.num_uav}_ep{ep+1}.pt")
                self.save(interim)
                logger.info(f"周期性存盘: {interim}")

        return stats

    # ------------------------------------------------------------
    # 评估（用 decentralized q_net argmax，与 eval_dynamic_marl 一致）
    # ------------------------------------------------------------
    def evaluate_on_env(self, env: MultiAgentEnv, num_eval: int = 10) -> Tuple[List[float], List[float]]:
        probs, sinrs = [], []
        for _ in range(num_eval):
            env.reset()
            self.select_actions(env, evaluate=True)
            probs.append(env.get_interference_prob())
            sinr_vals = [rx.sinr for rx in env.base_env.receivers
                         if rx.sinr != float('-inf') and rx.sinr != float('inf')]
            sinrs.append(np.mean(sinr_vals) if sinr_vals else 0.0)
        return probs, sinrs

    def evaluate_on_layouts(self, env: MultiAgentEnv, layouts: List[dict]) -> Tuple[List[float], List[float]]:
        probs, sinrs = [], []
        for layout in layouts:
            env.load_layout(layout)
            self.select_actions(env, evaluate=True)
            probs.append(env.get_interference_prob())
            sinr_vals = [rx.sinr for rx in env.base_env.receivers
                         if rx.sinr != float('-inf') and rx.sinr != float('inf')]
            sinrs.append(np.mean(sinr_vals) if sinr_vals else 0.0)
        return probs, sinrs

    # ------------------------------------------------------------
    # 存取
    # ------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "q_central": self.q_central.state_dict(),
            "q_central_target": self.q_central_target.state_dict(),
            "num_uav": self.num_uav,
            "obs_dim": self.obs_dim,
            "n_power": self.n_power,
            "n_freq": self.n_freq,
            "global_feat_dim": self.global_feat_dim,
        }, path)
        logger.info(f"CARLTON 模型保存至 {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_central.load_state_dict(ckpt["q_central"])
        self.q_central_target.load_state_dict(ckpt["q_central_target"])
        self.num_uav = ckpt["num_uav"]
        self.obs_dim = ckpt["obs_dim"]
        logger.info(f"CARLTON 模型从 {path} 加载完成 (N={self.num_uav}, obs_dim={self.obs_dim})")
