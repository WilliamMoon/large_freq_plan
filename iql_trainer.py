"""Independent Q-Learning (IQL) 训练器。

每个 agent 独立学习一个 Q 网络：Q(local_obs, discrete_action) -> reward
单步 episode 下 Q target = r_i（无折扣未来）
动作选择：epsilon-greedy 或 softmax

这是最简单但可靠的基线 RL 方法，用于验证 MARL 框架是否可行。
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

from config import ENV_CONFIG, ACTION_CONFIG, MADDPG_CONFIG, CHECKPOINT_DIR, RESULTS_DIR, LoggerSingleton
from marl_env import MultiAgentEnv
from models import Actor

logger = LoggerSingleton.get_instance()


class QNetwork(nn.Module):
    """Q 网络：local_obs -> Q values for each discrete action。
    
    动作空间：n_power * n_freq 个离散组合
    """
    def __init__(self, obs_dim: int, n_power: int, n_freq: int, hidden_dim: int = 128):
        super().__init__()
        self.n_power = n_power
        self.n_freq = n_freq
        self.n_actions = n_power * n_freq
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions),
        )

    def forward(self, obs: torch.Tensor):
        """返回所有动作的 Q 值 (B, n_actions)"""
        return self.net(obs)


class TrainingStats:
    def __init__(self):
        self.episode_rewards = []
        self.episode_interf_probs = []
        self.eval_episodes = []
        self.eval_interf_probs = []
        self.eval_avg_sinrs = []

    def to_dict(self):
        return {
            "episode_rewards": self.episode_rewards,
            "episode_interf_probs": self.episode_interf_probs,
            "eval_episodes": self.eval_episodes,
            "eval_interf_probs": self.eval_interf_probs,
            "eval_avg_sinrs": self.eval_avg_sinrs,
        }


class IQLTrainer:
    """Independent Q-Learning: 每个 agent 共享一个 Q 网络。"""
    
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
        
        # 参数共享的 Q 网络
        self.q_net = QNetwork(obs_dim, self.n_power, self.n_freq, hidden_dim).to(self.device)
        self.q_target = QNetwork(obs_dim, self.n_power, self.n_freq, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # 离散动作档位
        self.power_levels = np.linspace(0, ENV_CONFIG["max_power"], self.n_power)
        freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
        freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
        self.freq_levels = np.linspace(freq_lo, freq_hi, self.n_freq)
        
        # 所有离散动作组合 (n_actions, 2): [power, freq]
        self.action_table = np.array([
            [p, f] for p in self.power_levels for f in self.freq_levels
        ], dtype=np.float32)  # (n_power*n_freq, 2)
        
        self.tau_soft = 0.005
        self.gamma = 0.95
        
    def _action_idx_to_values(self, idx: int) -> Tuple[float, float]:
        p_idx = idx // self.n_freq
        f_idx = idx % self.n_freq
        return float(self.power_levels[p_idx]), float(self.freq_levels[f_idx])
    
    def _values_to_action_idx(self, power: float, freq: float) -> int:
        p_idx = int(np.argmin(np.abs(self.power_levels - power)))
        f_idx = int(np.argmin(np.abs(self.freq_levels - freq)))
        return p_idx * self.n_freq + f_idx
    
    def _normalized_action(self, power: float, freq: float) -> np.ndarray:
        """转换为 env 接受的 [-1,1] 归一化动作"""
        freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
        freq_span = ENV_CONFIG["freq_max"] - ENV_CONFIG["freq_min"] - ENV_CONFIG["bandwidth"]
        return np.array([
            (power / ENV_CONFIG["max_power"]) * 2 - 1,
            ((freq - freq_lo) / freq_span) * 2 - 1,
        ], dtype=np.float32)
    
    def select_actions(self, env: MultiAgentEnv, epsilon: float = 0.0,
                       evaluate: bool = False) -> Dict[str, np.ndarray]:
        """epsilon-greedy 动作选择（顺序决策）"""
        actions = {}
        node_ids = list(env.base_env.nodes.keys())
        
        if evaluate:
            start_offset = 0
        else:
            start_offset = np.random.randint(0, self.num_uav)
        ordered_ids = node_ids[start_offset:] + node_ids[:start_offset]
        
        for exec_idx, agent_id in enumerate(ordered_ids):
            obs = env.get_sequential_observation(agent_id, exec_idx)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.q_net(obs_t)  # (1, n_actions)
            
            if not evaluate and np.random.random() < epsilon:
                action_idx = np.random.randint(self.n_actions)
            else:
                action_idx = int(q_values.argmax(dim=-1).item())
            
            power_val, freq_val = self._action_idx_to_values(action_idx)
            normalized = self._normalized_action(power_val, freq_val)
            env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
            actions[agent_id] = np.array([power_val, freq_val], dtype=np.float32)
        
        return actions
    
    def train(self, num_episodes: int, warmup: int = 30, eval_env: MultiAgentEnv = None,
              eval_layouts: list = None, eval_interval: int = 50,
              epsilon_start: float = 1.0, epsilon_end: float = 0.05,
              epsilon_decay: float = 0.995) -> TrainingStats:
        
        stats = TrainingStats()
        
        train_env = MultiAgentEnv(
            num_uav=self.num_uav,
            observation_radius=ENV_CONFIG["observation_radius"],
            area_size=ENV_CONFIG["area_size"],
            limit_neighbors=ENV_CONFIG["limit_neighbors"],
        )
        
        # 经验回放
        from collections import deque
        import random
        buffer = deque(maxlen=10000)
        batch_size = 64
        update_per_episode = 10
        epsilon = epsilon_start
        
        pbar = tqdm(range(num_episodes), desc=f"IQL N={self.num_uav}", ncols=150)
        for ep in pbar:
            # 收集 experience（顺序决策）
            train_env.reset()
            node_ids = list(train_env.base_env.nodes.keys())
            start_offset = np.random.randint(0, self.num_uav)
            ordered_ids = node_ids[start_offset:] + node_ids[:start_offset]
            
            transitions = []  # (obs, action_idx, reward)
            for exec_idx, agent_id in enumerate(ordered_ids):
                obs = train_env.get_sequential_observation(agent_id, exec_idx)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    q_values = self.q_net(obs_t)
                
                if ep < warmup or np.random.random() < epsilon:
                    action_idx = np.random.randint(self.n_actions)
                else:
                    action_idx = int(q_values.argmax(dim=-1).item())
                
                power_val, freq_val = self._action_idx_to_values(action_idx)
                normalized = self._normalized_action(power_val, freq_val)
                train_env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
                transitions.append((obs, action_idx, agent_id))
            
            # 计算所有 agent 的 reward
            rewards = train_env.calculate_rewards()
            for obs, action_idx, agent_id in transitions:
                buffer.append((obs, action_idx, rewards[agent_id]))
            
            total_reward = float(np.mean(list(rewards.values())))
            interf_prob = train_env.get_interference_prob()
            stats.episode_rewards.append(total_reward)
            stats.episode_interf_probs.append(interf_prob)
            
            # Q 网络更新
            avg_q_loss = 0.0
            if ep >= warmup and len(buffer) >= batch_size:
                for _ in range(update_per_episode):
                    batch = random.sample(buffer, batch_size)
                    obs_batch = np.array([t[0] for t in batch], dtype=np.float32)
                    act_batch = np.array([t[1] for t in batch], dtype=np.int64)
                    rew_batch = np.array([t[2] for t in batch], dtype=np.float32)
                    
                    obs_t = torch.tensor(obs_batch, device=self.device)
                    act_t = torch.tensor(act_batch, device=self.device)
                    rew_t = torch.tensor(rew_batch, device=self.device)
                    
                    q_values = self.q_net(obs_t)  # (B, n_actions)
                    q_sa = q_values.gather(1, act_t.unsqueeze(-1)).squeeze(-1)  # (B,)
                    
                    # 单步 episode: target = reward (no next state)
                    loss = nn.functional.mse_loss(q_sa, rew_t)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    avg_q_loss += float(loss.item())
                
                avg_q_loss /= update_per_episode
                
                # Target 软更新
                with torch.no_grad():
                    for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
                        tp.data.mul_(1 - self.tau_soft).add_(self.tau_soft * p.data)
            
            # epsilon 衰减
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # 评估
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
                    "eps": f"{epsilon:.3f}",
                    "P_int": f"{interf_prob:.3f}",
                    "eval": f"{avg_eval_prob:.3f}",
                    "loss": f"{avg_q_loss:.3f}",
                })
            else:
                pbar.set_postfix({
                    "eps": f"{epsilon:.3f}",
                    "P_int": f"{interf_prob:.3f}",
                    "loss": f"{avg_q_loss:.3f}",
                })
        
        return stats
    
    def evaluate_on_env(self, env: MultiAgentEnv, num_eval: int = 10) -> Tuple[List[float], List[float]]:
        probs, sinrs = [], []
        for _ in range(num_eval):
            env.reset()
            self.select_actions(env, epsilon=0.0, evaluate=True)
            probs.append(env.get_interference_prob())
            sinr_vals = [rx.sinr for rx in env.base_env.receivers
                         if rx.sinr != float('-inf') and rx.sinr != float('inf')]
            sinrs.append(np.mean(sinr_vals) if sinr_vals else 0.0)
        return probs, sinrs
    
    def evaluate_on_layouts(self, env: MultiAgentEnv, layouts: List[dict]) -> Tuple[List[float], List[float]]:
        probs, sinrs = [], []
        for layout in layouts:
            env.load_layout(layout)
            self.select_actions(env, epsilon=0.0, evaluate=True)
            probs.append(env.get_interference_prob())
            sinr_vals = [rx.sinr for rx in env.base_env.receivers
                         if rx.sinr != float('-inf') and rx.sinr != float('inf')]
            sinrs.append(np.mean(sinr_vals) if sinr_vals else 0.0)
        return probs, sinrs
    
    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "num_uav": self.num_uav,
            "obs_dim": self.obs_dim,
            "n_power": self.n_power,
            "n_freq": self.n_freq,
        }, path)
        logger.info(f"模型保存至 {path}")
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.num_uav = ckpt["num_uav"]
        self.obs_dim = ckpt["obs_dim"]
        logger.info(f"模型从 {path} 加载完成 (N={self.num_uav}, obs_dim={self.obs_dim})")
