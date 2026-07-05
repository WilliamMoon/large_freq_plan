"""行为克隆（BC）训练器：用 Greedy-Sequential 作为 teacher 生成专家轨迹并训练 actor。

核心思路：
1. 在随机布局上运行 greedy-sequential，收集 (obs, expert_action_idx) 对
2. 训练分类网络模仿专家决策
3. BC 预训练后可直接评估，或加载到 IQL trainer 做 RL 微调

关键设计：
- 复用 IQL 的 QNetwork 结构（输出 n_actions 个 Q 值，取 argmax 即动作）
- 这样 BC 预训练的权重可以直接加载到 IQL trainer 中微调
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

from config import ENV_CONFIG, ACTION_CONFIG, CHECKPOINT_DIR, RESULTS_DIR, LAYOUTS_DIR, LoggerSingleton
from marl_env import MultiAgentEnv
from iql_trainer import IQLTrainer, QNetwork
from test_cgreedy import (
    _enumerate_best_frequency_sequential,
    _choose_best_freq_power,
)

logger = LoggerSingleton.get_instance()


class BCTrainer:
    """行为克隆：用 greedy-sequential teacher 训练 actor。"""
    
    def __init__(
        self,
        num_uav: int,
        obs_dim: int,
        n_power: int = None,
        n_freq: int = None,
        lr: float = 1e-3,
        hidden_dim: int = 256,
        device: str = "cuda",
    ):
        self.num_uav = num_uav
        self.obs_dim = obs_dim
        self.n_power = n_power or ACTION_CONFIG["n_power"]
        self.n_freq = n_freq or ACTION_CONFIG["n_freq"]
        self.n_actions = self.n_power * self.n_freq
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 复用 QNetwork 结构，方便后续迁移到 IQL
        self.q_net = QNetwork(obs_dim, self.n_power, self.n_freq, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # 离散动作档位
        self.power_levels = np.linspace(0, ENV_CONFIG["max_power"], self.n_power)
        freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
        freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
        self.freq_levels = np.linspace(freq_lo, freq_hi, self.n_freq)
    
    def _action_idx_to_values(self, idx: int) -> Tuple[float, float]:
        p_idx = idx // self.n_freq
        f_idx = idx % self.n_freq
        return float(self.power_levels[p_idx]), float(self.freq_levels[f_idx])
    
    def _values_to_action_idx(self, power: float, freq: float) -> int:
        p_idx = int(np.argmin(np.abs(self.power_levels - power)))
        f_idx = int(np.argmin(np.abs(self.freq_levels - freq)))
        return p_idx * self.n_freq + f_idx
    
    def _normalized_action(self, power: float, freq: float) -> np.ndarray:
        freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
        freq_span = ENV_CONFIG["freq_max"] - ENV_CONFIG["freq_min"] - ENV_CONFIG["bandwidth"]
        return np.array([
            (power / ENV_CONFIG["max_power"]) * 2 - 1,
            ((freq - freq_lo) / freq_span) * 2 - 1,
        ], dtype=np.float32)
    
    def collect_expert_data(self, num_episodes: int) -> List[Tuple[np.ndarray, int]]:
        """用 greedy-sequential 收集专家轨迹。
        
        Returns:
            list of (obs, expert_action_idx)
        """
        env = MultiAgentEnv(
            num_uav=self.num_uav,
            observation_radius=ENV_CONFIG["observation_radius"],
            area_size=ENV_CONFIG["area_size"],
            limit_neighbors=ENV_CONFIG["limit_neighbors"],
        )
        
        # 贪心候选频点/功率（与 baseline 对齐）
        freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
        freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
        freq_candidates = np.linspace(freq_lo, freq_hi, self.n_freq)
        power_candidates = np.linspace(0, ENV_CONFIG["max_power"], self.n_power)
        
        expert_data = []
        pbar = tqdm(range(num_episodes), desc="Collecting expert data", ncols=120)
        for ep in pbar:
            env.reset()
            env.base_env.update_sinr()
            
            # 顺序贪心决策
            node_list = list(env.base_env.nodes.values())
            rng = np.random.default_rng()
            rng.shuffle(node_list)
            
            for exec_idx, node in enumerate(node_list):
                # 收集此时刻的观测
                obs = env.get_sequential_observation(node.node_id, exec_idx)
                
                # greedy 选最优动作
                best_sinr, best_freq, best_power = _choose_best_freq_power(
                    env, node, freq_candidates, power_candidates,
                    use_neighbors=True, neighbor_sample=None,
                )
                expert_action_idx = self._values_to_action_idx(best_power, best_freq)
                expert_data.append((obs, expert_action_idx))
                
                # 应用专家动作到环境
                tx = node.tx
                rx = tx.peer
                tx.frequency = best_freq
                tx.power = best_power
                rx.frequency = best_freq
                env.base_env.update_sinr()
            
            if (ep + 1) % 100 == 0:
                pbar.set_postfix({"samples": len(expert_data)})
        
        logger.info(f"收集 {len(expert_data)} 条专家数据（{num_episodes} episodes × {self.num_uav} agents）")
        return expert_data
    
    def train(self, expert_data: List[Tuple[np.ndarray, int]], num_epochs: int = 50,
              batch_size: int = 256, eval_env: MultiAgentEnv = None,
              eval_layouts: list = None, eval_interval: int = 10) -> dict:
        """训练行为克隆模型。"""
        # 准备数据
        obs_array = np.array([d[0] for d in expert_data], dtype=np.float32)
        act_array = np.array([d[1] for d in expert_data], dtype=np.int64)
        n_samples = len(obs_array)
        logger.info(f"BC 训练数据: {n_samples} samples, obs_dim={obs_array.shape[1]}")
        
        history = {"epoch": [], "loss": [], "acc": [], "eval_interf_probs": []}
        
        pbar = tqdm(range(num_epochs), desc="BC training", ncols=120)
        for epoch in pbar:
            # shuffle
            perm = np.random.permutation(n_samples)
            obs_shuf = obs_array[perm]
            act_shuf = act_array[perm]
            
            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                obs_batch = torch.tensor(obs_shuf[start:end], device=self.device)
                act_batch = torch.tensor(act_shuf[start:end], device=self.device)
                
                q_values = self.q_net(obs_batch)  # (B, n_actions)
                loss = nn.functional.cross_entropy(q_values, act_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += float(loss.item())
                pred = q_values.argmax(dim=-1)
                epoch_correct += int((pred == act_batch).sum().item())
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            acc = epoch_correct / n_samples
            
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["acc"].append(acc)
            
            # 评估
            if (epoch + 1) % eval_interval == 0 and eval_env is not None and eval_layouts:
                probs, sinrs = self.evaluate_on_layouts(eval_env, eval_layouts)
                avg_prob = float(np.mean(probs))
                history["eval_interf_probs"].append(avg_prob)
                # 保存最佳模型
                if not hasattr(self, '_best_eval_prob') or avg_prob < self._best_eval_prob:
                    self._best_eval_prob = avg_prob
                    self._best_state = {k: v.clone() for k, v in self.q_net.state_dict().items()}
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.3f}", "eval": f"{avg_prob:.4f}", "best": f"{self._best_eval_prob:.4f}"})
            else:
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.3f}"})
        
        # 恢复最佳模型
        if hasattr(self, '_best_state'):
            self.q_net.load_state_dict(self._best_state)
            logger.info(f"恢复最佳模型 (eval P_int={self._best_eval_prob:.4f})")
        
        return history
    
    def select_actions(self, env: MultiAgentEnv, evaluate: bool = True) -> Dict[str, np.ndarray]:
        """用 BC 训练的策略选择动作（顺序决策）。"""
        actions = {}
        node_ids = list(env.base_env.nodes.keys())
        start_offset = 0 if evaluate else np.random.randint(0, self.num_uav)
        ordered_ids = node_ids[start_offset:] + node_ids[:start_offset]
        
        for exec_idx, agent_id in enumerate(ordered_ids):
            obs = env.get_sequential_observation(agent_id, exec_idx)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.q_net(obs_t)
            action_idx = int(q_values.argmax(dim=-1).item())
            
            power_val, freq_val = self._action_idx_to_values(action_idx)
            normalized = self._normalized_action(power_val, freq_val)
            env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
            actions[agent_id] = np.array([power_val, freq_val], dtype=np.float32)
        
        return actions
    
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
    
    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "num_uav": self.num_uav,
            "obs_dim": self.obs_dim,
            "n_power": self.n_power,
            "n_freq": self.n_freq,
        }, path)
        logger.info(f"BC 模型保存至 {path}")
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.num_uav = ckpt["num_uav"]
        self.obs_dim = ckpt["obs_dim"]
        logger.info(f"BC 模型从 {path} 加载完成 (N={self.num_uav}, obs_dim={self.obs_dim})")
