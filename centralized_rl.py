"""中心化 RL 训练器：观测全局，输出所有 UAV 的动作。

作为 RL 方法的上界——使用全局信息（所有 UAV 的位置+频率+功率+SINR），
单个网络输出所有 N 个 UAV 的频率/功率选择。

与 BC/MARL 的区别：
- BC/IQL: 每个 UAV 独立观测局部信息，各自决策（去中心化）
- Centralized RL: 单个网络观测全局信息，一次性输出所有动作（中心化）

训练方式：行为克隆（用 Greedy-Sequential 作为 teacher）
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
from test_cgreedy import _choose_best_freq_power

logger = LoggerSingleton.get_instance()


class CentralizedNet(nn.Module):
    """中心化网络：全局状态 → 所有 UAV 的动作。
    
    输入：全局状态（所有 UAV 的位置 + 频率 + 功率 + SINR + 配对信息）
    输出：每个 UAV 的 Q 值（n_power * n_freq 个离散动作）
    """
    def __init__(self, global_state_dim: int, num_uav: int, n_power: int, n_freq: int, hidden_dim: int = 256):
        super().__init__()
        self.num_uav = num_uav
        self.n_power = n_power
        self.n_freq = n_freq
        self.n_actions = n_power * n_freq
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 每个 UAV 的输出头
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.n_actions) for _ in range(num_uav)
        ])

    def forward(self, global_state: torch.Tensor):
        """返回所有 UAV 的 Q 值。
        
        Args:
            global_state: (B, global_state_dim)
        Returns:
            q_values: (B, num_uav, n_actions)
        """
        encoded = self.shared_encoder(global_state)  # (B, hidden)
        outputs = [head(encoded) for head in self.output_heads]  # list of (B, n_actions)
        return torch.stack(outputs, dim=1)  # (B, num_uav, n_actions)


class CentralizedBC:
    """中心化行为克隆：用全局状态预测所有 UAV 的最优动作。"""
    
    def __init__(
        self,
        num_uav: int,
        global_state_dim: int,
        n_power: int = None,
        n_freq: int = None,
        lr: float = 1e-3,
        hidden_dim: int = 256,
        device: str = "cuda",
    ):
        self.num_uav = num_uav
        self.global_state_dim = global_state_dim
        self.n_power = n_power or ACTION_CONFIG["n_power"]
        self.n_freq = n_freq or ACTION_CONFIG["n_freq"]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.net = CentralizedNet(global_state_dim, num_uav, self.n_power, self.n_freq, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        # 离散动作档位
        self.power_levels = np.linspace(0, ENV_CONFIG["max_power"], self.n_power)
        freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
        freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
        self.freq_levels = np.linspace(freq_lo, freq_hi, self.n_freq)
    
    def _build_global_state(self, env: MultiAgentEnv) -> np.ndarray:
        """构建全局状态向量：所有 UAV 的位置+频率+功率+SINR。"""
        state = []
        node_ids = list(env.base_env.nodes.keys())
        for node_id in node_ids:
            node = env.base_env.nodes[node_id]
            tx = node.tx
            rx = node.tx.peer if node.tx.peer else node.rx
            
            # 归一化位置
            state.extend([
                node.position[0] / env.area_size * 2,
                node.position[1] / env.area_size * 2,
                (node.position[2] - env.z_center) / env.z_span,
            ])
            # 归一化载荷
            state.extend([
                (tx.power - tx.min_power) / max(1e-6, tx.max_power - tx.min_power),
                (tx.frequency - tx.min_frequency) / max(1e-6, tx.max_frequency - tx.min_frequency),
            ])
            # SINR
            sinr = rx.sinr if rx.sinr != float('-inf') and rx.sinr != float('inf') else -50.0
            state.append(np.tanh((sinr - rx.threshold) / 10.0))
        
        return np.array(state, dtype=np.float32)
    
    def collect_expert_data(self, num_episodes: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """用 Greedy-Sequential 收集专家数据。
        
        Returns:
            list of (global_state, expert_actions) 
            expert_actions: (num_uav,) 每个UAV的最优动作index
        """
        env = MultiAgentEnv(
            num_uav=self.num_uav,
            observation_radius=ENV_CONFIG["observation_radius"],
            area_size=ENV_CONFIG["area_size"],
            limit_neighbors=ENV_CONFIG["limit_neighbors"],
        )
        
        freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
        freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
        freq_candidates = np.linspace(freq_lo, freq_hi, self.n_freq)
        power_candidates = np.linspace(0, ENV_CONFIG["max_power"], self.n_power)
        
        expert_data = []
        pbar = tqdm(range(num_episodes), desc="Centralized BC collecting", ncols=120)
        for ep in pbar:
            env.reset()
            env.base_env.update_sinr()
            
            # 收集初始全局状态
            global_state = self._build_global_state(env)
            
            # 顺序贪心决策
            node_list = list(env.base_env.nodes.values())
            rng = np.random.default_rng()
            rng.shuffle(node_list)
            
            expert_actions = np.zeros(self.num_uav, dtype=np.int64)
            node_id_to_idx = {n.node_id: i for i, n in enumerate(env.base_env.nodes.values())}
            
            for node in node_list:
                best_sinr, best_freq, best_power = _choose_best_freq_power(
                    env, node, freq_candidates, power_candidates,
                    use_neighbors=True, neighbor_sample=None,
                )
                # 动作 index
                p_idx = int(np.argmin(np.abs(self.power_levels - best_power)))
                f_idx = int(np.argmin(np.abs(self.freq_levels - best_freq)))
                action_idx = p_idx * self.n_freq + f_idx
                
                node_idx = node_id_to_idx[node.node_id]
                expert_actions[node_idx] = action_idx
                
                # 应用到环境
                tx = node.tx
                rx = tx.peer
                tx.frequency = best_freq
                tx.power = best_power
                rx.frequency = best_freq
                env.base_env.update_sinr()
            
            expert_data.append((global_state, expert_actions))
            
            if (ep + 1) % 100 == 0:
                pbar.set_postfix({"samples": len(expert_data)})
        
        logger.info(f"收集 {len(expert_data)} 条中心化专家数据")
        return expert_data
    
    def train(self, expert_data: List[Tuple[np.ndarray, np.ndarray]], num_epochs: int = 200,
              batch_size: int = 256, eval_env: MultiAgentEnv = None,
              eval_layouts: list = None, eval_interval: int = 10) -> dict:
        """训练中心化 BC。"""
        states = np.array([d[0] for d in expert_data], dtype=np.float32)
        actions = np.array([d[1] for d in expert_data], dtype=np.int64)  # (N_samples, num_uav)
        n_samples = len(states)
        logger.info(f"Centralized BC 训练: {n_samples} samples, state_dim={states.shape[1]}")
        
        history = {"epoch": [], "loss": [], "acc": [], "eval_interf_probs": []}
        best_eval = float('inf')
        best_state = None
        
        pbar = tqdm(range(num_epochs), desc="Centralized BC", ncols=120)
        for epoch in pbar:
            perm = np.random.permutation(n_samples)
            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = perm[start:end]
                state_batch = torch.tensor(states[idx], device=self.device)
                action_batch = torch.tensor(actions[idx], device=self.device)  # (B, num_uav)
                
                q_values = self.net(state_batch)  # (B, num_uav, n_actions)
                
                # 对每个 UAV 计算交叉熵
                loss = 0
                for i in range(self.num_uav):
                    loss += nn.functional.cross_entropy(q_values[:, i, :], action_batch[:, i])
                loss /= self.num_uav
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += float(loss.item())
                pred = q_values.argmax(dim=-1)  # (B, num_uav)
                epoch_correct += int((pred == action_batch).all(dim=-1).sum().item())
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            acc = epoch_correct / n_samples
            
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["acc"].append(acc)
            
            if (epoch + 1) % eval_interval == 0 and eval_env is not None and eval_layouts:
                probs = self._evaluate(eval_env, eval_layouts)
                avg_prob = float(np.mean(probs))
                history["eval_interf_probs"].append(avg_prob)
                if avg_prob < best_eval:
                    best_eval = avg_prob
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.3f}", "eval": f"{avg_prob:.4f}"})
            else:
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.3f}"})
        
        if best_state:
            self.net.load_state_dict(best_state)
            logger.info(f"恢复最佳中心化模型 (eval P_int={best_eval:.4f})")
        
        return history
    
    def _evaluate(self, env: MultiAgentEnv, layouts: list) -> List[float]:
        """评估中心化策略。"""
        probs = []
        for layout in layouts:
            env.load_layout(layout)
            global_state = self._build_global_state(env)
            state_t = torch.tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.net(state_t)  # (1, num_uav, n_actions)
                actions = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()  # (num_uav,)
            
            # 应用动作
            env.reset_commit_state()
            node_ids = list(env.base_env.nodes.keys())
            for exec_idx, (node_id, action_idx) in enumerate(zip(node_ids, actions)):
                p_idx = action_idx // self.n_freq
                f_idx = action_idx % self.n_freq
                power_val = float(self.power_levels[p_idx])
                freq_val = float(self.freq_levels[f_idx])
                
                freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
                freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
                power_norm = (power_val / ENV_CONFIG["max_power"]) * 2 - 1
                freq_norm = ((freq_val - freq_lo) / (freq_hi - freq_lo)) * 2 - 1
                
                normalized = np.array([power_norm, freq_norm], dtype=np.float32)
                env.apply_sequential_action(node_id, normalized, commit_index=exec_idx)
            
            probs.append(env.get_interference_prob())
        return probs
    
    def save(self, path: str):
        torch.save({
            "net": self.net.state_dict(),
            "num_uav": self.num_uav,
            "global_state_dim": self.global_state_dim,
            "n_power": self.n_power,
            "n_freq": self.n_freq,
        }, path)
        logger.info(f"中心化模型保存至 {path}")
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        logger.info(f"中心化模型加载完成 (N={ckpt['num_uav']})")
