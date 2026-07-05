"""Actor/Critic 网络定义，支持 Gumbel-Softmax 离散动作。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """Actor 网络：局部观测 → 离散动作 logits（功率 + 频率）。
    
    参数共享：所有 agent 使用同一个网络实例。
    输出两组 logits：功率 logits (n_power) 和频率 logits (n_freq)。
    """
    def __init__(self, obs_dim: int, n_power: int, n_freq: int, hidden_dim: int = 128):
        super().__init__()
        self.n_power = n_power
        self.n_freq = n_freq
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_power + n_freq),
        )

    def forward(self, obs: torch.Tensor):
        """返回功率 logits 和频率 logits。
        
        Args:
            obs: (B, obs_dim) 或 (N, obs_dim)
        Returns:
            power_logits: (B, n_power)
            freq_logits: (B, n_freq)
        """
        logits = self.net(obs)
        power_logits = logits[:, :self.n_power]
        freq_logits = logits[:, self.n_power:]
        return power_logits, freq_logits


class Critic(nn.Module):
    """Critic 网络：全局观测 + 联合动作 → Q 值。
    
    参数共享：所有 agent 使用同一个网络实例。
    输入：全局观测拼接 (N*obs_dim) + 联合动作 (N*2)，其中动作用连续值表示
         （Gumbel-Softmax 输出的 power/freq 连续值）。
    """
    def __init__(self, global_obs_dim: int, joint_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = global_obs_dim + joint_action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs: torch.Tensor, joint_actions: torch.Tensor):
        """返回 Q 值。
        
        Args:
            global_obs: (B, N*obs_dim)
            joint_actions: (B, N*action_dim)  action_dim=2 (power, freq 连续值)
        Returns:
            q: (B, 1)
        """
        x = torch.cat([global_obs, joint_actions], dim=-1)
        return self.net(x)


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float, hard: bool = False):
    """Gumbel-Softmax 采样。
    
    Args:
        logits: (B, n) 未归一化 logits
        temperature: 温度参数
        hard: True 时返回 one-hot（直通估计），False 时返回软概率
    Returns:
        (B, n) 概率向量
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    y = (logits + gumbel_noise) / temperature
    y_soft = F.softmax(y, dim=-1)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft  # straight-through
    return y_soft


def logits_to_continuous_action(power_logits: torch.Tensor, freq_logits: torch.Tensor,
                                 power_levels: np.ndarray, freq_levels: np.ndarray,
                                 temperature: float, hard: bool = False):
    """将 logits 通过 Gumbel-Softmax 转为连续动作值 (power, freq)。
    
    Args:
        power_logits: (B, n_power)
        freq_logits: (B, n_freq)
        power_levels: (n_power,) 离散功率档位值 [0..5] W
        freq_levels: (n_freq,) 离散频率档位值 [1205..1295] MHz
        temperature: Gumbel 温度
        hard: 是否 hard 采样
    Returns:
        actions: (B, 2) 连续动作 [power, freq]
        power_probs: (B, n_power)
        freq_probs: (B, n_freq)
    """
    power_probs = gumbel_softmax_sample(power_logits, temperature, hard=hard)
    freq_probs = gumbel_softmax_sample(freq_logits, temperature, hard=hard)
    
    power_levels_t = torch.tensor(power_levels, dtype=torch.float32, device=power_logits.device)
    freq_levels_t = torch.tensor(freq_levels, dtype=torch.float32, device=freq_logits.device)
    
    power_val = (power_probs * power_levels_t).sum(dim=-1, keepdim=True)  # (B, 1)
    freq_val = (freq_probs * freq_levels_t).sum(dim=-1, keepdim=True)    # (B, 1)
    actions = torch.cat([power_val, freq_val], dim=-1)  # (B, 2)
    return actions, power_probs, freq_probs
