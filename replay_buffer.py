"""经验回放缓冲区"""
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, global_obs, joint_actions, rewards, global_obs_next, dones):
        """存储一条全局transition。
        
        Args:
            global_obs: (N, obs_dim) 全局观测
            joint_actions: (N, action_dim) 联合动作
            rewards: (N,) 每个agent的奖励
            global_obs_next: (N, obs_dim) 下一时刻全局观测
            dones: bool 是否结束
        """
        self.buffer.append((
            np.array(global_obs, dtype=np.float32),
            np.array(joint_actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(global_obs_next, dtype=np.float32),
            dones,
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        global_obs, joint_actions, rewards, global_obs_next, dones = zip(*batch)
        return (
            np.array(global_obs, dtype=np.float32),       # (B, N, obs_dim)
            np.array(joint_actions, dtype=np.float32),    # (B, N, action_dim)
            np.array(rewards, dtype=np.float32),          # (B, N)
            np.array(global_obs_next, dtype=np.float32),  # (B, N, obs_dim)
            np.array(dones, dtype=np.float32),            # (B,)
        )

    def __len__(self):
        return len(self.buffer)
