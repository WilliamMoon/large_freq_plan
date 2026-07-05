"""MADDPG 训练器：集中训练、分布执行，Gumbel-Softmax 离散动作。

核心设计：
- 参数共享：所有 agent 共享同一个 actor/critic 网络
- Actor 输入：局部观测 (obs_dim, 与 N 无关) → 功率/频率 logits
- Critic 输入：全局观测拼接 + 联合连续动作 → Q 值
- 训练时用 Gumbel-Softmax 软采样，评估时用 argmax 硬采样
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from config import ENV_CONFIG, ACTION_CONFIG, MADDPG_CONFIG, CHECKPOINT_DIR, LoggerSingleton
from marl_env import MultiAgentEnv
from models import Actor, Critic, gumbel_softmax_sample, logits_to_continuous_action
from replay_buffer import ReplayBuffer

logger = LoggerSingleton.get_instance()


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


class MADDPGTrainer:
    def __init__(
        self,
        num_uav: int,
        obs_dim: int,
        n_power: int = None,
        n_freq: int = None,
        hparams: dict = None,
        device: str = "cuda",
    ):
        self.num_uav = num_uav
        self.obs_dim = obs_dim
        self.n_power = n_power or ACTION_CONFIG["n_power"]
        self.n_freq = n_freq or ACTION_CONFIG["n_freq"]
        self.hparams = hparams or MADDPG_CONFIG
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        h = self.hparams["hidden_dim"]
        
        # 参数共享：单一网络
        self.actor = Actor(obs_dim, self.n_power, self.n_freq, h).to(self.device)
        self.actor_target = Actor(obs_dim, self.n_power, self.n_freq, h).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        global_obs_dim = obs_dim * num_uav
        joint_action_dim = 2 * num_uav  # 每个 agent 输出 (power, freq) 连续值
        self.critic = Critic(global_obs_dim, joint_action_dim, h).to(self.device)
        self.critic_target = Critic(global_obs_dim, joint_action_dim, h).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.hparams["actor_lr"])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.hparams["critic_lr"])
        
        self.buffer = ReplayBuffer(self.hparams["buffer_capacity"])
        
        # 离散动作档位
        self.power_levels = np.linspace(0, ENV_CONFIG["max_power"], self.n_power)
        freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
        freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
        self.freq_levels = np.linspace(freq_lo, freq_hi, self.n_freq)
        
        # 温度退火
        self.tau_init = self.hparams["tau_init"]
        self.tau_final = self.hparams["tau_final"]
        self.tau_decay = self.hparams["tau_decay"]
        
        self.gamma = self.hparams["gamma"]
        self.tau_soft = self.hparams["tau"]  # target network soft update
        
    def get_temperature(self, episode: int, total_episodes: int, warmup: int) -> float:
        if episode < warmup:
            return self.tau_init
        progress = (episode - warmup) / max(1, total_episodes - warmup)
        return self.tau_final + (self.tau_init - self.tau_final) * np.exp(-self.tau_decay * progress)
    
    def select_actions(self, env: MultiAgentEnv, temperature: float, evaluate: bool = False,
                       sequential: bool = True) -> Tuple[Dict[str, np.ndarray], Dict]:
        """为所有 agent 选择动作。
        
        Args:
            env: 环境
            temperature: Gumbel 温度（评估时忽略，用 argmax）
            evaluate: 评估模式（无噪声、argmax）
            sequential: 是否使用顺序决策协议
        
        Returns:
            actions: {agent_id: [power, freq]}
            info: 额外信息
        """
        actions = {}
        node_ids = list(env.base_env.nodes.keys())
        
        if sequential:
            # 顺序决策：轮流作为起点
            start_offset = np.random.randint(0, self.num_uav) if not evaluate else 0
            ordered_ids = node_ids[start_offset:] + node_ids[:start_offset]
            
            for exec_idx, agent_id in enumerate(ordered_ids):
                obs = env.get_sequential_observation(agent_id, exec_idx)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    power_logits, freq_logits = self.actor(obs_t)
                    if evaluate:
                        # argmax
                        power_idx = power_logits.argmax(dim=-1)
                        freq_idx = freq_logits.argmax(dim=-1)
                        power_val = float(self.power_levels[power_idx.item()])
                        freq_val = float(self.freq_levels[freq_idx.item()])
                    else:
                        action_val, _, _ = logits_to_continuous_action(
                            power_logits, freq_logits,
                            self.power_levels, self.freq_levels,
                            temperature, hard=False,
                        )
                        power_val = float(action_val[0, 0].item())
                        freq_val = float(action_val[0, 1].item())
                
                normalized = np.array([
                    (power_val / ENV_CONFIG["max_power"]) * 2 - 1,   # [-1, 1]
                    ((freq_val - (ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"]/2)) / 
                     (ENV_CONFIG["freq_max"] - ENV_CONFIG["freq_min"] - ENV_CONFIG["bandwidth"])) * 2 - 1
                ], dtype=np.float32)
                
                env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
                actions[agent_id] = np.array([power_val, freq_val], dtype=np.float32)
        else:
            # 同步决策
            observations = env.get_observations()
            for agent_id in node_ids:
                obs = observations[agent_id]
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    power_logits, freq_logits = self.actor(obs_t)
                    if evaluate:
                        power_idx = power_logits.argmax(dim=-1)
                        freq_idx = freq_logits.argmax(dim=-1)
                        power_val = float(self.power_levels[power_idx.item()])
                        freq_val = float(self.freq_levels[freq_idx.item()])
                    else:
                        action_val, _, _ = logits_to_continuous_action(
                            power_logits, freq_logits,
                            self.power_levels, self.freq_levels,
                            temperature, hard=False,
                        )
                        power_val = float(action_val[0, 0].item())
                        freq_val = float(action_val[0, 1].item())
                
                normalized = np.array([
                    (power_val / ENV_CONFIG["max_power"]) * 2 - 1,
                    ((freq_val - (ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"]/2)) / 
                     (ENV_CONFIG["freq_max"] - ENV_CONFIG["freq_min"] - ENV_CONFIG["bandwidth"])) * 2 - 1
                ], dtype=np.float32)
                actions[agent_id] = normalized
        
        return actions, {}
    
    def _collect_transition(self, env: MultiAgentEnv, temperature: float):
        """收集一步transition（顺序决策，单步episode）。"""
        env.reset()
        
        # 顺序决策，收集所有agent的观测和动作
        node_ids = list(env.base_env.nodes.keys())
        start_offset = np.random.randint(0, self.num_uav)
        ordered_ids = node_ids[start_offset:] + node_ids[:start_offset]
        
        agent_obs_list = []
        agent_action_list = []  # 连续动作 [power, freq]
        
        for exec_idx, agent_id in enumerate(ordered_ids):
            obs = env.get_sequential_observation(agent_id, exec_idx)
            agent_obs_list.append(obs)
            
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                power_logits, freq_logits = self.actor(obs_t)
                action_val, _, _ = logits_to_continuous_action(
                    power_logits, freq_logits,
                    self.power_levels, self.freq_levels,
                    temperature, hard=False,
                )
                power_val = float(action_val[0, 0].item())
                freq_val = float(action_val[0, 1].item())
            
            agent_action_list.append([power_val, freq_val])
            
            normalized = np.array([
                (power_val / ENV_CONFIG["max_power"]) * 2 - 1,
                ((freq_val - (ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"]/2)) / 
                 (ENV_CONFIG["freq_max"] - ENV_CONFIG["freq_min"] - ENV_CONFIG["bandwidth"])) * 2 - 1
            ], dtype=np.float32)
            env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
        
        # 此时所有agent已执行完毕，计算reward
        rewards = env.calculate_rewards()
        # 顺序决策是单步episode：obs_next = 最终观测
        next_observations = env.get_observations()
        
        # 按node_ids顺序整理
        global_obs = np.array([agent_obs_list[i] for i in range(self.num_uav)], dtype=np.float32)
        # 注意: ordered_ids 的顺序和 node_ids 可能不同，需要对齐
        # 这里直接用 ordered_ids 顺序存储
        joint_actions = np.array(agent_action_list, dtype=np.float32)
        reward_vec = np.array([rewards[ordered_ids[i]] for i in range(self.num_uav)], dtype=np.float32)
        global_obs_next = np.array([
            next_observations[ordered_ids[i]] for i in range(self.num_uav)
        ], dtype=np.float32)
        
        # 存储transition（按ordered_ids顺序，critic输入也是这个顺序）
        self.buffer.push(global_obs, joint_actions, reward_vec, global_obs_next, True)
        
        total_reward = float(np.mean(reward_vec))
        interf_prob = env.get_interference_prob()
        return total_reward, interf_prob
    
    def update(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return 0.0, 0.0
        
        global_obs, joint_actions, rewards, global_obs_next, dones = self.buffer.sample(batch_size)
        # shapes: (B, N, ...)
        B, N, _ = global_obs.shape
        
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # (B, N)
        
        # === Critic 更新 ===
        global_obs_flat = global_obs.reshape(B, -1)
        joint_actions_flat = joint_actions.reshape(B, -1)
        global_obs_t = torch.tensor(global_obs_flat, dtype=torch.float32, device=self.device)
        joint_actions_t = torch.tensor(joint_actions_flat, dtype=torch.float32, device=self.device)
        
        avg_reward = rewards_t.mean(dim=-1, keepdim=True)  # (B, 1)
        current_q = self.critic(global_obs_t, joint_actions_t)  # (B, 1)
        critic_loss = nn.functional.mse_loss(current_q, avg_reward.detach())
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()
        
        # === Actor 更新（策略梯度 + 归一化 advantage）===
        actor_loss_total = 0.0
        
        for i in range(N):
            obs_i = global_obs[:, i, :]  # (B, obs_dim)
            obs_i_t = torch.tensor(obs_i, dtype=torch.float32, device=self.device)
            p_logits, f_logits = self.actor(obs_i_t)
            
            p_log_probs = nn.functional.log_softmax(p_logits, dim=-1)
            f_log_probs = nn.functional.log_softmax(f_logits, dim=-1)
            
            p_actions = joint_actions[:, i, 0]
            f_actions = joint_actions[:, i, 1]
            p_idx = torch.tensor(
                [np.argmin(np.abs(self.power_levels - v)) for v in p_actions],
                dtype=torch.long, device=self.device
            )
            f_idx = torch.tensor(
                [np.argmin(np.abs(self.freq_levels - v)) for v in f_actions],
                dtype=torch.long, device=self.device
            )
            
            log_prob = p_log_probs.gather(1, p_idx.unsqueeze(-1)).squeeze(-1) + \
                       f_log_probs.gather(1, f_idx.unsqueeze(-1)).squeeze(-1)
            
            # advantage: 个体 reward - batch 内该 agent 的 reward 均值
            with torch.no_grad():
                r_i = rewards_t[:, i]  # (B,)
                r_mean = r_i.mean()
                r_std = r_i.std() + 1e-6
                advantage = (r_i - r_mean) / r_std  # 标准化
                advantage = advantage.clamp(-3, 3)  # clip
            
            actor_loss_total += -(log_prob * advantage).mean()
        
        actor_loss = actor_loss_total / N
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()
        
        # Target 软更新
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau_soft).add_(self.tau_soft * p.data)
        
        return float(critic_loss.item()), float(actor_loss.item())
    
    def train(self, num_episodes: int, warmup: int = None, eval_env: MultiAgentEnv = None,
              eval_layouts: list = None, eval_interval: int = None, gpu_id: int = 1) -> TrainingStats:
        """训练 MADDPG。
        
        Args:
            num_episodes: 训练轮数
            warmup: 热身轮数
            eval_env: 评估环境（可不同规模，用于泛化评估）
            eval_layouts: 评估布局列表
            eval_interval: 评估间隔
            gpu_id: GPU编号
        """
        warmup = warmup or self.hparams["warmup_episodes"]
        eval_interval = eval_interval or self.hparams["eval_interval"]
        
        # 离散档位 tensor
        self.power_levels_t = torch.tensor(self.power_levels, dtype=torch.float32, device=self.device)
        self.freq_levels_t = torch.tensor(self.freq_levels, dtype=torch.float32, device=self.device)
        
        stats = TrainingStats()
        
        # 训练环境
        train_env = MultiAgentEnv(
            num_uav=self.num_uav,
            observation_radius=ENV_CONFIG["observation_radius"],
            area_size=ENV_CONFIG["area_size"],
            limit_neighbors=ENV_CONFIG["limit_neighbors"],
        )
        
        update_interval = self.hparams["update_interval"]
        batch_size = min(self.hparams["batch_size"], 16)  # 单步episode下降低batch_size
        # 单步episode下多次更新以加速收敛
        update_per_episode = 10
        
        pbar = tqdm(range(num_episodes), desc=f"MADDPG N={self.num_uav}", ncols=150)
        for ep in pbar:
            temperature = self.get_temperature(ep, num_episodes, warmup)
            
            total_reward, interf_prob = self._collect_transition(train_env, temperature)
            stats.episode_rewards.append(total_reward)
            stats.episode_interf_probs.append(interf_prob)
            
            # 网络更新：每个episode多次更新（单步episode下buffer增长慢）
            if ep >= warmup and len(self.buffer) >= batch_size:
                cl_sum, al_sum = 0.0, 0.0
                for _ in range(update_per_episode):
                    cl, al = self.update(batch_size)
                    cl_sum += cl
                    al_sum += al
                cl, al = cl_sum / update_per_episode, al_sum / update_per_episode
            else:
                cl = al = 0.0
            
            # 评估
            if (ep + 1) % eval_interval == 0:
                if eval_env is not None and eval_layouts:
                    eval_probs, eval_sinrs = self.evaluate_on_layouts(eval_env, eval_layouts)
                else:
                    # 在训练环境上评估
                    eval_probs, eval_sinrs = self.evaluate_on_env(train_env, num_eval=10)
                avg_eval_prob = float(np.mean(eval_probs))
                avg_eval_sinr = float(np.mean(eval_sinrs))
                stats.eval_episodes.append(ep)
                stats.eval_interf_probs.append(avg_eval_prob)
                stats.eval_avg_sinrs.append(avg_eval_sinr)
                pbar.set_postfix({
                    "T": f"{temperature:.3f}",
                    "P_int": f"{interf_prob:.3f}",
                    "eval": f"{avg_eval_prob:.3f}",
                    "cl": f"{cl:.4f}",
                    "al": f"{al:.4f}",
                    "buf": f"{len(self.buffer)}",
                })
            else:
                pbar.set_postfix({
                    "T": f"{temperature:.3f}",
                    "P_int": f"{interf_prob:.3f}",
                    "cl": f"{cl:.4f}",
                    "al": f"{al:.4f}",
                    "buf": f"{len(self.buffer)}",
                })
        
        return stats
    
    def evaluate_on_env(self, env: MultiAgentEnv, num_eval: int = 10) -> Tuple[List[float], List[float]]:
        """在随机布局上评估。"""
        probs, sinrs = [], []
        for _ in range(num_eval):
            env.reset()
            self.select_actions(env, temperature=self.tau_final, evaluate=True, sequential=True)
            probs.append(env.get_interference_prob())
            sinr_vals = []
            for rx in env.base_env.receivers:
                if rx.sinr != float('-inf') and rx.sinr != float('inf'):
                    sinr_vals.append(rx.sinr)
            sinrs.append(np.mean(sinr_vals) if sinr_vals else 0.0)
        return probs, sinrs
    
    def evaluate_on_layouts(self, env: MultiAgentEnv, layouts: List[dict]) -> Tuple[List[float], List[float]]:
        """在固定布局上评估。"""
        probs, sinrs = [], []
        for layout in layouts:
            env.load_layout(layout)
            self.select_actions(env, temperature=self.tau_final, evaluate=True, sequential=True)
            probs.append(env.get_interference_prob())
            sinr_vals = []
            for rx in env.base_env.receivers:
                if rx.sinr != float('-inf') and rx.sinr != float('inf'):
                    sinr_vals.append(rx.sinr)
            sinrs.append(np.mean(sinr_vals) if sinr_vals else 0.0)
        return probs, sinrs
    
    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "num_uav": self.num_uav,
            "obs_dim": self.obs_dim,
            "n_power": self.n_power,
            "n_freq": self.n_freq,
            "hparams": self.hparams,
        }, path)
        logger.info(f"模型保存至 {path}")
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.num_uav = ckpt["num_uav"]
        self.obs_dim = ckpt["obs_dim"]
        logger.info(f"模型从 {path} 加载完成 (N={self.num_uav}, obs_dim={self.obs_dim})")
