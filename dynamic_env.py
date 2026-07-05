"""动态多智能体环境：UAV 移动 + 多时隙 episode。

在静态 MultiAgentEnv 基础上增加：
1. UAV 移动模型（Random Waypoint）
2. 多时隙 episode（每时隙：移动→重配对→决策→计算SINR）
3. 累积互扰概率统计

用法：
    env = DynamicMultiAgentEnv(num_uav=10, num_slots=50, uav_speed=20.0)
    obs = env.reset()
    for slot in range(env.num_slots):
        actions = policy(obs)
        obs, rewards, done, info = env.step(actions)
"""
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

from env import Env
from node import Node
from config import ENV_CONFIG, LoggerSingleton
from marl_env import MultiAgentEnv

logger = LoggerSingleton.get_instance()


class DynamicMultiAgentEnv(MultiAgentEnv):
    """动态 UAV 集群环境，支持多时隙 episode 和 UAV 移动。"""

    def __init__(
        self,
        num_uav: int,
        num_slots: int = 50,
        uav_speed: float = 20.0,       # m/s, 0表示静态
        relink_interval: int = 5,       # 每隔多少时隙重新配对
        observation_radius: float = 600.0,
        area_size: float = 2000.0,
        limit_neighbors: int = 5,
        freq_min: float = 1200.0,
        freq_max: float = 1300.0,
        num_channels: int = 11,
        bandwidth: float = 10.0,
        effective_range: float = 500.0,
    ):
        # 父类初始化（会生成初始布局）
        super().__init__(
            num_uav=num_uav,
            observation_radius=observation_radius,
            area_size=area_size,
            limit_neighbors=limit_neighbors,
            freq_min=freq_min,
            freq_max=freq_max,
            num_channels=num_channels,
            bandwidth=bandwidth,
        )

        self.num_slots = num_slots
        self.uav_speed = uav_speed
        self.relink_interval = relink_interval
        self.effective_range = effective_range

        # 动态状态
        self.current_slot = 0
        self._waypoints: Dict[str, np.ndarray] = {}  # 每个UAV的目标航点
        self._velocities: Dict[str, np.ndarray] = {}  # 每个UAV的速度向量

        # 累积统计
        self.slot_interf_probs: List[float] = []

    # ============================================================
    # UAV 移动
    # ============================================================

    def _init_waypoints(self):
        """为每个 UAV 初始化随机航点。"""
        for node_id in self.base_env.nodes:
            self._waypoints[node_id] = self._random_position()

    def _random_position(self) -> np.ndarray:
        """在区域内随机生成一个位置。"""
        half = self.area_size / 2
        return np.array([
            np.random.uniform(-half, half),
            np.random.uniform(-half, half),
            np.random.uniform(self.z_center - self.z_span / 2, self.z_center + self.z_span / 2),
        ])

    def _update_velocities(self):
        """更新每个 UAV 的速度向量（朝向航点）。"""
        for node_id, node in self.base_env.nodes.items():
            pos = np.array(node.position)
            target = self._waypoints[node_id]
            direction = target - pos
            dist = np.linalg.norm(direction)

            if dist < 50:  # 到达航点，选新目标
                self._waypoints[node_id] = self._random_position()
                direction = self._waypoints[node_id] - pos
                dist = np.linalg.norm(direction)

            if dist > 0:
                # 水平方向移动，高度保持小范围波动
                horiz_dir = direction[:2]
                horiz_dist = np.linalg.norm(horiz_dir)
                if horiz_dist > 0:
                    horiz_vel = horiz_dir / horiz_dist * self.uav_speed
                else:
                    horiz_vel = np.zeros(2)
                # 高度微小随机波动（2 m/s）
                vert_vel = np.random.uniform(-2, 2)
                self._velocities[node_id] = np.array([horiz_vel[0], horiz_vel[1], vert_vel])
            else:
                self._velocities[node_id] = np.zeros(3)

    def _move_uavs(self, dt: float = 1.0):
        """移动所有 UAV 一个时间步。

        Args:
            dt: 时间步长（秒）
        """
        if self.uav_speed <= 0:
            return  # 静态模式

        self._update_velocities()
        half = self.area_size / 2

        for node_id, node in self.base_env.nodes.items():
            vel = self._velocities[node_id]
            new_pos = np.array(node.position) + vel * dt

            # 边界反射
            for i in range(2):  # x, y
                if new_pos[i] > half:
                    new_pos[i] = half - (new_pos[i] - half)
                    self._waypoints[node_id] = self._random_position()
                elif new_pos[i] < -half:
                    new_pos[i] = -half + (-half - new_pos[i])
                    self._waypoints[node_id] = self._random_position()

            # 高度限制
            new_pos[2] = np.clip(
                new_pos[2],
                self.z_center - self.z_span / 2,
                self.z_center + self.z_span / 2,
            )

            node.position = tuple(new_pos)

    def _relink(self):
        """重新配对通信链路。"""
        # 保存当前的频率/功率设置
        old_settings = {}
        for node in self.base_env.nodes.values():
            tx = node.tx
            old_settings[node.node_id] = {
                'power': tx.power,
                'frequency': tx.frequency,
            }

        # 清除旧配对，重新匹配
        for tx in self.base_env.transmitters:
            if hasattr(tx, 'peer'):
                tx.peer = None
        for rx in self.base_env.receivers:
            if hasattr(rx, 'peer'):
                rx.peer = None

        self.base_env._match(self.effective_range)
        self._apply_spectrum_limits()

        # 恢复频率/功率设置
        for node in self.base_env.nodes.values():
            s = old_settings.get(node.node_id)
            if s:
                node.tx.power = s['power']
                node.tx.frequency = s['frequency']
                if node.tx.peer:
                    node.tx.peer.frequency = s['frequency']

    # ============================================================
    # 多时隙 episode
    # ============================================================

    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境，开始新的动态 episode。"""
        self.base_env.reset()
        self.base_env.generate_random_uav_layout(self.num_uav)
        self._apply_spectrum_limits()
        self.reset_commit_state()
        self._random_initialize_payloads()
        self.base_env.update_sinr()

        self.current_slot = 0
        self.slot_interf_probs = []
        self.interference_prob_history.clear()
        self.episode_rewards.clear()

        self._init_waypoints()

        return self.get_observations()

    def step_dynamics(self):
        """执行一个时隙的动态变化（移动 + 重配对），不涉及决策。

        在 step() 之前调用，模拟环境的自然演化。
        """
        self._move_uavs(dt=1.0)

        # 定期重新配对
        if self.current_slot > 0 and self.current_slot % self.relink_interval == 0:
            self._relink()

        self.base_env.update_sinr()

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict, bool]:
        """环境步进一个时隙。

        流程：
        1. 应用所有 agent 的动作（频率/功率选择）
        2. 计算 SINR 和 reward
        3. 记录互扰概率
        4. 推进环境动态（UAV移动 + 重配对）
        5. 返回下一时刻观测

        Args:
            actions: {agent_id: [power_normalized, freq_normalized]} 或 {agent_id: [power, freq]}
        Returns:
            observations, rewards, info, done
        """
        # 1. 应用动作
        for agent_id, action in actions.items():
            node = self.base_env.nodes[agent_id]
            action_dict = self._normalize_action(action)
            self._apply_action(node, action_dict)

        # 2. 计算 SINR 和 reward
        self.base_env.update_sinr()
        rewards = self._calculate_rewards()
        observations = self.get_observations()

        # 3. 记录互扰概率
        interf_prob = self.base_env.calc_interf_prob()
        self.slot_interf_probs.append(interf_prob)
        self.interference_prob_history.append(interf_prob)

        info = {
            'interference_prob': interf_prob,
            'slot': self.current_slot,
            'individual_rewards': rewards,
        }

        self.current_slot += 1
        done = self.current_slot >= self.num_slots

        # 4. 如果未结束，推进环境动态（为下一时隙准备）
        if not done:
            self.step_dynamics()
            observations = self.get_observations()

        return observations, rewards, info, done

    def get_cumulative_interf_prob(self) -> float:
        """获取累积互扰概率（所有时隙的均值）。"""
        if not self.slot_interf_probs:
            return 0.0
        return float(np.mean(self.slot_interf_probs))

    def get_interf_prob_history(self) -> List[float]:
        """获取每个时隙的互扰概率。"""
        return self.slot_interf_probs.copy()

    # ============================================================
    # 顺序决策支持（动态场景下每时隙顺序决策）
    # ============================================================

    def run_sequential_decisions(self, get_action_fn, evaluate: bool = False) -> Dict[str, float]:
        """在一个时隙内执行顺序决策。

        Args:
            get_action_fn: callback(obs: np.ndarray, agent_id: str, exec_idx: int) -> (power, freq)
                          返回归一化动作 [power_norm, freq_norm] in [-1, 1]
            evaluate: 评估模式（固定顺序）

        Returns:
            该时隙的 reward 字典
        """
        self.reset_commit_state()
        node_ids = list(self.base_env.nodes.keys())

        if evaluate:
            start_offset = 0
        else:
            start_offset = np.random.randint(0, self.num_uav)
        ordered_ids = node_ids[start_offset:] + node_ids[:start_offset]

        for exec_idx, agent_id in enumerate(ordered_ids):
            obs = self.get_sequential_observation(agent_id, exec_idx)
            action = get_action_fn(obs, agent_id, exec_idx)
            self.apply_sequential_action(agent_id, action, commit_index=exec_idx)

        return self.calculate_rewards()

    # ============================================================
    # 固定布局加载（用于动态评估）
    # ============================================================

    def load_layout(self, layout: dict) -> Dict[str, np.ndarray]:
        """加载固定布局的初始位置（动态 episode 会从这里开始移动）。"""
        self.base_env.reset()
        positions = layout["positions"]
        for idx, pos in enumerate(positions):
            node = Node(node_id=f"uav{idx}", position=tuple(pos))
            node._add_payloads()
            self.base_env.add_node(node)
        self._apply_spectrum_limits()

        pairing = layout.get("pairing", [])
        for tx_idx, rx_idx in pairing:
            tx_node = self.base_env.nodes[f"uav{tx_idx}"]
            rx_node = self.base_env.nodes[f"uav{rx_idx}"]
            tx_node.tx.set_peer(rx_node.rx)

        self._random_initialize_payloads()
        self.base_env.update_sinr()
        self.reset_commit_state()
        self.current_slot = 0
        self.slot_interf_probs = []
        self.interference_prob_history.clear()
        self._init_waypoints()
        return self.get_observations()
