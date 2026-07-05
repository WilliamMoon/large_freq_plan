import numpy as np
import math
from typing import List, Dict, Tuple, Optional

from env import Env
from node import Node
from config import LoggerSingleton
logger = LoggerSingleton.get_instance()


class MultiAgentEnv:
    def __init__(
            self, 
            num_uav: int, 
            observation_radius: float = 600.0, 
            area_size: float = 2000.0,
            limit_neighbors: int = 5,
            freq_min: float = 1200.0,
            freq_max: float = 1300.0,
            num_channels: int = 11,
            bandwidth: float = 10.0,
    ):
        self.num_uav = num_uav
        self.area_size = area_size
        self.z_center = 100.0
        self.z_span = 100.0
        self.base_env = Env(area_size, self.z_center, self.z_span)
        # 频谱栅格信息，与训练动作一致
        self.base_env.freq_min = freq_min
        self.base_env.freq_max = freq_max
        self.base_env.num_channels = num_channels
        # self.base_env.bandwidth = bandwidth
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_channels = num_channels
        self.bandwidth = bandwidth
        self.base_env.generate_random_uav_layout(num_uav)
        self._apply_spectrum_limits()
        self.observation_radius = observation_radius  # 观测半径，单位米
        self.limit_neighbors = limit_neighbors  # 最大观测邻居数
        
        # 性能记录
        self.interference_prob_history = []
        self.episode_rewards = []  # todo 似乎没用
    
    def get_neighbors(self, agent_node: Node, limit: int = 99) -> List[Node]:
        """获取指定智能体的邻居Node实例列表"""
        neighbors = []
        for other_node in self.base_env.nodes.values():
            if other_node.node_id == agent_node.node_id:
                continue
            distance = math.dist(agent_node.position, other_node.position)
            if distance <= self.observation_radius:
                neighbors.append((distance, other_node))
        neighbors.sort(key=lambda x: x[0])
        return [n[1] for n in neighbors[:limit]]
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """获取所有智能体的观测（统一含schedule信息，维度与 sequential 一致）"""
        observations = {}
        executed_count = sum(1 for n in self.base_env.nodes.values() if n.commit_index >= 0)
        for node in self.base_env.nodes.values():
            neighbor_nodes = self.get_neighbors(node, limit=self.limit_neighbors)
            obs = self._build_observation(
                node, neighbor_nodes,
                include_schedule=True,
                executed_fraction=executed_count / max(1, self.num_uav),
            )
            observations[node.node_id] = obs
        return observations

    def _apply_spectrum_limits(self):
        """将频率上下限与带宽参数同步到所有载荷"""
        freq_lo = self.freq_min
        freq_hi = self.freq_max
        bw = self.bandwidth
        for node in self.base_env.nodes.values():
            tx = node.tx
            rx = node.rx
            tx.min_frequency = freq_lo
            tx.max_frequency = freq_hi
            tx.bandwidth = bw
            rx.min_frequency = freq_lo
            rx.max_frequency = freq_hi
            rx.bandwidth = bw
    
    def _build_observation(
        self,
        node: Node,
        neighbor_nodes: List[Node],
        include_schedule: bool = False,
        executed_fraction: float = 0.0,
    ) -> np.ndarray:
        """构建观测向量，包含自身状态和邻居状态
        每个状态包含以下信息：
            - x / 1000 (归一化坐标，单位km)
            - y / 1000
            - z / 100 (归一化坐标，单位100m)
            - 通信载荷功率 / 5
            - 通信载荷频率 / 2000
            - 通信载荷SINR / 50 (若无通信则为-1)
        """
        # 构建观测向量，包含自身状态和邻居状态
        obs = []
        
        # 自身状态
        obs.extend([
            node.position[0] / self.area_size * 2,  # 归一化x坐标 (km)
            node.position[1] / self.area_size * 2,  # 归一化y坐标 (km)
            (node.position[2] - self.z_center) / self.z_span,   # 归一化z坐标 (100m)
        ])
        
        # 自身载荷状态
        tx = node.tx
        rx = node.rx
        # SINR 在接收机上计算；tx.peer 即配对的接收机
        peer_rx = tx.peer if tx.peer else rx
        sinr_val = peer_rx.sinr if peer_rx and peer_rx.sinr != float('-inf') and peer_rx.sinr != float('inf') else float('-inf')
        obs.extend([
            (tx.power - tx.min_power) / (tx.max_power - tx.min_power),  # 归一化功率
            (tx.frequency - tx.min_frequency) / (tx.max_frequency - tx.min_frequency),  # 归一化频率
            np.tanh((sinr_val - peer_rx.threshold) / 10.0) if sinr_val != float('-inf') else -1.0,  # 归一化SINR
            (rx.frequency - rx.min_frequency) / (rx.max_frequency - rx.min_frequency)  # 接收机归一化频率
        ])

        if include_schedule:
            obs.extend([
                float(np.clip(executed_fraction, 0.0, 1.0)),
            ])
            # 已执行 agent 的频率使用直方图（归一化）
            freq_hist = self._compute_freq_histogram()
            obs.extend(freq_hist.tolist())
            # 全场景频率占用直方图（让 agent 能看到远处干扰源的频率分布）
            all_freq_hist = self._compute_all_freq_histogram()
            obs.extend(all_freq_hist.tolist())
        
        # 邻居状态
        for i in range(self.limit_neighbors):
            if i < len(neighbor_nodes):
                neighbor = neighbor_nodes[i]
                # 邻居存在标志
                obs.append(1.0)  # 邻居存在
                
                # 相对位置
                rel_x = (neighbor.position[0] - node.position[0]) / self.area_size * 2
                rel_y = (neighbor.position[1] - node.position[1]) / self.area_size * 2
                rel_z = (neighbor.position[2] - node.position[2]) / self.z_span
                obs.extend([rel_x, rel_y, rel_z])
                
                # 邻居载荷状态
                tx = neighbor.tx
                rx = neighbor.rx
                obs.extend([
                    (tx.power - tx.min_power) / (tx.max_power - tx.min_power),
                    (tx.frequency - tx.min_frequency) / (tx.max_frequency - tx.min_frequency),
                    np.tanh((tx.peer.sinr - tx.peer.threshold) / 10.0) if tx.peer and tx.peer.sinr != float('-inf') else -1.0,
                    (rx.frequency - rx.min_frequency) / (rx.max_frequency - rx.min_frequency)
                ])
                if include_schedule:
                    committed_flag = 1.0 if neighbor.commit_index >= 0 else 0.0
                    commit_norm = neighbor.commit_index / self.num_uav if committed_flag > 0 else 0.0
                    obs.extend([committed_flag, float(np.clip(commit_norm, 0.0, 1.0))])
            else:
                # 填充空邻居
                obs.append(.0)  # 邻居不存在
                obs.extend([.0, .0, .0, .0, .0, .0, .0])  # 其他值可以为0，因为有存在标志
                if include_schedule:
                    obs.extend([.0, .0])
        
        return np.array(obs, dtype=np.float32)

    def _compute_freq_histogram(self, n_bins: int = None) -> np.ndarray:
        """计算已执行 agent 的频率使用直方图（归一化）。
        
        将频率范围 [freq_min+bw/2, freq_max-bw/2] 均匀分为 n_bins 个 bin，
        统计已提交决策的 agent 中各 bin 的频率使用比例。
        
        Args:
            n_bins: 直方图 bin 数量（默认与动作空间频率离散数对齐）
        Returns:
            (n_bins,) 归一化直方图，和为1（若无已执行agent则全0）
        """
        if n_bins is None:
            from config import ACTION_CONFIG
            n_bins = ACTION_CONFIG["n_freq"]
        freq_lo = self.freq_min + self.bandwidth / 2
        freq_hi = self.freq_max - self.bandwidth / 2
        freq_span = max(1e-6, freq_hi - freq_lo)
        
        hist = np.zeros(n_bins, dtype=np.float32)
        committed_count = 0
        for node in self.base_env.nodes.values():
            if node.commit_index >= 0:
                freq = node.tx.frequency
                bin_idx = int((freq - freq_lo) / freq_span * n_bins)
                bin_idx = int(np.clip(bin_idx, 0, n_bins - 1))
                hist[bin_idx] += 1.0
                committed_count += 1
        
        if committed_count > 0:
            hist /= committed_count
        return hist

    def _compute_all_freq_histogram(self, n_bins: int = None) -> np.ndarray:
        """计算所有 agent 的频率占用直方图（归一化）。
        
        统计场景中所有 UAV 的当前频率分布（不管是否已顺序执行）。
        这让每个 agent 能看到全局频率拥挤度，即使远处干扰源不在 K 邻居中。
        """
        if n_bins is None:
            from config import ACTION_CONFIG
            n_bins = ACTION_CONFIG["n_freq"]
        freq_lo = self.freq_min + self.bandwidth / 2
        freq_hi = self.freq_max - self.bandwidth / 2
        freq_span = max(1e-6, freq_hi - freq_lo)
        
        hist = np.zeros(n_bins, dtype=np.float32)
        count = 0
        for node in self.base_env.nodes.values():
            freq = node.tx.frequency
            bin_idx = int((freq - freq_lo) / freq_span * n_bins)
            bin_idx = int(np.clip(bin_idx, 0, n_bins - 1))
            hist[bin_idx] += 1.0
            count += 1
        
        if count > 0:
            hist /= count
        return hist

    def reset_commit_state(self):
        for node in self.base_env.nodes.values():
            node.commit_index = -1

    def get_sequential_observation(self, agent_id: str, executed_count: int) -> np.ndarray:
        node = self.base_env.nodes[agent_id]
        neighbor_nodes = self.get_neighbors(node, limit=self.limit_neighbors)
        executed_fraction = executed_count / self.num_uav
        return self._build_observation(
            node,
            neighbor_nodes,
            include_schedule=True,
            executed_fraction=executed_fraction,
        )

    def apply_sequential_action(self, agent_id: str, normalized_action: np.ndarray, commit_index: int):
        node = self.base_env.nodes[agent_id]
        action_dict = self._normalize_action(normalized_action)
        self._apply_action(node, action_dict)
        node.commit_index = commit_index
        self.base_env.update_sinr()

    def calculate_rewards(self) -> Dict[str, float]:
        return self._calculate_rewards()

    def get_interference_prob(self) -> float:
        return self.base_env.calc_interf_prob()
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict]:
        """环境步进"""
        # 应用所有智能体的动作
        for agent_id, action in actions.items():
            node = self.base_env.nodes[agent_id]
            action_dict = self._normalize_action(action)
            self._apply_action(node, action_dict)

        self.base_env.update_sinr()
        rewards = self._calculate_rewards()
        next_observations = self.get_observations()
        
        # 记录性能指标
        interference_prob = self.base_env.calc_interf_prob()
        self.interference_prob_history.append(interference_prob)
        
        info = {
            'interference_prob': interference_prob,
            'individual_rewards': rewards
        }
        
        return next_observations, rewards, info
    
    def _normalize_action(self, action: np.ndarray) -> Dict[str, float]:
        """将神经网络输出转换为动作字典"""
        # 假设动作空间为 [power, frequency]
        action = np.clip(action, -1, 1)  # 确保在[-1, 1]范围内
        freq_span = max(1e-6, (self.freq_max - self.freq_min - self.bandwidth))
        freq_center_min = self.freq_min + self.bandwidth / 2
        freq = (action[1] + 1) / 2 * freq_span + freq_center_min

        return {
            'power': (action[0] + 1) / 2 * 5.0,  # 映射到[0, 5]W
            'frequency': freq  # 映射到可用中心频点区间，确保带宽不越界
        }
    
    def _apply_action(self, node: Node, action_dict: Dict[str, float]):
        """将动作应用到节点的载荷上"""
        tx = node.tx
        tx.power = action_dict['power']
        tx.frequency = action_dict['frequency']
        tx.peer.frequency = action_dict['frequency']
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """计算智能体奖励"""
        rewards = {}
        
        # 全局奖励 - 基于互扰概率（降低量级以稳定训练）
        interference_prob = self.base_env.calc_interf_prob()
        global_reward = -20.0 * interference_prob

        sinr_high_margin = 10.0  # dB above threshold allowed without extra penalty
        sinr_high_lambda = 2.0   # penalty weight for excessive SINR
        
        for node in self.base_env.nodes.values():
            reward = global_reward
            
            rx = node.tx.peer
            sinr = rx.sinr if rx.sinr != float('-inf') else rx.threshold - 30.0
            sinr_margin = sinr - rx.threshold

            # 对过高的 SINR 施加惩罚，避免过大功率造成外部干扰
            excessive = max(0.0, sinr_margin - sinr_high_margin)
            high_sinr_penalty = -sinr_high_lambda * excessive

            # 链路项奖励越界越大，未达门限时给予额外惩罚
            link_term = 30.0 * np.tanh(sinr_margin / 5.0)
            threshold_term = 15.0 if sinr_margin >= 0 else -25.0

            # 功率惩罚鼓励使用更低发射功率
            tx = node.tx
            power_ratio = tx.power / max(tx.max_power, 1e-6)
            power_penalty = -5.0 * power_ratio

            # 本地干扰源数量提示当前节点周围的拥挤程度
            interference_sources = len(rx.interference_sources) if hasattr(rx, 'interference_sources') else 0
            local_penalty = -3.0 * min(1.0, interference_sources / 3.0)

            reward += link_term + threshold_term + power_penalty + local_penalty + high_sinr_penalty

            rewards[node.node_id] = reward
        return rewards
    
    def load_layout(self, layout: dict) -> Dict[str, np.ndarray]:
        """加载固定布局，用于公平评估。layout 格式: {positions: [...], pairing: [[tx_idx, rx_idx], ...]}"""
        self.base_env.reset()
        positions = layout["positions"]
        for idx, pos in enumerate(positions):
            node = Node(node_id=f"uav{idx}", position=tuple(pos))
            node._add_payloads()
            self.base_env.add_node(node)
        self._apply_spectrum_limits()

        # 加载固定配对
        pairing = layout.get("pairing", [])
        for tx_idx, rx_idx in pairing:
            tx_node = self.base_env.nodes[f"uav{tx_idx}"]
            rx_node = self.base_env.nodes[f"uav{rx_idx}"]
            tx_node.tx.set_peer(rx_node.rx)

        # 随机初始化载荷参数（评估时会被具体方法覆盖）
        self._random_initialize_payloads()
        self.base_env.update_sinr()
        self.reset_commit_state()
        self.interference_prob_history.clear()
        self.episode_rewards.clear()
        return self.get_observations()

    @property
    def obs_dim(self) -> int:
        """返回单agent观测维度（含schedule信息）"""
        return len(self._build_observation(
            list(self.base_env.nodes.values())[0],
            self.get_neighbors(list(self.base_env.nodes.values())[0], limit=self.limit_neighbors),
            include_schedule=True,
            executed_fraction=0.0,
        ))

    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        # 重置环境
        self.base_env.reset()
        self.base_env.generate_random_uav_layout(self.num_uav)
        self._apply_spectrum_limits()
        self.reset_commit_state()
        
        # 清空历史记录
        self.interference_prob_history.clear()
        self.episode_rewards.clear()
        
        # 随机初始化载荷参数
        self._random_initialize_payloads()
        
        # 更新环境状态
        self.base_env.update_sinr()
        
        # 获取初始观测
        observations = self.get_observations()
        
        return observations

    def set_layout(
        self,
        positions: List[Tuple[float, float, float]],
        randomize_payloads: bool = True,
        pair_range: float = 500.0,
        payload_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, np.ndarray]:
        """使用指定坐标强制设置布局，方便可视化或定制评估"""
        if len(positions) != self.num_uav:
            raise ValueError(f"positions数量{len(positions)}与num_uav {self.num_uav}不一致")

        self.base_env.reset()
        for idx, pos in enumerate(positions):
            node = Node(node_id=f"uav{idx}", position=pos)
            node._add_payloads()
            self.base_env.add_node(node)

        self._apply_spectrum_limits()

        # 重新配对链路
        self.base_env._match(pair_range)

        if payload_overrides:
            for node_id, overrides in payload_overrides.items():
                if node_id not in self.base_env.nodes:
                    continue
                tx = self.base_env.nodes[node_id].tx
                if 'power' in overrides:
                    tx.power = float(overrides['power'])
                if 'frequency' in overrides:
                    tx.frequency = float(overrides['frequency'])
                    tx.peer.frequency = float(overrides['frequency'])
        elif randomize_payloads:
            self._random_initialize_payloads()
        else:
            for node in self.base_env.nodes.values():
                tx = node.tx
                tx.power = tx.max_power * 0.5
                freq_low = self.freq_min + self.bandwidth / 2
                freq_high = self.freq_max - self.bandwidth / 2
                tx.frequency = (freq_high + freq_low) / 2
                tx.peer.frequency = tx.frequency

        self.base_env.update_sinr()
        self.interference_prob_history.clear()
        self.episode_rewards.clear()
        self.reset_commit_state()
        return self.get_observations()
    
    def _random_initialize_payloads(self):
        """随机初始化载荷参数"""
        # 定义动作边界
        for node in self.base_env.nodes.values():
            tx = node.tx
            tx.power = np.random.uniform(tx.min_power, tx.max_power)
            freq_low = self.freq_min + self.bandwidth / 2
            freq_high = self.freq_max - self.bandwidth / 2
            tx.frequency = np.random.uniform(freq_low, freq_high)
            tx.peer.frequency = tx.frequency
    
    def render(self, save_path: Optional[str] = None):
        """可视化当前环境状态"""
        if save_path:
            self.base_env.plot_layout(save_path)
        else:
            self.base_env.plot_layout()
    
    def describe_layout(self) -> None:
        nodes = list(self.base_env.nodes.values())
        area_size = self.base_env.area_size
        num_uav = len(nodes)
        density = num_uav / (area_size / 1000) ** 2
        logger.debug(
            "区域大小%dx%d米，节点%d个，面密度%.2f个每平方千米。" %
            (area_size, area_size, num_uav, density)
        )
        avg_neighbors = np.mean([
            len(self.get_neighbors(node))
            for node in nodes
        ])
        avg_observable = np.mean([
            len(self.get_neighbors(node, limit=self.limit_neighbors))
            for node in nodes
        ])
        logger.debug(
            "观测半径%.1f米，节点平均邻居%.2f个，平均可观测%.2f个。" %
            (self.observation_radius, avg_neighbors, avg_observable)
        )
        nearest_distances = []
        for node_a in nodes:
            min_distance = float("inf")
            for node_b in nodes:
                if node_a is node_b:
                    continue
                dist = math.dist(node_a.position, node_b.position)
                min_distance = min(min_distance, dist)
            nearest_distances.append(min_distance)
        logger.debug("平均最近邻距离为%.2f米。" % np.mean(nearest_distances))