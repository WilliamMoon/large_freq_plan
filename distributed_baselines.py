"""传统分布式基线算法。

1. Distributed Greedy: 每个 UAV 用局部信息（自身+K邻居SINR）贪心选最优频率/功率
   - 类似 GADIA (Larsson 2010) 的思路，但用正交信道离散选择
   - 每个 UAV 只需知道邻居的频率和自身 SINR，不需要全局信息

2. JAR (Jamming Avoidance Response): 经典分布式避障算法
   - 受电鱼避障启发的增量信道切换
   - 仅当切换到相邻信道能显著提升信道质量时才切换
   - 参考: Cohen et al. 2024, IEEE TCCN

3. Distributed Graph Coloring (局部): 基于局部干扰图的贪心着色
   - 每个 UAV 只知道通信范围内的邻居，构建局部干扰图
   - 贪心选择与已分配邻居不同的颜色（频率）
"""
import numpy as np
import time
from typing import Dict, List, Tuple

from config import ENV_CONFIG, ACTION_CONFIG, LoggerSingleton
from dynamic_env import DynamicMultiAgentEnv

logger = LoggerSingleton.get_instance()


def _get_local_neighbors(env, node, radius=None):
    """获取通信范围内的邻居节点。"""
    if radius is None:
        radius = ENV_CONFIG["observation_radius"]
    import math
    neighbors = []
    for other in env.base_env.nodes.values():
        if other.node_id == node.node_id:
            continue
        dist = math.dist(node.position, other.position)
        if dist <= radius:
            neighbors.append((dist, other))
    neighbors.sort(key=lambda x: x[0])
    return [n[1] for n in neighbors]


def run_distributed_greedy(env: DynamicMultiAgentEnv) -> List[float]:
    """分布式贪心：每个 UAV 用局部信息贪心选最优频率/功率。

    流程：
    1. 所有 UAV 同时（或随机顺序）决策
    2. 每个 UAV 遍历所有正交频率，选择使自身 SINR 最大的频率
    3. 只需知道邻居的当前频率（通过局部感知），不需要全局信息
    4. 迭代多轮直到收敛或达到最大轮数
    """
    env.reset()

    n_freq = ACTION_CONFIG["n_freq"]
    n_power = ACTION_CONFIG["n_power"]
    freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
    freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
    freq_candidates = np.linspace(freq_lo, freq_hi, n_freq)
    power_candidates = np.linspace(0, ENV_CONFIG["max_power"], n_power)

    slot_probs = []
    done = False
    max_iter = 5  # 每个时隙最多迭代5轮

    while not done:
        # 迭代式局部贪心
        for iteration in range(max_iter):
            changed = False
            node_ids = list(env.base_env.nodes.keys())
            np.random.shuffle(node_ids)

            for node_id in node_ids:
                node = env.base_env.nodes[node_id]
                tx = node.tx
                rx = tx.peer
                if rx is None:
                    continue

                best_sinr = rx.sinr if rx.sinr != float('-inf') else -999
                best_freq = tx.frequency
                best_power = tx.power

                # 遍历所有频率和功率组合，找局部最优
                for freq in freq_candidates:
                    for power in power_candidates:
                        # 临时设置
                        old_freq = tx.frequency
                        old_power = tx.power
                        old_rx_freq = rx.frequency
                        tx.frequency = freq
                        tx.power = power
                        rx.frequency = freq
                        env.base_env.update_sinr()

                        new_sinr = rx.sinr if rx.sinr != float('-inf') else -999
                        if new_sinr > best_sinr + 0.1:  # 有显著提升才切换
                            best_sinr = new_sinr
                            best_freq = freq
                            best_power = power
                            changed = True

                        # 恢复
                        tx.frequency = old_freq
                        tx.power = old_power
                        rx.frequency = old_rx_freq

                # 应用最优
                tx.frequency = best_freq
                tx.power = best_power
                rx.frequency = best_freq

            env.base_env.update_sinr()
            if not changed:
                break  # 收敛

        # 推进环境
        actions = {}
        for node_id, node in env.base_env.nodes.items():
            tx = node.tx
            power_norm = (tx.power / ENV_CONFIG["max_power"]) * 2 - 1
            freq_norm = ((tx.frequency - freq_lo) / (freq_hi - freq_lo)) * 2 - 1
            actions[node_id] = np.array([power_norm, freq_norm], dtype=np.float32)

        obs, rewards, info, done = env.step(actions)
        slot_probs.append(info['interference_prob'])

    return slot_probs


def run_jar(env: DynamicMultiAgentEnv, improvement_threshold: float = 0.05) -> List[float]:
    """JAR (Jamming Avoidance Response) 分布式避障算法。

    参考: Cohen et al. 2024, "SINR-Aware DRL for Distributed DCA"

    流程：
    1. 初始随机分配频率
    2. 每个 UAV 检测当前信道质量（SINR）
    3. 如果切换到相邻信道能将 SINR 提升超过阈值，则切换
    4. 只需局部感知自身 SINR，不需要全局信息
    """
    env.reset()

    n_freq = ACTION_CONFIG["n_freq"]
    freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
    freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
    freq_candidates = np.linspace(freq_lo, freq_hi, n_freq)

    # 初始随机分配
    for node in env.base_env.nodes.values():
        freq = np.random.choice(freq_candidates)
        node.tx.frequency = freq
        node.tx.power = ENV_CONFIG["max_power"] * 0.5  # 固定中等功率
        if node.tx.peer:
            node.tx.peer.frequency = freq

    slot_probs = []
    done = False

    while not done:
        env.base_env.update_sinr()

        # 每个 UAV 检查是否需要切换到相邻信道
        for node in env.base_env.nodes.values():
            tx = node.tx
            rx = tx.peer
            if rx is None:
                continue

            current_sinr = rx.sinr if rx.sinr != float('-inf') else -999
            current_freq_idx = int(np.argmin(np.abs(freq_candidates - tx.frequency)))

            # 检查相邻信道（±1, ±2）
            best_sinr = current_sinr
            best_freq = tx.frequency
            for offset in [-2, -1, 1, 2]:
                new_idx = current_freq_idx + offset
                if 0 <= new_idx < n_freq:
                    new_freq = freq_candidates[new_idx]
                    old_freq = tx.frequency
                    old_rx_freq = rx.frequency
                    tx.frequency = new_freq
                    rx.frequency = new_freq
                    env.base_env.update_sinr()
                    new_sinr = rx.sinr if rx.sinr != float('-inf') else -999

                    # 归一化 SINR 到 [0,1] 作为信道质量
                    cq_current = 1.0 / (1.0 + np.exp(-current_sinr / 10))
                    cq_new = 1.0 / (1.0 + np.exp(-new_sinr / 10))

                    if cq_new - cq_current > improvement_threshold:
                        best_sinr = new_sinr
                        best_freq = new_freq

                    # 恢复
                    tx.frequency = old_freq
                    rx.frequency = old_rx_freq

            tx.frequency = best_freq
            rx.frequency = best_freq

        env.base_env.update_sinr()

        # 推进环境
        actions = {}
        for node_id, node in env.base_env.nodes.items():
            tx = node.tx
            power_norm = (tx.power / ENV_CONFIG["max_power"]) * 2 - 1
            freq_norm = ((tx.frequency - freq_lo) / (freq_hi - freq_lo)) * 2 - 1
            actions[node_id] = np.array([power_norm, freq_norm], dtype=np.float32)

        obs, rewards, info, done = env.step(actions)
        slot_probs.append(info['interference_prob'])

    return slot_probs


def run_distributed_coloring(env: DynamicMultiAgentEnv) -> List[float]:
    """分布式图着色：基于局部干扰图的贪心着色。

    流程：
    1. 每个 UAV 只知道通信范围内的邻居
    2. 随机顺序决策，选择与已分配邻居频率不同的信道
    3. 如果所有正交信道都被占用，选干扰最小的
    """
    env.reset()

    n_freq = ACTION_CONFIG["n_freq"]
    freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
    freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
    freq_candidates = np.linspace(freq_lo, freq_hi, n_freq)
    fixed_power = ENV_CONFIG["max_power"] * 0.5

    slot_probs = []
    done = False

    while not done:
        # 顺序着色
        env.reset_commit_state()
        node_ids = list(env.base_env.nodes.keys())
        np.random.shuffle(node_ids)

        for exec_idx, node_id in enumerate(node_ids):
            node = env.base_env.nodes[node_id]
            tx = node.tx
            rx = tx.peer
            if rx is None:
                continue

            # 获取局部邻居
            neighbors = _get_local_neighbors(env, node, radius=ENV_CONFIG["effective_range"])

            # 找到已被占用的频率
            used_freqs = set()
            for neighbor in neighbors:
                if neighbor.commit_index >= 0:
                    used_freqs.add(neighbor.tx.frequency)

            # 选择未被占用的频率
            available = [f for f in freq_candidates if f not in used_freqs]
            if available:
                best_freq = np.random.choice(available)
            else:
                # 所有频率都被占用，选 SINR 最好的
                best_sinr = -999
                best_freq = freq_candidates[0]
                for freq in freq_candidates:
                    old_freq = tx.frequency
                    old_rx = rx.frequency
                    tx.frequency = freq
                    rx.frequency = freq
                    env.base_env.update_sinr()
                    sinr = rx.sinr if rx.sinr != float('-inf') else -999
                    if sinr > best_sinr:
                        best_sinr = sinr
                        best_freq = freq
                    tx.frequency = old_freq
                    rx.frequency = old_rx

            tx.frequency = best_freq
            tx.power = fixed_power
            rx.frequency = best_freq
            node.commit_index = exec_idx

        env.base_env.update_sinr()

        # 推进环境
        actions = {}
        for node_id, node in env.base_env.nodes.items():
            tx = node.tx
            power_norm = (tx.power / ENV_CONFIG["max_power"]) * 2 - 1
            freq_norm = ((tx.frequency - freq_lo) / (freq_hi - freq_lo)) * 2 - 1
            actions[node_id] = np.array([power_norm, freq_norm], dtype=np.float32)

        obs, rewards, info, done = env.step(actions)
        slot_probs.append(info['interference_prob'])

    return slot_probs
