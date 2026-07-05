"""动态场景下的基线方法。

1. Fixed-Orthogonal: 初始时正交分配频率，之后不随拓扑变化调整
2. Periodic Greedy: 每隔 N 个时隙用 greedy 重新规划一次
3. Per-slot Greedy: 每个时隙都用 greedy 规划（Oracle 上界）
4. Random: 每个时隙随机选择
"""
import numpy as np
import time
from typing import Dict, List, Tuple, Callable

from config import ENV_CONFIG, ACTION_CONFIG, LoggerSingleton
from dynamic_env import DynamicMultiAgentEnv
from test_cgreedy import _enumerate_best_frequency_sequential

logger = LoggerSingleton.get_instance()


def run_fixed_orthogonal(env: DynamicMultiAgentEnv) -> List[float]:
    """固定正交分配：初始时轮流分配 11 个正交信道，之后不调整。

    功率固定为中等值（2.5W）。
    """
    env.reset()
    # 正交信道频率
    freq_lo = env.freq_min + env.bandwidth / 2
    freq_hi = env.freq_max - env.bandwidth / 2
    n_orthogonal = int((freq_hi - freq_lo) / env.bandwidth)
    orthogonal_freqs = np.linspace(freq_lo, freq_hi, n_orthogonal)
    fixed_power = ENV_CONFIG["max_power"] * 0.5  # 2.5W

    # 初始分配
    node_ids = list(env.base_env.nodes.keys())
    for i, node_id in enumerate(node_ids):
        node = env.base_env.nodes[node_id]
        freq = orthogonal_freqs[i % n_orthogonal]
        node.tx.power = fixed_power
        node.tx.frequency = freq
        if node.tx.peer:
            node.tx.peer.frequency = freq

    slot_probs = []
    done = False
    while not done:
        env.base_env.update_sinr()
        probs = env.base_env.calc_interf_prob()
        slot_probs.append(probs)

        # 动作不变，只推进环境
        actions = {}
        for node_id in node_ids:
            actions[node_id] = np.array([fixed_power, orthogonal_freqs[0]], dtype=np.float32)
        # 用 step 但动作就是保持不变
        obs, rewards, info, done = env.step(actions)

    return slot_probs


def run_random_dynamic(env: DynamicMultiAgentEnv, seed: int = 42) -> List[float]:
    """每个时隙随机选择频率和功率。"""
    rng = np.random.default_rng(seed)
    env.reset()

    freq_lo = env.freq_min + env.bandwidth / 2
    freq_hi = env.freq_max - env.bandwidth / 2

    slot_probs = []
    done = False
    while not done:
        actions = {}
        for node_id in env.base_env.nodes:
            power = float(rng.uniform(0, ENV_CONFIG["max_power"]))
            freq = float(rng.uniform(freq_lo, freq_hi))
            # 归一化到 [-1, 1]
            power_norm = (power / ENV_CONFIG["max_power"]) * 2 - 1
            freq_norm = ((freq - freq_lo) / (freq_hi - freq_lo)) * 2 - 1
            actions[node_id] = np.array([power_norm, freq_norm], dtype=np.float32)

        obs, rewards, info, done = env.step(actions)
        slot_probs.append(info['interference_prob'])

    return slot_probs


def run_periodic_greedy(env: DynamicMultiAgentEnv, replan_interval: int = 5,
                         greedy_mode: str = "sequential") -> Tuple[List[float], float]:
    """周期性贪心重规划。

    每隔 replan_interval 个时隙，用 greedy 重新分配频率/功率。
    间隔期内保持上次分配不变。

    Args:
        replan_interval: 重规划间隔（时隙数）。1 = 每时隙都重规划（Per-slot Greedy）
        greedy_mode: "sequential" 或 "central"
    Returns:
        (每时隙互扰概率列表, 总规划耗时秒)
    """
    env.reset()

    # 贪心候选
    n_freq = ACTION_CONFIG["n_freq"]
    n_power = ACTION_CONFIG["n_power"]
    freq_lo = env.freq_min + env.bandwidth / 2
    freq_hi = env.freq_max - env.bandwidth / 2
    freq_candidates = np.linspace(freq_lo, freq_hi, n_freq)
    power_candidates = np.linspace(0, ENV_CONFIG["max_power"], n_power)

    slot_probs = []
    total_planning_time = 0.0
    done = False

    while not done:
        # 检查是否需要重规划
        if env.current_slot % replan_interval == 0:
            t_start = time.time()
            env.base_env.update_sinr()
            env.reset_commit_state()

            if greedy_mode == "sequential":
                _enumerate_best_frequency_sequential(
                    env, freq_candidates, power_candidates,
                    shuffle_order=True, neighbor_sample=None,
                    freq_sample=None, power_sample=None,
                )
            else:
                from test_cgreedy import _enumerate_best_frequency_central
                _enumerate_best_frequency_central(env, freq_candidates, power_candidates)

            total_planning_time += time.time() - t_start

        # 应用当前分配（保持不变），推进环境
        actions = {}
        for node_id, node in env.base_env.nodes.items():
            tx = node.tx
            power_norm = (tx.power / ENV_CONFIG["max_power"]) * 2 - 1
            freq_norm = ((tx.frequency - freq_lo) / (freq_hi - freq_lo)) * 2 - 1
            actions[node_id] = np.array([power_norm, freq_norm], dtype=np.float32)

        obs, rewards, info, done = env.step(actions)
        slot_probs.append(info['interference_prob'])

    return slot_probs, total_planning_time


def run_per_slot_greedy(env: DynamicMultiAgentEnv, greedy_mode: str = "sequential") -> Tuple[List[float], float]:
    """每时隙都用 greedy 重新规划（Oracle 上界）。"""
    return run_periodic_greedy(env, replan_interval=1, greedy_mode=greedy_mode)


def run_marl_dynamic(env: DynamicMultiAgentEnv, policy_fn: Callable,
                      evaluate: bool = True) -> Tuple[List[float], float]:
    """用 MARL 策略运行动态 episode。

    Args:
        env: 动态环境
        policy_fn: callback(env) -> actions_dict
                   每个时隙调用，返回 {agent_id: [power_norm, freq_norm]}
        evaluate: 评估模式
    Returns:
        (每时隙互扰概率列表, 总推理耗时秒)
    """
    env.reset()
    slot_probs = []
    total_inference_time = 0.0
    done = False

    while not done:
        t_start = time.time()
        actions = policy_fn(env)
        total_inference_time += time.time() - t_start
        obs, rewards, info, done = env.step(actions)
        slot_probs.append(info['interference_prob'])

    return slot_probs, total_inference_time
