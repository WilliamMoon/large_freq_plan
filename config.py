"""集中管理配置：日志、超参数、路径"""
import logging
import os
import sys


class LoggerSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = logging.getLogger("freq_plan")
            cls._instance.setLevel(logging.DEBUG)
            if not cls._instance.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(
                    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
                )
                cls._instance.addHandler(handler)
        return cls._instance


# ============================================================
# 环境参数
# ============================================================
ENV_CONFIG = {
    "area_size": 2000.0,
    "z_center": 100.0,
    "z_span": 100.0,
    "freq_min": 1200.0,
    "freq_max": 1300.0,
    "num_channels": 11,
    "bandwidth": 10.0,
    "max_power": 5.0,
    "noise_floor": -90,          # dBm
    "sinr_threshold": 0.0,       # dB
    "effective_range": 500.0,    # m, 配对范围
    "observation_radius": 600.0, # m
    "limit_neighbors": 5,
}

# ============================================================
# 动作空间（离散化）
# ============================================================
ACTION_CONFIG = {
    "n_power": 10,   # 功率离散档数
    "n_freq": 20,    # 频率离散档数
}

# ============================================================
# MADDPG 训练超参数
# ============================================================
MADDPG_CONFIG = {
    "actor_lr": 3e-4,
    "critic_lr": 1e-3,
    "gamma": 0.95,
    "tau": 0.005,              # soft update
    "batch_size": 192,
    "buffer_capacity": 150_000,
    "hidden_dim": 128,
    "tau_init": 1.5,           # Gumbel-Softmax 初始温度
    "tau_final": 0.05,         # 最终温度
    "tau_decay": 5.0,          # 温度衰减系数
    "warmup_episodes": 50,     # 热身轮数（按规模缩放）
    "update_interval": 5,      # 每隔多少step更新一次
    "eval_interval": 25,       # 每隔多少episode评估一次
    "noise_scale": 0.1,        # OU噪声或高斯噪声
}

# ============================================================
# 评估参数
# ============================================================
EVAL_CONFIG = {
    "num_eval_layouts": 50,
    "seed": 42,
}

# ============================================================
# 路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LAYOUTS_DIR = os.path.join(PROJECT_ROOT, "eval_layouts")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

for _d in [RESULTS_DIR, LAYOUTS_DIR, CHECKPOINT_DIR]:
    os.makedirs(_d, exist_ok=True)
