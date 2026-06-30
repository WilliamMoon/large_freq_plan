from typing import List, Literal, Tuple
import numpy as np

class Node:
    """电磁节点类，表示无人机或其他设备"""
    def __init__(self, node_id: str, position: Tuple[float, float, float]):
        self.node_id = node_id
        self.position = position
        self.speed = 20  # 无人机速度，单位m/s
        self.commit_index = -1  # 顺序调度时记录执行次序

    def _add_payloads(self):
        """为节点添加通信载荷（发射机和接收机）"""
        tx = Payload(
            pid=f"{self.node_id}_tx",
            mode='tx'
        )
        tx.node = self
        self.tx = tx

        rx = Payload(
            pid=f"{self.node_id}_rx",
            mode='rx'
        )
        rx.node = self
        self.rx = rx


class Payload:
    """载荷类"""
    def __init__(
            self, 
            pid: str, 
            mode: Literal['tx', 'rx'], 
            frequency: float = .0, max_frequency: float = 1300.0, min_frequency: float = 1200.0, 
            power: float = .0, max_power: float = 5.0, min_power: float = 0.0,
            antenna_gain: float = .0,
            threshold: float = 0.0
        ):
        self.pid = pid  # 载荷ID
        self.mode = mode  # 载荷模式：tx（发射机）或rx（接收机）
        self.node: 'Node'  # 载荷所属节点

        self.frequency: float = frequency  # 单位MHz
        self.max_frequency: float = max_frequency  # 最大频率，单位MHz
        self.min_frequency: float = min_frequency  # 最小频率，单位MHz
        self.bandwidth = 10.0  # 带宽，单位MHz

        self.power: float = power if power != 0 else 5  # 发射功率，单位W
        self.max_power: float = max_power if max_power != 0 else 5  # 最大发射功率，单位W
        self.min_power: float = min_power if min_power != 0 else 0  # 最小发射功率，单位W

        self.antenna_gain: float = antenna_gain  # 主瓣增益，单位dB
        self.threshold: float = threshold  # SINR阈值
        self.band_interf_mw = np.zeros(1)  # 各频段干扰功率，单位mW
        self.band_interf_dbm = np.zeros(1)  # 各频段干扰功率，单位dBm

        # 接收机
        self.signal_dbm = float('-inf')  # 信号功率，初始化为负无穷
        self.interference_dbm = float('-inf')  # 干扰功率，初始化为负无穷
        self.interference_sources: List[dict] = []  # 记录干扰源 [{'id': ..., 'power_dbm': ...}, ...]
        self.sinr: float = float('-inf')  # 信噪比
        self.status: str = ''  # 用于说明载荷受扰状态
        self.peer: 'Payload'

    def set_peer(self, peer: 'Payload'):
        """设置载荷的对端或目标"""
        self.peer = peer
        peer.peer = self
