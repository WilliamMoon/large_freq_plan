import math
import numpy as np
from node import Node, Payload
from typing import List, Union, Tuple, Dict, Optional
from config import LoggerSingleton
import random
import numpy as np

logger = LoggerSingleton.get_instance()

class Env:
    def __init__(self, area_size: float = 2000.0, z_center: float = 100.0, z_span: float = 100.0):
        self.area_size = area_size  # 环境区域大小（米）
        self.z_center = z_center
        self.z_span = z_span
        self.nodes: Dict[str, Node] = {}  # 改为字典，key为node_id，value为Node实例
        self.receivers: List[Payload] = []
        self.transmitters: List[Payload] = []  # 通信发射机
        self.noise_floor = -90  # 噪声底，单位dBm
        self.interf_receivers: List[Payload] = []
        self.freq_min: float
        self.freq_max: float
        self.num_channels: int

    def add_node(self, node: Node):
        """添加节点到环境"""
        self.nodes[node.node_id] = node  # 使用node_id作为key添加到字典
        self.transmitters.append(node.tx)
        self.receivers.append(node.rx)

    def reset(self):
        self.nodes.clear()
        self.receivers.clear()
        self.transmitters.clear()

    def remove_node(self, node_id: str) -> bool:
        """移除指定节点，返回是否成功移除"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            del self.nodes[node_id]
            # 同时移除相关的发射机和接收机
            self.transmitters = [tx for tx in self.transmitters if tx.node != node]
            self.receivers = [rx for rx in self.receivers if rx.node != node]
            return True
        return False

    def generate_random_uav_layout(self, num_uavs: int = 10, effective_range: float = 500.0):
        """
        随机部署指定数量的无人机节点，并在有效通信范围内匹配发射机和接收机。

        Args:
            num_uavs: 无人机数量
            effective_range: 有效通信范围（米），优先在此范围内匹配
        """

        # 清空现有节点
        self.reset()

        # 生成随机位置的无人机节点
        for i in range(num_uavs):
            node_id = f"uav{i}"

            # 随机生成位置
            x = random.uniform(-self.area_size / 2, self.area_size / 2)
            y = random.uniform(-self.area_size / 2, self.area_size / 2)
            z = random.uniform(self.z_center - self.z_span / 2, self.z_center + self.z_span / 2)
            position = (x, y, z)

            # 创建无人机节点
            node = Node(node_id=node_id, position=position)
            node._add_payloads()
            self.add_node(node)

        # 匹配较近的发射机和接收机
        self._match(effective_range)

    def _match(self, effective_range: float):
        """
        匹配通信发射机和接收机。
        优先在有效通信范围内匹配，剩余的随机匹配。

        Args:
            effective_range: 有效通信范围（米）
        """
        # 记录已配对的设备
        paired_tx = set()
        paired_rx = set()
        # 记录已建立连接的节点对，避免双向连接
        connected_node_pairs = set()

        # 按距离分类发射机和接收机
        in_range_pairs: List[Tuple[Payload, Payload]] = []
        out_of_range_pairs: List[Tuple[Payload, Payload]] = []

        for tx in self.transmitters:
            for rx in self.receivers:
                # 避免同一节点的发射机和接收机配对
                if tx.node == rx.node:
                    continue

                dist = math.dist(tx.node.position, rx.node.position)
                pair = (tx, rx)

                if dist <= effective_range:
                    in_range_pairs.append(pair)
                else:
                    out_of_range_pairs.append(pair)

        # 优先匹配有效范围内的发射机和接收机
        random.shuffle(in_range_pairs) 
        for tx, rx in in_range_pairs:
            # 检查是否已配对
            if tx in paired_tx or rx in paired_rx:
                continue

            node_pair_1 = (tx.node.node_id, rx.node.node_id)
            node_pair_2 = (rx.node.node_id, tx.node.node_id)

            # 检查节点对是否已连接，避免双向连接
            if node_pair_1 not in connected_node_pairs and node_pair_2 not in connected_node_pairs:
                tx.set_peer(rx)
                paired_tx.add(tx)
                paired_rx.add(rx)
                connected_node_pairs.add(node_pair_1)

        # 随机匹配剩余的发射机和接收机
        random.shuffle(out_of_range_pairs)
        for tx, rx in out_of_range_pairs:
            if tx in paired_tx or rx in paired_rx:
                continue

            node_pair_1 = (tx.node.node_id, rx.node.node_id)
            node_pair_2 = (rx.node.node_id, tx.node.node_id)

            if node_pair_1 not in connected_node_pairs and node_pair_2 not in connected_node_pairs:
                tx.set_peer(rx)
                paired_tx.add(tx)
                paired_rx.add(rx)
                connected_node_pairs.add(node_pair_1)

        # 辅助函数：配对并记录
        def _force_pair(tx: Payload, rx: Payload):
            tx.set_peer(rx)
            paired_tx.add(tx)
            paired_rx.add(rx)
            connected_node_pairs.add((tx.node.node_id, rx.node.node_id))

        # 尝试忽略重复连接限制，为剩余的发射机寻找不同节点的接收机
        remaining_txs = [tx for tx in self.transmitters if tx not in paired_tx]
        remaining_rxs = [rx for rx in self.receivers if rx not in paired_rx]
        for tx in remaining_txs:
            candidate = None
            for rx in remaining_rxs:
                # 存在不同节点的接收机，就匹配
                if rx.node != tx.node and rx not in paired_rx:
                    candidate = rx
                    break
            # 不存在不同节点的接收机，就强制匹配第一个剩余接收机
            if candidate is None and remaining_rxs:
                candidate = remaining_rxs[0]
            if candidate is not None:
                _force_pair(tx, candidate)
                remaining_rxs.remove(candidate)

    def calc_pathloss_db(self, tx: Payload, rx: Union[Payload, Node]):
        if isinstance(rx, Payload):
            d = math.dist(tx.node.position, rx.node.position) / 1000
        else:
            d = math.dist(tx.node.position, rx.position) / 1000  # 转换为km
        freq = tx.frequency
        # 同一节点的设备距离设置为0.5m
        if d == 0:
            d = 0.0005
        # 自由空间路径损耗，d为距离（km），freq为频率（MHz）
        fspl = 20 * math.log10(d) + 20 * math.log10(freq) + 32.44
        return fspl
    
    def calc_overlap_freq(self, tx: Payload, rx: Payload):
        if rx == tx:
            return tx.bandwidth
        f1_min = tx.frequency - tx.bandwidth / 2
        f1_max = tx.frequency + tx.bandwidth / 2
        f2_min = rx.frequency - rx.bandwidth / 2
        f2_max = rx.frequency + rx.bandwidth / 2
        overlap_min = max(f1_min, f2_min)
        overlap_max = min(f1_max, f2_max)
        return max(0, overlap_max - overlap_min)
    
    def calc_signal_dbm(self, tx: Payload, rx: Payload):
        '''计算从发射机到接收机的信号功率（dBm）。
        如果发射机没有功率或者频谱无重叠，则返回负无穷。'''
        # 发射机没开机
        if tx.power <= 0:
            return float('-inf')
        # 计算重叠频谱
        overlap_ratio = self.calc_overlap_freq(tx, rx) / tx.bandwidth
        if overlap_ratio <= 0:
            return float('-inf')
        # 假设功率谱密度均匀分布
        pt_dbm = 10 * math.log10(tx.power * 1000 * overlap_ratio)
        pl = self.calc_pathloss_db(tx, rx)
        gt = tx.antenna_gain
        gr = rx.antenna_gain
        return pt_dbm + gt - pl + gr

    def update_sinr(self):
        """
        计算所有节点的接收机载荷的SINR。
        返回: dict {receiver: SINR}
        """
        # 频段桶用于按训练频点栅格聚合干扰功率
        freq_min = getattr(self, "freq_min", 1200.0)
        freq_max = getattr(self, "freq_max", 1300.0)
        num_channels = max(1, int(getattr(self, "num_channels", 1)))
        band_edges = np.linspace(freq_min, freq_max, num_channels + 1)
        # 遍历所有我方接收机
        for rx in self.receivers:
            rx.signal_dbm = float('-inf')
            rx.interference_dbm = float('-inf')
            rx.interference_sources = []  # 清空干扰源列表
            rx.sinr = float('-inf')
            interf_mw = 0
            band_interf_mw = np.zeros(num_channels, dtype=float)

            # 检查接收机是否有配对的发射机
            if not hasattr(rx, 'peer') or not rx.peer:
                rx.status = 'unpaired'
                rx.sinr = float('inf')
                rx.band_interf_mw = band_interf_mw
                continue

            # 计算信号和干扰
            for tx in self.transmitters:
                # 信号
                if tx == rx.peer:
                    rx.signal_dbm = self.calc_signal_dbm(tx, rx)
                    continue
                # 干扰
                interf_dbm = self.calc_signal_dbm(tx, rx)
                if interf_dbm == float('-inf'):
                    continue
                rx.interference_sources.append({
                    'id': tx.pid,
                    'power_dbm': interf_dbm
                })
                interf_mw += 10 ** (interf_dbm / 10)
                # 将干扰功率累加到对应频段桶
                freq = tx.frequency
                # 找到所在频段索引
                band_idx = np.searchsorted(band_edges, freq, side='right') - 1
                band_idx = int(np.clip(band_idx, 0, num_channels - 1))
                band_interf_mw[band_idx] += 10 ** (interf_dbm / 10)
            rx.interference_dbm = 10 * math.log10(interf_mw) if interf_mw > 0 else float('-inf')
            rx.band_interf_mw = band_interf_mw
            rx.band_interf_dbm = np.where(
                band_interf_mw > 0,
                10 * np.log10(band_interf_mw),
                float('-inf')
            )

            # 计算SINR
            rx.sinr = rx.signal_dbm - 10 * math.log10(
                10 ** (rx.interference_dbm / 10) + 10 ** (self.noise_floor / 10))

    def calc_interf_prob(self):
        """
        计算当前环境下的互扰概率。
        返回: float 互扰概率
        """
        interf_count = 0
        for rx in self.receivers:
            if rx.sinr < rx.threshold:
                interf_count += 1
        return interf_count / len(self.receivers)

    def get_sinr(self) -> Dict[str, float]:
        """
        获取所有接收机的SINR值。
        返回: dict {node_id: SINR}
        """
        sinr_dict = {}
        for rx in self.receivers:
            sinr_dict[rx.node.node_id] = rx.sinr
        return sinr_dict
    
    def plot_layout(self, save_name=None, show_only_sinr=False):
        """
        绘制节点布局图
        
        Args:
            save_name: 保存文件名，如果为None则直接显示
            show_only_sinr: 是否只显示SINR信息（减少文字重叠）
        """
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        x = [node.position[0] for node in self.nodes.values()]
        y = [node.position[1] for node in self.nodes.values()]

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c='black', marker='o', label='UAV')

        # 标注节点ID、载荷ID和SINR
        for node in self.nodes.values():
            plt.text(node.position[0], node.position[1], node.node_id, fontsize=8)

        # 绘制通信业务
        for tx in self.transmitters:
            # 检查发射机是否有配对的接收机
            if not hasattr(tx, 'peer') or not tx.peer:
                continue
            rx = tx.peer
            color = 'green'
            x0, y0, z0 = tx.node.position
            x1, y1, z1 = rx.node.position
            dx, dy = x1 - x0, y1 - y0
            # 发射机到接收机的箭头
            plt.arrow(x0, y0, dx, dy, length_includes_head=True,
                      head_width=30, head_length=50, fc=color, ec=color, alpha=0.5)
            # 接收机信息
            if rx.sinr == float('inf'):
                if tx.power == 0:
                    description = "poweroff"
                else:
                    description = "diff freq"
                text = f"{rx.pid}:\n {description}"
            else:
                if show_only_sinr:
                    text = f"{rx.sinr:.1f}"
                else:
                    text = f"{rx.pid}:\n{rx.sinr:.1f}dB\n{rx.frequency:.1f}MHz\n{rx.power:.1f}W"
            arrow_frac = 0.85
            label_x = x1 - dx * (1 - arrow_frac)
            label_y = y1 - dy * (1 - arrow_frac)

            plt.text(label_x, label_y, text, color=color, fontsize=8, va='top')

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Node Layout')
        # plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name, dpi=200)
            plt.close()
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            buf.close()
            return img