"""生成固定的评估布局集合，保证所有方法在同一批布局上公平对比。"""
import argparse
import json
import os
import random
import math
import numpy as np

from config import ENV_CONFIG, LAYOUTS_DIR, LoggerSingleton
from env import Env
from node import Node

logger = LoggerSingleton.get_instance()


def generate_layout(num_uav: int, area_size: float, z_center: float, z_span: float,
                    effective_range: float, rng: random.Random) -> dict:
    """生成一个随机布局（位置 + 配对），不依赖 marl_env 以避免循环导入。"""
    positions = []
    for _ in range(num_uav):
        x = rng.uniform(-area_size / 2, area_size / 2)
        y = rng.uniform(-area_size / 2, area_size / 2)
        z = rng.uniform(z_center - z_span / 2, z_center + z_span / 2)
        positions.append([round(x, 2), round(y, 2), round(z, 2)])

    # 用 Env 的 _match 逻辑生成配对
    env = Env(area_size, z_center, z_span)
    for idx, pos in enumerate(positions):
        node = Node(node_id=f"uav{idx}", position=tuple(pos))
        node._add_payloads()
        env.add_node(node)
    env._match(effective_range)

    # 提取配对信息
    pairing = []
    for tx in env.transmitters:
        if hasattr(tx, 'peer') and tx.peer:
            tx_idx = int(tx.node.node_id.replace("uav", ""))
            rx_idx = int(tx.peer.node.node_id.replace("uav", ""))
            pairing.append([tx_idx, rx_idx])

    return {"positions": positions, "pairing": pairing}


def main():
    parser = argparse.ArgumentParser(description="生成固定评估布局")
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 20, 25, 30, 40, 50])
    parser.add_argument("--num_layouts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=LAYOUTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    for count in args.counts:
        layouts = {}
        for i in range(args.num_layouts):
            layouts[f"layout_{i}"] = generate_layout(
                num_uav=count,
                area_size=ENV_CONFIG["area_size"],
                z_center=ENV_CONFIG["z_center"],
                z_span=ENV_CONFIG["z_span"],
                effective_range=ENV_CONFIG["effective_range"],
                rng=rng,
            )
        out_path = os.path.join(args.output_dir, f"N{count}_{args.num_layouts}layouts.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(layouts, f, ensure_ascii=False)
        logger.info(f"生成 {args.num_layouts} 个 N={count} 布局 → {out_path}")


if __name__ == "__main__":
    main()
