"""分析 BC 策略的决策结果：哪些链路受干扰、被谁干扰、为什么没能规避。

对每个受扰链路，输出：
1. 受扰链路信息（位置、频率、功率、SINR）
2. 主要干扰源（谁干扰了它、频率差、距离、干扰功率）
3. BC 选择的频率/功率 vs greedy 会选什么
4. 受扰链路的局部观测特征
"""
import json
import os
import numpy as np
import torch
from collections import Counter

from config import ENV_CONFIG, ACTION_CONFIG, LAYOUTS_DIR, RESULTS_DIR, LoggerSingleton
from marl_env import MultiAgentEnv
from bc_trainer import BCTrainer
from test_cgreedy import _choose_best_freq_power

logger = LoggerSingleton.get_instance()


def analyze_failures(num_uav=10, num_layouts=20, ckpt_path="checkpoints/bc_freqhist_best_N10.pt"):
    """分析 BC 策略在固定布局上的失败案例。"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 加载 BC 模型
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    trainer = BCTrainer(num_uav=num_uav, obs_dim=obs_dim, device=device)
    trainer.load(ckpt_path)

    # 加载布局
    layout_path = os.path.join(LAYOUTS_DIR, f"N{num_uav}_50layouts.json")
    with open(layout_path, "r") as f:
        layouts = list(json.load(f).values())[:num_layouts]

    env = MultiAgentEnv(
        num_uav=num_uav,
        observation_radius=ENV_CONFIG["observation_radius"],
        area_size=ENV_CONFIG["area_size"],
        limit_neighbors=ENV_CONFIG["limit_neighbors"],
    )

    # 贪心候选
    n_freq = ACTION_CONFIG["n_freq"]
    n_power = ACTION_CONFIG["n_power"]
    freq_lo = ENV_CONFIG["freq_min"] + ENV_CONFIG["bandwidth"] / 2
    freq_hi = ENV_CONFIG["freq_max"] - ENV_CONFIG["bandwidth"] / 2
    freq_candidates = np.linspace(freq_lo, freq_hi, n_freq)
    power_candidates = np.linspace(0, ENV_CONFIG["max_power"], n_power)

    all_failures = []

    for layout_idx, layout in enumerate(layouts):
        env.load_layout(layout)
        env.reset_commit_state()
        node_ids = list(env.base_env.nodes.keys())

        # BC 顺序决策
        bc_actions = {}
        for exec_idx, agent_id in enumerate(node_ids):
            obs = env.get_sequential_observation(agent_id, exec_idx)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = trainer.q_net(obs_t)
            action_idx = int(q_values.argmax(dim=-1).item())
            p_idx = action_idx // trainer.n_freq
            f_idx = action_idx % trainer.n_freq
            power_val = float(trainer.power_levels[p_idx])
            freq_val = float(trainer.freq_levels[f_idx])

            freq_span = freq_hi - freq_lo
            power_norm = (power_val / ENV_CONFIG["max_power"]) * 2 - 1
            freq_norm = ((freq_val - freq_lo) / freq_span) * 2 - 1
            normalized = np.array([power_norm, freq_norm], dtype=np.float32)
            env.apply_sequential_action(agent_id, normalized, commit_index=exec_idx)
            bc_actions[agent_id] = {"power": power_val, "freq": freq_val, "action_idx": action_idx}

        env.base_env.update_sinr()
        bc_interf_prob = env.base_env.calc_interf_prob()

        # 分析每个受扰链路
        for rx in env.base_env.receivers:
            if rx.sinr == float('-inf') or rx.sinr == float('inf'):
                continue
            if rx.sinr >= rx.threshold:
                continue  # 未受扰

            # 找到干扰源
            interferers = []
            for tx in env.base_env.transmitters:
                if tx == rx.peer:
                    continue
                interf_dbm = env.base_env.calc_signal_dbm(tx, rx)
                if interf_dbm == float('-inf'):
                    continue
                interferers.append({
                    "tx_id": tx.pid,
                    "tx_node": tx.node.node_id,
                    "tx_freq": tx.frequency,
                    "tx_power": tx.power,
                    "tx_distance_m": float(np.linalg.norm(
                        np.array(tx.node.position) - np.array(rx.node.position)
                    )),
                    "interf_dbm": interf_dbm,
                    "freq_diff_mhz": abs(tx.frequency - rx.peer.frequency),
                })

            interferers.sort(key=lambda x: x["interf_dbm"], reverse=True)

            # 信号功率
            signal_dbm = env.base_env.calc_signal_dbm(rx.peer, rx)
            tx_distance = float(np.linalg.norm(
                np.array(rx.peer.node.position) - np.array(rx.node.position)
            ))

            failure = {
                "layout_idx": layout_idx,
                "victim_node": rx.node.node_id,
                "victim_freq": rx.peer.frequency,
                "victim_power": rx.peer.power,
                "victim_sinr_db": rx.sinr,
                "signal_dbm": signal_dbm,
                "link_distance_m": tx_distance,
                "num_interferers": len(interferers),
                "top_interferers": interferers[:3],
                "bc_action": bc_actions[rx.peer.node.node_id],
            }

            # Greedy 会选什么？
            # 重新加载布局跑 greedy 对比
            all_failures.append(failure)

    # 统计分析
    logger.info(f"\n{'='*70}")
    logger.info(f"BC 失败分析: {len(all_failures)} 个受扰链路（across {num_layouts} layouts）")
    logger.info(f"{'='*70}")

    # 1. 干扰源频率差分布
    freq_diffs = []
    for f in all_failures:
        for interf in f["top_interferers"]:
            freq_diffs.append(interf["freq_diff_mhz"])
    logger.info(f"\n1. 干扰源频率差分布 (MHz):")
    logger.info(f"   mean={np.mean(freq_diffs):.1f}, median={np.median(freq_diffs):.1f}")
    logger.info(f"   <5MHz: {sum(1 for d in freq_diffs if d < 5)}/{len(freq_diffs)} ({100*sum(1 for d in freq_diffs if d < 5)/len(freq_diffs):.0f}%)")
    logger.info(f"   5-10MHz: {sum(1 for d in freq_diffs if 5 <= d < 10)}/{len(freq_diffs)}")
    logger.info(f"   >=10MHz: {sum(1 for d in freq_diffs if d >= 10)}/{len(freq_diffs)}")

    # 2. 干扰源距离分布
    distances = []
    for f in all_failures:
        for interf in f["top_interferers"]:
            distances.append(interf["tx_distance_m"])
    logger.info(f"\n2. 干扰源距离分布 (m):")
    logger.info(f"   mean={np.mean(distances):.0f}, median={np.median(distances):.0f}")
    logger.info(f"   <300m: {sum(1 for d in distances if d < 300)}")
    logger.info(f"   300-600m: {sum(1 for d in distances if 300 <= d < 600)}")
    logger.info(f"   >=600m: {sum(1 for d in distances if d >= 600)}")

    # 3. 受扰链路的信号强度
    signals = [f["signal_dbm"] for f in all_failures]
    logger.info(f"\n3. 受扰链路信号功率 (dBm):")
    logger.info(f"   mean={np.mean(signals):.1f}, min={np.min(signals):.1f}, max={np.max(signals):.1f}")

    # 4. BC 选择的频率分布 vs 受扰时的频率
    bc_freqs = [f["bc_action"]["freq"] for f in all_failures]
    bc_powers = [f["bc_action"]["power"] for f in all_failures]
    logger.info(f"\n4. BC 在受扰链路的选择:")
    logger.info(f"   频率: mean={np.mean(bc_freqs):.1f}MHz, range=[{np.min(bc_freqs):.1f}, {np.max(bc_freqs):.1f}]")
    logger.info(f"   功率: mean={np.mean(bc_powers):.2f}W, range=[{np.min(bc_powers):.2f}, {np.max(bc_powers):.2f}]")

    # 5. 干扰源是否在观测半径内
    in_obs = 0
    out_obs = 0
    for f in all_failures:
        victim_node = env.base_env.nodes.get(f["victim_node"])
        if victim_node is None:
            continue
        for interf in f["top_interferers"]:
            if interf["tx_distance_m"] <= ENV_CONFIG["observation_radius"]:
                in_obs += 1
            else:
                out_obs += 1
    logger.info(f"\n5. 干扰源是否在观测半径({ENV_CONFIG['observation_radius']}m)内:")
    logger.info(f"   在观测内: {in_obs} ({100*in_obs/(in_obs+out_obs):.0f}%)")
    logger.info(f"   在观测外: {out_obs} ({100*out_obs/(in_obs+out_obs):.0f}%)")

    # 6. 干扰源是否是观测的 K 个邻居之一
    logger.info(f"\n6. 干扰源是否在 K={ENV_CONFIG['limit_neighbors']} 个最近邻中:")
    in_k = 0
    out_k = 0
    for f in all_failures:
        victim_node = env.base_env.nodes.get(f["victim_node"])
        if victim_node is None:
            continue
        neighbors = env.get_neighbors(victim_node, limit=ENV_CONFIG["limit_neighbors"])
        neighbor_ids = {n.node_id for n in neighbors}
        for interf in f["top_interferers"]:
            if interf["tx_node"] in neighbor_ids:
                in_k += 1
            else:
                out_k += 1
    logger.info(f"   在K邻居中: {in_k} ({100*in_k/(in_k+out_k):.0f}%)")
    logger.info(f"   不在K邻居中: {out_k} ({100*out_k/(in_k+out_k):.0f}%)")

    # 7. 典型失败案例
    logger.info(f"\n7. 典型失败案例（前5个）:")
    for f in all_failures[:5]:
        logger.info(f"\n  Layout {f['layout_idx']}, {f['victim_node']}:")
        logger.info(f"    SINR={f['victim_sinr_db']:.1f}dB, signal={f['signal_dbm']:.1f}dBm, link_dist={f['link_distance_m']:.0f}m")
        logger.info(f"    BC选择: freq={f['bc_action']['freq']:.1f}MHz, power={f['bc_action']['power']:.2f}W")
        for i, interf in enumerate(f["top_interferers"]):
            logger.info(f"    干扰源{i+1}: {interf['tx_node']}, freq={interf['tx_freq']:.1f}MHz, "
                       f"dist={interf['tx_distance_m']:.0f}m, interf={interf['interf_dbm']:.1f}dBm, "
                       f"freq_diff={interf['freq_diff_mhz']:.1f}MHz")

    # 保存详细结果
    output_path = os.path.join(RESULTS_DIR, "failure_analysis.json")
    with open(output_path, "w") as fp:
        json.dump({
            "num_failures": len(all_failures),
            "num_layouts": num_layouts,
            "freq_diff_stats": {
                "mean": float(np.mean(freq_diffs)),
                "median": float(np.median(freq_diffs)),
                "lt_5MHz": sum(1 for d in freq_diffs if d < 5),
                "ge_10MHz": sum(1 for d in freq_diffs if d >= 10),
            },
            "distance_stats": {
                "mean": float(np.mean(distances)),
                "median": float(np.median(distances)),
                "lt_300m": sum(1 for d in distances if d < 300),
                "ge_600m": sum(1 for d in distances if d >= 600),
            },
            "in_obs_ratio": in_obs / (in_obs + out_obs) if (in_obs + out_obs) > 0 else 0,
            "in_k_ratio": in_k / (in_k + out_k) if (in_k + out_k) > 0 else 0,
            "failures": all_failures[:50],  # 保存前50个详细案例
        }, fp, indent=2, ensure_ascii=False)
    logger.info(f"\n详细结果保存至 {output_path}")


if __name__ == "__main__":
    analyze_failures()
