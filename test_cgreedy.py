import argparse
import json
import math
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import LoggerSingleton
from marl_env import MultiAgentEnv

logger = LoggerSingleton.get_instance()


def _choose_best_freq_power(
    env: MultiAgentEnv,
    node,
    freq_candidates: np.ndarray,
    power_candidates: np.ndarray,
    use_neighbors: bool,
    neighbor_sample: int | None,
) -> tuple[float, float, float]:
    tx = node.tx
    rx = tx.peer
    threshold = getattr(rx, 'threshold', float('-inf'))
    neighbor_nodes_full = env.get_neighbors(node, limit=env.limit_neighbors) if use_neighbors else []
    if use_neighbors and neighbor_sample is not None and neighbor_sample > 0:
        neighbor_nodes = neighbor_nodes_full[:neighbor_sample]
    else:
        neighbor_nodes = neighbor_nodes_full

    orig_freq = tx.frequency
    orig_power = tx.power
    orig_rx_freq = rx.frequency

    # central: self-only constraint (threshold, min power, tie by higher SINR)
    # sequential: self+neighbors constraint (all above threshold, min power, tie by neighbor margin then SINR)
    feasible_choice: tuple[float, float, float, float] | None = None  # (power, margin_metric, sinr, freq)
    best_margin_choice: tuple[float, float, float, float] = (
        tx.power,
        float('-inf'),
        rx.sinr,
        tx.frequency,
    )
    for freq in freq_candidates:
        for power in power_candidates:
            tx.frequency = float(freq)
            tx.power = float(power)
            rx.frequency = float(freq)
            env.base_env.update_sinr()
            candidate_sinr = rx.sinr
            self_margin = candidate_sinr - threshold
            if use_neighbors:
                neighbor_margins = [n.rx.sinr - getattr(n.rx, 'threshold', 0.0) for n in neighbor_nodes]
                min_neighbor_margin = min(neighbor_margins) if neighbor_margins else float('inf')
                worst_margin = min(self_margin, min_neighbor_margin)
                if candidate_sinr >= threshold and min_neighbor_margin >= 0:
                    if (
                        feasible_choice is None
                        or power < feasible_choice[0]
                        or (power == feasible_choice[0] and min_neighbor_margin > feasible_choice[1])
                        or (
                            power == feasible_choice[0]
                            and min_neighbor_margin == feasible_choice[1]
                            and candidate_sinr > feasible_choice[2]
                        )
                    ):
                        feasible_choice = (float(power), float(min_neighbor_margin), float(candidate_sinr), float(freq))

                if (
                    worst_margin > best_margin_choice[1]
                    or (worst_margin == best_margin_choice[1] and power < best_margin_choice[0])
                    or (
                        worst_margin == best_margin_choice[1]
                        and power == best_margin_choice[0]
                        and candidate_sinr > best_margin_choice[2]
                    )
                ):
                    best_margin_choice = (float(power), float(worst_margin), float(candidate_sinr), float(freq))
            else:
                # central: only self constraint
                if candidate_sinr >= threshold:
                    if (
                        feasible_choice is None
                        or power < feasible_choice[0]
                        or (power == feasible_choice[0] and candidate_sinr > feasible_choice[2])
                    ):
                        feasible_choice = (float(power), self_margin, float(candidate_sinr), float(freq))
                if (
                    self_margin > best_margin_choice[1]
                    or (self_margin == best_margin_choice[1] and power < best_margin_choice[0])
                    or (
                        self_margin == best_margin_choice[1]
                        and power == best_margin_choice[0]
                        and candidate_sinr > best_margin_choice[2]
                    )
                ):
                    best_margin_choice = (float(power), self_margin, float(candidate_sinr), float(freq))

    if feasible_choice:
        _, _, sinr_val, freq_val = feasible_choice
        power_val = feasible_choice[0]
        tx.frequency = orig_freq
        tx.power = orig_power
        rx.frequency = orig_rx_freq
        env.base_env.update_sinr()
        return sinr_val, freq_val, power_val
    _, _, sinr_val, freq_val = best_margin_choice
    power_val = best_margin_choice[0]
    tx.frequency = orig_freq
    tx.power = orig_power
    rx.frequency = orig_rx_freq
    env.base_env.update_sinr()
    return sinr_val, freq_val, power_val


def _enumerate_best_frequency_central(
    env: MultiAgentEnv,
    freq_candidates: np.ndarray,
    power_candidates: np.ndarray,
) -> List[Dict]:
    episode_details = []
    env.base_env.update_sinr()
    for node in env.base_env.nodes.values():
        best_sinr, best_freq, best_power = _choose_best_freq_power(
            env, node, freq_candidates, power_candidates, use_neighbors=False, neighbor_sample=None
        )
        tx = node.tx
        rx = tx.peer
        tx.frequency = best_freq
        tx.power = best_power
        rx.frequency = best_freq
        env.base_env.update_sinr()
        episode_details.append({
            'agent': node.node_id,
            'selected_freq': best_freq,
            'selected_power': best_power,
            'sinr': rx.sinr
        })
    return episode_details


def _enumerate_best_frequency_sequential(
    env: MultiAgentEnv,
    freq_candidates: np.ndarray,
    power_candidates: np.ndarray,
    shuffle_order: bool,
    neighbor_sample: int | None,
    freq_sample: int | None,
    power_sample: int | None,
) -> List[Dict]:
    env.base_env.update_sinr()
    node_list = list(env.base_env.nodes.values())
    if shuffle_order:
        rng = np.random.default_rng()
        rng.shuffle(node_list)

    episode_details = []
    for node in node_list:
        freq_pool = freq_candidates
        power_pool = power_candidates
        if freq_sample is not None and freq_sample > 0 and freq_sample < len(freq_candidates):
            rng = np.random.default_rng()
            freq_pool = rng.choice(freq_candidates, size=freq_sample, replace=False)
        if power_sample is not None and power_sample > 0 and power_sample < len(power_candidates):
            rng = np.random.default_rng()
            power_pool = rng.choice(power_candidates, size=power_sample, replace=False)
        best_sinr, best_freq, best_power = _choose_best_freq_power(
            env, node, freq_pool, power_pool, use_neighbors=True, neighbor_sample=neighbor_sample
        )
        tx = node.tx
        rx = tx.peer
        tx.frequency = best_freq
        tx.power = best_power
        rx.frequency = best_freq
        env.base_env.update_sinr()
        episode_details.append({
            'agent': node.node_id,
            'selected_freq': best_freq,
            'selected_power': best_power,
            'sinr': rx.sinr
        })
    return episode_details


def _enumerate_best_frequency_independent(
    env: MultiAgentEnv,
    freq_candidates: np.ndarray,
    power_candidates: np.ndarray,
) -> List[Dict]:
    env.base_env.update_sinr()
    node_list = list(env.base_env.nodes.values())

    # 先为所有节点独立选出方案但不立即生效
    decisions = {}
    for node in node_list:
        best_sinr, best_freq, best_power = _choose_best_freq_power(
            env, node, freq_candidates, power_candidates, use_neighbors=False, neighbor_sample=None
        )
        decisions[node.node_id] = (best_sinr, best_freq, best_power)

    # 一次性应用所有节点的决策
    for node in node_list:
        best_sinr, best_freq, best_power = decisions[node.node_id]
        tx = node.tx
        rx = tx.peer
        tx.frequency = best_freq
        tx.power = best_power
        rx.frequency = best_freq

    env.base_env.update_sinr()

    episode_details = []
    for node in node_list:
        tx = node.tx
        rx = tx.peer
        episode_details.append({
            'agent': node.node_id,
            'selected_freq': tx.frequency,
            'selected_power': tx.power,
            'sinr': rx.sinr
        })

    return episode_details


def _summarize_episode(
    env: MultiAgentEnv,
    details: List[Dict],
    episode_idx: int,
    include_details: bool = True,
) -> Dict:
    sinrs = [item['sinr'] for item in details if item['sinr'] != float('-inf')]
    env.base_env.update_sinr()
    interf_prob = env.base_env.calc_interf_prob()
    summary = {
        'episode': episode_idx,
        'avg_sinr': float(np.mean(sinrs)) if sinrs else 0.0,
        'min_sinr': float(np.min(sinrs)) if sinrs else 0.0,
        'max_sinr': float(np.max(sinrs)) if sinrs else 0.0,
        'interference_prob': float(interf_prob),
    }
    if include_details:
        summary['per_agent'] = details
    return summary


def _plot_freq_distribution(details: List[Dict], episode: int, save_dir: str):
    freqs = [d['selected_freq'] for d in details]
    unique_freqs, counts = np.unique(freqs, return_counts=True)
    
    plt.figure(figsize=(8, 5))
    plt.bar(unique_freqs, counts, width=1.0, alpha=0.7, edgecolor='black')
    plt.title(f'Frequency Usage Distribution (Episode {episode})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Node Count')
    plt.grid(True, alpha=0.3, axis='y')
    
    dist_dir = os.path.join(save_dir, 'freq_dists')
    os.makedirs(dist_dir, exist_ok=True)
    plot_path = os.path.join(dist_dir, f'freq_dist_ep{episode:04d}.png')
    plt.savefig(plot_path, dpi=100)
    plt.close()


def _plot_metrics(metrics: List[Dict], save_dir: str):
    if not metrics:
        return
    episodes = [item['episode'] for item in metrics]
    probs = [item['interference_prob'] for item in metrics]
    avg_sinr = [item['avg_sinr'] for item in metrics]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episodes, probs, label='Interference Prob', color='tab:red')
    plt.xlabel('Episode')
    plt.ylabel('Interference Probability')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(episodes, avg_sinr, label='Avg SINR', color='tab:blue')
    plt.xlabel('Episode')
    plt.ylabel('Average SINR (dB)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'greedy_metrics.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info(f"贪婪指标图已保存: {plot_path}")


def _save_metrics(metrics: List[Dict], save_dir: str):
    if not metrics:
        return
    json_path = os.path.join(save_dir, 'greedy_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)
    csv_path = os.path.join(save_dir, 'greedy_metrics.csv')
    with open(csv_path, 'w', encoding='utf-8') as fp:
        fp.write('episode,interference_prob,avg_sinr,min_sinr,max_sinr\n')
        for item in metrics:
            fp.write(
                f"{item['episode']},{item['interference_prob']:.6f},{item['avg_sinr']:.6f},{item['min_sinr']:.6f},{item['max_sinr']:.6f}\n"
            )
    logger.info(f"贪婪指标已保存: {json_path}, {csv_path}")


def _run_greedy_for_count(
    num_uav: int,
    num_episodes: int,
    freq_candidates: np.ndarray,
    power_candidates: np.ndarray,
    greedy_mode: str,
    shuffle_order: bool,
    neighbor_sample: int | None,
    freq_sample: int | None,
    power_sample: int | None,
    area_size: float,
    limit_neighbors: int,
    save_dir: str,
    save_episode_details: bool,
) -> List[Dict]:
    env = MultiAgentEnv(
        num_uav=num_uav,
        observation_radius=600.0,
        area_size=area_size,
        limit_neighbors=limit_neighbors,
    )
    env.describe_layout()

    metrics: List[Dict] = []
    episode_times: List[float] = []
    progress = tqdm(range(num_episodes), desc=f'UAV={num_uav}', leave=False, ncols=200)
    for episode in progress:
        start = time.time()
        env.reset()
        if greedy_mode == 'sequential':
            details = _enumerate_best_frequency_sequential(
                env, freq_candidates, power_candidates, shuffle_order,
                neighbor_sample=neighbor_sample,
                freq_sample=freq_sample,
                power_sample=power_sample,
            )
        elif greedy_mode == 'independent':
            details = _enumerate_best_frequency_independent(
                env, freq_candidates, power_candidates
            )
        else:
            details = _enumerate_best_frequency_central(env, freq_candidates, power_candidates)
        
        if save_episode_details:
            _plot_freq_distribution(details, episode, save_dir)

        summary = _summarize_episode(env, details, episode, include_details=save_episode_details)
        metrics.append(summary)
        duration = time.time() - start
        episode_times.append(duration)
        progress.set_postfix({
            'prob': f"{summary['interference_prob']:.3f}",
            'avg_sinr': f"{summary['avg_sinr']:.2f}",
            't(s)': f"{duration:.2f}",
        })

    if save_episode_details:
        _save_metrics(metrics, save_dir)
        _plot_metrics(metrics, save_dir)

    return metrics


def _save_summary(summary_records: List[Dict], save_dir: str):
    if not summary_records:
        return
    json_path = os.path.join(save_dir, 'cgreedy_summary.json')
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(summary_records, fp, ensure_ascii=False, indent=2)
    csv_path = os.path.join(save_dir, 'cgreedy_summary.csv')
    with open(csv_path, 'w', encoding='utf-8') as fp:
        fp.write('uav_count,avg_interference_prob,std_interference_prob,avg_sinr\n')
        for item in summary_records:
            fp.write(
                f"{item['uav_count']},{item['avg_interference_prob']:.6f},"
                f"{item['std_interference_prob']:.6f},{item['avg_sinr']:.6f}\n"
            )
    logger.info(f"贪婪汇总已保存: {json_path}, {csv_path}")


def _plot_summary(summary_records: List[Dict], save_dir: str):
    if not summary_records:
        return
    summary_records = sorted(summary_records, key=lambda x: x['uav_count'])
    counts = [item['uav_count'] for item in summary_records]
    probs = [item['avg_interference_prob'] for item in summary_records]

    plt.figure(figsize=(6, 4))
    plt.plot(counts, probs, marker='o', color='tab:red')
    plt.xlabel('UAV Count')
    plt.ylabel('Average Interference Probability')
    plt.title('Greedy Interference vs UAV Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'cgreedy_summary.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info(f"贪婪汇总图已保存: {plot_path}")


def greedy(
    num_uav: int = 10,
    num_episodes: int = 100,
    output: str | None = None,
    num_freq_candidates: int = 10,
    freq_min: float = 1205.0,
    freq_max: float = 1295.0,
    uav_counts: List[int] | None = None,
    power_candidates: List[float] | None = None,
    greedy_mode: str = 'central',
    shuffle_order: bool = True,
    neighbor_sample: int | None = None,
    freq_sample: int | None = None,
    power_sample: int | None = None,
    area_size: float = 2000.0,
    limit_neighbors: int = 5,
    save_episode_details: bool = False,
):
    counts = uav_counts if uav_counts else [num_uav]
    if output:
        base_dir = output
    else:
        base_dir = 'cgreedy_sweep' if len(counts) > 1 else f'cgreedy_{counts[0]}'
    os.makedirs(base_dir, exist_ok=True)

    freq_candidates = np.linspace(freq_min, freq_max, num_freq_candidates)
    power_candidates_arr = np.array(power_candidates if power_candidates else [1, 2, 3, 4, 5], dtype=float)

    summary_records: List[Dict] = []
    for count in counts:
        if len(counts) == 1:
            count_dir = base_dir
        else:
            count_dir = os.path.join(base_dir, f'uav_{count}')
            os.makedirs(count_dir, exist_ok=True)
        metrics = _run_greedy_for_count(
            num_uav=count,
            num_episodes=num_episodes,
            freq_candidates=freq_candidates,
            power_candidates=power_candidates_arr,
            greedy_mode=greedy_mode,
            shuffle_order=shuffle_order,
            neighbor_sample=neighbor_sample,
            freq_sample=freq_sample,
            power_sample=power_sample,
            area_size=area_size,
            limit_neighbors=limit_neighbors,
            save_dir=count_dir,
            save_episode_details=save_episode_details,
        )
        probs = [item['interference_prob'] for item in metrics]
        avg_prob = float(np.mean(probs)) if probs else 0.0
        std_prob = float(np.std(probs)) if probs else 0.0
        avg_sinr = float(np.mean([item['avg_sinr'] for item in metrics])) if metrics else 0.0
        logger.info(
            f"UAV={count} mode={greedy_mode}: 平均干扰概率{avg_prob:.4f}±{std_prob:.4f}, "
            f"平均SINR{avg_sinr:.2f}"
        )
        summary_records.append({
            'uav_count': count,
            'avg_interference_prob': avg_prob,
            'std_interference_prob': std_prob,
            'avg_sinr': avg_sinr,
            'episodes': num_episodes,
        })

    _save_summary(summary_records, base_dir)
    _plot_summary(summary_records, base_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Greedy baseline for frequency selection')
    parser.add_argument('--num_uav', type=int, default=10, help='无人机数量')
    parser.add_argument('--num_episodes', type=int, default=100, help='贪婪轮数')
    parser.add_argument('--output', type=str, default=None, help='输出目录，为空则自动命名')
    parser.add_argument('--num_freq_candidates', type=int, default=10, help='遍历频率数量')
    parser.add_argument('--freq_min', type=float, default=1205.0, help='最小频率 (MHz)')
    parser.add_argument('--freq_max', type=float, default=1295.0, help='最大频率 (MHz)')
    parser.add_argument('--area_size', type=float, default=2000.0, help='场景区域尺寸 (米)')
    parser.add_argument('--uav_counts', type=int, nargs='+', help='批量运行的无人机数量列表')
    parser.add_argument('--limit_neighbors', type=int, default=5, help='每个节点的邻居数量上限')
    parser.add_argument('--power_candidates', type=float, nargs='+', help='可选功率列表 (W)')
    parser.add_argument('--greedy_mode', choices=['central', 'sequential', 'independent'], default='central', help='central 模拟 CDDPG，sequential 模拟 MADDPG 顺序决策，independent 模拟 IDDPG 同步独立决策')
    parser.add_argument('--no_shuffle_order', action='store_true', help='顺序模式下不在每个 episode 打乱决策顺序')
    parser.add_argument('--seq_neighbor_sample', type=int, default=None, help='顺序模式下邻居裁剪数量，缺省为不限')
    parser.add_argument('--seq_freq_sample', type=int, default=None, help='顺序模式下每节点随机抽取的频点数量，缺省为全量')
    parser.add_argument('--seq_power_sample', type=int, default=None, help='顺序模式下每节点随机抽取的功率数量，缺省为全量')
    parser.add_argument('--save_episode_details', action='store_true', help='是否保存每个episode的详细数据和曲线')

    args = parser.parse_args()

    np.random.seed(42)

    greedy(
        num_uav=args.num_uav,
        num_episodes=args.num_episodes,
        output=args.output,
        num_freq_candidates=args.num_freq_candidates,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        area_size=args.area_size,
        limit_neighbors=args.limit_neighbors,
        uav_counts=args.uav_counts,
        power_candidates=args.power_candidates,
        greedy_mode=args.greedy_mode,
        shuffle_order=not args.no_shuffle_order,
        neighbor_sample=args.seq_neighbor_sample,
        freq_sample=args.seq_freq_sample,
        power_sample=args.seq_power_sample,
        save_episode_details=args.save_episode_details,
    )