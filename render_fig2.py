"""渲染 Fig.2：性能-计算权衡散点图 (P_int vs 决策时间, log-x)。

数据来自 results_data/ 下的 dist_N{n}_speed0.json (baselines) 与
dynamic_bc_*/dynamic_carlton_* (BC / CARLTON)。
- N=10/20 保留 dist_greedy；N=30/50 按决策去掉 dist_greedy (点自然缺失)。
- BC / CARLTON 作为彩色星号点叠加在 Pareto 前沿。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'results_data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def gs(dyn, speed, field):
    """从动态结果按速度取值，兼容 speed_X / speed_X.0；缺失返回 nan。"""
    if not isinstance(dyn, dict) or 'results' not in dyn:
        return np.nan
    res = dyn['results']
    for k in (f'speed_{speed}', f'speed_{speed}.0', f'speed_{int(speed)}'):
        if k in res:
            return res[k].get(field, np.nan)
    return np.nan


sizes = [10, 20, 30, 50]

# 仅保留实际存在的方法（按决策：N=30/50 无 dist_greedy）
methods = ['random', 'dist_coloring', 'dist_greedy']
colors = {'random': '#888888', 'dist_coloring': '#4daf4a', 'dist_greedy': '#ff7f00'}
markers = {10: 'o', 20: 's', 30: '^', 50: 'D'}
labels = {'random': 'Random', 'dist_coloring': 'Dist-Color', 'dist_greedy': 'Dist-Greedy'}

fig, ax = plt.subplots(figsize=(9.5, 6.2))

# ---- baseline 散点：方法=颜色，N=形状；同方法按决策时间串成 trade-off 曲线 ----
seen_method = set()
for m in methods:
    pts = []  # (time, P_int, N)
    for n in sizes:
        d = load_json(f'{DATA_DIR}/dist_N{n}_speed0.json')
        sm = {s['method']: s for s in d['summary']}
        if m in sm:
            pts.append((sm[m]['avg_time_s'], sm[m]['cum_interf_prob'], n))
    if not pts:
        continue
    pts.sort(key=lambda x: x[0])  # 按决策时间排序以连成曲线
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # 淡色连线：体现同一方法随 N 增大的 trade-off 走向
    ax.plot(xs, ys, color=colors[m], alpha=0.35, linewidth=1.5, zorder=2)
    for t, p, n in pts:
        lab = labels[m] if m not in seen_method else None
        seen_method.add(m)
        ax.scatter(t, p, c=colors[m], marker=markers[n], s=95, label=lab,
                   alpha=0.9, edgecolors='k', linewidths=0.6, zorder=3)

# ---- N 形状图例（proxy artists）----
from matplotlib.lines import Line2D
n_handles = [Line2D([0], [0], marker=markers[n], color='w', markerfacecolor='#444444',
                    markeredgecolor='k', markersize=9, label=f'N={n}') for n in sizes]

# ---- BC / CARLTON 彩色星号点 ----
for key, col, lab in [('bc', '#984ea3', 'BC (Ours)'), ('carlton', '#a65628', 'CARLTON')]:
    xs, ys = [], []
    for n in sizes:
        if key == 'bc':
            bc = load_json(f'{DATA_DIR}/dynamic_bc_bc_orthogonal_N{n}.json')
            t = gs(bc, 0, 'avg_total_inference_time_s')
            p = gs(bc, 0, 'cum_interf_prob_mean')
        else:
            cf = f'{DATA_DIR}/dynamic_carlton_N{n}.json'
            if os.path.exists(cf):
                ca = load_json(cf)
                t = gs(ca, 0, 'avg_total_inference_time_s')
                p = gs(ca, 0, 'cum_interf_prob_mean')
            else:
                t, p = np.nan, np.nan
        if not np.isnan(t) and not np.isnan(p):
            xs.append(t)
            ys.append(p)
    if xs:
        ax.scatter(xs, ys, c=col, marker='*', s=240, label=lab,
                   edgecolors='k', linewidths=0.6, zorder=5)

ax.set_xscale('log')
ax.set_xlabel('Decision Time per Episode (s, log scale)', fontsize=12)
ax.set_ylabel('Cumulative Interference Probability', fontsize=12)
ax.set_title('Performance–Computation Tradeoff (v = 0)', fontsize=13)
# 方法图例（上右）+ N 形状图例（下右）
leg1 = ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=n_handles, fontsize=9, loc='lower right', title='Scale (N)',
          title_fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig2_tradeoff.pdf')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Fig.2 rendered -> {out}')

# 打印各点坐标，便于核对
print('--- baseline points (method: (time_s, P_int)) ---')
for m in methods:
    for n in sizes:
        d = load_json(f'{DATA_DIR}/dist_N{n}_speed0.json')
        sm = {s['method']: s for s in d['summary']}
        if m in sm:
            print(f'  {m:14s} N={n:2d}: t={sm[m]["avg_time_s"]:.4f}  P={sm[m]["cum_interf_prob"]:.4f}')
