"""把 experiment_figures_executed.ipynb 的 Fig.2 cell (两子图折线) 替换为单图散点版。"""
import json

NB = 'paper/experiment_figures_executed.ipynb'

new_source = '''# Fig.2: 性能-计算权衡散点图（P_int vs 决策时间，单图，合并原两子图）
sizes = [10, 20, 30, 50]
methods = ['random', 'dist_coloring', 'dist_greedy']
method_colors = {'random': '#888888', 'dist_coloring': '#4daf4a', 'dist_greedy': '#ff7f00'}
method_labels = {'random': 'Random', 'dist_coloring': 'Dist-Color', 'dist_greedy': 'Dist-Greedy'}
markers = {10: 'o', 20: 's', 30: '^', 50: 'D'}

fig, ax = plt.subplots(figsize=(9.5, 6.2))

# baseline: 方法=颜色, N=形状; 同方法按决策时间串成 trade-off 曲线
seen_method = set()
for m in methods:
    pts = []
    for n in sizes:
        df = f'{DATA_DIR}/dist_N{n}_speed0.json'
        if os.path.exists(df):
            d = load_json(df)
            sm = {s['method']: s for s in d['summary']}
            if m in sm:
                pts.append((sm[m]['avg_time_s'], sm[m]['cum_interf_prob'], n))
    if not pts:
        continue
    pts.sort(key=lambda x: x[0])
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=method_colors[m], alpha=0.35, linewidth=1.5, zorder=2)
    for t, p, n in pts:
        lab = method_labels[m] if m not in seen_method else None
        seen_method.add(m)
        ax.scatter(t, p, c=method_colors[m], marker=markers[n], s=95, label=lab,
                   alpha=0.9, edgecolors='k', linewidths=0.6, zorder=3)

# BC / CARLTON 彩色星号点（Pareto 前沿）
for key, col, lab in [('bc', '#984ea3', 'BC (Ours)'), ('carlton', '#a65628', 'CARLTON')]:
    xs, ys = [], []
    for n in sizes:
        if key == 'bc':
            bc = load_json(f'{DATA_DIR}/dynamic_bc_bc_orthogonal_N{n}.json')
            t = gs(bc, 0, 'avg_total_inference_time_s'); p = gs(bc, 0, 'cum_interf_prob_mean')
        else:
            cf = f'{DATA_DIR}/dynamic_carlton_N{n}.json'
            if os.path.exists(cf):
                ca = load_json(cf)
                t = gs(ca, 0, 'avg_total_inference_time_s'); p = gs(ca, 0, 'cum_interf_prob_mean')
            else:
                t, p = np.nan, np.nan
        if not np.isnan(t) and not np.isnan(p):
            xs.append(t); ys.append(p)
    if xs:
        ax.scatter(xs, ys, c=col, marker='*', s=260, label=lab, edgecolors='k', linewidths=0.6, zorder=5)

ax.set_xscale('log')
ax.set_xlabel('Decision Time per Episode (s, log scale)', fontsize=12)
ax.set_ylabel('Cumulative Interference Probability', fontsize=12)
ax.set_title('Performance\\u2013Computation Tradeoff (v = 0)', fontsize=13)
from matplotlib.lines import Line2D
n_handles = [Line2D([0], [0], marker=markers[n], color='w', markerfacecolor='#444444',
                    markeredgecolor='k', markersize=9, label=f'N={n}') for n in sizes]
leg1 = ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=n_handles, fontsize=9, loc='lower right', title='Scale (N)', title_fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
plt.tight_layout(); fig.savefig(f'{FIG_DIR}/fig2_tradeoff.pdf', dpi=150, bbox_inches='tight')
plt.show()
print('Fig.2 rendered.')
'''.splitlines(keepends=True)

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

replaced = 0
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell['source'])
    if '性能-决策用时' in src or 'Performance vs Scale' in src:
        cell['source'] = new_source
        replaced += 1
        print('replaced Fig.2 cell')

assert replaced == 1, f'expected exactly 1 Fig.2 cell, found {replaced}'
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('notebook saved')
