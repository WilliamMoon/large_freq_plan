"""把 experiment_figures_executed.ipynb 的 Fig.2 单元改写为 N=10/20/30/50 的折线图。"""
import json

NB = "paper/experiment_figures_executed.ipynb"

new_src = '''# 性能-决策用时 折线图（N=10/20/30/50, speed=0）
sizes = [10, 20, 30, 50]
baseline_methods = ['random', 'dist_coloring', 'jar', 'dist_greedy', 'greedy_periodic_5', 'greedy_per_slot']
baseline_labels = {
    'random': 'Random', 'dist_coloring': 'Dist-Color', 'jar': 'JAR',
    'dist_greedy': 'Dist-Greedy', 'greedy_periodic_5': 'Greedy-Per5',
    'greedy_per_slot': 'Per-slot Greedy'}
baseline_colors = {
    'random': '#888888', 'dist_coloring': '#4daf4a', 'jar': '#9999cc',
    'dist_greedy': '#ff7f00', 'greedy_periodic_5': '#377eb8', 'greedy_per_slot': '#e41a1c'}

probs_by_method = {m: [] for m in baseline_methods + ['bc', 'carlton']}
times_by_method = {m: [] for m in baseline_methods + ['bc', 'carlton']}

for n in sizes:
    d = load_json(f'{DATA_DIR}/dist_N{n}_speed0.json')
    sm = {s['method']: s for s in d['summary']}
    for m in baseline_methods:
        if m in sm:
            probs_by_method[m].append(sm[m]['cum_interf_prob'])
            times_by_method[m].append(sm[m]['avg_time_s'])
        else:
            probs_by_method[m].append(None); times_by_method[m].append(None)
    bc = load_json(f'{DATA_DIR}/dynamic_bc_bc_orthogonal_N{n}.json')
    probs_by_method['bc'].append(bc['results']['speed_0']['cum_interf_prob_mean'])
    times_by_method['bc'].append(bc['results']['speed_0']['avg_total_inference_time_s'])
    cf = f'{DATA_DIR}/dynamic_carlton_N{n}.json'
    if os.path.exists(cf):
        ca = load_json(cf)
        probs_by_method['carlton'].append(ca['results']['speed_0']['cum_interf_prob_mean'])
        times_by_method['carlton'].append(ca['results']['speed_0']['avg_total_inference_time_s'])
    else:
        probs_by_method['carlton'].append(None); times_by_method['carlton'].append(None)

method_style = {**baseline_colors, 'bc': '#984ea3', 'carlton': '#a65628'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(sizes))

for m, color in method_style.items():
    ys = probs_by_method[m]
    if any(v is not None for v in ys):
        lbl = m.upper() if m in ('bc', 'carlton') else baseline_labels[m]
        ax1.plot(x, ys, marker='o', linewidth=2, color=color, label=lbl)

for m, color in method_style.items():
    ts = times_by_method[m]
    if any(v is not None for v in ts):
        lbl = m.upper() if m in ('bc', 'carlton') else baseline_labels[m]
        ax2.plot(x, ts, marker='s', linewidth=2, color=color, label=lbl)

ax1.set_xticks(x); ax1.set_xticklabels([f'N={n}' for n in sizes])
ax1.set_xlabel('Number of UAVs (N)', fontsize=12)
ax1.set_ylabel('Cumulative Interference Probability', fontsize=12)
ax1.set_title('Performance vs Scale (v=0)', fontsize=13)
ax1.legend(fontsize=8.5, ncol=2); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1.05)

ax2.set_xticks(x); ax2.set_xticklabels([f'N={n}' for n in sizes])
ax2.set_xlabel('Number of UAVs (N)', fontsize=12)
ax2.set_ylabel('Decision Time per Episode (s)', fontsize=12)
ax2.set_title('Decision Time vs Scale (v=0)', fontsize=13)
ax2.legend(fontsize=8.5, ncol=2); ax2.grid(alpha=0.3); ax2.set_yscale('log')

plt.tight_layout(); plt.show()
print('Fig.2 rendered.')
'''

# 转为 ipynb source 列表（每行一个字符串，末尾带 \\n，最后一行不带）
lines = new_src.split('\n')
source = [l + '\n' for l in lines[:-1]] + [lines[-1]]

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

replaced = False
for cell in nb['cells']:
    if cell.get('cell_type') == 'code' and any('speed_idx = 0' in s for s in cell.get('source', [])):
        cell['source'] = source
        replaced = True
        break

assert replaced, "未找到 Fig.2 单元格（含 'speed_idx = 0'）"
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Fig.2 单元格已更新。")
