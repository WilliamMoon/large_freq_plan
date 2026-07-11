import json, re

NB = 'paper/experiment_figures_executed.ipynb'

with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

# 找到第一个 code cell 的索引（保留其 savefig/print）
first_code_idx = None
for i, c in enumerate(nb['cells']):
    if c.get('cell_type') == 'code':
        first_code_idx = i
        break

savefig_re = re.compile(r';\s*fig\.savefig\(.*\)')  # 仅删 ; fig.savefig(...) 片段

removed = 0
for i, c in enumerate(nb['cells']):
    if c.get('cell_type') != 'code':
        continue
    if i == first_code_idx:
        continue
    new_src = []
    for line in c['source']:
        # savefig 处理
        if 'savefig' in line:
            stripped = savefig_re.sub('', line)
            if 'savefig' in stripped:
                # 整行就是 savefig -> 删除
                removed += 1
                continue
            # 仅删除了后半段，保留前半段（如 plt.tight_layout()）
            new_src.append(stripped)
            removed += 1
            continue
        # print 处理（整行删除）
        if 'print(' in line:
            removed += 1
            continue
        new_src.append(line)
    c['source'] = new_src

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'第一个 code cell 索引 = {first_code_idx}（已保留）')
print(f'共删除/改写 savefig+print 行数 = {removed}')
