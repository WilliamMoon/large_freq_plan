#!/bin/bash
# 批量训练+评估 from-scratch DRL 基线 (IQL, MADDPG)，供论文对比。
# 仅使用 GPU 4（BC 会话在 3/4 且为 CPU 瓶颈，GPU 基本闲置）。
# 不修改任何共享源码文件；MADDPG 动态评估用独立脚本 eval_maddpg_dynamic.py。
set -u
cd /mnt/data1/liushiyang/large_freq_plan
PY=/home/lsy/.conda/envs/uav/bin/python
LOG=/mnt/data1/liushiyang/large_freq_plan/baselines_run.log
echo "BASELINES START $(date)" >> "$LOG"

for N in 10 20 30 50; do
  echo "== IQL TRAIN N=$N $(date) ==" >> "$LOG"
  $PY train.py --method iql --num_uav $N --gpu 4 --episodes 2000 --warmup 100 --save_name iql_N$N >> "$LOG" 2>&1
  echo "== IQL EVAL  N=$N $(date) ==" >> "$LOG"
  # eval_dynamic_marl.py 在汇总打印处有 float 格式 bug 会崩溃，但 json 已先保存，故 || true
  CUDA_VISIBLE_DEVICES=4 $PY eval_dynamic_marl.py --method iql --ckpt checkpoints/iql_N$N.pt \
      --num_uav $N --speeds 0 10 20 30 40 --num_episodes 20 >> "$LOG" 2>&1 || true
done

for N in 10 20 30 50; do
  echo "== MADDPG TRAIN N=$N $(date) ==" >> "$LOG"
  $PY train.py --method maddpg --num_uav $N --gpu 4 --episodes 5000 --warmup 200 --save_name maddpg_N$N >> "$LOG" 2>&1
  echo "== MADDPG EVAL  N=$N $(date) ==" >> "$LOG"
  $PY eval_maddpg_dynamic.py --ckpt checkpoints/maddpg_N$N.pt \
      --num_uav $N --speeds 0 10 20 30 40 --num_episodes 20 --gpu 4 >> "$LOG" 2>&1
done

echo "BASELINES DONE $(date)" >> "$LOG"
