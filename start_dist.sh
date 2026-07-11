#!/bin/bash
cd /mnt/data1/liushiyang/large_freq_plan
export PYTHONPATH=$(pwd)

# 只杀真正的 python 进程，不会匹配到本脚本自身
for p in $(pgrep -f "python run_dist_comparison.py"); do
  kill -9 "$p" 2>/dev/null
done
sleep 2

# N=10/20 保留 dist_greedy（慢，约 20min / 1.3h）
screen -dmS dist_small /home/lsy/.conda/envs/uav/bin/python run_dist_comparison.py \
  --num_uav 10 20 --speeds 0 10 20 30 40 --num_episodes 20 \
  --methods random dist_coloring dist_greedy --num_workers 20 --output results \
  > night_logs/dist_small.log 2>&1

# N=30/50 只跑 random + dist_coloring（分钟级出齐）
screen -dmS dist_large /home/lsy/.conda/envs/uav/bin/python run_dist_comparison.py \
  --num_uav 30 50 --speeds 0 10 20 30 40 --num_episodes 20 \
  --methods random dist_coloring --num_workers 20 --output results \
  > night_logs/dist_large.log 2>&1

echo launched
