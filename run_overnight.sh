#!/bin/bash
# 通宵实验启动脚本（服务器 /mnt/data1/liushiyang/large_freq_plan）
# 用户: lsy  | 环境: /home/lsy/.conda/envs/uav/bin/python (torch 2.5.1)
# 任务:
#   1) Zero-shot 多速度 (CPU, eval-only) -> 补齐 Fig.6: speed 0/10/30/40 (speed20 已有)
#   2) Centralized BC 扩 N=20/30/50 (GPU 2/3/4) -> 上界基线跨尺度 (N=10 已有)
set -u
cd /mnt/data1/liushiyang/large_freq_plan
PY=/home/lsy/.conda/envs/uav/bin/python
LOG=/mnt/data1/liushiyang/large_freq_plan/night_logs
mkdir -p "$LOG"

# ---------- Task 1: Zero-shot 多速度 (CPU, 强制不占 GPU) ----------
for s in 0 10 30 40; do
  setsid env CUDA_VISIBLE_DEVICES="" "$PY" run_zeroshot.py \
    --ckpt checkpoints/bc_orthogonal_N10.pt \
    --speed "$s" --deploy_sizes 10 20 30 50 --num_episodes 10 --seed 42 \
    > "$LOG/zs_speed$s.log" 2>&1 < /dev/null &
done

# ---------- Task 2: Centralized BC 扩 N=20/30/50 (GPU 2/3/4, 避开 0 号卡) ----------
setsid env CUDA_VISIBLE_DEVICES=2 "$PY" train_centralized.py \
  --num_uav 20 --gpu 2 --save_name centralized_bc_N20 --seed 42 \
  > "$LOG/cb_N20.log" 2>&1 < /dev/null &
setsid env CUDA_VISIBLE_DEVICES=3 "$PY" train_centralized.py \
  --num_uav 30 --gpu 3 --save_name centralized_bc_N30 --seed 42 \
  > "$LOG/cb_N30.log" 2>&1 < /dev/null &
setsid env CUDA_VISIBLE_DEVICES=4 "$PY" train_centralized.py \
  --num_uav 50 --gpu 4 --save_name centralized_bc_N50 --seed 42 \
  > "$LOG/cb_N50.log" 2>&1 < /dev/null &

echo "launched all 7 jobs (4 zeroshot CPU + 3 centralized BC GPU 2/3/4)"
