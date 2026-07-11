#!/bin/bash
# 重训 CARLTON（修复 mellowmax omega 调度 + 去掉 reward 标准化），覆盖 N=10/20/30/50。
set -u
PROJ=/mnt/data1/liushiyang/large_freq_plan
cd "$PROJ"
PY=/home/lsy/.conda/envs/uav/bin/python
LOG=night_logs/carlton_fix
mkdir -p "$LOG"

for n in 10 20 30 50; do
  echo "[$(date +%H:%M:%S)] ===== CARLTON N=$n fix train ====="
  CUDA_VISIBLE_DEVICES=4 $PY train.py --method carlton --num_uav $n \
    --episodes 2500 --warmup 50 --eval_interval 50 \
    --save_name carlton_N${n}_fix --seed 42 --gpu 4 \
    > "$LOG/train_N$n.log" 2>&1
  echo "[$(date +%H:%M:%S)] train done (N=$n)"

  if [ -f checkpoints/carlton_N${n}_fix.pt ]; then
    CUDA_VISIBLE_DEVICES=4 $PY eval_dynamic_marl.py --method carlton \
      --ckpt checkpoints/carlton_N${n}_fix.pt --num_uav $n \
      --speeds 0 10 20 30 40 --num_episodes 20 --num_slots 50 \
      --output results_data \
      > "$LOG/eval_N$n.log" 2>&1
    mv -f "results_data/dynamic_carlton_carlton_N${n}_fix.json" \
          "results_data/dynamic_carlton_N${n}.json"
    echo "[$(date +%H:%M:%S)] eval done -> results_data/dynamic_carlton_N${n}.json"
  else
    echo "[$(date +%H:%M:%S)] NO CKPT for N=$n"
  fi
done
echo "[$(date +%H:%M:%S)] ALL CARLTON FIX DONE" | tee "$LOG/done.sentinel"
