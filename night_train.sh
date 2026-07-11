#!/bin/bash
# 夜间训练+评估驱动脚本（服务器侧运行）
# GPU 分配（每张卡顺序执行，避免显存争用）：
#   GPU2: BC N=10 -> BC N=50 -> CARLTON N=10 re-eval -> CARLTON N=20
#   GPU3: BC N=20 -> CARLTON N=30
#   GPU4: BC N=30 -> CARLTON N=50
set -u
cd /mnt/data1/liushiyang/large_freq_plan
export PYTHON=/home/lsy/.conda/envs/uav/bin/python
export PYTHONPATH=$(pwd)
LOG=night_logs
mkdir -p "$LOG"

run_bc() {
  local n=$1; local expert=$2; local epochs=$3; local gpu=$4
  echo "[$(date +%H:%M)] BC N=$n start (expert=$expert epochs=$epochs gpu=$gpu)" >> "$LOG/schedule.log"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON train.py --method bc --num_uav $n \
    --bc_expert_episodes $expert --bc_epochs $epochs --eval_interval 10 \
    --save_name bc_orthogonal_N$n --gpu $gpu --seed 42 >> "$LOG/bc_N$n.log" 2>&1
  echo "[$(date +%H:%M)] BC N=$n train done" >> "$LOG/schedule.log"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON eval_dynamic_marl.py --method bc \
    --ckpt checkpoints/bc_orthogonal_N$n.pt --num_uav $n \
    --speeds 0 10 20 30 40 --num_episodes 20 --num_slots 50 \
    --output results >> "$LOG/eval_bc_N$n.log" 2>&1
  echo "[$(date +%H:%M)] BC N=$n eval done" >> "$LOG/schedule.log"
}

run_carlton() {
  local n=$1; local episodes=$2; local gpu=$3
  echo "[$(date +%H:%M)] CARLTON N=$n start (episodes=$episodes gpu=$gpu)" >> "$LOG/schedule.log"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON train.py --method carlton --num_uav $n \
    --episodes $episodes --eval_interval 50 --save_name carlton_N$n --gpu $gpu \
    --seed 42 >> "$LOG/carlton_N$n.log" 2>&1
  echo "[$(date +%H:%M)] CARLTON N=$n train done" >> "$LOG/schedule.log"
  if [ -f checkpoints/carlton_N$n.pt ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON eval_dynamic_marl.py --method carlton \
      --ckpt checkpoints/carlton_N$n.pt --num_uav $n \
      --speeds 0 10 20 30 40 --num_episodes 20 --num_slots 50 \
      --output results >> "$LOG/eval_carlton_N$n.log" 2>&1
    mv -f results/dynamic_carlton_carlton_N$n.json results/dynamic_carlton_N$n.json
    echo "[$(date +%H:%M)] CARLTON N=$n eval done" >> "$LOG/schedule.log"
  else
    echo "[$(date +%H:%M)] CARLTON N=$n no ckpt, skip eval" >> "$LOG/schedule.log"
  fi
}

# 分布式基线对比（CPU，无需GPU）：所有 N 的所有速度
( for n in 10 20 30 50; do
    echo "[$(date +%H:%M)] dist N=$n start" >> "$LOG/schedule.log"
    $PYTHON run_dist_comparison.py --num_uav $n --speeds 0 10 20 30 40 \
      --num_episodes 20 --output results >> "$LOG/dist_N$n.log" 2>&1
    echo "[$(date +%H:%M)] dist N=$n done" >> "$LOG/schedule.log"
  done ) &

# CARLTON N=10 重新评估（补 speed_40），用已有 ckpt
( CUDA_VISIBLE_DEVICES=2 $PYTHON eval_dynamic_marl.py --method carlton \
    --ckpt checkpoints/carlton_N10_s42.pt --num_uav 10 \
    --speeds 0 10 20 30 40 --num_episodes 20 --num_slots 50 \
    --output results >> "$LOG/eval_carlton_N10.log" 2>&1
  mv -f results/dynamic_carlton_carlton_N10.json results/dynamic_carlton_N10.json
  echo "[$(date +%H:%M)] CARLTON N=10 re-eval done" >> "$LOG/schedule.log"
) &

# GPU2: BC10 -> BC50 -> CARLTON20
( run_bc 10 2000 1000 2
  run_bc 50 1500 1000 2
  run_carlton 20 2500 2
) &

# GPU3: BC20 -> CARLTON30
( run_bc 20 1500 1000 3
  run_carlton 30 2500 3
) &

# GPU4: BC30 -> CARLTON50
( run_bc 30 1500 1000 4
  run_carlton 50 2500 4
) &

wait
echo "[$(date +%H:%M)] ALL NIGHT JOBS DONE" >> "$LOG/schedule.log"
