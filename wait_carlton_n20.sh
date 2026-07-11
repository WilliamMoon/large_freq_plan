#!/usr/bin/env bash
# 等 CARLTON N=20 训练结束，再跑其动态评估，最后写 sentinel。
set -e
PROJ=/mnt/data1/liushiyang/large_freq_plan
cd "$PROJ"
export PYTHONPATH="$PROJ"
TRAIN_PID=2753572
PY=/home/lsy/.conda/envs/uav/bin/python

echo "[$(date +%H:%M:%S)] waiting for CARLTON N=20 training (PID $TRAIN_PID)..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] training done; checking for existing dynamic eval..."

# 若 schedule 脚本已产出动态评估文件则跳过，避免重复
if ls "$PROJ"/results/dynamic_carlton_carlton_N20*.json >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] dynamic eval already present, skip"
else
    echo "[$(date +%H:%M:%S)] running CARLTON N=20 dynamic eval..."
    "$PY" eval_dynamic_marl.py --method carlton --ckpt checkpoints/carlton_N20.pt \
        --num_uav 20 --speeds 0 10 20 30 40 --num_episodes 20 --num_slots 50 \
        --output results > night_logs/carlton_N20_eval.log 2>&1
fi

echo "[$(date +%H:%M:%S)] DONE" > "$PROJ"/results/dynamic_carlton_N20_done.sentinel
echo "[$(date +%H:%M:%S)] sentinel written, chain finished"
