#!/bin/bash
# ============================================================
# 批量实验脚本：area=750m 动态场景下 BC vs 周期性贪心对比
# 服务器路径: /mnt/data1/liushiyang/large_freq_plan
# ============================================================
set -e

PROJECT_DIR="/mnt/data1/liushiyang/large_freq_plan"
CONDA_ENV="uav"
PYTHON="/home/lsy/.conda/envs/uav/bin/python"
RESULTS_DIR="$PROJECT_DIR/results_data"

cd $PROJECT_DIR
mkdir -p $RESULTS_DIR

# ============================================================
# Phase 1: 训练 BC N=10, area=750m (GPU 1)
# ============================================================
echo "=============================================="
echo "Phase 1: 训练 BC (N=10, area=750m, GPU 1)"
echo "=============================================="
CUDA_VISIBLE_DEVICES=1 $PYTHON train.py \
    --method bc \
    --num_uav 10 \
    --area_size 750 \
    --gpu 1 \
    --save_name bc_area750_N10 \
    --bc_expert_episodes 500 \
    --bc_epochs 50 \
    --seed 42

echo "Phase 1 完成！模型保存在 checkpoints/bc_area750_N10.pt"

# ============================================================
# Phase 2: 并行评估（多GPU + 后台CPU任务）
# ============================================================
echo ""
echo "=============================================="
echo "Phase 2: 并行评估"
echo "=============================================="

CKPT="$PROJECT_DIR/checkpoints/bc_area750_N10.pt"

# --- GPU 1: BC 动态评估 N=10 ---
echo "启动 GPU1: BC evaluation N=10"
CUDA_VISIBLE_DEVICES=1 $PYTHON eval_dynamic_marl.py \
    --ckpt $CKPT \
    --method bc \
    --num_uav 10 \
    --area_size 750 \
    --num_slots 50 \
    --num_episodes 10 \
    --seed 42 \
    --output $RESULTS_DIR/dynamic_marl_area750 \
    > $RESULTS_DIR/log_bc_eval_N10.txt 2>&1 &
PID_BC_EVAL=$!

# --- CPU: 动态基线对比 (各speed并行) ---
for speed in 0 20; do
    echo "启动 CPU: baseline comparison speed=$speed"
    $PYTHON run_dynamic_comparison.py \
        --num_uav 10 \
        --num_slots 50 \
        --uav_speed $speed \
        --area_size 750 \
        --replan_intervals 5 10 20 \
        --num_episodes 10 \
        --seed 42 \
        --output $RESULTS_DIR/dynamic_area750 \
        > $RESULTS_DIR/log_baseline_speed${speed}.txt 2>&1 &
done

# --- CPU: 分布式基线对比 (各speed并行) ---
for speed in 0 20; do
    echo "启动 CPU: dist baseline speed=$speed"
    $PYTHON run_dist_comparison.py \
        --num_uav 10 \
        --num_slots 50 \
        --speeds $speed \
        --area_size 750 \
        --num_episodes 10 \
        --seed 42 \
        --output $RESULTS_DIR/dist_area750 \
        > $RESULTS_DIR/log_dist_speed${speed}.txt 2>&1 &
done

# --- GPU 2: BC 动态评估 N=20 (zero-shot from N=10 model) ---
echo "启动 GPU2: BC zero-shot N=20"
CUDA_VISIBLE_DEVICES=2 $PYTHON eval_dynamic_marl.py \
    --ckpt $CKPT \
    --method bc \
    --num_uav 20 \
    --area_size 750 \
    --num_slots 50 \
    --num_episodes 10 \
    --seed 42 \
    --output $RESULTS_DIR/dynamic_marl_area750 \
    > $RESULTS_DIR/log_bc_eval_N20_zeroshot.txt 2>&1 &
PID_BC_N20=$!

# --- GPU 3: BC 动态评估 N=30 (zero-shot) ---
echo "启动 GPU3: BC zero-shot N=30"
CUDA_VISIBLE_DEVICES=3 $PYTHON eval_dynamic_marl.py \
    --ckpt $CKPT \
    --method bc \
    --num_uav 30 \
    --area_size 750 \
    --num_slots 50 \
    --num_episodes 10 \
    --seed 42 \
    --output $RESULTS_DIR/dynamic_marl_area750 \
    > $RESULTS_DIR/log_bc_eval_N30_zeroshot.txt 2>&1 &
PID_BC_N30=$!

# --- GPU 4: 额外speed下的baseline (speed=10,30 和 baseline) ---
for speed in 10 30; do
    echo "启动 CPU: baseline comparison speed=$speed"
    $PYTHON run_dynamic_comparison.py \
        --num_uav 10 \
        --num_slots 50 \
        --uav_speed $speed \
        --area_size 750 \
        --replan_intervals 5 10 20 \
        --num_episodes 10 \
        --seed 42 \
        --output $RESULTS_DIR/dynamic_area750 \
        > $RESULTS_DIR/log_baseline_speed${speed}.txt 2>&1 &
done

# ============================================================
# 等待所有后台任务完成
# ============================================================
echo ""
echo "等待所有评估任务完成..."
wait $PID_BC_EVAL
wait $PID_BC_N20
wait $PID_BC_N30
wait  # 等待所有后台任务

echo ""
echo "=============================================="
echo "全部实验完成！结果保存在 $RESULTS_DIR"
echo "=============================================="

# 打印结果汇总
echo ""
echo "========== BC 评估结果 =========="
cat $RESULTS_DIR/dynamic_marl_area750/dynamic_bc_bc_area750_N10.json 2>/dev/null | $PYTHON -c "
import json, sys
d = json.load(sys.stdin)
for k, v in d.get('results', {}).items():
    print(f\"  {k}: P_int={v['cum_interf_prob_mean']:.4f}±{v['cum_interf_prob_std']:.4f}, time={v['avg_total_inference_time_s']:.3f}s\")
" 2>/dev/null || echo "(BC eval results pending)"

echo ""
echo "========== Baseline 结果 =========="
for f in $RESULTS_DIR/dynamic_area750/dynamic_N10_speed*.json; do
    [ -f "$f" ] || continue
    echo "--- $f ---"
    $PYTHON -c "
import json, sys
d = json.load(open('$f'))
for s in d.get('summary', []):
    print(f\"  {s['method']:25s}: P_int={s['cum_interf_prob_mean']:.4f}±{s['cum_interf_prob_std']:.4f}, time={s['avg_total_time_s']:.1f}s\")
" 2>/dev/null
done
