#!/bin/bash
# ============================================================
# 并行改进实验：4条路线同时跑，各占一块GPU
# ============================================================
PROJECT_DIR="/mnt/data1/liushiyang/large_freq_plan"
PYTHON="/home/lsy/.conda/envs/uav/bin/python"
RESULTS_DIR="$PROJECT_DIR/results_data"
cd $PROJECT_DIR

# ============================================================
# Route A: BC 重训 area=750m, 2000 expert episodes (GPU 1)
# ============================================================
cat > $PROJECT_DIR/run_routeA.sh << 'EOF'
#!/bin/bash
PROJECT_DIR="/mnt/data1/liushiyang/large_freq_plan"
PYTHON="/home/lsy/.conda/envs/uav/bin/python"
cd $PROJECT_DIR
echo "=== Route A: BC retrain area=750m, 2000 episodes, GPU 1 ==="

# 训练
CUDA_VISIBLE_DEVICES=1 $PYTHON train.py \
    --method bc --num_uav 10 --area_size 750 --gpu 1 \
    --save_name bc_area750_N10_v2 \
    --bc_expert_episodes 2000 --bc_epochs 80 --seed 42

CKPT="$PROJECT_DIR/checkpoints/bc_area750_N10_v2.pt"

# 动态评估
CUDA_VISIBLE_DEVICES=1 $PYTHON eval_dynamic_marl.py \
    --ckpt $CKPT --method bc --num_uav 10 --area_size 750 \
    --num_slots 50 --num_episodes 10 --seed 42 \
    --output $PROJECT_DIR/results_data/routeA_dynamic

echo "=== Route A 完成 ==="
EOF
chmod +x $PROJECT_DIR/run_routeA.sh

# ============================================================
# Route B: BC→IQL 微调 area=750m (GPU 2)
# ============================================================
cat > $PROJECT_DIR/run_routeB.sh << 'EOF'
#!/bin/bash
PROJECT_DIR="/mnt/data1/liushiyang/large_freq_plan"
PYTHON="/home/lsy/.conda/envs/uav/bin/python"
cd $PROJECT_DIR
echo "=== Route B: BC→IQL fine-tune area=750m, GPU 2 ==="

BC_CKPT="$PROJECT_DIR/checkpoints/bc_area750_N10.pt"

# BC+IQL 微调
CUDA_VISIBLE_DEVICES=2 $PYTHON train.py \
    --method bc_iql --num_uav 10 --area_size 750 --gpu 2 \
    --bc_ckpt $BC_CKPT \
    --save_name bc_iql_area750_N10 \
    --episodes 300 --warmup 20 --seed 42

CKPT="$PROJECT_DIR/checkpoints/bc_iql_area750_N10.pt"

# 动态评估
CUDA_VISIBLE_DEVICES=2 $PYTHON eval_dynamic_marl.py \
    --ckpt $CKPT --method iql --num_uav 10 --area_size 750 \
    --num_slots 50 --num_episodes 10 --seed 42 \
    --output $PROJECT_DIR/results_data/routeB_dynamic

echo "=== Route B 完成 ==="
EOF
chmod +x $PROJECT_DIR/run_routeB.sh

# ============================================================
# Route C: BC 重训 area=1000m (GPU 3)
# ============================================================
cat > $PROJECT_DIR/run_routeC.sh << 'EOF'
#!/bin/bash
PROJECT_DIR="/mnt/data1/liushiyang/large_freq_plan"
PYTHON="/home/lsy/.conda/envs/uav/bin/python"
cd $PROJECT_DIR
echo "=== Route C: BC retrain area=1000m, GPU 3 ==="

# 训练
CUDA_VISIBLE_DEVICES=3 $PYTHON train.py \
    --method bc --num_uav 10 --area_size 1000 --gpu 3 \
    --save_name bc_area1000_N10 \
    --bc_expert_episodes 1000 --bc_epochs 60 --seed 42

CKPT="$PROJECT_DIR/checkpoints/bc_area1000_N10.pt"

# 动态评估 BC
CUDA_VISIBLE_DEVICES=3 $PYTHON eval_dynamic_marl.py \
    --ckpt $CKPT --method bc --num_uav 10 --area_size 1000 \
    --num_slots 50 --num_episodes 10 --seed 42 \
    --output $PROJECT_DIR/results_data/routeC_dynamic

# 基线对比 area=1000m
for speed in 0 20; do
    $PYTHON run_dynamic_comparison.py \
        --num_uav 10 --num_slots 50 --uav_speed $speed \
        --area_size 1000 --replan_intervals 5 10 20 \
        --num_episodes 10 --seed 42 \
        --output $PROJECT_DIR/results_data/routeC_baseline
done

echo "=== Route C 完成 ==="
EOF
chmod +x $PROJECT_DIR/run_routeC.sh

# ============================================================
# Route D: area=2000m 大间隔周期性贪心退化 (GPU 4, 仅评估)
# ============================================================
cat > $PROJECT_DIR/run_routeD.sh << 'EOF'
#!/bin/bash
PROJECT_DIR="/mnt/data1/liushiyang/large_freq_plan"
PYTHON="/home/lsy/.conda/envs/uav/bin/python"
cd $PROJECT_DIR
echo "=== Route D: area=2000m baseline with large replan intervals ==="

# 用已有的 bc_orthogonal_N10.pt 做动态评估
BC_CKPT="$PROJECT_DIR/checkpoints/bc_orthogonal_N10.pt"
if [ -f "$BC_CKPT" ]; then
    CUDA_VISIBLE_DEVICES=4 $PYTHON eval_dynamic_marl.py \
        --ckpt $BC_CKPT --method bc --num_uav 10 --area_size 2000 \
        --num_slots 50 --num_episodes 10 --seed 42 \
        --output $PROJECT_DIR/results_data/routeD_dynamic
fi

# 基线对比 with 大间隔
for speed in 0 10 20 30; do
    $PYTHON run_dynamic_comparison.py \
        --num_uav 10 --num_slots 50 --uav_speed $speed \
        --area_size 2000 --replan_intervals 5 10 20 30 50 \
        --num_episodes 10 --seed 42 \
        --output $PROJECT_DIR/results_data/routeD_baseline
done

echo "=== Route D 完成 ==="
EOF
chmod +x $PROJECT_DIR/run_routeD.sh

# ============================================================
# 启动4个tmux session并行
# ============================================================
tmux new-session -d -s routeA "bash $PROJECT_DIR/run_routeA.sh 2>&1 | tee $RESULTS_DIR/log_routeA.txt"
tmux new-session -d -s routeB "bash $PROJECT_DIR/run_routeB.sh 2>&1 | tee $RESULTS_DIR/log_routeB.txt"
tmux new-session -d -s routeC "bash $PROJECT_DIR/run_routeC.sh 2>&1 | tee $RESULTS_DIR/log_routeC.txt"
tmux new-session -d -s routeD "bash $PROJECT_DIR/run_routeD.sh 2>&1 | tee $RESULTS_DIR/log_routeD.txt"

echo "4条路线已启动："
echo "  Route A (GPU1): BC重训 area=750m, 2000 episodes"
echo "  Route B (GPU2): BC→IQL微调 area=750m"
echo "  Route C (GPU3): BC重训 area=1000m"
echo "  Route D (GPU4): area=2000m 大间隔基线"
echo ""
echo "监控: tmux attach -t routeA / routeB / routeC / routeD"
