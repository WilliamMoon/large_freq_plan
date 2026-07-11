#!/bin/bash
cd /mnt/data1/liushiyang/large_freq_plan
for i in $(seq 1 90); do
  [ -f checkpoints/bc_dynamic_s10_N20.pt ] && break
  sleep 30
done
if [ ! -f checkpoints/bc_dynamic_s10_N20.pt ]; then
  echo "CKPT MISSING after wait"; tail -n 5 logs/bc_dyn_s10_N20.log; exit 1
fi
echo "ckpt ready: $(ls -la checkpoints/bc_dynamic_s10_N20.pt)"
CUDA_VISIBLE_DEVICES=4 /home/lsy/.conda/envs/uav/bin/python eval_dynamic_marl.py --method bc --ckpt checkpoints/bc_dynamic_s10_N20.pt --num_uav 20 --speeds 0 10 20 30 40 --num_episodes 20 --num_slots 50 --output results 2>&1 | tail -n 20
cp results/dynamic_bc_bc_dynamic_s10_N20.json results_data/dynamic_bc_bc_dynamic_s10_N20.json && echo COPIED_TO_RESULTS_DATA
