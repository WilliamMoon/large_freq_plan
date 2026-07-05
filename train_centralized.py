"""训练中心化 BC 并评估。"""
import argparse
import os
import json
import time
import numpy as np
import torch

from config import ENV_CONFIG, ACTION_CONFIG, CHECKPOINT_DIR, RESULTS_DIR, LAYOUTS_DIR, LoggerSingleton
from marl_env import MultiAgentEnv
from centralized_rl import CentralizedBC

logger = LoggerSingleton.get_instance()


def main():
    parser = argparse.ArgumentParser(description="训练中心化BC")
    parser.add_argument("--num_uav", type=int, default=10)
    parser.add_argument("--expert_episodes", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_name", type=str, default="centralized_bc_N10")
    args = parser.parse_args()

    assert args.gpu >= 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 确定全局状态维度: num_uav * 6 (x,y,z,power,freq,sinr)
    global_state_dim = args.num_uav * 6
    
    trainer = CentralizedBC(
        num_uav=args.num_uav,
        global_state_dim=global_state_dim,
        device=device,
    )
    
    # 加载评估布局
    layout_path = os.path.join(LAYOUTS_DIR, f"N{args.num_uav}_50layouts.json")
    eval_layouts = []
    if os.path.exists(layout_path):
        with open(layout_path, "r") as f:
            eval_layouts = list(json.load(f).values())
    
    eval_env = MultiAgentEnv(
        num_uav=args.num_uav,
        observation_radius=ENV_CONFIG["observation_radius"],
        area_size=ENV_CONFIG["area_size"],
        limit_neighbors=ENV_CONFIG["limit_neighbors"],
    )
    
    # 收集专家数据
    start_time = time.time()
    expert_data = trainer.collect_expert_data(num_episodes=args.expert_episodes)
    
    # 训练
    history = trainer.train(
        expert_data, num_epochs=args.epochs,
        eval_env=eval_env, eval_layouts=eval_layouts,
        eval_interval=10,
    )
    elapsed = time.time() - start_time
    logger.info(f"训练完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    # 保存
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.save_name}.pt")
    trainer.save(ckpt_path)
    
    stats_path = os.path.join(RESULTS_DIR, f"{args.save_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(history, f, indent=2)
    
    # 最终评估
    if eval_layouts:
        probs = trainer._evaluate(eval_env, eval_layouts)
        logger.info(f"中心化BC最终评估: P_int={np.mean(probs):.4f}±{np.std(probs):.4f}")


if __name__ == "__main__":
    main()
