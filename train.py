"""统一训练入口：支持 BC / IQL / BC+IQL微调。"""
import argparse
import os
import json
import time
import numpy as np
import torch

from config import ENV_CONFIG, ACTION_CONFIG, MADDPG_CONFIG, CHECKPOINT_DIR, RESULTS_DIR, LAYOUTS_DIR, LoggerSingleton
from marl_env import MultiAgentEnv
from maddpg_trainer import MADDPGTrainer
from iql_trainer import IQLTrainer
from bc_trainer import BCTrainer
from carlton_trainer import CarltonTrainer

logger = LoggerSingleton.get_instance()


def main():
    parser = argparse.ArgumentParser(description="训练")
    parser.add_argument("--method", type=str, default="iql", choices=["maddpg", "iql", "bc", "bc_iql", "carlton"], help="训练方法")
    parser.add_argument("--num_uav", type=int, default=10, help="训练规模")
    parser.add_argument("--episodes", type=int, default=500, help="训练轮数")
    parser.add_argument("--warmup", type=int, default=50, help="热身轮数")
    parser.add_argument("--gpu", type=int, default=1, help="GPU编号(1-5)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--eval_interval", type=int, default=25, help="评估间隔")
    parser.add_argument("--save_name", type=str, default=None, help="保存名称")
    # BC 相关
    parser.add_argument("--bc_expert_episodes", type=int, default=500, help="BC收集专家数据的episode数")
    parser.add_argument("--bc_epochs", type=int, default=50, help="BC训练轮数")
    parser.add_argument("--area_size", type=float, default=None, help="区域大小（默认使用ENV_CONFIG值）")
    # BC+IQL 微调相关
    parser.add_argument("--bc_ckpt", type=str, default=None, help="BC预训练checkpoint路径（用于bc_iql微调）")
    args = parser.parse_args()

    assert args.gpu >= 1, "禁止使用 GPU 0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    area_size = args.area_size or ENV_CONFIG["area_size"]

    # 创建环境获取 obs_dim
    env = MultiAgentEnv(
        num_uav=args.num_uav,
        observation_radius=ENV_CONFIG["observation_radius"],
        area_size=area_size,
        limit_neighbors=ENV_CONFIG["limit_neighbors"],
    )
    env.reset()
    obs_dim = env.obs_dim
    logger.info(f"Method={args.method}, N={args.num_uav}, area_size={area_size}, obs_dim={obs_dim}, device={device}")
    
    # 加载评估布局
    layout_path = os.path.join(LAYOUTS_DIR, f"N{args.num_uav}_50layouts.json")
    eval_layouts = []
    if os.path.exists(layout_path):
        with open(layout_path, "r") as f:
            layouts_dict = json.load(f)
        eval_layouts = list(layouts_dict.values())
        logger.info(f"加载 {len(eval_layouts)} 个评估布局")
    
    eval_env = MultiAgentEnv(
        num_uav=args.num_uav,
        observation_radius=ENV_CONFIG["observation_radius"],
        area_size=area_size,
        limit_neighbors=ENV_CONFIG["limit_neighbors"],
    )
    
    save_name = args.save_name or f"{args.method}_N{args.num_uav}"
    
    # ==================== BC 训练 ====================
    if args.method == "bc":
        trainer = BCTrainer(num_uav=args.num_uav, obs_dim=obs_dim, device=device)
        
        start_time = time.time()
        # 收集专家数据
        expert_data = trainer.collect_expert_data(num_episodes=args.bc_expert_episodes, area_size=area_size)
        collect_time = time.time() - start_time
        logger.info(f"专家数据收集完成，耗时 {collect_time:.1f}s")
        
        # BC 训练
        bc_history = trainer.train(
            expert_data, num_epochs=args.bc_epochs,
            batch_size=256, eval_env=eval_env, eval_layouts=eval_layouts,
            eval_interval=5,
        )
        elapsed = time.time() - start_time
        logger.info(f"BC 训练完成，总耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
        
        # 保存
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{save_name}.pt")
        trainer.save(ckpt_path)
        
        stats_path = os.path.join(RESULTS_DIR, f"{save_name}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(bc_history, f, indent=2)
        
        # 最终评估
        if eval_layouts:
            probs, sinrs = trainer.evaluate_on_layouts(eval_env, eval_layouts)
            avg_prob = float(np.mean(probs))
            std_prob = float(np.std(probs))
            avg_sinr = float(np.mean(sinrs))
            logger.info(f"BC 最终评估 N={args.num_uav}: P_int={avg_prob:.4f}±{std_prob:.4f}, avg_SINR={avg_sinr:.2f}dB")
            
        result = {
            "method": "bc", "uav_count": args.num_uav,
            "area_size": area_size,
            "avg_interference_prob": avg_prob, "std_interference_prob": std_prob,
            "avg_sinr": avg_sinr, "training_time_s": elapsed,
            "bc_expert_episodes": args.bc_expert_episodes, "bc_epochs": args.bc_epochs,
        }
        with open(os.path.join(RESULTS_DIR, f"{save_name}_result.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return
    
    # ==================== BC+IQL 微调 ====================
    if args.method == "bc_iql":
        # 先加载 BC 预训练模型
        bc_ckpt = args.bc_ckpt or os.path.join(CHECKPOINT_DIR, f"bc_N{args.num_uav}.pt")
        if not os.path.exists(bc_ckpt):
            logger.error(f"BC checkpoint 不存在: {bc_ckpt}，请先运行 BC 训练")
            return
        
        # 创建 IQL trainer 并加载 BC 权重（hidden_dim 需匹配 BC 模型=256）
        trainer = IQLTrainer(num_uav=args.num_uav, obs_dim=obs_dim, device=device, hidden_dim=256)
        bc_ckpt_data = torch.load(bc_ckpt, map_location=device, weights_only=False)
        trainer.q_net.load_state_dict(bc_ckpt_data["q_net"])
        trainer.q_target.load_state_dict(bc_ckpt_data["q_net"])
        logger.info(f"从 BC checkpoint 加载权重: {bc_ckpt}")
        
        # 先评估 BC 预训练模型的性能
        if eval_layouts:
            probs, sinrs = trainer.evaluate_on_layouts(eval_env, eval_layouts)
            logger.info(f"BC 预训练模型评估: P_int={np.mean(probs):.4f}±{np.std(probs):.4f}")
        
        # RL 微调（低epsilon避免破坏BC策略）
        start_time = time.time()
        stats = trainer.train(
            num_episodes=args.episodes, warmup=args.warmup,
            eval_env=eval_env, eval_layouts=eval_layouts,
            eval_interval=args.eval_interval,
            epsilon_start=0.1, epsilon_end=0.02, epsilon_decay=0.999,
        )
        elapsed = time.time() - start_time
        logger.info(f"IQL 微调完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
        
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{save_name}.pt")
        trainer.save(ckpt_path)
        
        stats_path = os.path.join(RESULTS_DIR, f"{save_name}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        
        if eval_layouts:
            probs, sinrs = trainer.evaluate_on_layouts(eval_env, eval_layouts)
            avg_prob = float(np.mean(probs))
            std_prob = float(np.std(probs))
            avg_sinr = float(np.mean(sinrs))
            logger.info(f"BC+IQL 最终评估 N={args.num_uav}: P_int={avg_prob:.4f}±{std_prob:.4f}, avg_SINR={avg_sinr:.2f}dB")
            
            result = {
                "method": "bc_iql", "uav_count": args.num_uav,
                "avg_interference_prob": avg_prob, "std_interference_prob": std_prob,
                "avg_sinr": avg_sinr, "training_time_s": elapsed,
                "episodes": args.episodes, "bc_ckpt": bc_ckpt,
            }
            with open(os.path.join(RESULTS_DIR, f"{save_name}_result.json"), "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        return
    
    # ==================== CARLTON 训练（CTDE value-based, DeepMellow 风格基线）====================
    if args.method == "carlton":
        trainer = CarltonTrainer(num_uav=args.num_uav, obs_dim=obs_dim, device=device)

        start_time = time.time()
        stats = trainer.train(
            num_episodes=args.episodes, warmup=args.warmup,
            eval_env=eval_env, eval_layouts=eval_layouts,
            eval_interval=args.eval_interval,
        )
        elapsed = time.time() - start_time
        logger.info(f"CARLTON 训练完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{save_name}.pt")
        trainer.save(ckpt_path)

        stats_path = os.path.join(RESULTS_DIR, f"{save_name}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        logger.info(f"训练曲线保存至 {stats_path}")

        if eval_layouts:
            probs, sinrs = trainer.evaluate_on_layouts(eval_env, eval_layouts)
            avg_prob = float(np.mean(probs))
            std_prob = float(np.std(probs))
            avg_sinr = float(np.mean(sinrs))
            logger.info(f"CARLTON 最终评估 N={args.num_uav}: P_int={avg_prob:.4f}±{std_prob:.4f}, avg_SINR={avg_sinr:.2f}dB")
            result = {
                "method": "carlton", "uav_count": args.num_uav,
                "avg_interference_prob": avg_prob, "std_interference_prob": std_prob,
                "avg_sinr": avg_sinr, "training_time_s": elapsed, "episodes": args.episodes,
            }
            with open(os.path.join(RESULTS_DIR, f"{save_name}_result.json"), "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        return

    # ==================== MADDPG / IQL 训练 ====================
    if args.method == "maddpg":
        trainer = MADDPGTrainer(num_uav=args.num_uav, obs_dim=obs_dim, device=device)
    else:
        trainer = IQLTrainer(num_uav=args.num_uav, obs_dim=obs_dim, device=device)
    
    start_time = time.time()
    stats = trainer.train(
        num_episodes=args.episodes, warmup=args.warmup,
        eval_env=eval_env, eval_layouts=eval_layouts,
        eval_interval=args.eval_interval,
    )
    elapsed = time.time() - start_time
    logger.info(f"训练完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{save_name}.pt")
    trainer.save(ckpt_path)
    
    stats_path = os.path.join(RESULTS_DIR, f"{save_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)
    logger.info(f"训练曲线保存至 {stats_path}")
    
    if eval_layouts:
        probs, sinrs = trainer.evaluate_on_layouts(eval_env, eval_layouts)
        avg_prob = float(np.mean(probs))
        std_prob = float(np.std(probs))
        avg_sinr = float(np.mean(sinrs))
        logger.info(f"最终评估 N={args.num_uav}: P_int={avg_prob:.4f}±{std_prob:.4f}, avg_SINR={avg_sinr:.2f}dB")
        
        result = {
            "method": args.method, "uav_count": args.num_uav,
            "avg_interference_prob": avg_prob, "std_interference_prob": std_prob,
            "avg_sinr": avg_sinr, "training_time_s": elapsed, "episodes": args.episodes,
        }
        result_path = os.path.join(RESULTS_DIR, f"{save_name}_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
