"""诊断 CARLTON：加载训练好的 q_net，实测 P_int 并检查策略是否退化成随机。"""
import json, numpy as np, torch
from carlton_trainer import CarltonTrainer, mellowmax_dist
from marl_env import MultiAgentEnv
from config import ENV_CONFIG

ckpt = "checkpoints/carlton_N10_fix.pt"
d = json.load(open("results_data/dynamic_carlton_N10.json"))
num_uav = d["config"]["num_uav"]

dev = "cpu"
_obs_dim = torch.load(ckpt, map_location="cpu", weights_only=False)["obs_dim"]
tr = CarltonTrainer(num_uav=num_uav, obs_dim=int(_obs_dim), device=dev)
tr.load(ckpt)
qnet = tr.q_net.to(dev)
qnet.eval()

# 1) 静态环境实测 P_int（用 argmax 执行，与 eval 一致）
env = MultiAgentEnv(num_uav=num_uav,
                    observation_radius=ENV_CONFIG["observation_radius"],
                    area_size=ENV_CONFIG["area_size"],
                    limit_neighbors=ENV_CONFIG["limit_neighbors"])
ps = []
for _ in range(50):
    env.reset()
    tr.select_actions(env, evaluate=True)
    ps.append(env.get_interference_prob())
print(f"[CARLTON static] mean P_int = {np.mean(ps):.4f}  (dist_coloring=0.2069, random=0.4159)")

# 2) 检查 q_net 输出分布：策略是否接近随机？
# 采样 200 个随机 obs
rng = np.random.default_rng(0)
obs_dim = tr.obs_dim
obs = rng.standard_normal((200, obs_dim)).astype(np.float32)
with torch.no_grad():
    Q = qnet(torch.tensor(obs, device=dev)).cpu().numpy()  # (200, n_actions)
per_row_std = Q.std(axis=1).mean()
argmax = Q.argmax(axis=1)
# argmax 在所有动作上的频率均匀度：均匀分布下每个动作占比=1/n_actions
counts = np.bincount(argmax, minlength=tr.n_actions)
freq = counts / counts.sum()
uniformity = freq.std()  # 越小越接近均匀(随机)
print(f"[Q dist] n_actions={tr.n_actions},  per-row Q std = {per_row_std:.4f}  (小=>各动作Q相近)")
print(f"[Q dist] argmax 频率分布 std = {uniformity:.4f}  (随机策略时≈{ (1/tr.n_actions)*(1-1/tr.n_actions):.4f})")
print(f"[Q dist] top-1 动作占比 = {freq.max():.3f}  (越接近1越确定，越接近{1/tr.n_actions:.3f}越随机)")

# 3) 对比：如果策略是纯随机，P_int 应≈ random(0.4159)；若确定则更低
# 计算 argmax 复用度（同一 obs 重复前向是否一致）
with torch.no_grad():
    Q2 = qnet(torch.tensor(obs, device=dev)).cpu().numpy()
am2 = Q2.argmax(axis=1)
agree = (argmax == am2).mean()
print(f"[stability] 同 obs 两次 argmax 一致率 = {agree:.3f}")
