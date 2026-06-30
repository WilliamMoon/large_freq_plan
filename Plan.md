# 项目：大规模无人集群 MARL 用频规划（服务器端，从0重写RL代码）

## 环境
- 工作区: [服务器上的项目路径]
- Python: 见 .codebuddy/CODEBUDDY.md

## 策略决定
上一session已审计全部现有代码。决定：
✅ 保留: env.py, node.py（物理层建模，已验证正确）
✅ 保留: marl_env.py（环境包装，review后使用）
✅ 保留: test_cgreedy.py（贪心基线，作为传统方法baseline）
❌ 丢弃: 所有 *_trainer.py, *_agent.py, another_ddpg.py, greedy.py, test_dgreedy.py
→ 从零重写统一的MARL训练框架

## 任务

### Step 1: 清理 + 建立新框架
删除所有废弃文件，创建统一接口：

config.py — 集中管理所有超参数（UAV数量、频率范围、学习率、网络尺寸等）
rl_env.py — 基于 marl_env.py 整理（review观测拼接逻辑，统一动作空间为离散Gumbel-Softmax）

### Step 2: 实现四种MARL方法（统一接口）
所有 trainer 必须：
- 共用同一个 env 接口
- 动作空间统一为离散（power levels × freq levels，Gumbel-Softmax）
- seed=42 固定
- train() 返回 TrainingStats（含 episode_rewards, interference_probs, eval_results）
- 实现 evaluate() 方法（无噪声，返回 avg_interference_prob）

1. maddpg_trainer.py — 核心MARL方法
   - Actor: 输入局部观测→输出离散动作的Gumbel-Softmax分布
   - Critic: 输入全局状态+所有agent动作→输出Q值
   - 集中训练，分布式执行

2. iddpg_trainer.py — 独立RL基线
   - Actor: 输入完整局部观测（含邻居信息！）→输出动作
   - Critic: 仅输入自身观测+自身动作
   - ⚠️ 这是上一版的关键bug：不能只取自身6维观测

3. cddpg_trainer.py — 中心式基线
   - 单个全局Actor输出所有agent动作（2N维）
   - 注意：大规模下动作空间爆炸，这正是证明MARL优势的对比点

4. qmix_trainer.py — 值分解方法（加分项，时间允许再做）
   - Agent网络输出Q值，Mixer网络组合为全局Q_total

### Step 3: 新增 random_baseline.py
随机均匀选择功率和频率，作为最弱对照

### Step 4: 实验执行（自纠正循环）
⚠️ 这不是一次性任务。每轮实验后必须：
- 读取结果，判断 MADDPG 是否在互扰概率上显著优于 baselines
- 如果不优 → 分析原因（reward/架构/超参数），修改，重跑
- 10 UAV 验证通过 → 再扩展到 25, 50, 75, 100

UAV 规模: [10, 25, 50, 75, 100]
基准方法: random, greedy-central, greedy-sequential, greedy-independent
MARL方法: MADDPG, IDDPG, CDDPG（50+可跳过）, QMIX（可选）

每个规模跑完所有方法后立即汇总对比，确认MARL优于baseline再进下一规模。

### Step 5: 消融实验（核心结果OK后）
MADDPG @50UAVs:
- 消融1: 仅全局奖励 vs 全局+局部
- 消融2: limit_neighbors = [3, 5, 10]

### 最终产出
results/ 目录：
- summary.csv: 方法×规模的互扰概率矩阵
- summary.png: 折线对比图
- convergence/: 各方法训练曲线
- ablation/: 消融结果

### DDL
7月底前完成所有实验，产出可插入论文的图表和数据
