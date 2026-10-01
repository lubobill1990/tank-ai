# 泛化训练战役记录（2026-10-01，接手 copilot 后的首个 24 小时）

目标：训练出有泛化能力的模型——至少"见过一关后能通关这关"，最终争取全关通关。
训练面使用自研 Rust 并行引擎（批量取数据、批量推进）；**真实 ROM 全关卡（12,000 帧）
为唯一正式验收面**，Rust 通关不声称 ROM 胜利。

## 方法流水线

1. **MPC 前瞻否决 teacher**（`mpc_teacher.py`）：`TeacherControllerV2`/`GuardedTeacher`
   提案；每个决策把当前 RAM 克隆成 10 动作 × 3 seed × 深度 12 的 Rust 批量推演，
   以引擎折扣稀疏奖励评分，否决会死/丢基地的首动作。基地威胁惩罚 + 提前回防
   （`GuardedTeacher`）。全 35 关 teacher 通关率约 54%（450/840），而历史规则
   teacher 为 33–36%、裸启发式 0%。
2. **大规模演示采集**：每个并行环境一个 `DemonstrationWriter` 流；
   `filter_success.py` 按字节切片抽取 clear 局。约 1,200+ teacher 通关局。
3. **行为克隆**（`bc_fast.py`：线程池解码 + 预取）+ **DAgger 迭代**
   （`mpc_teacher.py --student`：学生执行、MPC teacher 对学生访问态打标签，
   β=0.05）。11 轮 DAgger，学生自主通关率逐轮上升：1→8→15→28→34→49→64→77→84/560。
4. **表征更换**：`EgocentricGridCNN`（玩家中心重采样，learning_lab/ego_encoder.py）
   替代 Grid26CNN 后，同数据下通关率约 ×3，ROM 迁移显著改善。

## 结果（截至本文档撰写，仍在迭代）

Rust 引擎（全 35 关 × 8 局，确定性）：

| 模型 | 通关 | 覆盖关数 |
|---|---|---|
| bc-v1（CNN，纯 BC） | 0/280 | 0 |
| bc-v4（CNN + 3 轮 DAgger） | 20/280 | 15 |
| bc-ego-v1（ego，从零） | 55/280 | 24 |
| bc-ego-v2 | 75/280 | 25（stage10 8/8） |
| bc-ego-v3 | 81/280 | 25 |

真实 ROM（全 35 关 × 2 seed，确定性，12,000 帧，普通复位、无 RAM 写入）：

| 模型 | 通关局 | 至少通过一次的关 |
|---|---|---|
| bc-v4（CNN） | 4/30（部分关） | 3（stage3 2/2） |
| bc-ego-v1 | 12/70 | 11 |
| **bc-ego-v2（最终冠军）** | **16/70** | **16** |
| bc-ego-v3 | 8/70 | 7 |
| bc-ego-v4 | 9/70 | 8（stage1 2/2） |
| bc-ego-v5-rom（ROM-DAgger×1） | 12/70 | 10（stage10、28 各 2/2） |
| bc-ego-v6-rom（ROM-DAgger×2，v2 父本） | 12/70 | 11（stage4 2/2） |

- Rust 分数更高不保证 ROM 更好（v3/v4 > v2 于 Rust，但 ROM 更差）：对 Rust
  分布拟合更深会损失 ROM 鲁棒性。**最终模型按 ROM 成绩选择。**
- 各 checkpoint 在 ROM 上通关的关卡子集差异大（方差高）；全部 ego checkpoint
  合计曾在 **24/35** 个真实 ROM 关卡上取得通关：
  1,2,3,4,5,6,8,9,10,12,13,14,15,16,17,22,23,24,27,28,29,30,31,33。
- ROM-DAgger（`rom_dagger.py`：学生在 ROM 上玩、每步 RAM 快照导入 Rust 引擎
  做 MPC 规划打标签，三轮共 770 局、~50 万标签、原生导入零失败）使学生在
  ROM 采集期的自主通关率从 14/210 提升到 28/210 与 47/350。
- **最终交付 bc-ego-v8-romanneal**（v6-rom 基础上仅用 ROM 标签低学习率退火）：
  seeds 0,1 15/70（14 关，含**首次通关的 stage20、stage35**）；seeds 2,3 9/70。
  四 seed 合计 24/140（17.1%），与 ego-v2 的 25/140（17.9%）统计打平；两者覆盖
  的关卡子集互补。2-seed 70 局终测的单次方差约 ±4~5 胜，比较必须用多 seed。
- 全部 ego checkpoint 合计曾在 **26/35** 个真实 ROM 关卡上取得通关（新增 20、35）。
  从未在 ROM 通关的 9 关：7,11,18,19,21,25,26,32,34（其中 21/25/32/34 是
  teacher 不可达关）。

## 负结果（已验证，勿重复）

- 原生稀疏奖励与击杀导向重塑的 PPO 微调（BC 初始化、value 预热、KL 约束）
  均使策略退化为被动躲避（击杀下降、base_lost 上升），两次独立失败。
- 离线验证 CE 选型误导：bc-v2 的 best-by-CE 2/280 vs 过拟合 final 15/280。
- teacher 威胁权重调大（10/20）整体更差（畏缩不清敌）；仅 stage25 破零。
- teacher 不可达关：21/34 约百局全败（32 关学生曾有 1 次自主通关）。

## 工件

- 模型与评测 JSON：`runs/generalization-20261001/models/`（主仓库，durable）。
- 演示语料（~20GB）：worktree `runs/mpc-demos/`（round1/2、dagger1–11、defense-probe）。
- 管线脚本（本分支）：`mpc_teacher.py`、`mpc_sweep.py`、`filter_success.py`、
  `repair_first_frame.py`、`bc_fast.py`、`bc_run.py`、`ego_wide.py`、
  `ppo_finetune.py`、`eval_rust.py`、`eval_rom.py`。

## 下一步建议

1. 继续 ego 线 DAgger 飞轮（尚未收敛），但以 ROM 成绩做 checkpoint 选择，
   或在训练语料中混入少量 ROM 轨迹对齐分布。
2. 攻坚 teacher 不可达关（21/25/32/34）：需要针对性策略（如地形工事维修、
   长期蹲守）而非通用参数。
3. 55 通道 PrecisionObservation 移植到 Rust 引擎，解决已审计的 2px 观测混叠。
4. 更大 ego 网络（`ego_wide.py` 已就绪）与更大规模采集。
