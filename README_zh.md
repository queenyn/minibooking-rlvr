# MiniBooking-RLVR（中文说明）

[English README](README.md) | [架构设计](docs/architecture.md) | [完整实验报告](docs/experiments.md) | [典型失败与轨迹剖析](docs/case_studies.md) | [交互式轨迹回放器](demos/trajectory_viewer.html) | [演进路线图](docs/roadmap.md)

**面向小参数大模型世界模型的可验证状态增量与可逆性基准、纯 PyTorch 手写 GRPO（含 H/D/S/C 在线诊断与熵反馈控制器），以及风险感知一步前瞻 MPC 规划闭环。**

本项目研究可验证奖励强化学习（RLVR）如何塑造小模型（`Qwen/Qwen3-1.7B + LoRA`）预测**精确状态变化量（`changes`）**、**即时经济损失（`cost`）**、**深度受限可逆性（`reversible`）**与**置信度校准（`confidence`）**的能力，并在同分布（`test_id`）、组合泛化（`test_comp`）和闭环多步规划任务上进行系统诊断。

---

## 核心工程特性

- **零漂移确定性环境与深度-3 BFS 可逆性验证器（`minibooking_rlvr.env`）**
  - 全程采用整数“分”与整数比退款规则（`0/1`、`1/2`、`1/1`），严格保证金额守恒（`balance + paid + loss = const`）与损失单调不减。
  - 区分即时损失 `cost` 与多步可逆性 `reversible`：例如半额退款规则下的 `confirm()` 当步无手续费（`cost = 0`），但因无法在 3 步内无损恢复原状态而被判定为不可逆（`reversible = False`）。
- **反泄漏场景族数据集协议（`minibooking_rlvr.data`）**
  - 先按 240 个 `(规则, 初始余额)` 场景族切分 `train (2400) / dev (300) / test_id (300) / test_comp (300)`，再在各切分内独立随机游走与规范化输入 SHA256 去重，保证跨切分零样本重叠、零场景族重叠。
  - `test_comp` 留出 `(退款 1/2, 改期费 0)` 与 `(退款 1/1, 改期费 20)` 两组未见规则搭配（各分量均在训练集出现过），并支持 `--check` 逐字节复现校验。
- **纯 PyTorch 手写 GRPO 与 H/D/S/C 在线诊断（`minibooking_rlvr.rl`）**
  - **不使用 TRL 或 vLLM**：在 `bf16` 中将冻结 SFT LoRA 合并为起点与 reference，通过 `disable_adapter()` 零额外显存计算参考策略 `logprob`。
  - 完整实现组内优势（总体标准差）、PPO 风格 clipped surrogate、`k3` 非负 KL 估计、分块全词表监测熵与可求导熵正则（解析梯度经 autograd 校验）、策略/熵梯度范数比分解（`ent_grad_ratio`）、在线 D/S/C 组统计及带准确率门控的 `EntropyController`。
- **双口径冻结评测与一步前瞻规划闭环（`minibooking_rlvr.eval`、`minibooking_rlvr.planner`）**
  - 并列报告主口径（全样本，无效输出严格罚分）与有效输出诊断口径，同步输出置信度坍缩直方图与 `test_comp` 上 `allow_change=True/False` 子集分解。
  - 提供风险感知 1-step MPC 规划闭环及 `null`（空预测）/ `oracle`（真值预测对照）双 CPU 基线。

---

## 已核实单种子实验结论与严谨边界

实验基于 `Qwen/Qwen3-1.7B + LoRA (r=16, alpha=32)`、单种子 `seed=20261025`（LoRA SFT 150 步 + 手写 GRPO G1/G2/G3 各 200 步）。以下全部数字均经原始日志核实，并如实列出负结果与边界条件：

1. **冻结评测（每集 `n=300`，主口径全样本）**：
   - `test_id` exact：SFT `0.8833` vs RL 三组 `0.9133 (G1) / 0.9033 (G2) / 0.8967 (G3)`；三组间最大差 `0.0166`，落在同配置 `bf16` 重复运行噪声（`dev[:100]` 约 `0.06`）内。
   - `test_comp` 指标分化：RL 的 `exact` 低于 SFT（`0.8433 / 0.8400 / 0.8233` vs `0.8767`，主要丢失在非空 `cancel` 漏 `balance_cents`），但可逆性 `macro-F1` 显著高于 SFT（`0.7256 / 0.7401 / 0.7435` vs `0.5163`，主要来自 `退款 1/2 + 改期费 0` 组合）。
2. **离线难度过滤与零方差组诊断**：
   - 离线剔除 SFT `K=4` 全对的 1,098 道题后，RL 初始零方差组比例由 `63.9%` 降至 `7.4%`；训练集中被完全剔除的 `inspect` 在 `dev / test_id / test_comp` 上四模型均保持 `1.0000` 零回归。
   - 200 步（800 组）训练中，复合奖励组 G2/G3 的有效非零方差组数高于单一 `DeltaF1` 组 G1（`211 / 196` vs `162`），并将全 0 零方差组从 `47` 降至 `0`。
3. **如实报告校准坍缩与 G3 控制器边界**：
   - 由于 SFT 目标中 `confidence` 恒为 `1.0`，四模型有效输出的 `confidence` 100% 坍缩在 `1.0`，Brier 未携带 `exact` 之外的校准信息。
   - G3 控制器在 200 步（20 次检查）中仅在第 170 步触发 1 次（`lambda=1e-4` 仅持续第 171–180 步，熵项/策略梯度比最大 `7.67e-5`），因此本单种子运行中 `G3 ≈ G2`，**不声称验证了控制器有效**，亦未复现 DACE-RL 的 ARB / DUAL-EDC。
4. **一步前瞻规划闭环（100 测试任务：62 可解 / 38 无解，冻结权重 `w_risk=0, w_irr=0, w_conf=0.1, tau=0`）**：
   - 可解成功率：G2 `0.500 (31/62)` > SFT/G1 `0.387 (24/62)` > G3 `0.355 (22/62)` > `null 0.000`；真值预测对照 `oracle` 为 `0.565 (35/62)`（同一步前瞻规划器下的诊断对照，而非多步上界）。
   - G3 较低的不可逆失败率（`0.040` vs G2 `0.080`）源自非法 `create` 打平空转至 `max_steps=5` 的副作用，而非策略更谨慎。

---

## 研究背景与论文引用边界

本仓库是一个**独立构建的轻量级可验证工程测试床**，问题设定与诊断思路受以下两篇 NeurIPS 2026 Poster 论文启发，但**并非**两篇论文的官方代码发布仓：

- **DACE-RL**：*DACE RL for Compute Efficient Reinforcement Learning in Small Model Reasoning*（NeurIPS 2026 Poster，[OpenReview](https://openreview.net/forum?id=asPNY0YBKr)）—— 借鉴了 H/D/S/C 在线训练动态诊断与单旋钮准确率门控熵反馈思路（`lambda_entropy`，区别于固定 `beta_kl`）；未复现论文中的 ARB、DUAL-EDC 多轴控制与数学推理基准。
- **CDWM**：*Predicting What Changes: Causal Delta World Models for Risk-Aware LLM Agent Planning*（NeurIPS 2026 Poster，[OpenReview](https://openreview.net/forum?id=AiYvJPM2Oy)）—— 借鉴了“预测状态增量（`changes`）+ 可逆性 + 代价 + 置信度”的世界模型建模与风险感知闭环规划思路；本仓库基于独立实现的 MiniBooking 确定性订单状态机，不含论文中的因果稀疏掩码与 WebArena/ALFWorld 实验。

---

## 本地快速验证（纯 CPU，5 秒内完成）

```bash
python -m pip install -e ".[dev]"
python -X utf8 -m pytest -v
python -X utf8 scripts/generate_dataset.py --check
python -X utf8 scripts/reproduce_eval_tables.py --verify
python -X utf8 scripts/run_planning_eval.py --split test --predictors oracle null --no-save
```

如需浏览 8 条真实多模型闭环规划对比轨迹（含非法 `create` 打平空转、1-step 视野受限停止与无解不可逆陷阱），直接用浏览器打开 [demos/trajectory_viewer.html](demos/trajectory_viewer.html) 即可离线交互回放。
