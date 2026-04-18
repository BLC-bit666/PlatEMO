# PRBCCMO Boundary-Data Design

> Superseded on 2026-04-18 by the current pair-centric `fix.md` contract.
> This document contains historical `trusted sector / bridge-gated B` assumptions and is no longer authoritative for implementation.
> Current diagnostics should prioritize `anchor_count`, `pair_count`, `mean_pair_dist`, `lowmargin_pair_hit`, and `boundary_seg_cross_dist`.

日期：2026-04-15

## 1. 背景与目标

本次修改严格遵循 `fix.md`，目标不是继续增强分类器技巧，而是把 `PRBCCMO` 的边界学习任务收缩为：

> 让 plain BCE 在局部双侧支撑的样本分布上学习边界附近的 feasible / infeasible 分界。

因此本次主线只允许修改三类内容：

1. 边界定义
2. 边界存档 `B` 的入选规则
3. MLP 训练数据分布

本次明确不进入主线的机制：

1. `BoundWeight`
2. class weighting
3. source weighting
4. calibration buffer
5. temperature scaling
6. predicted-opposite helper
7. novelty 或额外排序分支
8. 新损失函数或新模型家族

## 2. 影响范围

本次修改只覆盖 `PRBCCMO` 主线生态：

1. `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m`
2. `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m`
3. `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`
4. `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_smoke.m`
5. `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m`
6. `Algorithms/Multi-objective optimization/PRBCCMO/summarize_PRBCCMO_t_data.m`
7. 如有必要，同目录下与 `PRBCCMO_t` CSV 列结构强绑定的分析脚本

不修改：

1. `PRBCCMO1/2/3`
2. 其他算法目录
3. 问题定义、环境选择主逻辑、算子实现

## 3. 设计概览

推荐并确认采用的方案是“完全按 `fix.md` 做语义清理，主线、追踪版、测试、诊断一起收缩到 plain BCE”。

核心原则如下：

1. MLP 仍然保留 warm start，但训练和预测不再依赖任何校准或额外权重。
2. `trusted sector` 只由真实双侧支撑决定，不由模型预测补齐。
3. `B` 在 trusted sector 中做边界细排，在 untrusted sector 中只允许桥接样本进入。
4. 训练集主体只来自 `B + RecentBoundaryOff`，`P_C/P_U` representative 只在局部缺侧时补一个锚点。
5. 训练档案裁剪采用 sector-wise quota，而不是全局优先级 + 权重排序。

## 4. 主循环与模型接口收缩

### 4.1 `PRBCCMO.m`

主循环将移除：

1. `MaxCalib`
2. `CalibArchive`
3. `UpdateCalibrationArchive`

边界模型更新接口从：

`UpdateBoundaryModelIfDue(Model,TrainArchive,CalibArchive,...)`

收缩为只依赖训练档案的形式：

`UpdateBoundaryModelIfDue(Model,TrainArchive,...)`

### 4.2 `PRBCCMO_t.m`

追踪版保持与主线同构：

1. 移除 `MaxCalib` 与 `CalibArchive`
2. traced MLP 事件不再输出 calibration / temperature 相关字段
3. traced summary 不再聚合 calibration / temperature 指标

## 5. 边界定义与 trusted-sector 规则

### 5.1 trusted sector

`trusted sector` 的定义保持为：

1. 只看训练档案中的 core source，即 `B` 与 `RecentBoundaryOff`
2. 在某扇区及其邻扇区内，真实标签同时出现 feasible 和 infeasible
3. 满足上条时，该扇区才被视为 trusted

推论：

1. trusted sector 中，`p ~= 0.5` 可以作为边界接近度代理
2. untrusted sector 中，模型概率只能表示不确定，不能直接作为边界证据

### 5.2 boundary meta

`BuildBoundaryMeta` 统一维护以下字段：

1. `sector`
2. `prob`
3. `margin`
4. `objScore`
5. `oppSupport`
6. `oppDist`
7. `feasible`
8. `trusted`
9. `score`

统一排序分数保持为：

`score = 0.50 * margin + 0.30 * (1 - oppSupport) + 0.20 * objScore`

其中：

1. `margin = 2 * abs(prob - 0.5)`，越小越接近边界
2. `oppSupport` 来自真实 opposite-side support，越大越可信
3. `objScore` 衡量 PF 相关性，越小越优

## 6. 边界存档 `B` 的资格筛选与保留规则

### 6.1 trusted sector 内

在 trusted sector 中：

1. 候选资格 = 当前扇区全部候选
2. 按统一 `score` 排序
3. 每扇区最多保留 `kappa = 3`

### 6.2 untrusted sector 内

在 untrusted sector 中：

1. 不允许普通 low-margin 候选直接进入 `B`
2. 只有 bridge candidate 可以进入

bridge candidate 的定义：

1. 它所在扇区当前还不 trusted
2. 将该点与当前 core support 合并后，能在该扇区局部邻域内补齐缺失的一侧真实标签
3. 即该点能把“单侧证据”推进为“双侧证据”

### 6.3 soft minority reservation

soft minority reservation 保留，但只对已经通过资格筛选的候选生效：

1. 若某扇区当前候选同时包含 feasible 与 infeasible
2. 且 top-k 初选结果退化为单侧
3. 则检查最佳 opposite-side 候选
4. 若其分数仅略差于当前最差已选样本，则做一次软替换

这一步用于防止 `B` 在局部扇区中退化成单侧档案，但不会突破 trusted / bridge 资格约束。

## 7. 训练数据分布与 archive 配额

### 7.1 训练集组成

训练集主体固定为：

`D_t = B_t union O^B_t union R_C^F union R_U^I`

其中语义不是四类并列，而是：

1. `B_t` 与 `RecentBoundaryOff` 是主体
2. `P_C/P_U` reps 只负责补缺侧

### 7.2 representative 规则

对每个扇区 `s`：

1. 先检查 `B + RecentBoundaryOff` 在 `s` 及其邻扇区内是否已有双侧样本
2. 若已有双侧，不补任何 rep
3. 若缺 feasible，只补一个最近的 `P_C` feasible representative
4. 若缺 infeasible，只补一个最近的 `P_U` infeasible representative
5. 每扇区每侧至多补一个

### 7.3 sector-wise quota

训练档案裁剪改为按扇区和来源定额保留：

1. 每扇区最多 2 个 `B`
2. 每扇区最多 2 个 `RecentBoundaryOff`
3. 每扇区最多 1 个 feasible rep
4. 每扇区最多 1 个 infeasible rep

超额时按以下规则裁剪：

1. 更近时间优先
2. 原始稳定顺序打破平局

本次不再使用：

1. `BoundWeight`
2. 以 `BoundWeight` 为依据的 archive 排序
3. 额外 source 权重

## 8. plain BCE 训练与预测

### 8.1 训练

`TrainBoundaryMLP` 保留：

1. 单个 MLP
2. warm start
3. sigmoid 输出

删除：

1. class weighting
2. source weighting
3. boundary weighting
4. calibration / temperature scaling

因此训练将退化为真正的 plain BCE，样本差异只由训练数据组成体现。

### 8.2 预测

`PredictBoundaryMLP` 直接对 raw logits 做 sigmoid：

1. 不读取 `Model.T`
2. 不做温度缩放
3. 仍保留数值稳定用的概率截断

## 9. boundary sampling 约束

本次不改 boundary sampling 的基本骨架，但保持并强化以下约束：

1. helper 只来自 `P_C union P_U`
2. helper 必须是真实 opposite side
3. 没有 real opposite helper 时直接跳过该扇区
4. 不回退到 predicted-opposite helper
5. probe beta 固定为 `{0.25, 0.50, 0.75, 1.05}`
6. probe ranking 继续使用 `margin + opposite-side support + objective relevance` 统一语义

## 10. 诊断与 CSV 输出收缩

### 10.1 `mlp_events.csv`

保留：

1. `gap`
2. `tick`
3. `can_train`
4. `trained`
5. `warm_start`
6. `model_ready_before`
7. `model_ready_after`
8. `train_size`
9. `pos_count`
10. `neg_count`
11. `src_b`
12. `src_recent_boundary`
13. `src_rep_c`
14. `src_rep_u`
15. `acc_before`
16. `bal_acc_before`
17. `brier_before`
18. `logloss_before`
19. `acc_after`
20. `bal_acc_after`
21. `brier_after`
22. `logloss_after`

删除：

1. `mean_bound_weight`
2. `temperature`
3. `calib_size`
4. `calib_brier_holdout`
5. `calib_bal_acc`

### 10.2 `generation_summary.csv`

保留与当前主线一致的字段，例如：

1. `boundary_attempts`
2. `seed_b_size`
3. `seed_b_sector_coverage`
4. `seed_b_mixed_sectors`
5. `b_trusted_count`
6. `b_trusted_sector_coverage`
7. `b_trusted_lowmargin_count`
8. `lowmargin_mix_score`
9. `lowmargin_oppdist`
10. `train_minus_boundary_bal_gap`
11. `train_minus_b_bal_gap`

删除：

1. `helper_pred_opp_ratio`
2. `train_mean_bound_weight`
3. `calib_brier_holdout`
4. `calib_bal_acc_holdout`
5. `temperature`

必要时可新增能直接体现 bridge-gated `B` 或 quota 生效的摘要列，但优先复用现有列。

### 10.3 `summarize_PRBCCMO_t_data.m`

汇总脚本同步移除 calibration / temperature / bound-weight 相关 required columns 和输出列，只保留 plain-BCE 主线所需统计。

## 11. 验证计划

按由便宜到昂贵的顺序做最小验证：

1. `test_PRBCCMO_semantics`
2. `test_PRBCCMO_t_smoke`
3. `test_PRBCCMO_t_metrics`

验证目标：

1. 确认文本语义已删除旧机制并包含新主线关键条件
2. 确认 `PRBCCMO_t` 仍能生成完整 CSV 运行目录
3. 确认 traced 输出列与 summary 脚本、回归测试保持一致

## 12. 风险与边界

主要风险：

1. 删除 calibration / temperature / bound-weight 后，trace 和 summary 的列结构会发生变化，需要同步修改所有强绑定测试
2. bridge-gated `B` 可能减少 early-stage 边界档案规模，需要确认现有 `trusted sector` 与 helper 跳过逻辑不会导致前期完全停滞
3. sector-wise quota 会改变训练档案时间分布，需避免实现中出现按全局容量截断覆盖局部配额的回归

本次不处理：

1. 新损失函数
2. Deep Ensembles
3. 新的 helper 类型
4. `PRBCCMO2 / PRBCCMO3` 分支清理
5. 跨目录实验脚本的大规模重构

## 13. 实施顺序

1. 收缩 `PRBCCMO.m` 的模型接口和训练档案结构
2. 实现 bridge-gated `B` 与 sector-wise quota
3. 同步 `PRBCCMO_t.m` 的 traced 逻辑和 CSV 列
4. 更新 `test_PRBCCMO_semantics.m`
5. 更新 `test_PRBCCMO_t_metrics.m`
6. 更新 `summarize_PRBCCMO_t_data.m`
7. 依次执行最小验证
