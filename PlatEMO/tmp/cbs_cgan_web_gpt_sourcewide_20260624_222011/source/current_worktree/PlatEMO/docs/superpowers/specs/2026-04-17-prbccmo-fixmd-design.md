# PRBCCMO Current fix.md Design Note

日期：2026-04-18

本文件用于替换仓库中已过时的 `2026-04-17` 版设计说明。当前唯一权威需求是仓库根目录的 `fix.md`，本说明只记录已经落地的现行口径，不再保留旧的 `trusted sector / paired sector / bridgeGrowth` 方案。

## 1. 当前目标

`PRBCCMO` 与 `PRBCCMO_t` 必须严格遵循当前 `fix.md`：

1. 训练 core 采用 `anchor + opposite infeasible shell`
2. sectors 只负责锚点分布多样性，不再承担训练、档案或 probe 的准入门职责
3. `SeedPool` 固定为 `[B,OffspringC,OffspringU]`
4. BCE 训练采用 batch 级 `1:1` replay balancing
5. traced 侧的默认诊断口径从 `trusted/paired sector` 转为 `anchor/pair`

## 2. 主线语义

### 2.1 训练档案

训练档案不再按 sector 强行保双侧，而是：

1. 从 `B` 与 `RecentBoundaryOff` 中收集可行 anchors
2. 不足时只从 `P_C` 做全局 feasible 补足
3. 为每个 anchor 选择少量最近的 opposite infeasible
4. 用 `ReserveAnchorPairs` 而不是 `ReserveCoreSideCoverageBySector`

### 2.2 训练门

训练门不再依赖 `trusted sector` 或 `paired sector` 数量，而是：

1. `anchor_count >= max(2,M)`
2. `pair_count >= max(4,2*M)`

其中 `pair_count` 由训练档案中负类行的权重和给出。

### 2.3 helper / support / archive

当前实现中：

1. helper 从整个 `P_C ∪ P_U` 中选真实 opposite，不再受邻域 sector 限制
2. `OppSupport` 不再按 sector 邻域屏蔽 reference，只由 pair 证据与距离决定
3. `ResolveBridgeBoundaryMask` 不再承担 sector 准入语义

## 3. traced 指标

`PRBCCMO_t` 当前默认输出和汇总优先关注以下 pair 视角指标：

1. `anchor_count`
2. `pair_count`
3. `mean_pair_dist`
4. `lowmargin_pair_hit`
5. `boundary_seg_cross_dist`

其中：

1. `mean_pair_dist` 为 anchor 到最近 opposite shell 的平均距离
2. `lowmargin_pair_hit` 表示 low-margin 点是否落在真实 pair 证据附近
3. `boundary_seg_cross_dist` 是不额外消耗 FE 的 segment crossing 代理误差

## 4. 受影响文件

当前应与 `fix.md` 保持一致的文件包括：

1. `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m`
2. `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m`
3. `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`
4. `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m`
5. `Algorithms/Multi-objective optimization/PRBCCMO/summarize_PRBCCMO_t_run.m`
6. `Algorithms/Multi-objective optimization/PRBCCMO/summarize_PRBCCMO_t_data.m`
7. `Algorithms/Multi-objective optimization/PRBCCMO/benchmark_PRBCCMO_t_suite.m`

## 5. 验证

当前最小验证链路是：

1. `test_PRBCCMO_semantics`
2. `test_PRBCCMO_t_metrics`

本说明不再作为独立设计输入；若后续需求变化，应先改 `fix.md`，再同步本说明。
