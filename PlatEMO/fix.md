## 1. Evidence Map

下表列出我实际使用的源码、结果汇总和代表性 CSV。所有后续判断都回到这些证据。

| 证据                                                         | 我用它回答什么                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `CCMO_GAN_BDG.m`                                             | 当前主线配置、训练/采样主循环、`stageProbeN=200`、`ganTrainMode="epoch"`、每次 refresh 的 `BuildEvaluatedFeasibleSource_BDG -> BuildGANTrainingSet_BDG -> BoundaryGAN_BDG('traindirectboundary') -> BoundaryGAN_BDG('traindiagnose')` 流程。 |
| `BuildBoundaryTargetTriples_BDG.m`                           | `targetMode="near_segment_feasible"` 的真实实现、`nearSegmentMaxPerPair=5`、`candidateCount=nPair*sourceCount`、`1e6` 阈值、full scan / bbox 两条路径、`t` 与 condition 构造。 |
| `FilterBoundaryTargetTriples_BDG.m`                          | 训练集过滤只作用于 CGAN training set，不改 AF/AI archive；`condition_knn` 是后置过滤；其核心是 `pdist2(C,C)`。 |
| `BoundaryGAN_BDG.m`                                          | `epoch` 与 `iter` 两种训练模式的真实差异；当前 `epoch` 模式按全数据批次数放大更新；Huber 重构损失；`gan_g_loss_count` 的累加含义。 |
| `UpdateBoundaryArchive_BDG.m`                                | 当前 archive 主线是 `global_af_nd + af_not_dominates_ai + none + neighbor4`；统一 `neighbor4` 路径会池化 flip/cross/previous archive 再重配对；archive rebuild 的 Pareto rank / direction / pairRef / perRef 保留链。 |
| `Support/run_CCMO_GAN_BDG_fixmd_pairt_diagnostic_suite.m`    | 当前 review bundle 的 nearseg/endpoint 套件确实是 10 题、runs=3、worker=8、maxFE=100000。 |
| `fixmd_pairt_core_by_variant.csv` / `fixmd_endpoint_core_by_variant.csv` | nearseg vs endpoint 的 target 数量、nearseg 候选数、GAN 更新数、边界拟合指标。 |
| `fixmd_pairt_core_by_problem.csv` / `fixmd_endpoint_core_by_problem.csv` | 哪些问题 target 数和 boundary 指标最差，尤其 DASCMOP4/5。    |
| `fixmd_pairt_diagnostic_by_variant.csv` / `fixmd_endpoint_diagnostic_by_variant.csv` | `normal_obj_seg_dist90` 等诊断指标的 nearseg vs endpoint 对比。 |
| 代表性 `boundary_diagnostic_run_summary.csv` / `gan_diagnostic_metrics.csv` / snapshot CSV | 当前 suite 确实打开了 snapshot/diagnostic 输出；代表性诊断一条记录就包含 `normal=200, fixedz=30, zsweep=300`；snapshot CSV 是一个宽表，串行写出 population / archive / generated 各角色。 |

补充两条外部资料，只用于校准设计方向，不用于替代 bundle 证据：

1. unknown-constraint 的物理设计问题里，最优解常常就在 feasible/infeasible 边界附近；
2. “epoch” 本身就是对整套训练数据的一次完整遍历，所以 epoch 模式的成本会随训练样本数和 batch 数线性放大。([arXiv](https://arxiv.org/abs/2402.07692?utm_source=chatgpt.com))

------

## 2. Root Cause

### 当前算法运行慢的根本原因

| 耗时来源                                         | 源码定位                                                     | 复杂度表达式                                                 | 已被实验支持 | 结论                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ |
| **A. nearseg 几何查找**                          | `NearSegmentFeasibleTargets_BDG` / `NearSegmentFullScanTargets_BDG` | 若走 full scan：`O(nPair * sourceCount * M)` 距离计算，外加每 pair 的局部排序；输出上界 `O(nPair * nearSegmentMaxPerPair)` | **部分支持** | 这是实打实的成本，但不是当前 3x runtime gap 的首因。bundle 顶部已经直接写明“nearseg 慢的主要机制不是点到线段距离本身，而是 target triples 放大 + epoch 训练放大 GAN 更新次数”。 |
| **B. nearseg target 放大**                       | `nearSegmentMaxPerPair=5`，先构造 triples，再过滤。          | 预过滤输出规模 `T_pre ≈ nPair * nearSegmentMaxPerPair`；当前主线平均 `647.9` 条 pre-filter nearseg targets，最终只训练 `388.8` 条 | **强支持**   | 这是第一层放大。当前 nearseg 平均 `mean_target_pair_count=129.67`，`mean_target_near_segment_keep_count=647.9`，几乎就是 `5/pair` 满打满算；但最终 `mean_target_triple_count=388.8`，说明后面又删掉了约 40%。 |
| **C. condition_knn 后置过滤本身也重**            | `FilterBoundaryTargetTriples_BDG` 先 `pdist2(C,C)` 再逐行排序。 | `O(T_pre^2 * condDim + T_pre^2 log T_pre + T_pre * k * D)`，主导项近似 `O(T_pre^2)` | **强支持**   | 当前不是“先少量选样本再训练”，而是“先构造满额 nearseg triples，再做一个二次复杂度的全局 kNN 过滤”。 |
| **D. GAN epoch 训练成本被 target 数线性放大**    | `BoundaryGAN_BDG` 的 epoch 路径对每个 epoch 跑完整个 dataset 的 batch 循环。 | 设 `T=target_triple_count`, `B=miniBatch`, `G=ganIter`。当前主线 `gUpdates = G * ceil(T/B) * gSteps`，`dUpdates = dPretrainIter + G * ceil(T/B) * dSteps` | **强支持**   | 这是当前主因。nearseg 平均 `T=388.8`，`B=64`，`G=50`，预测约 `50*ceil(388.8/64)=350` 次 G 更新；实际观测 `331.7`。endpoint 平均 `T=80.2`，预测约 `100` 次，实际 `88.3`。和代码/统计高度一致。 |
| **E. 当前 mainline 之所以一定走 epoch 放大路径** | epoch 路径条件是 `Options.trainMode=="epoch" && isempty(sampleWeights)`；主线 filter 是 `condition_knn`，返回的是 keep mask，不是 sampleWeights。 | 同 D                                                         | **强支持**   | 这不是偶然，而是当前 retained mainline 的结构性结果。        |
| **F. always-on core diagnostics**                | 每次 refresh 都做 `traindiagnose(trainDiagN=100)`；每个 FE checkpoint 还会做 `stageProbeN=200` 的 probe。 | 每次 refresh 额外 `O(100)` 采样/判别；每个 stage 额外 `O(200)` probe，再叠加 boundary metrics 计算 | **部分支持** | 这肯定有成本，但 bundle 没有分函数 timer，无法定量它占总 runtime 的百分比。**证据不足：需要 profile**。 |
| **G. suite 级 snapshot / diagnostic 导出**       | 当前 suite 运行结果里 `snapshot_count=1` 且 `diagnostic_snapshot_count=1`；诊断配置是 `condition_count=30`, `z_per_condition=10`, `normal_n=200`。 | 额外采样 + 大 CSV I/O；规模取决于问题 D 和导出行数           | **部分支持** | 这是 wall-clock 的真实开销，但它更像实验 harness 成本，不是边界生成核心机制。**证据不足：需要 profile 区分 runtime_core / runtime_total**。 |
| **H. archive 更新**                              | `UpdateBoundaryArchive_BDG` retained 主线走 unified `neighbor4`，会池化 `flip1/flip2/cross + previous archive`，再重配对并 rebuild。 | 近似 `O(nFpool * nIpool * M + A^2 * M + A *                  | W            | )`                                                           |

### 关键归因结论

当前慢的**根本原因排序**是：

1. **epoch 模式下，GAN 训练更新数跟 `target_triple_count / miniBatch` 成正比放大**；
2. **nearseg 先构造满额 targets，再做后置 `condition_knn` 删除，导致 target 构造和过滤都做了冗余工作**；
3. **core diagnostics + suite 导出继续叠加 wall-clock**；
4. **archive 更新不是当前 nearseg vs endpoint 的主要差异来源**。

------

## 3. Redundancy / Fragmentation Analysis

### 3.1 当前数据流的重复与割裂

当前 retained mainline 的 refresh 路径是：

`UpdateBoundaryArchive_BDG`
→ `BuildEvaluatedFeasibleSource_BDG`
→ `BuildBoundaryTargetTriples_BDG`
→ `ArchiveDecisionRows_BDG`
→ `FilterBoundaryTargetTriples_BDG`
→ `BoundaryGAN_BDG('traindirectboundary')`。

这条链里有四类明显重复。

#### (1) 同一批 population / offspring 被重复扫描

- archive 更新已经读了一遍 `Population1, Offspring1, Population2, Offspring2` 去构造 pair skeleton。
- 训练前又把几乎同样的对象重新拼接成 `EvaluatedSource`。

这是**同源数据二次拼接**，但两边没有共享缓存。

#### (2) pair-level 信息被打平为 triple-level，再重新对齐

`BuildBoundaryTargetTriples_BDG` 的本质是 pair-level AF/AI skeleton 上挂多条 target。可它输出的是 flat `XBoundary` 和 flat `ConditionData`。之后又用 `ArchiveDecisionRows_BDG(AI, target_keep_index, D)` 把 pair-level `AI` 复制成 triple-level `DiagAI/GANTrainAI`。

这意味着：

- 同一个 pair 的 `AI.dec` 会被重复复制多次；
- 同一个 pair 的 `AFNorm / Direction / RefToken` 也会在 flat rows 中重复多次；
- downstream 再也看不到“这些 triple 属于哪个 pair 的 cell”，只能把它们当一堆扁平样本。

#### (3) nearseg 先满配构造，再被 `condition_knn` 删除

这一步是当前最明显的冗余：

- retained 主线 `nearSegmentMaxPerPair=5`。
- nearseg 平均 `target_pair_count=129.67`，`target_near_segment_keep_count=647.9`，正好接近 `5/pair`。
- 但 filter 后最终只剩 `target_triple_count=388.8`，`target_filter_retain_ratio=0.6003`。

也就是：**先造 5 条/ pair，再删到约 3 条/ pair**。
这不是小浪费，而是当前 nearseg 的典型行为。

#### (4) 当前 suite 基本不走 bbox 加速分支

代码只有当 `candidateCount >= 1e6` 才进入 `NearSegmentBBoxTargets_BDG`，否则一律 full scan。
而当前 10 题 30 runs 的 nearseg 平均 `target_near_segment_candidate_count=43302.6`，问题级最大也只有 `92845.3`。

结论很直接：**当前主线速度问题不是 bbox 分支没写好，而是这条加速分支根本没被触发**。

------

### 3.2 nearseg、archive、training set、GAN training、diagnostics 之间的割裂

| 环节                              | 当前职责                                       | 割裂点                                                       |
| --------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| `UpdateBoundaryArchive_BDG`       | 生成 AF/AI pair skeleton                       | 不知道 nearseg source，也不产出 `t` / per-pair target cell   |
| `BuildBoundaryTargetTriples_BDG`  | 用 pair skeleton + source 生成 nearseg triples | 不保留 pair cell 结构，直接打平成 flat triples               |
| `FilterBoundaryTargetTriples_BDG` | 对 flat triples 做后置删样/加权                | 过滤发生在 triple 已经构造完成之后，不能回收 upstream 几何计算 |
| `BoundaryGAN_BDG`                 | 把 flat dataset 当普通训练集                   | `epoch` 模式对 target 数敏感，训练成本随样本量直接放大       |
| diagnostics / snapshots           | 诊断与可视化                                   | 一部分在 core loop 内总是发生，一部分由 suite 打开；统计和算法耦合过紧 |

------

### 3.3 保留 / 合并 / 删除 / 待实验

| 项                                                           | 结论                       | 依据                                                         | 处理建议                                 |
| ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| `near_segment_feasible`                                      | **保留**                   | endpoint 只快，不够贴边界线；all10 上 `GAN_to_Segment_Dist90 0.613` vs `2.128`，`normal_obj_seg_dist90 0.677` vs `2.439`。 | 不能回退成 endpoint 主线                 |
| `global_af_nd + af_not_dominates_ai + neighbor4 + none`      | **保留**                   | retained 主线和 endpoint 对照都固定用这组 archive 选项。     | 把它当 production path                   |
| `yt_dt_t_ref`                                                | **保留**                   | retained nearseg/endpoint 都固定该 condition；当前 bundle 没有证据表明 condition 是速度瓶颈。 | 不优先改                                 |
| `conditional_adversarial_huber`                              | **保留**                   | nearseg 与 endpoint 都共享 Huber 监督，差异来自 targetMode，不来自 loss。 | 保留 supervision 信号                    |
| `BuildEvaluatedFeasibleSource` 与 archive source pooling     | **合并**                   | 两边都扫描同一批 Pop/Offspring。                             | 合并成 refresh-local source cache        |
| nearseg 构造与 `condition_knn` 过滤                          | **合并**                   | 当前先造 647.9 再留 388.8；filter 是 CGAN-only，不能回收 upstream 成本。 | 把 retain 逻辑前移/吸收进 target builder |
| triple-level `AI` 展开                                       | **合并**                   | `ArchiveDecisionRows_BDG` 只是把 pair-level AI 按 triple 重复展开。 | 改为按 pair id lazy fetch                |
| `ref_af_nd`, `sourceCapMode="limited"`, `pairRefMode="none"` 等 retained-dead 分支 | **删除出 production path** | retained variant table 只有 nearseg/endpoint 两条，且都不用这些旧分支。 | 保留到 support/legacy，不放运行主路径    |
| `NearSegmentBBoxTargets_BDG` 优化                            | **待实验 / 非优先**        | 当前 candidate count 全部 < `1e6`，bbox 分支未触发。         | 不是当前第一优先级                       |
| `trainGap`、`ganIter` 的 brute-force sweep                   | **待实验**                 | bundle 没有 fixmd 主线上的直接 speed-quality 证据            | 先做结构性减法，再做少量 sweep           |
| suite snapshot / diagnostic 导出进入 runtime 对比            | **删除出生产计时**         | 当前 run summary 明确每 run 都有 `snapshot_count=1`、`diagnostic_snapshot_count=1`。 | 生产 runtime 与分析 runtime 分离         |

------

## 4. Minimal Unified Redesign

### 4.1 结论

**可以统一，但不能把 nearseg 完全塞进 persistent AF/AI archive 本体。**

原因很明确：

- persistent archive 是跨代保留的 **pair skeleton**；
- nearseg target 依赖的是 refresh 时刻的 **当前 evaluated feasible source**，它来自 `Population1/Offspring1/Population2/Offspring2/AFTrain` 的即时拼接。

所以，**正确的统一层级不是“把 nearseg 塞进 archive struct”**，而是引入一个**每次 refresh 构造一次的统一训练 bundle**。

### 4.2 最小统一方案：`BoundaryTrainBundle`

#### 统一后的数据结构

```text
BoundaryTrainBundle
  .pairAFDec        [P, D]
  .pairAIDec        [P, D]
  .pairAFNorm       [P, M]      % y_t
  .pairDir          [P, M]      % d_t = AI-AF
  .pairRefToken     [P, R]
  .pairScore        [P, 1]
  .targets{p}.X     [k_p, D]    % nearseg feasible targets
  .targets{p}.t     [k_p, 1]
  .targets{p}.dist  [k_p, 1]    % 到 pair segment 的 obj-space 距离
  .targets{p}.fb    [k_p, 1]    % fallback 标记
```

它把现在分散在：

- `AF/AI`
- `EvaluatedSource`
- flat `TrainDecs`
- flat `ConditionData`
- triple-aligned `DiagAI`

这些对象里的信息，收敛到一个 refresh-local bundle 里。

### 4.3 统一后的主流程

1. `UpdateBoundaryArchive_BDG` 继续只做 pair skeleton。
2. 新增 `BuildBoundaryTrainBundle_BDG(AF,AI,EvaluatedSource,W,Control)`：
   - 用当前 AF/AI pairs 和当前 feasible source，按 pair 构建 nearseg target cell；
   - 在 bundle 内缓存 `y_t / d_t / t / ref` 所需元数据；
   - 不再生成 flat `TrainDecs / ConditionData / GANTrainAI`。
3. `BoundaryGAN_BDG` 增加 `DrawBoundaryMiniBatchFromBundle_BDG`：
   - 先抽 pair，再从该 pair 的 `targets{p}` 中抽 target；
   - 当场拼 condition `[y_t, d_t, t, ref]`；
   - 当场按 pair id fetch `AI.dec`；
   - 用 `iter` 模式训练固定步数。
4. `FilterBoundaryTargetTriples_BDG` 从 retained mainline 退出；
   它的保留逻辑如果还需要，变成 bundle 内的 sampling score，而不是 row deletion。

### 4.4 这个方案替代了什么、删除了什么、保留了什么

- **替代**
  - `BuildBoundaryTargetTriples_BDG` 的 flat triple 输出；
  - `ArchiveDecisionRows_BDG` 的 triple-level AI 复制；
  - `FilterBoundaryTargetTriples_BDG` 的后置删样；
  - `epoch` 全数据遍历训练。
- **删除**
  - materialize-then-filter；
  - pair-level 信息打平再重建；
  - retained mainline 不使用的 legacy runtime branches。
- **保留**
  - `near_segment_feasible` 的监督语义；
  - `yt_dt_t_ref` 的 condition；
  - `conditional_adversarial_huber`；
  - `global_af_nd + af_not_dominates_ai + neighbor4` 的 pair skeleton。

这个方向和外部 unknown-constraint 文献是同向的：边界建模本身是关键，不能为了提速把 boundary signal 删掉；同时 epoch 本身就是对整套数据的一次完整遍历，所以当 target 数膨胀时，epoch 成本一定会随 batch 数放大。([arXiv](https://arxiv.org/abs/2402.07692?utm_source=chatgpt.com))

------

## 5. Speedup Proposals Ranked

### Proposal 1 — 把 retained mainline GAN 训练从 `epoch` 改成 `iter`

**优先级：最高**

**改哪些函数**

- `CCMO_GAN_BDG.m`：把 retained mainline 的 `ganTrainMode` 设为 `"iter"`；
- `BoundaryGAN_BDG.m`：直接走已有 `iter` 路径，不必重写网络结构。

**删/合并什么**

- 不删 supervision；
- 只停止“每个 epoch 扫完整个 target set”的训练方式。

**复杂度为什么会降**
当前 retained mainline 的 generator 更新数是：
$$
[
N_G = ganIter \cdot \lceil T / miniBatch \rceil \cdot gSteps
]
$$
而 `iter` 模式变成：
$$
[
N_G = ganIter \cdot gSteps
]
$$
nearseg 当前 `T=388.8`、`miniBatch=64`，所以每个 epoch 大约 7 个 batch；endpoint 只有 2 个。代码和统计是一致的。

**可能影响的指标**

- 主要风险在 `GAN_to_Segment_Dist90`、`normal_obj_seg_dist90`、`BoundaryHit_all`、`RawGANFeasibleRate`。
- archive quality 不应直接受影响。

**最小验证实验**

- 只改 `ganTrainMode: epoch -> iter`，其余一律不动；
- 跑 `DASCMOP1_BC, DASCMOP5_BC, LIRCMOP7_BC, LIRCMOP9_BC`，`runs=1:3`。

**成功判据**

- runtime 降至少 30%；
- `GAN_to_Segment_Dist90`、`normal_obj_seg_dist90` 不劣于 baseline 10%；
- `BoundaryHit_all` 不低于 baseline 10%。

**失败判据**

- runtime 虽降，但 DASCMOP4/5 或 LIRCMOP7 类问题的边界线指标明显崩掉。

------

### Proposal 2 — 把 `nearseg` 选样和 `condition_knn` 保留逻辑合并，取消“先造后删”

**优先级：第二**

**改哪些函数**

- `BuildBoundaryTargetTriples_BDG.m`
- `FilterBoundaryTargetTriples_BDG.m`
- `BuildGANTrainingSet_BDG.m`

**删/合并什么**

- retained mainline 下，不再先生成 `647.9` 条 nearseg rows，再删到 `388.8` 条；
- `condition_knn` 不再是 post-hoc row filter，而改成 target builder 内部的保留策略。

**复杂度为什么会降**

- 去掉 `ConditionKNNKeepMask_BDG` 的 `pdist2(C,C)` 全矩阵成本，主导项从 `O(T_pre^2)` 降掉。
- 去掉 40% 的无效构造/内存/AI 对齐复制。

**可能影响的指标**

- `GAN_to_Segment_Dist90`、`normal_obj_seg_dist90`
- 尤其 DASCMOP4/5 这类难题更敏感。

**最小验证实验**

- 近似地把每 pair 输出数从 5 直接压到当前有效训练规模附近；
- 不改 loss，不改 archive，不改 condition 语义。

**成功判据**

- runtime 再降至少 15%；
- `GAN_to_Segment_Dist90`、`normal_obj_seg_dist90` 不劣于 baseline 10%；
- `target_near_segment_keep_count` 与 `target_triple_count` 更接近，不再出现明显 build-delete 差。

**失败判据**

- 最终 target 数降了，但 DASCMOP4/5 的边界命中率进一步恶化。

------

### Proposal 3 — 引入 `BoundaryTrainBundle`，改成 lazy mini-batch sampling

**优先级：第三**

**改哪些函数**

- `CCMO_GAN_BDG.m`
- `BuildEvaluatedFeasibleSource_BDG.m`
- `BuildGANTrainingSet_BDG.m`
- `BoundaryGAN_BDG.m`

**删/合并什么**

- 合并 `EvaluatedSource`、pair 元数据、nearseg targets、triple-level AI 展开；
- 删除 flat `TrainDecs / ConditionData / GANTrainAI` 作为 retained mainline 的中间表示。

**复杂度为什么会降**

- 去掉 pair→triple 的大规模复制；
- 训练只按 batch 抽样，不再受 flat dataset 形状牵制；
- 数据流更短，更容易做 profile 和进一步优化。

**可能影响的指标**

- 理论上不该影响边界拟合；
- 风险主要来自 sampler 写错，而不是算法思想本身。

**最小验证实验**

- 先做 batch-level 等价性检查：
  - condition 维度仍是 7；
  - pair coverage 不低于 baseline；
  - `t` 分布与 baseline 近似。
- 再跑 `DASCMOP1_BC` 和 `LIRCMOP7_BC`，`runs=1:3`。

**成功判据**

- 目标统计和 boundary metrics 在噪声范围内一致；
- 内存和 wall-clock 都下降。

**失败判据**

- bundle sampler 改变了 pair coverage 或 `t` 分布，导致 boundary 质量偏移。

------

### Proposal 4 — 把 diagnostics/export 从“计时主路径”里拆出去

**优先级：第四**

**改哪些函数**

- `CCMO_GAN_BDG.m`
- `run_CCMO_GAN_BDG_fixmd_pairt_diagnostic_suite.m`

**删/合并什么**

- `traindiagnose`、`stageProbeN`、snapshot / diagnostic CSV 导出改为显式 flags；
- 生产 runtime 报告只算 `runtime_core`；分析 suite 再算 `runtime_total`。

**复杂度为什么会降**

- 每次 refresh 少一次 `n=100` 的 train diagnose；
- 每个 stage 少一次 `n=200` 的 probe；
- 少大批 CSV I/O。

**可能影响的指标**

- **算法本体指标不应变化**；
- 变化的只是观测和导出。

**最小验证实验**

- 固定 seed，同一配置跑 diagnostics on/off 对照。

**成功判据**

- `GAN_to_Segment_Dist90`、`normal_obj_seg_dist90`、`BoundaryHit_all` 一致；
- runtime 明显下降。

**失败判据**

- 关掉 diagnostics 以后算法路径发生副作用。

------

## 6. Do Not Do

### 不推荐优先做的方向

| 方向                                                         | 为什么不应优先做                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **直接退回 endpoint**                                        | 虽然快，但边界拟合明显变差：all10 上 `runtime 648.6s` vs `1940.4s`，但 `GAN_to_Segment_Dist90 2.1279` vs `0.6134`，`normal_obj_seg_dist90 2.4393` vs `0.6768`；去掉 DASCMOP4/5 后也仍然更差。 |
| **先花时间优化 bbox 分支**                                   | 当前 suite 所有 nearseg `candidate_count` 都远低于 `1e6` 阈值，bbox 分支实际上没被触发。现在优化它，对当前主线 runtime 基本无效。 |
| **先做 point-to-segment 微优化/向量化，而不改 triple+epoch 结构** | bundle 顶部已经直接给出结论：慢的主要机制不是点到线段距离本身，而是 target triples 放大和 epoch 训练更新放大。 |
| **继续把 `ref_af_nd` / `limited` / `pairRefMode="none"` 当生产路径继续维护** | retained mainline 已经只保留 `global_af_nd + af_not_dominates_ai + none + neighbor4`，这些 legacy 分支不是当前主线。 |
| **先做 `trainGap` / `ganIter` 的 brute-force 大扫参**        | 当前 bundle 没有 fixmd 主线上的这类 speed-quality ablation。直接扫参会把结构问题掩盖掉。**证据不足：应先加分函数 timer，再做少量定向 ablation。** |

------

## 7. Next Experiment Matrix

### 实验 1：`epoch -> iter` 的最小改动验证

| 改什么                                                       | 跑哪些问题/runs                                              | 看哪些指标                                                   | 成功标准                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------- |
| 只把 retained mainline `ganTrainMode` 从 `epoch` 改成 `iter`；其他配置全不动 | `DASCMOP1_BC, DASCMOP5_BC, LIRCMOP7_BC, LIRCMOP9_BC`；`runs=1:3` | `runtime_total`、新增 timer：`t_target_build, t_filter, t_gan_train, t_gan_diag, t_archive`；以及 `GAN_to_Segment_Dist90`、`normal_obj_seg_dist90`、`BoundaryHit_all`、`RawGANFeasibleRate`、`gan_g_loss_count` | runtime 降 ≥ 30%；边界拟合主指标不劣于 baseline 10% |

### 实验 2：把 nearseg 构造与保留逻辑合并，取消 post-hoc row deletion

| 改什么                                                       | 跑哪些问题/runs | 看哪些指标                                                   | 成功标准                                                     |
| ------------------------------------------------------------ | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 在 `BuildBoundaryTargetTriples_BDG` 内部完成 retained mainline 的保留逻辑；`FilterBoundaryTargetTriples_BDG` 对 nearseg mainline 不再删行 | 同实验 1        | 除实验 1 指标外，再看 `target_near_segment_keep_count`、`target_triple_count`、`target_filter_retain_ratio`、`condition_knn` 耗时 | runtime 再降 ≥ 15%；`target_near_segment_keep_count` 与 `target_triple_count` 接近；边界指标不劣于 baseline 10% |

### 实验 3：组合版全量回归

| 改什么                                                       | 跑哪些问题/runs                                       | 看哪些指标                                                   | 成功标准                                                     |
| ------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 组合 `iter` + 合并后的 nearseg builder；保留 `near_segment_feasible + yt_dt_t_ref + conditional_adversarial_huber` 不变 | **全 10 题**，`runs=1:3`，与当前 suite 同 worker 设置 | `runtime_total`、`runtime_core`、全部 4 个核心指标、`target_triple_count`、`gan_g_loss_count`、分函数 timer | all10 上 `GAN_to_Segment_Dist90 ≤ 0.70`、`normal_obj_seg_dist90 ≤ 0.75`、`BoundaryHit_all ≥ 0.42`、`RawGANFeasibleRate ≥ 0.49`，同时 `runtime/run ≤ 1300s` |

### 必须顺带加入的 profile 字段

无论上面 3 个实验跑哪一个，都应把下面 6 个 timer 打进 `core_metrics` 或 run summary：

- `t_archive_update`
- `t_source_build`
- `t_target_build`
- `t_target_filter`
- `t_gan_train`
- `t_gan_diagnose`

否则，关于 diagnostics 与 archive 的精确耗时占比，仍然只能停留在“代码可推断、百分比未知”的层面。