### D. 解决路线

先修正一条我上一版里不够准确的建议：

我**不再建议**把 `UpdateBoundaryMemory_RC.m` 的 per-ref 保留直接改成“纯按 gap 选”。
更合理的是：

**先把同一 `c` 下的训练云压成单一局部边界带，再在这个局部带内用 PF 更好的可行 anchor 做种子保留邻域。**

原因是当前代码里，`UpdateBoundaryMemory_RC.m` 同时做了这几件事：
`frontDepth=2` 的 feasible front 保留、`maxAnchorsPerRef` 截断、`prev1_fair_union` 并入上一代 feasible anchors、以及“每个 anchor 自身都作为 cloud member 保留”。这会让**同一 ref 下出现多段/多层 boundary cloud 混合**。在这种混合状态下：

- **PF 更好**，不等于它一定来自“当前想学的那一条边界带”；
- **gap 更小**，也不等于它一定比 PF 更好地代表“当前边界”；
- 真正先要修的是：**同一 coarse condition `c` 下，训练分布是否尽量单值、单带、局部连续。**

所以我现在更倾向于：
**不是“PF vs gap 二选一”，而是“先单带化，再在单带内用 PF 做种子，gap 只做 proximity gate”。**

| 改什么                                                       | 为什么改                                                     | 对应证据                                                     | 预期视觉变化                                                 | 预期 CSV 指标变化                                            | 最小实验 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| **主线 1：把 BMem 改成“PF-seeded local band”，而不是当前的 mixed cloud** | 当前最根本的问题不是 GAN 目标本身，而是同一 `c` 下训练云混了多段边界。`BuildBoundaryDataset_RC.m` 直接把 `BMem.x_b` 全送进 `TrainX`，而 `TrainC` 只保留 `W(ref,:)`，所以只要同一 ref 内混了多段，`c` 就天然不够分辨。这里先不加新 condition，而是先把同一 ref 的训练云压成一条局部带。 | 代码：`UpdateBoundaryMemory_RC.m` 里 `frontDepth=2`、`capFeasibleAnchorsPerRef` 用 `CalFitness_CBS` 截断、`prev1_fair_union` 并入旧 feasible anchors；`BuildBoundaryDataset_RC.m` 明确 `Condition = W(region,:) only`，并把所有 `BMem.x_b` 作为 `TrainX`。图片：`experiments/selected_images/mainline_prev_anchor_repair/mainline_prev_anchor_repair_LIRCMOP5_BC_FE050000.png`、`...LIRCMOP7_BC_FE050000.png`、`...LIRCMOP8_BC_FE050000.png` 都能看到橙色训练点不是单一窄带。 | 橙色训练点会先从“多段/多层云”变成“每个 ref 一条更短、更连续的局部带”；红点会跟着减少远离橙带的竖散点和跨层点。 | `train_count` 可能小幅下降；但 `gap_ratio50`、`gap_ratio90` 应下降，`near_boundary_rate_gap1` 应上升；`generated_problems` 不应明显下降。 | 实验 1   |
| **主线 2：把 query 从“all W 随机”改成“有支撑的邻接 continuation”，不是退回纯 populated-only** | 你的创新点需要“由部分边界解继续生成尚未探索但仍属当前边界的解”。这意味着 query 不能回退成纯 replay。当前两个极端都不理想：`random_all_w` 太粗，`boundary_populated` 太保守。更合理的是：**以已支撑 ref 为中心，向一跳相邻 ref 做局部外推**。这样既保留“生成未探索边界”的能力，又避免全局无支撑喷散。 | `analysis_branch_summary.csv`：`query_boundary_populated` 相比 baseline 有明显更好的贴边（`near_gap1_mean 0.474 > 0.265`，`gap50_median 1.024 < 2.761`，`gap90_median 6.866 < 18.985`），但 coverage 没明显扩大；`random_iter100` 最终生成问题数 6，高于 `query_boundary_iter100` 的 4，但本地贴边更差。`analysis_final_fe_by_problem.csv` 里，在共同生成的问题上，`query_boundary_iter100` 对 `LIRCMOP7/8/9` 的 `near_gap1/gap50` 多数优于 `random_iter100`，说明 query 更贴边，但过于保守。另有事件统计显示 `query_boundary_populated` 的平均 `query_count≈22`，而 `random_all_w` 为 50；配合 `train_mean≈90`，前者每个 query 区域平均大约只有 4 行训练，后者摊到所有 ref 只有约 1.8 行/区，支撑太薄。 | 红点不应再在完全空白的 ref 上到处喷散；而应主要沿已有橙带的邻接方向延伸，形成“边界 continuation”，尤其在 `LIRCMOP7/8` 上应比 `random_all_w` 更像沿曲线外推。 | 相比 `query_boundary_populated`，`generated_problems` 应回升；相比 `random_iter100`，`gap50/gap90` 应更低、`near_gap1` 不下降。`query_unique_ref_count` 应介于 populated-only 和 all-W 之间。 | 实验 2   |
| **主线 3：在主线 1+2 固定后，只做小幅训练步数定标，优先测 75，不先动 z 和网络容量** | 当前证据不支持先改大网络，也不支持先把 z 当主因。对 30 维问题，当前 WGAN 结构其实很小：`cDim=2`、`zDim=6`、hidden `[32 32]` 时，生成器约 **2334** 个参数，critic 约 **2145** 个参数，总共约 **4479**。结合包内 `train_mean≈80~90`、`miniBatch=32`，`ganIter=50/75/100` 对生成器大致对应 **18/27/36** 次“有效样本遍历”，critic 还是它的 5 倍量级。也就是说，当前训练轮次更像**次因**，不是首因。 | 代码：`BoundaryWGAN_RC.m` 显示 `zDim=6`、`miniBatch=32`、`nCritic=5`、hidden `[32 32]`；`BoundaryCGAN_CBS.m` 的 BCE 版本也是同级小网络。CSV：`query_boundary_iter75` 的 `gap90_median=4.759` 已明显优于 baseline/random100；`query_boundary_iter150` 只有很小收益甚至 `gap90_median=4.950` 还略差，但 `runtime_mean_min` 从约 21.5 增到约 42.3。`refcap5_min32_CGAN_vs_WGAN` 上，WGAN 最终 `gap50≈1.782`、`gap90≈8.268`、`near_gap1≈0.533`，明显优于 CGAN 的 `2.976/12.796/0.258`。 | 50→75 应表现为同一条红色边界带更密、更稳；75→100 如果还有提升，更多应体现在尾部离群点再减少，而不是边界形状换掉。 | 预期 `75` 相比 `50` 会降 `gap90`；`100` 相比 `75` 只会有小幅收益。若 `100` 才能显著改善，则说明训练步数仍是次级瓶颈；若 `75≈100`，则说明该阶段已接近饱和。 | 实验 3   |

补充两点，我现在明确不把它们放进主线：

- **不把 z 当成主线旋钮。** 包内已有 `z_zero_sample`、`z_sigma_025`。它们影响很大，但高度问题依赖：例如 `z_zero_sample` 在某些问题上能收紧，但在 `LIRCMOP9_BC` 最终 `gap90` 仍极大；`z_sigma_025` 还会让 `LIRCMOP6_BC` 在 FE100000 时 `train_count=26 < minGANTrainCount=32`，直接无生成。
- **不把网络扩容列为主线。** 当前网络总参数量并不大，且证据不足以说明它是首因；先动它，风险是把“数据几何问题”误判成“容量问题”。

------

### E. 最小实验计划

#### 实验 1：`bmem_single_band_pfseed`

**验证什么假设**
主假设是：当前“生成不够贴边”的首因是 **BMem 在同一 ref 内混了多段/多层训练云**；只要把训练云压成单一局部带，哪怕 `c` 和 WGAN 都不变，红点也会明显变窄。

**改哪些源码文件**
只改：

- `source/current_CBS_CGAN/Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_RC.m`

具体改法我建议是：

- 保留现有 pairing 与 `adaptiveGapCap`；
- 但把每个 ref 的 feasible anchors 改成：
  1. 先用当前 `CalFitness_CBS` 选出一个 **PF-seed anchor**；
  2. 再按 objective-space 邻近性，只保留 seed 周围最近的若干 paired anchors；
  3. `prev1_fair_union` 并入的旧 feasible anchors，只在“同 ref 且落在当前 local band 邻域”时保留。
     这比“纯按 gap 排序”更稳，也更符合你对 PF 的理解。

**不改哪些东西**

- 不改 `BuildBoundaryDataset_RC.m`
- 不改 `RunRegionGAN_RC.m`
- 不改 `BoundaryWGAN_RC.m`
- 不改 `zDim=6`
- 不改 `sampleZMode/trainZMode`
- 不改 hidden `[32 32]`
- 不改 `nGen=30`
- 不改 `ganIter=50`

**运行规格**

- 问题：`LIRCMOP5_BC`–`LIRCMOP10_BC`
- `runs=1:3`
- `N=100`
- `maxFE=100000`
- stage 仍按主线已有快照设置

**观察哪些图片**
重点看：

- `LIRCMOP5_BC` FE50000 / FE100000
- `LIRCMOP7_BC` FE50000 / FE100000
- `LIRCMOP8_BC` FE50000 / FE100000
- `LIRCMOP9_BC` FE100000

**观察哪些 CSV 指标**

- `stage_snapshots_all.csv` 的
  `train_count`, `raw_generated_count`, `feasible_rate`,
  `gap_ratio50`, `gap_ratio90`, `near_boundary_rate_gap1`
- `analysis_final_fe_by_problem.csv`

**什么结果支持假设**

- 橙色训练点先明显从多层混合变成单一局部带；
- 红点随之收缩；
- `gap50/gap90` 下降，`near_gap1` 上升；
- `generated_problems` 不比当前主线更差。

**什么结果推翻假设**

- 橙点已经变薄，但红点仍然厚、仍然跨空区域；
- 或者 `train_count` 大幅跌破 32，导致大量 stage 直接无生成。

------

#### 实验 2：`bmem_single_band_pfseed + query_support_expand1 + ganIter75`

**验证什么假设**
主假设是：在训练云已单带化后，最合适的 query 不是 `random_all_w`，也不是纯 `boundary_populated`，而是
**“有支撑 ref + 一跳邻接 ref”的 boundary continuation query**。
这能同时保留“生成未探索当前边界”的能力和本地贴边性。

**改哪些源码文件**

- `UpdateBoundaryMemory_RC.m`（沿用实验 1）
- `RunRegionGAN_RC.m`
- 如果需要注册新模式，再改 `CBS_RegionGAN_Base.m` 或对应 branch runner

我建议的新 query 规则非常克制，不新增新参数：

1. 先给每个 populated ref 分 1 个 sample；
2. 预算有剩余时，再给其一跳邻接且未 populated 的 ref 分 1 个 sample；
3. 还有剩余，再回到 populated ref 做第二轮分配。

这样仍沿用现有 `nGen=30` 和 `queryPerCondition`，不引入新超参。

**不改哪些东西**

- 不改 `BuildBoundaryDataset_RC.m`
- 不改 `BoundaryWGAN_RC.m`
- 不改 `c` 的定义，仍然是 `W(ref,:)`
- 不改 `zDim=6`
- 不改 hidden `[32 32]`
- 不改 `sampleZMode=random`、`trainZMode=random`
- 不改 `nGen=30`

**运行规格**

- 问题：`LIRCMOP5_BC`–`LIRCMOP10_BC`
- `runs=1:3`
- `N=100`
- `maxFE=100000`
- `ganIter=75`
- 其他都和主线一致

我这里把 `ganIter` 直接放 75，是因为包内现有证据里，75 已经比 50 更贴边，而 150 没显示出足够大的额外收益来支撑更高代价。

**观察哪些图片**
重点看：

- `LIRCMOP5_BC` FE100000
- `LIRCMOP6_BC` FE100000
- `LIRCMOP7_BC` FE70000 / FE100000
- `LIRCMOP8_BC` FE70000 / FE100000
- `LIRCMOP9_BC` FE100000

**观察哪些 CSV 指标**

- `stage_snapshots_all.csv`
- `event_summary_all.csv` 的
  `query_count`, `query_sample_count`, `query_unique_ref_count`
- `analysis_branch_summary.csv`
- `analysis_final_fe_by_problem.csv`

**什么结果支持假设**

- 相比 `query_boundary_populated`，`LIRCMOP5/6` 不再长期无生成；
- 相比 `random_iter100`，共同生成的问题上 `gap50/gap90` 更低、`near_gap1` 不低；
- `query_unique_ref_count` 介于 populated-only 与 all-W 之间；
- 图上红点表现为“沿已有橙带继续延伸”，而不是“跳到完全空白 ref”。

**什么结果推翻假设**

- coverage 仍然只有 4 个问题左右；
- 或者本地贴边性退化到接近 `random_all_w`；
- 或者 `LIRCMOP9_BC` 仍然出现明显全局喷散，说明 query 还太粗。

------

#### 实验 3：`clean_pipeline_iter_sweep`（50 / 75 / 100）

**验证什么假设**
主假设是：在实验 1+2 修好训练几何和 query 之后，当前网络大小已经够小，`ganIter` 只需要做**轻量定标**；
也就是 **75 应该已接近够用，100 最多是次级修补，不会是决定性变化**。

**改哪些源码文件**
只改实验运行配置：

- branch runner / defaults（例如 `run_CBS_RegionWGAN_GP_ablation_branches.m` 或主配置入口里 `ganIter`）

**不改哪些东西**

- 不改 `UpdateBoundaryMemory_RC.m` 的新 local-band 逻辑
- 不改新的 query mode
- 不改 `BuildBoundaryDataset_RC.m`
- 不改 `BoundaryWGAN_RC.m`
- 不改 hidden `[32 32]`
- 不改 `zDim=6`
- 不改 z 采样方式
- 不改 `nGen=30`

**运行规格**

- 只跑代表性问题：`LIRCMOP7_BC`, `LIRCMOP8_BC`, `LIRCMOP9_BC`
- `runs=1:3`
- `maxFE=100000`
- 比较 `ganIter = 50 / 75 / 100`

**观察哪些图片**

- `LIRCMOP7_BC` FE70000 / FE100000
- `LIRCMOP8_BC` FE70000 / FE100000
- `LIRCMOP9_BC` FE100000

**观察哪些 CSV 指标**

- `stage_snapshots_all.csv`
- `run_summary.csv`
- branch summary 里的
  `gap_ratio50`, `gap_ratio90`, `near_boundary_rate_gap1`,
  `generated_problems`, `runtime_mean_min`

**什么结果支持假设**

- `75` 相比 `50` 有明显更低的 `gap90`；
- `100` 相比 `75` 只有小幅改进，或者几乎持平；
- 视觉上 75→100 主要是尾部离群点再少一点，而不是边界形状发生本质变化。

**什么结果推翻假设**

- 只有 `100` 才明显把红点拉回窄边界；
- 或者 `50`、`75` 仍然很厚，说明训练优化本身还没有到位。
  若出现这种情况，默认 `ganIter` 才值得上调到 100；但这也仍然是在**几何和 query 已修好之后**再做，而不是先做。

------

这版路线的核心变化只有一句话：

**先修正“同一 coarse condition 下训练分布被混成多条边界”的问题，再用“邻接 continuation query”去实现你要的“由部分边界继续生成未探索边界”，最后才轻量定标训练步数。**

这样更贴合你的主线，也比我上一版里“直接按 gap 选”更稳。
