我先遍历了 ZIP 全部 702 个条目，再核查了核心源码、分支汇总 CSV、stage/event CSV，以及主线 / prescreen / `nCritic` 对照的代表性 PNG。下面只基于 ZIP 内证据，不假设 ZIP 外还有任何数据；并且 ZIP 内确实没有 `.mat` 文件 。

## 1. ZIP 内可确认事实

1. ZIP 结构层面，可确认有 331 张 PNG、318 个 CSV、43 个 `.m`、4 个 `.md`、1 个 `.txt`，没有 `.mat`。主源码在 `Algorithms/Multi-objective optimization/CBS-CGAN/`，结果主目录在 `Data/CBS_RegionGAN_compare/`。
2. 当前 **exact mainline** 默认值是明确的：`queryMode=random_all_w`、`prevBMemMode=prev1_fair_union`、`bmemBandMode=current`、`trainGap=1`、`archiveGap=1`、`nGen=30`、`zDim=6`、`ganIter=50`、`ganMiniBatch=32`、`sampleSigma=0.3`、`gpLambda=10`、`nCritic=5`、`maxAnchorsPerRef=5`、`minGANTrainCount=32`。证据：`Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionWGAN_GP.m:84-110`；`Data/CBS_RegionGAN_compare/fixmd_all_20260701_1312/branch_summary.csv:2-11`。
3. 当前 WGAN 不是每次重置，而是 **warm-start 连续训练**；只有维度、`zDim` 或 hidden 结构不匹配时才重新初始化。证据：`Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryWGAN_RC.m:50-58`。
4. 当前一次训练事件的更新力度是：每个 outer iter 做 `nCritic` 次 critic 更新，再做 1 次 generator 更新；所以主线 `ganIter=50,nCritic=5` 实际是 **250 次 critic + 50 次 generator**。证据：`Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryWGAN_RC.m:60-79`。
5. 当前训练集不是“真实边界带”的直接表示，而是 **可行侧 boundary cloud**。`BuildBoundaryDataset_RC` 明写了 condition 只用粗 ref 方向，`objective y is deliberately dropped`；真正喂给 GAN 的是 `TrainX=BMem.x_b`、`TrainC=W(ref,:)`。证据：`Algorithms/Multi-objective optimization/CBS-CGAN/BuildBoundaryDataset_RC.m:3-7,28-41`。
6. `BMem` 本身保存了可行 anchor 与最近不可行端点的配对，但 **训练时丢掉了不可行侧与 gap 信息**。`UpdateBoundaryMemory_RC` 里 `y_b/x_b` 来自可行 anchor，`y_i/x_i` 是最近不可行端点；`prev1_fair_union` 只复用上一轮可行 anchors，不复用旧不可行端点。证据：`Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_RC.m:17-20,85-128,248-268`。
7. 当前 real samples 很可能天然就是“可行侧边界云”，而不一定是一条薄带。因为代码会从 **前 `frontDepth=2` 个可行 front** 取 anchors，并且每个 ref 最多保留 5 个 anchors。证据：`Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_RC.m:85-99,132-150`。
8. 当前已有 critic 训练/holdout 诊断，不应再以 `fix.md` 为最新事实源。`fix.md:17` 还写着“当前没有 holdout/generalization 训练诊断”，但现行代码已写入 `critic_train_gap` / `critic_holdout_gap`，stage CSV 头也已有这些字段。证据：`fix.md:17`；`Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryWGAN_RC.m:98-130`；`Data/CBS_RegionGAN_compare/fixmd_all_20260701_1312/fixmd_prescreen1_fixed50/stage_snapshots_all.csv:1`。
9. `prescreen` 已实现，但逻辑非常明确：同一条件先采 `prescreenMultiplier` 个候选，再按 critic 分数取最高者进入真实评估。证据：`Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryWGAN_RC.m:218-273`。默认 `prescreenMultiplier=1`：`BoundaryWGAN_RC.m:410-423`。
10. 现有 `gap50/gap90` 不是“到真实边界的距离”，而是 **到 BMem 中最近 feasible→infeasible 线段代理的归一化距离**。证据：`Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionGAN_Base.m:861-917,984-1032`。
11. `nCritic/ganIter` 的 4 组 exact-mainline 对照已经在 ZIP 里定义好了：`A_current_50x5`、`B_iter50_ncritic2`、`C_iter100_ncritic2`、`D_iter150_ncritic1`。证据：`tmp/run_ncritic_ratio_experiment.m:24-28,68-89,121-140`；`Data/CBS_RegionGAN_compare/ncritic_ratio_20260702_1220/branch_summary.csv:2-5`。

## 2. 当前问题的严格诊断

### 诊断 A：当前模型学到的更像“粗 ref 条件下的可行边界云”，不是“精确窄边界带”

**ZIP 内可确认事实**
`TrainC` 只有 `W(ref,:)`，`TrainX` 只有 `BMem.x_b`，精确目标位置 `y_b`、不可行侧 `y_i`、以及 `gap` 都没有进入 GAN 条件或训练目标。证据：`BuildBoundaryDataset_RC.m:3-7,28-41`；`UpdateBoundaryMemory_RC.m:120-127`。

**基于事实的推断**
这意味着当前 GAN 学到的是
`p(x | 粗ref)`，
而不是
`p(x | 精确边界位置/边界距离/边界两侧关系)`。
所以只要同一个 ref 下同时存在多层位置、厚带、或者多个局部边界片段，GAN 就会把它们混到一个 condition 里，天然容易生成厚带、偏移带、散点云，而不是窄带。

**证据不足**
仅凭现有 ZIP 还不能断言“只要加 1 个条件标量就一定完全修复”，但可以断言：**当前 condition 设计过粗**，这是源码层面的确定问题。

------

### 诊断 B：当前 real samples 本身就可能是厚带 / 多层带，而不是真正的薄边界

**ZIP 内可确认事实**
`UpdateBoundaryMemory_RC` 设计目标就是保留“boundary cloud”，不是 one-point skeleton；它从前 `frontDepth=2` 个可行 front 取 anchor，每个 ref 最多保留 5 个 anchor，并做 `prev1_fair_union`。证据：`UpdateBoundaryMemory_RC.m:3-20,85-99,132-150,248-268`。

**基于事实的推断**
所以当前 real samples 完全可能是：

- 可行侧边界云；
- 同一 ref 内的厚带；
- 多层带；
- 不同阶段拼出来的混合云。

这会直接污染“real 的几何定义”，使 critic 更像在学“当前云的密度/形状”，而不是“真实可行/不可行边界”。

**证据不足**
现有 ZIP 还不能仅凭代码就把责任 100% 归到训练集；因为 generator 也可能把原本不算太厚的云扩成更厚的云。这个要靠后面说的“真实边界距离 + 宽度”诊断来拆分。

------

### 诊断 C：PNG 显示当前输出形态并不稳定地像“窄边界带”

我看过主线和对照分支的代表性图，结论很一致：**当前输出更常见的是厚带、偏移带、散点云、局部塌缩和阶段漂移，而不是稳定窄带。**

**ZIP 内可确认事实（具体图）**

- `Data/CBS_RegionGAN_compare/mainline_prev1_fair_union_figs_runs1_10w_20260627_004228/domain_figures_all/LIRCMOP6_BC_run1_FE070000_domain_boundary.png`：红点大多位于橙色训练边界之上/之外，形成明显偏移散带，并有不少点落在大块不可行域中。
- `.../LIRCMOP5_BC_run1_FE100000_domain_boundary.png`：红点在左下主边界上方形成一团偏移带，右下小片边界覆盖明显不足。
- `.../LIRCMOP9_BC_run1_FE050000_domain_boundary.png`：红点大面积散到小可行岛之外，属于明显散点云/阶段漂移。
- `Data/CBS_RegionGAN_compare/ncritic_ratio_20260702_1220/B_iter50_ncritic2_run1_figures/LIRCMOP6_BC_run1/figures/LIRCMOP6_BC_run1_FE070000_domain_boundary.png`：`nCritic=2` 后仍是上方偏移散带，并没有收缩成窄边界带。

**基于事实的推断**
所以当前问题不是“某一张图偶然不好”，而是跨问题、跨 FE 阶段、跨分支都在反复出现：

- 厚带；
- 平行偏移带；
- 局部散点云；
- 只贴住一小段边界的局部塌缩；
- 某些阶段短暂变好，后期又漂移。

这与用户目标“形成一条贴近真实边界的窄带”并不一致。

**证据不足**
还不能说“所有问题都一样坏”。ZIP 里也确有少数阶段、少数问题看起来比别的阶段更贴边；所以更准确的结论是：**当前形态不稳定，且整体上未达到主目标。**

------

### 诊断 D：当前 critic score 不是可靠的 boundary-quality proxy；prescreen 因此没有被证明有效

**ZIP 内可确认事实**
prescreen 的唯一筛选依据就是 critic 分数：`BoundaryWGAN_RC.m:249-273`。
stage CSV 已经记录了 `critic_train_gap`、`critic_holdout_gap`、`generated_critic_score_mean`、`gap_ratio50/90`、`near_boundary_rate_gap1` 等字段：`fixmd_prescreen1_fixed50/stage_snapshots_all.csv:1`。

**基于 ZIP 内 CSV 计算的结果**
我把 `fixmd_all_20260701_1312/*/stage_snapshots_all.csv` 中 228 条可用 stage 记录合并后，按 stage 级做了 Spearman：

- `generated_critic_score_mean` vs `gap_ratio50` ≈ **+0.53**
- `generated_critic_score_mean` vs `gap_ratio90` ≈ **+0.60**
- `generated_critic_score_mean` vs `near_boundary_rate_gap1` ≈ **-0.37**

也就是：**critic 分数越高，当前 boundary proxy 反而越差**。这已经足够说明：当前 critic 更像在学“像不像训练云”，不是“是不是更贴真实边界”。

再看 exact mainline 下的 prescreen 对照：

- 分支定义：`fixmd_prescreen1_fixed50` vs `fixmd_prescreen8_fixed50`，只差 `prescreen_multiplier`，其余都相同。证据：`Data/CBS_RegionGAN_compare/fixmd_all_20260701_1312/branch_summary.csv:9-10`。
- 我对这两支共有的 25 个 stage snapshot 做一一对比，`prescreen8` 仅：
  - **11/25** 次 `gap50` 更低；
  - **13/25** 次 `gap90` 更低；
  - **8/25** 次 `near_boundary_rate_gap1` 更高；
  - **6/25** 次 `feasible_rate` 更高。
- 其 stage 汇总中位数也更差：`prescreen1_fixed50` 的 `gap50≈1.965, gap90≈8.664, near≈0.353, feasible≈0.521`，而 `prescreen8_fixed50` 为 `gap50≈2.495, gap90≈12.832, near≈0.338, feasible≈0.441`。数据来源：上述两支各自的 `stage_snapshots_all.csv`。

**结论**
当前 prescreen 没有稳定改善，并不能说明 GAN 一定失败；它只能说明：**当前 critic 分数还没有被证明能代理 boundary quality。**

## 3. 最多 3 个新的核心评价指标

我建议把当前主目标拆成 3 个几何上直接对应的指标。注意：这 3 个指标是给 **ZIP 里这批 2D objective-space 诊断问题** 用的；实现上不改主算法，只加诊断。`run_CBS_RegionCGAN_training_diagnostics.m:911-953` 已经能画出真实 feasible/infeasible 域，所以可以在同一诊断管线里算。

### 指标 1：`Bdist50_true`

定义：把 objective space 归一化到 `[0,1]^2` 后，计算每个生成点到**真实可行/不可行边界**的有符号距离 `d_true`，取 `median(|d_true|)`。
含义：越小越贴边。
它直接回答“贴得近不近”。

### 指标 2：`Bwidth90-10_true`

定义：对同一批生成点的有符号距离 `d_true`，计算 `Q90(d_true)-Q10(d_true)`。
含义：越小越窄。
它直接回答“是不是薄带，而不是厚带 / 多层带 / 散点云”。

### 指标 3：`Bcover@ε_true`

定义：把**真实边界弧长**均匀切成固定数量的 bins，例如 20 段；统计有多少 bins 至少被一个满足 `0 <= d_true <= ε` 的生成点命中，记为覆盖率。这里 `ε` 固定取归一化 objective box 对角线的 2%。
含义：越高越好。
它直接回答“是不是只塌到一小段，还是沿真实边界形成了一条带”。

------

### 现有 `gap50/gap90` 到底测了什么？

它们测的不是“到真实边界的距离”，而是：

1. 先把 `BMem.y_f -> BMem.y_i` 当作一组**代理边界线段**；
2. 再算生成点到最近代理线段的距离 `segment_dist`；
3. 最后除以该线段长度，得到 `gap_ratio`。
   证据：`CBS_RegionGAN_Base.m:984-1032`。

所以 `gap50/gap90` 实际上测的是：
**“生成点有没有靠近当前 BMem 里那批 feasible→infeasible 配对线段”**。

------

### 为什么 `gap50/gap90` 数值可能还行，但 PNG 仍然难看？

因为它们有 4 个先天盲点：

1. **它们对齐的是 BMem 代理，不是真实边界。**
   如果 BMem 自己就是厚带/偏移带，GAN 只要贴住 BMem，`gap` 也会好看。
2. **它们不看 sidedness。**
   只要离代理线段近，在线段哪一侧都可能被记成“不错”；但你真正要的是“可行侧薄带”。
3. **它们不直接看 band width。**
   一条又厚又平行的偏移带，只要整体离某些代理线段不太远，`gap50` 依然可能不差。
4. **它们不看沿边界的覆盖。**
   局部塌缩到一小段，也可能拿到不错的 `gap50`。

所以我的判断是：
**`gap50/gap90` 只能保留为辅助指标，不能再当核心指标。**

辅助用途很清楚：它们还能回答“GAN 有没有跟着当前 BMem 走”。
但它们不能回答“GAN 有没有学到你要的真实窄边界带”。

## 4. `ganIter/nCritic` 的最小实验方案

我建议就用 ZIP 里已经存在的 4 组 exact-mainline 方案，不再加新组：

1. `A_current_50x5`：`ganIter=50, nCritic=5`
2. `B_iter50_ncritic2`：`50, 2`
3. `C_iter100_ncritic2`：`100, 2`
4. `D_iter150_ncritic1`：`150, 1`

证据：`tmp/run_ncritic_ratio_experiment.m:24-28,68-89,121-140`；`Data/CBS_RegionGAN_compare/ncritic_ratio_20260702_1220/branch_summary.csv:2-5`。

### 为什么这 4 组够了

它们已经覆盖了 3 类比值：

- critic-heavy：`50x5`
- 较平衡：`50x2`
- 中等强度、较平衡：`100x2`
- generator-heavy：`150x1`

而且它们都保持了 exact mainline 的其他设定不变：`random_all_w`、`prev1_fair_union`、`sampleSigma=0.3`、`zDim=6`、`trainGap=1`、`prescreen=1`。

### 这个最小实验里，`ganIter` 应不应该固定？

**应该固定。**
因为你这一步的目的不是做最终调度策略，而是先把 `ganIter/nCritic` 的作用拆干净。
如果这一步再混入 `trainGap=2`、trigger、two-level schedule，就会再次把结论搞混。

### CGAN/WGAN 是否每代都需要训练？

**目前没有证据证明“必须每代都训”。**
但这一轮最小实验里，我建议先保持 `trainGap=1` 不动，只测 `ganIter/nCritic`。
原因是 exact-mainline 的 `trainGap=2` 分支在当前 proxy 上并不好：`fixmd_train_gap2_iter50/75` 的可比较生成阶段更少，`gap50/gap90` 也更差。我不建议在还没解决核心指标之前，又把 schedule 混进来。证据：`Data/CBS_RegionGAN_compare/fixmd_all_20260701_1312/branch_summary.csv:2-8`，以及对应 `stage_snapshots_all.csv` 汇总。

### 当前 `5 critic + 1 generator` 是否合适？

我的结论是：**不能说它最优，但也不能说它明显不合理。**

按当前 ZIP 中旧 proxy 来看，我汇总后得到：

- `A_current_50x5`：`gap50≈1.965, gap90≈8.664, near≈0.353`
- `B_iter50_ncritic2`：`gap50≈2.918, gap90≈10.963, near≈0.251`
- `C_iter100_ncritic2`：`gap50≈1.956, gap90≈8.699, near≈0.317`
- `D_iter150_ncritic1`：`gap50≈4.831, gap90≈11.114, near≈0.235`

在旧 proxy 下，排序更像 **A≈C > B > D**。
所以现有 ZIP 并不支持“立刻抛弃 5:1”。
但因为这些 proxy 本身就不对题，也**不能**据此宣称 5:1 已被验证最佳。

### 4 组实验如何同时回答 3 个问题

你要求同时确定：

1. 哪组最像窄边界带；
2. 哪组让 critic score 更接近 boundary-quality proxy；
3. prescreen 是否因此变有效。

做法不需要超过 4 组：

- **4 个 evolutionary groups** 只跑 A/B/C/D，`prescreen=1`。
- 在每个 stage snapshot 上，用**同一个已训练 GAN**、同一个 `QueryC`，额外做一次**离线 paired resampling**：
  - 一次 `prescreen=1`
  - 一次 `prescreen=8`
    这只是诊断采样，不再形成新的 evolutionary branch。

这样 4 组足够同时回答 3 个问题。

### 选择规则

我建议用下面顺序做 promotion：

1. 先比 `Bcover@ε_true`，选覆盖最高者；
2. 覆盖相近时，再比 `Bdist50_true`；
3. 再比 `Bwidth90-10_true`。

原因很简单：
你要的是“沿真实边界形成一条窄带”，不是“只在一个点上很准”。

### Critic proxy 的判定规则

对每组，在所有 stage 上计算：

- `corr(critic_score, Bdist50_true)` 应为负；
- `corr(critic_score, Bwidth90-10_true)` 应为负；
- `corr(critic_score, Bcover@ε_true)` 应为正。

如果某一组这 3 个方向在多数问题/阶段上都对，才说明 critic score 开始接近 boundary-quality proxy。
如果仍然方向错或很弱，就不要把 critic 用作 prescreen 代理。

### Prescreen 的判定规则

只要在同一 GAN 上，`prescreen=8` 相比 `prescreen=1`：

- `Bcover@ε_true` 提升，
- 且 `Bdist50_true`、`Bwidth90-10_true` 不恶化，

才算 prescreen 真有效。
否则就继续关掉。

## 5. 边界存档与训练集构造的最小优化建议

### 先回答你问的几个判断题

**当前训练集是否真的代表“目标空间可行/不可行边界”？**
不是。
更准确地说，它是**边界代理**：`BMem` 里有 feasible/infeasible 配对，但 GAN 真正训练时只用了可行侧 `x_b` 和粗 ref condition。证据：`BuildBoundaryDataset_RC.m:28-41`；`UpdateBoundaryMemory_RC.m:120-127`。

**当前 real samples 是否可能只是可行侧边界云、厚带或多层带？**
是，而且代码层面很可能。证据：`UpdateBoundaryMemory_RC.m:3-20,85-99,132-150,248-268`。

**条件变量是否过粗？**
是。当前只有 `W(ref,:)`。证据：`BuildBoundaryDataset_RC.m:3-7,28-41`。

**训练集中是否缺少不可行侧信息或边界距离信息？**
对 GAN 训练来说，是缺的。
虽然 `BMem` 里保留了 `y_i/x_i/gap`，但没有进 `TrainC` 或 loss。证据：`UpdateBoundaryMemory_RC.m:120-127`；`BuildBoundaryDataset_RC.m:28-41`。

------

### 我给的最小优化建议

#### 建议 1：先改 condition，不改主机制

把 `TrainC` 从
`W(ref,:)`
改成
`[W(ref,:), s_local]`
只加 **1 个标量** `s_local`，表示该 anchor 在本 ref 内的局部边界位置。

可以非常保守地取：

- `s_local =` 该 `y_b` 在本 ref 组内沿主边界方向的归一化投影；
- 或更简单地，按本 ref 组内 `y_b` 的一维排序位置归一化到 `[0,1]`。

这样做的理由非常直接：
你现在最大的结构性问题，就是 **同一个 ref 下不同边界位置被混成一个 condition**。
只补 1 个标量，就能把“同 ref 但不同位置”的样本拆开；这比加复杂新模块更对症。

#### 建议 2：如果你坚持不改 condition 维度，就先把 GAN 的 real cloud 变薄

最保守的做法是二选一，只改一个地方：

- 把 `frontDepth` 从 2 改为 1；或
- 在 `BuildBoundaryDataset_RC` 里只取每个 ref 中 `gap` 最小的少量 anchors 进入 `TrainX`。

这样不改算法框架、不改输出目标、不加新网络，只是让 GAN 面对的 real target 更接近“薄边界”。

#### 建议 3：`prev1_fair_union` 先不要动

我不建议现在拿掉 `prev1_fair_union`。
它会放大 coverage，也确实是 current mainline 的一部分。问题不是“是否有 previous anchors”，而是“previous anchors + coarse condition + thick real cloud 被一起喂给了 GAN”。
所以先修 target/condition，比先删 `prev1` 更合理。

## 6. 当前推荐主线

当前我建议的主线不是继续扩分支，而是**先收敛**：

1. **控制线**继续保持 exact mainline：
   `CBS_RegionWGAN_GP + random_all_w + prev1_fair_union + zDim=6 + sampleSigma=0.3 + prescreen=1 + ganIter=50 + nCritic=5`。
   这不是说它已被证明最优，而是说它现在是最干净的对照基线。证据：`CBS_RegionWGAN_GP.m:84-110`；`fixmd_all_20260701_1312/branch_summary.csv:2-11`。

2. **下一步唯一值得推进的主线候选**，我建议只选一个最小改动：

   - 优先：`TrainC = [W(ref,:), s_local]`
   - 备选：只做 train-set thinning / `frontDepth=1`

   两个不要同时上。

3. **`prescreen` 继续关**。
   exact mainline 下它没有稳定收益，当前 critic 也不是可靠 proxy。

4. **`boundary_populated` 先保留为强诊断方向，不直接升主线。**
   原因不是它不行，而是它和 exact mainline 的 `random_all_w` 不同，很多拆因子实验都在这个 query mode 下完成，不能直接并到主线结论里。证据：`Data/CBS_RegionGAN_compare/qw_plan_noplot_20260628_001/qw_experiment_report.md:20-28`；`formal_round2_query_20260628_214427/round2/promotion_report.csv:2-3`；而 exact mainline 代码默认仍是 `random_all_w`：`CBS_RegionWGAN_GP.m:87-110`。

5. **当前仍有主线价值的分支**

   - `fixmd_all_20260701_1312/*`：因为它们是 exact-mainline 家族；
   - `ncritic_ratio_20260702_1220/*`：因为它们只改 `ganIter/nCritic`，且 query mode 仍是 `random_all_w`。

6. **只具诊断价值、不宜直接进主线的分支**

   - `formal_round2_query_20260628_214427`、`formal_round3_train_20260628_230707`、`qw_plan_noplot_20260628_001`：主要是 `boundary_populated` 家族；
   - `fixmd_prescreen8_delta20_boost75`：叠了 prescreen、trigger、two-level training，多因素混杂；
   - `linear150_75_prescreen8` 视觉分支：更适合说明“早期更紧、中后期漂移”，不适合直接 promotion。

## 7. 哪些结论证据充分，哪些仍需补实验

### 证据充分

1. **当前主指标不对题。**
   `gap50/gap90` 只是在量“贴不贴当前 BMem 代理线段”，不是量“贴不贴真实边界”。
2. **当前输出没有稳定形成窄边界带。**
   代表性 PNG 已经反复显示厚带、偏移带、散点云、局部塌缩和阶段漂移。
3. **critic score 不是可靠的 boundary-quality proxy。**
   ZIP 内 stage 级相关分析已足够说明方向经常是反的。
4. **exact mainline 下 prescreen 没有被证明有效。**
   `prescreen8_fixed50` 对 `prescreen1_fixed50` 没有稳定优势。
5. **当前训练数据确实可能本身就是厚带/多层边界云。**
   这是代码结构直接允许的，不是纯猜测。

### 仍需补实验

1. **A/B/C/D 哪一组在“真实边界”指标下最好。**
   现有 ZIP 只能说在旧 proxy 下 A≈C 较强，不能作为最后结论。
2. **`boundary_populated` 是否应替换 `random_all_w`。**
   它有强证据，但还不是 exact mainline 复核。
3. **最小修复该选“1 维 condition”还是“薄化训练集”。**
   这要用新指标做一次干净对照。
4. **是否需要后续引入 trigger schedule。**
   这应放在 `ganIter/nCritic` 定下来之后，不该现在混入。

### 一个重要澄清

**prescreen 没有稳定改善，不能推出“CGAN 失败”。**
它只能推出：
**“critic 分数尚未被证明可用于 boundary-quality 筛选。”**

### 如何区分“训练集厚带 / Generator 学坏 / Critic 目标错 / 三者共同作用”

最小判别方法已经在 ZIP 现有诊断里有一半基础：

- 如果 `train_width` 本身大，而 `Bwidth90-10_true` 也大：更像训练集构造问题。
- 如果 `train_width` 小，但 `Bwidth90-10_true` 大：更像 generator 学习/采样问题。
- 如果 `Bdist/Bwidth/Bcover` 变好，但 critic 排序仍错：更像 critic 目标/代理问题。
- 如果三者都差：就是三者共同作用。

现有 `train_width50/90`、`gen_width50/90` 已在代码中实现，但它们是按 ref 内 centroid spread 算的，还不是真实边界法向宽度，所以只能当过渡诊断，不能当最终核心指标。证据：`CBS_RegionGAN_Base.m:686-759`。

## 8. 你认为不应该做的修改及原因

1. **不要把目标偷换成 Pareto front / objective-point generation。**
   你的核心创新点是“CGAN/WGAN 直接生成完整决策变量”，这一点必须守住。
2. **不要把 `gap50/gap90` 继续当主指标。**
   它们只能做辅助诊断。
3. **不要把 critic score 直接当主排序依据继续做 prescreen。**
   当前 ZIP 证据已经说明它经常与 boundary quality 反向。
4. **不要把多因素叠加分支直接升主线。**
   像 `prescreen8 + delta20 + boost75` 这种分支，解释不干净，继续推进只会把结论搞混。
5. **不要在还没修正 metrics/target 之前就继续加复杂新模块。**
   比如双网络、额外 boundary-distance head、两侧联合生成、复杂 surrogate。现在最大的问题还不是“模型不够复杂”，而是“训练目标和评估目标不对齐”。
6. **不要把 `boundary_populated`、`sampleSigma`、`zDim`、`ganIter` 的不同家族结果混着下结论。**
   ZIP 里确实存在 query mode、training dose、prescreen 等设置不一致的分支；这些不能直接合并成一个主线结论。
7. **不要把“prescreen 没改善”解释成“GAN 彻底失败”。**
   那只会把问题误诊成 generator，而忽略掉最关键的 critic-proxy 错配。
