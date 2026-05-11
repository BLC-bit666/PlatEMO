我重新想了一轮。最终版判断是：

**PRBCCMO 应该收敛成一条主线：`B` 只定义“当前阶段的目标空间边界”，refinement 只负责把 pair 在目标空间里收紧，MLP 只在 boundary band 内恢复不可行解的选择压力。**
按这条主线，**`FrontGap` 不应再参与运行时决策**；它至多保留为离线诊断列。当前代码里的 `FrontGap` 本质上只是 pair midpoint 相对 `PopulationC` 当前最佳可行 scalar 的差值，不是 CPF 距离；而它又被放在主不可行排序键的第一位，直接压住了 `prob`，这和你现在看到的主不可行 `mean_inf_prob_gain` 基本为 0 是一致的。  

同时，未知约束问题里“最优常贴边界”这个观察本身是成立的，但文献也明确说它**常见而非普适**；因此，把 B 讲成“当前阶段 objective boundary archive”比讲成“CPF-relevant boundary archive”更稳，也更诚实。BE-CBO 正是基于“未知物理约束下边界最优常见、边界模型精度关键”来设计边界探索，但它也明确承认并非所有问题都这样。([arXiv][1])

## 1. 关于 FrontGap：我的最终结论是“删出主逻辑，只留可选诊断”

你的质疑是对的，而且比我上一次的建议更接近 PRBCCMO 应该讲清楚的创新点。

你现在并不知道 CPF/主前沿在哪里；当前代码里的 `FrontGap` 也并没有真的刻画 CPF，它只是“pair midpoint 比当前 `PopulationC` 在该 sector 里最好的 feasible scalar 落后多少”。这会把理论叙事带偏成“B 追当前 feasible front”，而不是“B 拟合当前阶段边界”。

更重要的是，**你其实已经有“靠近原点/当前前沿方向”的 objective pressure 了**：当前 `BuildTargetBoundaryArchive` 里本来就有 per-sector 的 objective shell 选择，先把每个 sector 中 scalar 更靠前的 feasible / infeasible 候选留下，再去配 pair。也就是说，目标推进压力本来就该由 objective shell 来承担，而不是由一个名叫 `FrontGap` 的“伪 CPF gap”再额外压一遍。把两者都留着，反而会重复、冲突，而且会继续压住 MLP。

所以我的最终建议不是“删掉 FrontGap 后只剩 PairGap”，而是：

**把 `FrontGap` 换成更老实的 `MidScalar`，并且只用于 B 内部 tie-break；不要再把它带进主种群不可行排序。**

B 的运行时质量键改成：

`Q_B(pair) = (PairGap, MidScalar)`

其中：

* `PairGap = || f_F^N - f_I^N ||_2`，表示 pair 在目标空间里贴边的紧密程度。
* `MidScalar = scalar(0.5*(f_F^N + f_I^N))`，表示这条 pair 当前在目标推进方向上有多靠前。

这样就回答了你问的“字典序还是综合考虑”：

**不用加权综合。**
我建议用 **staged gating + lexicographic**：

先用现有的 objective shell 机制把每个 sector 中“更靠前”的 feasible / infeasible 候选筛出来；
再在 shell 内按 `Q_B = (PairGap, MidScalar)` 做字典序。

这比加权综合更稳定，也更符合“统一、减法、收敛”的要求。

## 2. 关于 `betaB / Kbis / rhoRef`：最终定为 `betaB=3, Kbis=1, rhoRef=0.10`

这里我给出明确决策，不留模糊区间。

### `betaB = 3`

你建议 3，我认同。原因不是拍脑袋，而是它正好是当前版本最合适的折中。

你现在正式实验里，`N=100`、`betaB=4` 时，B 预算是 400 pair；但实验汇总显示，每代平均 `replaced pair` 约为 DAS 34.23、LIR 28.02，而 `contracted pair` 只有 DAS 8.34、LIR 6.74，说明当前系统主要由“全局重组”驱动，而不是“同 pair 深挖”驱动。 

把 `betaB` 从 4 改到 3 后：

* B 不再稀释到 400 对那么松散；
* 在保持当前 `etaB` 配置不变时，全局 reshape 的有效预算规模会更接近你现在实际发生的 28~34 对/代，而不会像 `betaB=4` 那么宽，也不会像 `betaB=2` 那么紧。

所以 3 是最合适的收敛点。

### `Kbis = 1`

这一点我现在比上一次更确定。

按你当前代码，`rhoRef=0.10`、`N=100` 时，helper refinement 每代总预算大约就是 10 次评估；`Kbis=3` 时，实际只能覆盖约 3 个 pair；`Kbis=2` 也只能覆盖 5 个 pair。B 预算却是几百对，覆盖率明显不够。实验也已经说明这一点：全局重组比同 pair 的连续二分强得多。

所以这里的主矛盾是 **覆盖率**，不是 **单对挖掘深度**。
尤其当接受准则改成 objective-space first 以后，一次 midpoint 探测就足够判断这条 pair 是否值得收紧；没必要在同一代上连续挖同一对。

因此最终选择是：

* `Kbis = 1`
* `rhoRef = 0.10` 先不加

### `rhoRef = 0.10`

先不加大。因为 helper population 还承担探索作用；如果第一轮就把 `rhoRef` 提高，会直接挤占 `OffspringU` 的自由探索预算。先把 `Kbis=1`、objective-space acceptance、候选池和 MLP 入口都修正，再看 contracted/replaced 比例是否仍明显偏低。

## 3. 关于“二分目标不等于目标空间贴边”：接受准则必须改成 objective-space first

这一点你已经认同，我直接给最终实现建议：

**midpoint 保留；接受准则重写。**

也就是说，我不建议第一轮再搞更复杂的 τ-search、偏置线搜或多段采样。先保持最简单的 midpoint：

`x_ref = 0.5 * (x_f + x_i)`

原因很简单：
你真正缺的不是“更花的线段采样器”，而是“采到点之后，什么样的 pair 才允许留在 B”。

所以 `ContractBoundaryPairsByRefinedSamples` / `IsBracketTighter` 这条逻辑应改成：

* 若 `x_ref` feasible，则替换 feasible endpoint；
* 若 `x_ref` infeasible，则替换 infeasible endpoint；
* 只有当新 pair 的 `Q_B(new) = (PairGap_new, MidScalar_new)` 按字典序优于旧 pair，才接受；
* decision-space gap 只保留成最后的 tie-break 或 diagnostics，不再作为主判断。

这样你就不再需要讲“decision bracket 变紧，所以 objective boundary 也会贴近”。
你可以更硬地讲：**我在决策空间上线段取样，但我只按目标空间贴边质量保留 pair。**

这正好把你的动机和实现对齐了。 

## 4. 关于 B 候选池：`PopulationC` 和 `PopulationU` 都拿掉，但 `B` 不能拿掉

这一点我的最终判断是：

**把 `PopulationC` 和 `PopulationU` 从 B candidate pool 里删掉；保留 `B`、`OffspringC`、`OffspringU`。**

也就是把当前

`[B, PopulationC, PopulationU, OffspringC, OffspringU]`

改成

`[B, OffspringC, OffspringU]`。

原因有三点。

第一，**B 本身已经是记忆体**。
如果一条边界 pair 真有价值，它应该留在 B；没必要再靠 parent population 重复灌一次。

第二，**真正的新信息来自子代，不来自父代**。
父代更多是陈旧状态。你现在看到的大量远离真实边界的橙点，本来就有很大一部分来自无约束侧，而且不少是历史候选被 B 留下来的结果。DAS 最终 B 不可行端点约 63.91%、LIR 约 55.33% 来自无约束侧，这已经说明候选池输入过宽。

第三，这样改之后，helper 的作用会更像 CCMO 所说的“弱合作”：
helper 通过 **offspring** 提供有用提案，而不是通过 parent pool 强输入地污染主机制。CCMO 本来就强调辅助任务的价值在于共享搜索信息，而不是过强耦合。

所以这里不是“只留子代”，而是更准确地说：

**`B + 子代`，去掉父代。**

## 5. 关于 MLP：FrontGap 要完全退出 MLP 通路；MLP 角色现在可以收敛了

这里是你最核心的问题，我给出更明确的结论。

### 5.1 先说根因：当前 MLP 失效，主要不是模型太弱，而是链路没对齐

现在 MLP 效果差，根因有三条：

第一，当前主不可行排序键是
`[frontGap, -prob, supportDistObjToB, idx]`，
概率排第二。

第二，主 carry 通道通常只有 1 个位置；就算 carry 有轻微正增益，整体影响也很小。实验上 DAS/LIR 的主不可行 `mean_inf_prob_gain` 基本都接近 0。

第三，更关键的是，你当前 non-B train buffer 实际上被代码写成了 0。
按当前默认逻辑：

`MaxTrain = max(2*NBPair, 4*N)`
传入 `UpdateTrainingBufferT` 的是 `max(0, MaxTrain - 2*NBPair)`

在 `betaB=4, N=100` 时：

* `NBPair = 400`
* `2*NBPair = 800`
* `4*N = 400`
* 所以 `MaxTrain = 800`
* non-B budget = `800 - 800 = 0`

也就是说，recent refinement 和 boundary-band 样本会被全部裁掉。MLP 实际上几乎只在 B endpoints 上训练，却被拿去影响主种群不可行排序，当然很难有效。

### 5.2 所以 MLP 的最终角色应该这样定义

**MLP = B 条件下的 local boundary prior**

不是全局 classifier。
不是泛用 ranking model。
也不是 surrogate objective。

训练目标仍然是 feasibility classification；这和 NA-EMT 的“用 MLP 估计 infeasible solution value，再把 predicted value 真正写进主任务不可行解处理”是一致的。

但它的**使用场景**必须收缩到：

**只在 near-B infeasible shortlist 内排序。**

也就是说，MLP 不负责在整个 infeasible pool 上“通吃”；它只负责在“已经很靠近当前边界”的那一小撮不可行解里，判断谁更值得进入主种群。

### 5.3 对应的改法

1. **修 T-buffer**

   * `MaxTrain = 2*NBPair + 2*N`
   * refinement buffer 保留最新 `N` 条
   * boundary-band buffer 保留最新 `N` 条
   * `BuildTrainingBufferT` 拼接顺序改成 `[TrainBuffer ; B]`，不要再写成 `[B ; TrainBuffer]`

   这一步很关键，因为你现在的 `BalanceTrainingDataset` 是按出现顺序取 `Pos(1:Count)` 和 `Neg(1:Count)`；如果把 B 放前面，非 B 的 boundary-local 样本还是会被吃不进去。

2. **`Gstart` 改成 0**

   * 不再人为拖到 150 代以后
   * 真实能不能训练，交给 `pair_count` 和样本量门槛去控制

   这样更统一，也不增加分支。

3. **删除 carry/fill 双通道**

   * 统一成一个入口
   * 当 B ready 且 model ready 时，固定预留 `Kinf = ceil(0.05*N)` 个位置给 near-B infeasible
   * `N=100` 时就是 5 个位置

4. **先 boundary-band，再 MLP**

   * 在所有 infeasible 里，先按 `supportDistObjToB` 取最近的 `L = 3*Kinf`
   * 再在这 `L` 个里按 `[-prob, supportDistObjToB]` 排序

这样之后，`prob` 才真正成为主裁决器；而 `supportDist` 负责保证“这些点确实就在当前边界带里”。

### 5.4 这一轮不建议先上 ensemble

BE-CBO 的确说明了：在复杂边界和高维问题里，更强的 boundary classifier，尤其 deep ensembles，相比 GP 或弱模型会更稳。([arXiv][1])

但我不建议你第一轮就把单 MLP 换成 ensemble。
你现在更大的问题不是 model capacity，而是 **B 的定义、T 的来源、selection 的入口** 没对齐。先把这三点修正，再看单 MLP 还有没有必要升级。

## 6. 关于指标：对外只保留一个——`ΔIGD_MLP`

你说得对，不需要再分三层，也不需要再拿 accuracy 做主结果。

我建议对外只保留一个指标：

**same-seed paired `ΔIGD_MLP = IGD(MLP-off) - IGD(PRBCCMO)`**

解释很直接：

* `MLP-off`：把模型入口完全关掉，其余代码、参数、种子都不变；
* `ΔIGD_MLP > 0`：说明引入 MLP 以后最终结果更好；
* 这比 `mlp_train_acc` 更直接地回答：
  **B → T → MLP → 不可行选择 → 性能提升** 这条链条到底有没有闭合。

`mlp_train_acc` 不再作为论文/汇报指标。
它最多只保留成一个本地 debug 值，甚至可以不再写 CSV。

NA-EMT 里的 accuracy 是用来触发 model update 的；你现在已经是 fixed-period retrain，所以完全没必要再把 accuracy 当贡献证据。

---

## 在当前 PRBCCMO 基础上的完整修改方式

下面给你一个可直接落地的版本。

### A. 参数层

在 `PRBCCMO.m / PRBCCMO_t.m` 里，把默认参数改成：

* `betaB = 3`
* `Gstart = 0`

其余先不改。

在 `runObjectiveBoundaryCore` 里改成：

* `rhoRef = 0.10`
* `Kbis = 1`
* `NBPair = ceil(3*N)`
* `MaxTrain = 2*NBPair + 2*N`



### B. B 档案层

1. `InitBoundaryArchiveInfo`

   * 把 `FrontGap` 改成 `MidScalar`

2. `BuildBoundaryCandidatePool`

   * 改成 `A = [B, OffspringC, OffspringU]`

3. `BuildTargetBoundaryArchive`

   * 保留现有 sector 化与 `SelectBoundaryShellByScalar`
   * 不再计算/使用 `BestFeasibleScalar` 和 `FrontGap` 作为运行时键

4. `BuildSectorCandidatePairsObjective`

   * shell 内仍然构造 feasible/infeasible 配对
   * pair 质量键改成 `Q_B = (PairGap, MidScalar)`
   * 先按 `PairGap`，再按 `MidScalar`

5. `ComputeSectorPriority`

   * 改成 `log(1+Count)/(eps + bestPairGap)`

6. `AllocateSectorQuota`

   * 删除当前依赖 `FeRatio` 的 `CoverCount` 逻辑
   * 统一成：

     * 先给每个 available sector 1 个 quota
     * 再按 `Priority/(Quota+1)` 分配剩余 quota

7. `BlendBoundaryArchive`

   * add / drop / trim 全部只看 `PairGap`，年龄做 tie-break
   * `FrontGap` 不再参与

### C. refinement 层

1. `SelectBoundaryRefinementPairs`

   * 从当前 `[-DecGap, -Age, FrontGap]`
   * 改成 `[-PairGap, -Age]`

2. `InjectBoundaryRefinementIntoHelperOffspring`

   * midpoint 逻辑保持不变

3. `IsBracketTighter` / `CompareBoundaryPairObjectiveKeys` / `EvaluateBoundaryPairKeyFromNorm`

   * 目标键改成 `[PairGap, MidScalar]`
   * 接受条件改成 `Q_B(new) <lex Q_B(old)`
   * decision gap 只做最后 tie-break，不再主导

### D. MLP / 选择层

1. `BuildBoundaryMetaFromB`

   * 删除 `frontGap`
   * 只保留 `prob`、`supportDistObjToB`、`sector`

2. `SortBySelectionKey`

   * 改成 `[-prob, supportDistObjToB, idx]`

3. `EnvironmentalSelectionC_ObjectBoundary`

   * 删除 carry/fill 双通道
   * 统一预留 `Kinf = ceil(0.05*N)` 个 near-B infeasible 位置
   * 其余位置仍然优先给 feasible population

4. `SelectInfeasibleByBoundaryMeta`

   * 先按 `supportDistObjToB` 取最近的 `L = 3*Kinf`
   * 再按 `[-prob, supportDistObjToB]` 排序选 `Kinf`

### E. T-buffer 层

1. `UpdateTrainingBufferT`

   * non-B 不再共用一个 0 容量
   * 改成两个独立上限：

     * refinement: 最新 `N`
     * boundary-band: 最新 `N`

2. `TrimTrainingBufferNonB`

   * 分 source 分别裁剪，不再 source 2 吃完所有预算

3. `BuildTrainingBufferT`

   * 拼接顺序改成 `[TrainBuffer ; B]`

### F. 诊断与测试层

1. 把 `b_mean_front_gap / b_p90_front_gap` 系列列名整体改成 `b_mean_mid_scalar / ...`

   * 如果你嫌动脚本太多，至少把它们从论文叙事里删掉，不再叫 FrontGap

2. `mlp_train_acc`

   * 不再作为对外指标
   * 可选保留为本地 debug，或直接不再写 summary

3. 新增一个 paired summary

   * `delta_igd_mlp`

4. 同步更新：

   * `test_PRBCCMO_semantics.m`
   * `test_PRBCCMO_t_metrics.m`
   * `summarize_PRBCCMO_t_run.m`
   * `summarize_PRBCCMO_t_data.m`

---

## 最后一句收束

这版 PRBCCMO 的最终形态，我建议你这样讲：

**helper offspring 提供边界提案；B 用 objective shell + 最小 objective pair gap 定义当前阶段边界；midpoint refinement 只做 objective tightening；MLP 只在 boundary band 内恢复不可行解选择压力。**

这比“FrontGap 追 CPF + 二分类 accuracy + 辅助侧大量灌入 B”更统一，也更能把你的核心创新点讲硬。
DRMCMO 说明了在二元约束下 CV/ε 的指导会明显变弱；CCMO 说明 helper 应该弱合作；NA-EMT 说明 predicted value 必须真正进入主任务选择；BE-CBO 说明边界模型精度对未知约束问题确实关键。你现在最该做的，不是再加机制，而是把这四件事收敛成一条机制链。   ([arXiv][1])

[1]: https://arxiv.org/abs/2402.07692 "https://arxiv.org/abs/2402.07692"
