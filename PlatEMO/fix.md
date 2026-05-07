当前 PRBCCMO 仍然把 `Model` 传进 `GenerateBoundaryOffspring` 和 `UpdateBoundaryArchive`，训练源仍合并 `B + BoundaryEvidence + PopulationC + PopulationU`，不可行排序仍用 `boundaryGap/gapDec/gapObj/margin/objScore` 复合键，类注释仍写着 “Boundary-band MLP driven CCMO”；这些都说明它还混合着路线 1、2、3 的语义，需要一次性收束。

另外，现有 `kappa=10` 版本虽然扩大了 B 的覆盖，`B` 数量可到 166/130，但在难问题上已经出现明显厚边界带征兆：例如 LIRCMOP12_BC 的 `final_b_mean_dist_to_true_boundary = 0.328`，`p90 = 1.515`，`archive hit rate = 0.577`。这正是为什么重构后应让 `B` 变薄，把学习能力转交给 `D`。

## A. 直接保留，不动主逻辑骨架

这些部分继续保留：

* 双群体框架 `PopulationC / PopulationU`
* `GenerateDEOffspring`
* 参考向量扇区化
* `midpoint + one short bisection`
* 当前轻量 MLP 的一层结构、`tanh`、warm-start normalization rebase、temperature scaling
* trace / csv 指标框架

这些是你现有包里最值得保留的部分。

## B. 立即删除或降级

### 1. 让 MLP 退出边界采样和边界归档

把这两处全部改掉：

```matlab
GenerateBoundaryOffspring(..., W, Model, rho)
UpdateBoundaryArchive(..., W, Model, kappa)
```

改成：

```matlab
GenerateBoundaryOffspring(..., W, rho)
UpdateBoundaryArchive(..., W, kappaPair)
```

也就是说：

* `Model` 不再进入 `GenerateBoundaryOffspring`
* `Model` 不再进入 `UpdateBoundaryArchive`
* `margin` 不再参与边界事件筛选、archive 排序、archive admission

### 2. 删除点级厚 archive 语义

删除：

```matlab
kappa = 10
SideQuota = ceil(kappa/2)
feasible-side / infeasible-side independent point quota
```

`B` 改成 **pair-level archive**。

### 3. 删除全局训练源

删除训练源中的：

```matlab
PopulationC
PopulationU
```

训练源只能来自：

```matlab
PairEvents.finalPairs
revalidated B pairs
```

### 4. 删除复杂复合分数

删除或降级为 debug 字段：

```matlab
objScore
oppSupport
score
betweenScoreObj
shellDistDec
```

`objScore` 直接替换成更单一、可解释的 `frontGap`。

## C. 必须重写的函数

### 1. `BuildBoundaryPairsFromArrays`

重写目标：

* 先扇区化，再局部配对
* same/neighbor sector only
* mutual nearest cross-label
* fallback 仅在本地扇区
* 距离尺度改成确定性 median，不再 `randperm`

新增输出字段：

```matlab
PairId
PairGap
FrontGap
Sector
Time
MidDec
MidObj
```

### 2. `GenerateBoundaryOffspring`

重写目标：

* 不再接收 `Model`
* pair 排序改成：

```matlab
[FrontGap, PairGap, Time]
```

* pair 预算按扇区 round-robin
* 只做：

```matlab
midpoint contraction
+ one short bisection
```

* 返回：

```matlab
[BoundaryOff, PairEvents]
```

其中 `PairEvents` 至少包含：

```matlab
finalPairs
probes
pair metadata
```

### 3. `UpdateBoundaryArchive`

重写目标：

* 输入不再是点级 `BoundarySamples + TightEndpoints + RevalidatedB`
* 而是 pair-level：

```matlab
CandidatePairs = [E_t.finalPairs, RevalidatedOldPairs]
```

* 每扇区仅保留 `kappaPair=3` 对
* pair 排序键：

```matlab
[sourcePriority, frontGap, pairGap, -time]
```

如果你暂时不想彻底改 `B` 的外部接口，可以在内部按 pair 选好，再 `FlattenPairsToPopulation(B)`。

### 4. `BuildTrainingCandidateSet / UpdateTrainingBuffer / CollectSectorBoundaryBand`

这三个模块建议整体替换成显式 `PairBuffer` 逻辑：

```matlab
D = UpdatePairBuffer(D, PairEvents, B, FE)
Dataset = BuildDatasetFromPairBuffer(D)
```

`D` 推荐按 pair 存，而不是按散点存。

### 5. `BuildBoundaryMeta`

压缩成最小版本，只保留：

```matlab
sector
feasible
prob
margin
localPairExists
pairGap
sectorCovered
supportDistDec
frontGap
```

说明：

* `localPairExists / pairGap`：只服务于 evidence / archive
* `sectorCovered / supportDistDec / frontGap`：服务于 infeasible selection
* `margin`：只服务于 model-ready 的 infeasible selection

### 6. `SelectInfeasibleByBoundaryMeta`

这是重构后 MLP 的唯一主战场。

当前思路应该整体替换为：

```matlab
ApplicableI = infeasible & sectorCovered
```

然后：

```matlab
if ModelReady
    key = [margin, supportDistDec, frontGap, idx]
else
    key = [supportDistDec, frontGap, idx]
end
```

保持扇区 round-robin，不要再用：

```matlab
[boundaryGap, gapDec, gapObj, margin, objScore]
```

### 7. `ShouldRetrainBoundaryModel`

简化成：

```matlab
FirstTrain:
    pairCount >= 10
    mixedSectorCount >= 3
    both labels exist

Retrain:
    every 10 generations
    and newPairRatio >= 0.2
```

不要把复杂 validation/Brier 触发留在主逻辑里；它可以保留为 trace 指标，但不做默认训练开关。

## D. 建议新增的最小函数

为了让语义彻底清晰，建议新增四个最小函数，而不是继续在旧函数里打补丁：

```matlab
BuildLocalWitnessPairs(...)
EvaluatePairEvents(...)
UpdatePairMemory(...)
FindModelApplicableInfeasible(...)
```

其中：

### `FindModelApplicableInfeasible(...)`

核心逻辑只做两件事：

```matlab
1) 判断 sectorCovered
2) 计算 supportDistDec
```

不再复用 `BoundaryEligible(...)`，因为：

* `BoundaryEligible` 是给 **evidence / archive**
* `ModelApplicableInfeasible` 是给 **selection**

这两个语义本来就不该是同一个 gate。

## E. 参数建议

建议默认参数改为：

```text
rho       = 0.12
kappaPair = 3
tauE      = 1.25   % loose evidence gate
tauB      = 0.55   % strict archive gate
hidden    = 20     % 先保留当前值
epoch     = 20
lr        = 0.01
```

其中最重要的改变不是数值，而是：

```text
kappa 的单位从 point per sector 改成 pair per sector
```

## F. 指标体系如何跟着改

你现在的 trace 框架很好，不要废掉，只做字段替换和补充。

### 保留

保留这些：

```text
archive_hit_rate_eps
final_b_mean_dist_to_true_boundary
final_b_p90_dist_to_true_boundary
final_b_margin_true_boundary_corr
final_inf_margin_gain
```

### 改名

把：

```text
objScore -> frontGap
```

### 新增

新增两个更贴路线 2 的指标：

```text
final_inf_supportDist_gain
final_inf_frontGap_gain
```

这样你能直接验证：

* 被选中的 infeasible 是否更靠近 supported boundary
* 被选中的 infeasible 是否更接近当前 feasible frontier
* margin 是否真的在恢复选择压力，而不是单纯做分类

## G. 一张最关键的“改前 / 改后”对照

```text
改前：
B = 点级厚边界存档
MLP = 同时沾采样、归档、选择
Train = B + BoundaryEvidence + PopulationC + PopulationU
Selection = boundaryGap/gapDec/gapObj/margin/objScore 混合排序

改后：
B = pair-level thin boundary memory
MLP = 只做不可行解选择压力恢复
Train = PairEvents.finalPairs + revalidated B pairs
Selection = margin/supportDistDec/frontGap
```

## H. 最终的代码改造优先级

### 第一优先级：关闭路线 3 残留

先改这三处：

1. `GenerateBoundaryOffspring` 去掉 `Model`
2. `UpdateBoundaryArchive` 去掉 `Model`
3. `SelectInfeasibleByBoundaryMeta` 改成 `[margin, supportDistDec, frontGap]`

### 第二优先级：建立 `PairBuffer D`

把训练源从全局种群切回 pair-supported boundary evidence。

### 第三优先级：`B` 改成 pair-level thin memory

把 `kappa=10 + SideQuota` 改成 `kappaPair=3`。

### 第四优先级：trace 字段与语义测试更新

语义测试应该改成检查：

* `Boundary offspring must not use Model`
* `Boundary archive must be pair-level`
* `Training dataset must come from PairBuffer`
* `Infeasible selection must use margin as first key after applicability gate`

---

**最终结论**

你这次重构不该再写成：

```text
Boundary-band MLP driven CCMO
```

而应写成：

```text
Pair-supported boundary recoverability CCMO
```

核心只有三件事：

```text
1) pair 造证据
2) thin B 存证据
3) MLP 在 covered sectors 内恢复不可行解选择压力
```

这一版的关键词只有三个：**pair、thin memory、pressure recovery**。