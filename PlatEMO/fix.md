你这次的纠正是对的。

按你的创新链路，**B 的最终职责不应该继续和 MLP 并列做不可行解排序**；B 应该退回到“在目标空间中构造边界相关训练分布”的角色，**MLP 才是 I-I selection 的主判器**。这和 NA-EMT 里“用预测值识别 valuable infeasible solutions”的思路是一致的；也和 BE-CBO 的核心观察一致：unknown constraints 下，最优往往贴在 feasible / infeasible 边界上，因此边界模型的质量本身就是决定优化效果的关键。DRMCMO 则从另一侧说明，在二元约束下，如果 feasible-only pressure 过强，搜索会被卡住，必须恢复边界附近 infeasible 的选择价值。EADMM 的启发是：互补模块应承担不同职责，而不是长期把两个信号并列混用。([Springer][1])

但我要同时明确一件更关键的事：**现在这份 PRBCCMO 代码，还不能直接硬切成 pure MLP-only**。不是因为你的链路不对，而是因为**当前 MLP 还没有被训练成“recoverability 判别器”**；它更像一个 B-driven、near-boundary 的弱 feasibility-side classifier。源码里，`TrainBuffer` 目前只接收 `accepted_refined`，而 `rejected_refined` 只在诊断里算 `prob_rejected_refinement`，并没有进入主训练；同时当前 I-I comparator 仍然是 `BRank + ProbRank`，也就是 B 和 MLP 还在并列决策，而不是 “B 训练，MLP 主判”。此外，训练节奏还是 `Tretrain=20`，warm update 只有 `40` epoch、`3e-4` 学习率。   

这也正好解释了你现在看到的实验现象。附件汇总里，`current feasible - infeasible` 的平均概率差约 `+0.0949`，`B-core feasible - infeasible` 约 `+0.1287`，但你真正想证明的 `accepted refinement - rejected refinement` 只有 `+0.0089`；同时 `mlp_effective_win_rate` 只有 `0.00081`，`selected probability - pool probability` 只有 `0.00037`。这说明问题不是“MLP 完全没被调用”，而是**它学到的不是你要的 recoverability 主信号，而且当前排序器也没有把那一点点概率差真正放大成选择压力**。最终 IGD 上，当前 paired 结果也仍是 `6/2/74/10`（win/loss/tie/missing），主异常点仍是 `DASCMOP4_BC run 3 = -0.0627365`。所以，你要改的不是“再给 B 一点权重”或“加保险丝”，而是**让 MLP 真正成为主判器之前，先把它训练成主判器**。 

### 重新分析后的结论

我现在给你的结论是：

**第一，方向上，应该改成 `MLP-first`，而不是 `BRank + ProbRank`。**
你说得对，若论文故事是“B 为了训练 MLP，MLP 恢复 infeasible 的选择压力”，那最终比较器就不该再让 B 和 MLP 并列。正确形式应当是：

* **MLP 主判**
* **B 只在 MLP 无法区分时裁决**

而且这件事可以做得很统一，不需要再加 branch 或 safety fuse。

**第二，当前最主要的根因，不是 MLP 权重太弱，而是监督语义不对。**
现在的 `TrainBuffer` 只吃 `accepted_refined`，而 `rejected_refined` 只是事后诊断，所以模型没有被直接训练成 “accepted vs rejected 的 recoverability 判别器”。这正是为什么 accepted / rejected gap 很弱。 

**第三，当前训练参数本身不是“完全不够”，但训练制度不够。**
从代码看，你已经不是旧版单层 tanh + 手写 GD；现在是比较正常的 Adam + BCE + early stopping 路线。这个 backbone 作为第一阶段够用。真正不够的是：**训练触发太慢、warm update 太轻、训练数据没有围绕 recoverability 重排**。NA-EMT 也是把“predicted value”作为识别 valuable infeasible solutions 的核心，并强调模型需要随着进化过程更新；你的版本现在的问题，不是一定要更深，而是要让更新节奏和数据语义对准 recoverability。([Springer][1])

---

## 我建议的最小修改方案

下面这套方案完全按你的链路来，而且遵守“统一、减法、收敛”。

### 1) 把训练目标改成“recoverability-first”，而不是“accepted-only + feasibility”

这是最关键的一刀。

当前的核心缺口是：

```matlab
TrainBuffer = UpdateTrainingBufferT(TrainBuffer,ContractionDiag.accepted_refined,...)
```

这会让 `TrainBuffer` 只看到 accepted。

我建议改成：

```matlab
TrainBuffer = UpdateTrainingBufferT( ...
    TrainBuffer, ...
    ContractionDiag.accepted_refined, ...
    ContractionDiag.rejected_refined, ...
    Generation,MaxRefinementBufPerClass);
```

并且把 `TrainBuffer.Label` 的语义改掉：

* `accepted_refined -> 1`
* `rejected_refined -> 0`

也就是说，**进入 refinement buffer 的标签，不再表示 feasibility，而表示 recoverability outcome**。

这一步非常重要。因为 accepted / rejected 本身才是你创新点最想证明的语义。
B 端点和当前种群样本以后仍然可以保留，但它们只做**anchor /补样**，不再主导 MLP 的训练语义。

我建议把训练集重构成：

[
\mathcal D_t=\mathcal D_{\text{refine}} \cup \mathcal D_{\text{anchor}}
]

其中：

* `D_refine`：最近若干代的 `accepted_refined` 和 `rejected_refined`，这是主训练集；
* `D_anchor`：只有在 `D_refine` 数量不足、或某一类缺失时，才补进 B endpoints 和 near-B 当前种群样本。

这样 B 仍然是训练引擎，但**它通过 boundary refinement 产生 recoverability supervision**，而不是再让模型主要学 “feasible / infeasible”。

这比当前做法更符合你的叙事，也比 NA-EMT 的 generic infeasible value 更贴你的创新点：**你的预测值不是一般的 feasibility value，而是 B-conditioned recoverability value**。([Springer][1])

### 2) 让排序器变成真正的 “MLP 主判，B 只做平手裁决”

这一刀要改 `ComputeInfeasibleUtilityMeta` 及其后续排序。

当前代码是：

```matlab
BRank = NormalizeLexRankAsc([SupportDist,ScalarObj]);
ProbRank = NormalizeTieAwareRankDesc(Prob);
Utility = BRank + ProbRank;
```

这不符合你的链路，因为 B 还在和 MLP 并列主判。

我建议改成：

```matlab
BRank    = NormalizeLexRankAsc([SupportDist,ScalarObj]);
ProbRank = NormalizeTieAwareRankDesc(Prob, tauProb);

Utility = ProbRank;   % MLP 主判
```

然后排序 key 改成：

```matlab
Key = [-ProbRank, SupportDist, ScalarObj, idx];
```

意思很简单：

* 先按 `ProbRank` 排；
* 只有当 `ProbRank` 相同，也就是 MLP 认为“区分不开”时，
  才用 `SupportDist -> ScalarObj` 去裁决。

注意，这里我不是在加新 branch。
关键在 `NormalizeTieAwareRankDesc(Prob, tauProb)` 里的 `tauProb`：

* 如果两个概率差小于 `tauProb`，就视为同一 rank；
* 一旦同 rank，自然就落到后面的 B tie-break 上。

也就是说，**“只有当 MLP 无法区分时才采用 B”**，可以用一个统一的 tie-aware rank 直接实现，不需要 safety fuse，不需要额外 if-else 分支。

这正是你想要的形式。

我建议第一轮先用：

```matlab
tauProb = 0.02;
```

原因很直接：当前 accepted / rejected 的平均 gap 只有 `0.0089`，selected gain 只有 `0.00037`，所以目前 0.5 附近那点细小差值大概率只是噪声，不应该被当作“MLP 真在主判”。把 `tauProb` 提到 `0.02`，可以先把噪声压成 tie；等你把训练语义修正后，如果 recoverability gap 拉大，它自然会穿过这个阈值，真正接管 I-I 比较。

这一步做完后，你的 selector 逻辑就变成了：

> **MLP 主判；B 仅在 MLP 不可分时裁决。**

这和你的算法链路是严格一致的。

### 3) 训练参数：结构先不动，训练制度要动

你问得很准：`epoch`、`lr` 够不够？

我的判断是：

* **网络结构先够用**
* **训练制度不够**

当前结构已经不是瓶颈。第一阶段我不建议先加更深 MLP、更多层、更多 feature。
先保留：

* `hidden = 64`
* 两层 ReLU MLP
* `cold epoch = 200`
* `cold lr = 1e-3`
* BCE
* validation early stopping

真正该改的是这三处：

**第一，`Tretrain` 从 `20` 改到 `10`。**
当前模型更新太慢。unknown/binary constraints 下边界相关分布是移动的，20 代才训一次，对 recoverability 信号偏慢。NA-EMT 也是强调随着进化推进去更新模型。 ([Springer][1])

**第二，`WarmEpoch` 从 `40` 改到 `80`。**
原因不是“让模型更猛”，而是当前 warm update 太轻，难以追上训练分布变化。你又已经有 best-validation / early stopping 机制，所以把 warm epoch 翻倍的风险可控。

**第三，refinement buffer 改成按类保留，而不是总量短截。**
当前 `MaxRefinementBuf = 0.6N` 是基于“只存 accepted_refined”设计的。你现在如果要同时存 accepted / rejected，我建议改成：

```matlab
MaxRefinementBufPerClass = round(0.5 * Problem.N);
```

对 `N=100`，就是每类最多 50，总共最多 100。
这既不会变成历史堆料，又能保证 recoverability 正负样本都留得住。

另外，训练触发门槛也要稍微提前一点。我建议：

```matlab
MinTrain = max(64, round(0.8 * Problem.N));
```

而不是当前的 `max(96, 1.0N)`。
因为你现在主训练集会更偏 refinement outcome，前期样本量会变小；如果门槛还卡得太高，MLP 会很晚才开始有效介入。

一句话总结参数：

* **hidden 不动**
* **cold 参数基本不动**
* **Tretrain 20 -> 10**
* **WarmEpoch 40 -> 80**
* **WarmLR 保持 3e-4**
* **MinTrain 下调到 64~80**
* **buffer 改成 per-class 保存**

这符合“减法和收敛”，不是盲目调参。

### 4) 当前种群样本继续保留，但只能当补样，不能再主导 MLP 语义

现在 `BuildBoundaryDrivenDataset` 会把：

* B 端点
* TrainBuffer
* current feasible / infeasible supplement

一起拼起来。

这个框架不必推翻，但主次必须改：

* **主：refinement outcome**
* **次：B endpoints**
* **末：current population supplement**

而且 current supplement 只在缺样时启用。
否则你会再次把 MLP 拉回“near-B 的 feasibility classifier”。

所以这里的实现建议是：

```matlab
Dataset = CollectRefinementOutcomeSamples(...);   % priority 1
if insufficient
    Dataset = CollectBoundaryEndpointAnchors(...); % priority 2
end
if still insufficient
    Dataset = CollectCurrentPopulationAnchors(...); % priority 3
end
Dataset = FinalizeBoundaryDrivenDataset(...);
```

也就是说，**训练集的默认中心要从 B-core + current-pop，改成 refinement outcomes**。
这才是让 MLP 真正对齐 recoverability 的关键。

---

## 对你三个问题的直接回答

### 1. “我认为应该只用 MLP 判断，或者只有当 MLP 无法区分时才采用 B”

结论：

**方向上，我支持你的判断。**

但在当前版本上，我不支持“一步到位 pure MLP-only”；
我支持的是：

> **MLP-first + tie-aware B fallback**

原因不是保守，而是这是唯一同时满足下面三件事的方案：

* 符合你的论文叙事；
* 不推翻现有框架；
* 不在当前弱信号下放大噪声。

而且它可以通过一个统一 comparator 完成，不需要额外分支。

### 2. “MLP 继续训练的参数够吗？”

结论：

**主干够，制度不够。**

* `64-32 + Adam + BCE + early stopping`：先够；
* `Tretrain=20`：偏慢；
* `WarmEpoch=40`：偏轻；
* `accepted-only buffer`：根本性不够；
* `MinTrain≈100`：对 recoverability-first 版本偏高。

所以不要先加深网络。
先把：

* 标签语义
* retrain cadence
* warm budget
* buffer policy

改对。

### 3. “不要加 safety fuse”

我同意。

`DASCMOP4` 这类负异常，不应该靠保险丝去遮蔽。
正确做法是：

* 先把 MLP 训练成 recoverability 判别器；
* 再把 selector 改成 MLP-first；
* 再用 `tauProb` 让“不可分”自然回退到 B。

这已经足够机制化，不需要再套一层外部 fuse。

---

## 下一轮实验应该怎么做

我建议按下面顺序，不做大网格调参。

### 实验 1：只修训练语义，不改选择器

目的：确认 accepted / rejected gap 是否被真正拉开。

你要看的第一指标是：

[
\Delta_{\text{refine}}
======================

## \text{prob_accepted_refinement}

\text{prob_rejected_refinement}
]

当前只有 `+0.0089`。下一轮如果这项还是接近 0，就说明 MLP 还没被训练成 recoverability 模型。

### 实验 2：在实验 1 基础上，改成 MLP-first comparator

目的：确认 MLP 是否真的接管 I-I selection。

这一轮不要再看当前的 `mlp_effective_win_rate` 作为主指标，因为那是给 `BRank + ProbRank` 设计的。
你应该新增两个指标：

* `mlp_primary_resolution_rate`
  两个 infeasible 比较时，`ProbRank` 是否已经分出不同 rank；
* `b_fallback_rate`
  `ProbRank` 相同、回退给 B 的比例。

如果 MLP 真接管了，这个 `mlp_primary_resolution_rate` 就不能再接近 0。
工程上，我会把：

* `> 0.05` 看作开始有主判能力；
* `> 0.10` 看作比较像真的在主导。

### 实验 3：看选择压力是否真的变化

这里继续看你已有的：

[
\text{selected_prob_gain}
=========================

## \text{inf_selected_mean_prob}

\text{inf_pool_mean_prob}
]

当前只有 `0.00037`。
下一轮至少要上到 `10^{-3}` 量级，最好再高一截。否则很难传到最终 IGD。

### 最终结果：只报 win / loss / tie

这里按你的要求来。

主文只报：

* DASCMOP_BC：win / loss / tie
* LIRCMOP_BC：win / loss / tie

不要再把 mean delta 当主结论。
差值列只放附录。

---

## 我建议你在论文里这样改写 claim

当前版本最稳妥的写法应当是：

> **PRBCCMO 构建了一个由目标空间边界存档 B 驱动的数据生成机制，并利用 MLP 对边界相关 infeasible 解的可恢复性进行建模。当前实验验证了该机制的实现、边界相关训练数据的形成以及非随机内部概率信号；但优化性能收益仍具问题依赖性，尚不能声称该机制已在基准上稳定提升最终 IGD。**

如果你完成我上面的两刀之后，再写得积极一点，可以改成：

> **The revised PRBCCMO treats B as a boundary-data scaffold and makes the MLP the primary comparator for infeasible solutions, with B used only as a fallback when the MLP is non-discriminative.**

这比“B 和 MLP 共同排序”更符合你的原始创新点，也更容易被审稿人理解。

---

最后给你一句最核心的判断：

**你这条链路要闭合，不是让 B 更聪明，而是让 B 彻底退到“训练 MLP”的位置；然后把 MLP 真正训练成 recoverability 判别器，再让它成为 I-I selection 的一等裁判。**

如果你愿意，我下一条可以直接把这套方案展开成**逐函数的 MATLAB 级补丁清单**。

[1]: https://link.springer.com/chapter/10.1007/978-981-96-9805-9_11 "A Network-Assisted Evolutionary Multitask Framework for Multi-objective Optimization Problems with Unknown Constraints | Springer Nature Link"
