你现在这版实现里，真正需要改的**不是先换更复杂的 MLP**，而是把 `p≈0.5` 的解从“一个排序信号”升级成“一个有明确后续动作的边界触发器”。当前代码里，MLP 是**每代 from-scratch 重训**，`LocalBoundaryPerturbation` 本质上是围绕 seed 的各向同性扰动，`p≈0.5` 主要被用来给 infeasible 候选排序和补充种群，而不是显式地执行“找到更好的可行域”或“确认局部无可行域”的闭环动作。原 NA-EMT 则是把 MLP 用于 infeasible value perception、并按准确率阈值触发重训；你当前实现已经走向了 boundary-oriented 版本，但闭环还没完成。

## 1）每代训练一个轻量 MLP 是否有必要？是否应该隔几代训练一次？

我的结论是：**没必要每代都从头训练；更合理的是“批式触发 + warm-start 更新”**。

原因有两个。第一，你当前代码是在每一代 `while` 循环里都调用 `TrainBoundaryMLP`，而该函数每次都会重新随机初始化 `W1/W2`，所以这不是增量更新，而是**每代重训一个新模型**；这会让 `p≈0.5` 的判定带有不必要的抖动。 第二，你这里真正依赖的是“`p≈0.5` 是否有语义”，而这首先是**校准问题**。神经网络概率输出往往并不天然校准，温度缩放是一个很常用、而且往往很有效的后处理办法；主动学习也并不要求每拿到一批新标签就立刻重训，batch-mode 的做法本来就很常见。([Proceedings of Machine Learning Research][1])

我建议你改成下面这个策略：

* **初始阶段**：用初始化样本做一次完整训练。
* **常规阶段**：每 **5 代**更新一次模型；如果你担心边界变化太快，可以改成 **3 代**。
* **触发式提前更新**：满足任一条件就提前更新

  1. 新增真实边界样本数达到 `max(20, 0.1*TrainMax)`
  2. 最近一段时间内，`p∈[0.4,0.6]` 的样本，其真实可行率明显偏离 0.5
  3. 校准误差变差，例如 reliability diagram 明显偏斜，或 Brier/ECE 恶化。校准图本来就是检查概率预测是否可靠的标准工具。([scikit-learn][2])

实现上也不要再每次随机重启。更好的做法是：

* **平时 warm-start 小步更新**：沿用上一代权重，只训练 5–10 个 epoch；
* **偶尔 full restart**：比如每 20–30 代完整重训一次，避免长期漂移。

所以这题的直接答案是：

**不建议每代 from-scratch 训练。推荐“每 3–5 代一次 + 事件触发 + warm-start”。**

---

## 2）如何把 `p≈0.5` 的局部搜索做强，真正找到更好的可行域，或者确认局部无可行域？

这里我建议你把当前的 `LocalBoundaryPerturbation` 换成一个**标签感知的边界局部搜索**，我给它起个直观名字：

**LABS：Label-Aware Boundary Search**

当前代码的局部搜索太弱，原因很明确：它只是对同一个 seed 做 `OperatorGAhalf` 扰动，没有区分“这个 seed 最后被真实评估成 feasible 还是 infeasible”，因此**没有方向性**。

你想要的其实是两套完全不同的动作。

### 情况 A：`p≈0.5` 的点被评估后是 **可行解**

这时它的价值不是“把自己放进种群”，而是作为**新可行域入口**。

建议动作：

1. 在 `A_I ∪ P_U` 中，找与它**同 sector** 或最近的 infeasible 邻居 (x^{-})。
2. 定义一个“法向推进方向”
   [
   d_n=\frac{x_f-x^-}{|x_f-x^-|}
   ]
   其中 (x_f) 是这个新发现的 feasible 点。
3. 做**可行侧扩张采样**，而不是各向同性撒点：
   [
   x' = \Pi_\Omega\big(x_f+\alpha d_n+\beta \xi^\perp\big)
   ]
   其中 (\xi^\perp) 是与 (d_n) 正交的随机扰动，(\alpha>0) 让你沿“从不可行到可行”的方向继续推进，(\beta) 只负责少量横向探索。
4. 对这些点做真实评估；若出现更优 feasible 点，并且它在该 sector 上比当前 `P_C` 的代表更好，就只对这个 sector 做**局部迁移**，替换掉该 sector 内最差的 10%–20% 个体，而不是整群迁移。

这里我特别强调一句：
**不要整体迁移整个约束种群。**
多目标下，旧可行域可能仍贡献 PF 的另一段；整体迁移太激进，审稿人会质疑你会不会丢掉 disconnected PF。

### 情况 B：`p≈0.5` 的点被评估后是 **不可行解**

这时你要做的不是“继续围着它乱搜”，而是分两步：

#### 第一步：找附近是否真有可行域

1. 从 `A_F ∪ P_C` 中找同 sector 或最近的 feasible anchor (x^{+})。
2. 对每个 ((x^{+},x^{-})) 做**线段括逼 / 二分**：
   [
   x(\lambda)=x^+ + \lambda(x^- - x^+),\quad \lambda\in(0,1)
   ]
3. 每次评估中点，如果中点 feasible，就继续向 infeasible 侧二分；如果中点 infeasible，就向 feasible 侧二分。
4. 如果在很少几次查询内找到了一个 tight bracket，你就得到了“附近有边界，而且有可行域”的强证据，然后把中点转入上面的 feasible-case 扩张。

这一步特别适合你的问题，因为你只有 0/1 约束标签，没有 violation degree；**线段括逼在这种 setting 下比普通局部扰动更自然、更省预算**。

#### 第二步：如果一直找不到 feasible

那就不要再骗自己“这里大概有可行域”。
此时应该明确把这一带记成**局部 hard-negative neighborhood**：

* 在该 infeasible 点附近做一小圈半径递减的采样；
* 如果全部仍是 infeasible，就把这些样本作为**hard negatives**放进训练缓冲区；
* 后续候选评分时，对这个局部区域加惩罚，降低再次采样概率。

也就是说，你不是在逻辑上宣布“这里绝对没有可行域”，而是更严谨地说：

**在当前半径、当前方向、当前预算下，没有观察到可行证据，所以要让模型学会把这一块压到 0.5 以下。**

这和主动学习里“采最不确定点”后继续用新标签修正边界模型是同一个逻辑；同时也符合未知约束优化里“主动收集约束函数信息”的方向。只不过你这里是进化式的，不是 BO 式的。([Burr Settles][3])

### 这部分的核心建议，压缩成一句话

**可行的 `0.5` 点要拿来“扩张并迁移”；不可行的 `0.5` 点要拿来“括逼边界或确认负区域”。**

这才是你想要的闭环。

---

## 3）MLP 结构是否合理？`tanh + sigmoid` 和 `ReLU + SCG` 哪个更合适？

这题我给你的判断是：

**在你的问题里，结构不是第一矛盾；校准和更新策略比激活函数更重要。**

当前代码是**单隐藏层 + tanh + sigmoid**，原论文写的是**ReLU + sigmoid + SCG**。
从“边界概率场是否平滑、是否稳定”这个目标看，我更偏向：

**主版本继续用单隐藏层 + sigmoid 输出；隐藏层用 tanh 就可以，不必急着切 ReLU。**

原因是：

* 你不是在做很深的表征学习；
* 你只需要一个**稳定的二分类边界近似器**；
* 你真正依赖的是 `p≈0.5` 的语义，而不是多 0.2% 的分类精度。

至于 SCG，本身是个很好的小网络训练器。Møller 的原始工作就强调了 SCG 的二阶信息利用、超线性收敛和 (O(N)) 内存特性。([SCI2S][4])
但它更适合**周期性的 full-batch from-scratch 训练**。如果你按我上面的建议改成“3–5 代 warm-start 小更新”，那一个简单的梯度型优化器反而更顺手。

所以我的具体建议是：

* **主版本**：`1 hidden layer (16–32 units) + tanh + sigmoid`
* **训练目标**：class-balanced BCE
* **必须补的东西**：留一小块 calibration buffer，训练后做一次 temperature scaling。现代网络概率往往不校准，而 temperature scaling 往往有效。([Proceedings of Machine Learning Research][1])
* **论文对比/消融**：再放一个 `ReLU + SCG` 版本做 ablation 即可

如果你后面发现单个 MLP 的 `p≈0.5` 还是不稳，最干净的升级不是换更深网络，而是**3 个浅层 MLP 的小型 ensemble**。Deep ensembles 在不确定性质量上通常比单模型更稳。([NeurIPS 会议录][5])
但这一步我建议作为**备选增强**，不是主版本必选，否则创新点会被冲淡。

---

## 4）候选池 4 个来源是否多余？能否合并？候选池大小是否应固定？FIFO 是否合理？

我认为：**可以简化，而且建议简化。**

你当前 4 个来源是：

1. `P_C × P_U`
2. `A_F` 局部扰动
3. `A_I` 局部扰动
4. `A_I × P_C`

严格看，1 和 4 都属于“**可行–不可行桥接**”；2 和 3 都属于“**边界 seed 局部搜索**”。所以完全可以压成 **2 个来源**：

### 来源 S1：Bridge source

统一包括
[
(P_C \cup A_F)\times(P_U^{inf}\cup A_I)
]
也就是所有“可行端”和“不可行/不确定端”的桥接配对。

### 来源 S2：Boundary-local source

统一包括 `A_F ∪ A_I` 上的 label-aware local search。
如果 seed 是 feasible，就走“扩张”；如果 seed 是 infeasible，就走“括逼/负确认”。

这样一来，你的算法逻辑就清楚了：

* S1 负责**找边界入口**
* S2 负责**沿边界做局部动作**

这比 4 个 source 再叠 source-load 排序，更好讲，也更好做实验。

### 候选池大小要不要动态变化？

为了**便于实验归因**，我建议主版本先不要玩复杂自适应。

最干净的做法是：

* 每代真实 boundary evaluation budget 固定：
  [
  B=\lfloor b_\rho N \rfloor
  ]
* 只生成一个固定大小的未评估候选池：
  [
  |Pool| = 4B
  ]
* 其中 Bridge 和 Local 各占一半即可

也就是说，**先把评估预算固定**，这样实验里任何提升都更容易归因到“边界选择与局部搜索机制”，而不是“预算调度技巧”。

### FIFO 有没有道理？

有，但**只适合训练档案，不适合候选池**。

* **候选池**是每代临时生成的，不该 FIFO；
* **训练档案**用 FIFO 是合理的，因为分布在变；
* 但不要是纯 FIFO，建议改成
  **Balanced FIFO + Protected hard cases**：

  1. 一半保留近期 feasible
  2. 一半保留近期 infeasible
  3. 另外预留一小块只保留 tight bracket points 和被模型误判的 `p≈0.5` hard cases

因为纯 FIFO 很容易把稀有但关键的边界拓扑忘掉。你当前代码的 balanced FIFO 思路已经是对的，只是还缺一个“保护边界 hard cases”的小口袋。

---

## 5）当前环境选择策略是否合适？是否能合理利用 `p≈0.5` 的解？哪些技巧华而不实？

我的判断是：

**当前环境选择“方向没错，但层次太多，`p≈0.5` 没被用到最该用的地方”。**

你现在 `EnvironmentalSelectionC` 的逻辑是：

* 先把 feasible 解按目标选出来；
* 不够再用 near-boundary infeasible 解去补；
* `HelperU` 也会把一部分 near-boundary infeasible 解送回 constrained update。

这有两个问题。

### 问题 1：`p≈0.5` 的解被当成“种群填充物”，而不是“边界工作单元”

我更推荐的逻辑是：

* `P_C` 的主职能就是维护 feasible PF；
* `p≈0.5` 的点主要应进入**boundary worker branch**；
* 只有当 feasible 个体严重不足时，才允许少量 boundary points 临时填补 `P_C`。

这样 `p≈0.5` 的解才不会“混在环境选择里被稀释掉”，而是明确承担边界探测职责。

### 问题 2：你现在叠了太多 ranking 技巧

当前代码里反复出现这些量：

* `score = 1 - 2|p-0.5|`
* sector sparsity
* crowding
* front number
* source load
* same-sector dominated filter

这里面真正必要的，我认为只有两个：

1. **feasible-first**
2. **一个统一的 diversity 机制**（用 sector 就够了）

其余像 `source load` 这种，我更倾向于放到附录做 ablation，不要放进主算法。因为它很容易让审稿人觉得：
“你这里不是一个清楚的机制，而是一堆 heuristics 叠加出来的效果。”

### 我建议的更干净的环境选择

* **`P_C`**：只从真实评估后的 feasible 个体中做 NSGA-II/NSGA-III 式选择
* **若 feasible 不足**：再从“成功的 boundary workers”里补，不是从所有 `p≈0.5` infeasible 中补
* **`P_U`**：继续纯 objective-driven 即可
* **boundary information**：主要存在 `A_F/A_I` 或 boundary archive 中，不依赖长期占据 `P_C` 个体位

这样改完以后，`p≈0.5` 的解就真的被“合理利用”了：
**不是拿来长期存活，而是拿来触发边界搜索。**

---

## 6）梳理整体流程，重新优化算法逻辑；改完后重新评估创新点

我建议你把整套算法整理成下面这个版本。逻辑会比现在清楚很多。

### 重构后的整体流程

#### Step 0：初始化

* 随机采样初始训练集 `D0`
* 初始化 `P_C, P_U`
* 训练并校准 boundary MLP `M`

#### Step 1：常规双种群演化

* `P_C -> O_C`
* `P_U -> O_U`

#### Step 2：生成边界候选

只保留两个来源：

* **Bridge candidates**：从 `(P_C ∪ A_F)` 和 `(P_U^{inf} ∪ A_I)` 做 sector-aware pairing
* **Boundary-local candidates**：从 `A_F ∪ A_I` 上做 label-aware local operator

#### Step 3：边界候选评分

不要再只按 `|p-0.5|` 排序，改成：
[
U_b(x)=H(\tilde p(x))\cdot \widehat{\Delta HV^+}(x)\cdot \text{sector-novelty}(x)
]
其中：

* (\tilde p(x)) 是校准后的概率
* (H(\tilde p)) 体现边界不确定性
* (\widehat{\Delta HV^+}) 体现若它最终变 feasible，对 PF 的潜在价值
* sector-novelty 防止都采到一个方向

这里也正好呼应主动学习文献的结论：二分类里 `p≈0.5` 是最基本的 uncertainty signal，但 uncertainty-only 容易采到 outlier，所以要加 representativeness / diversity。另一方面，做 unknown constraint design 时，单纯看“离预测边界近”也不够，更合理的是看**概率不确定性本身**。([Burr Settles][3])

#### Step 4：真实评估 top-B 边界工作单元

* 对 top-B 候选做真实评估
* 然后按真实标签分两支：

  * feasible -> 扩张/迁移
  * infeasible -> 括逼/负确认

#### Step 5：更新边界档案

* `A_F`：保存边界来源的高价值 feasible seeds
* `A_I`：保存高价值 infeasible seeds
* 可选：额外保存 tight bracket pairs

#### Step 6：触发式更新 MLP

* 若满足新边界标签数/校准漂移等触发条件，则 warm-start 更新
* 每次更新后做一次 temperature scaling

#### Step 7：环境选择

* `P_C`：feasible-first
* `P_U`：objective-only
* sector 只作为配对与局部迁移工具，不再到处重复打分

### 改完后，创新点怎么评估？

我建议你在论文里**仍然只主打 1 个核心创新点**，不要硬写成 3 个。

#### 核心创新点

**Pareto-relevant active boundary search under binary unknown constraints**

再说得更具体一点：

**利用校准后的边界概率，主动挑选对 Pareto 前沿有价值的边界点，并对其执行标签感知的局部边界搜索，从而发现新的可行域入口或修正局部无可行域判断。**

#### 支撑性方法点

**Calibration-aware triggered retraining**

这可以写成第二个 contribution，但不要把它包装成和核心创新并列的“大创新”。
双种群、MLP、FIFO、sector matching，这些都不应该单独报成创新。

### 改完后的严格判断

* **严格大创新点数量**：还是 **1 个**
* **但这个 1 个比你现在版本更硬、更容易被实验验证**
* 审稿人会更容易接受，因为它从“堆 heuristic”变成了“边界搜索闭环机制”

---

## 7）根据改完后的创新点，再设计一套真正有说服力的实验

这里我只围绕**最大创新点**来设计实验，也就是：

**Pareto-relevant active boundary search with label-aware local search**

想证明它成立，必须建立一条完整证据链：

### 先验 sanity check：`p≈0.5` 到底有没有语义？

先别急着报 HV/IGD。先做这个。

#### Step 0：校准实验

在 held-out calibration buffer 上做：

* reliability diagram
* Brier score
* `p` 分箱后的真实 feasible rate

重点检查：

* `p∈[0.4,0.6]` 区间的真实 feasible rate 是否接近 0.5

如果这一步不过关，你后面所有“0.5附近主动采样”的结论都会被审稿人质疑。概率校准和 reliability diagrams 本来就是这一类概率模型该做的基本检查。([Proceedings of Machine Learning Research][1])

---

### 第一层证据：你挑到的点，是否真的更接近“真实边界”？

这一层最关键，也最能说服人。

#### Step 1：做一个“oracle boundary audit”

DASCMOP-UC 是从 DASCMOP 改来的。你可以在**实验分析阶段**额外记录原始连续约束值 `g_raw(x)`；算法运行时仍只能看到 binary feasibility，不能偷看连续约束。

定义一个只用于审计的边界距离：
[
d_B(x)=\min_j \frac{|g^{raw}_j(x)|}{s_j+\epsilon}
]
其中 (s_j) 是第 (j) 个约束的归一化尺度。

然后比较不同方法对“boundary budget 选出来的初始 query 点”的 `d_B` 分布。

#### 对照版本只需要 4 个

1. **Full**：完整方法
2. **Uncertain-only**：只按 `p≈0.5` 选，不加 Pareto relevance
3. **No-local-label**：保留边界选点，但把局部搜索改回普通各向同性扰动
4. **Random-boundary**：同样预算，随机选 boundary 候选

#### 看哪些指标

* median `d_B`
* mean `d_B`
* `#(d_B<τ)` 的累计曲线

如果 Full 不能显著更接近真实边界，你的最大创新点就站不住。

---

### 第二层证据：你找到的不是“普通边界点”，而是“有用边界点”

仅仅靠近边界还不够。你要证明这些边界点能更快转成**对 PF 有贡献的 feasible discoveries**。

#### Step 2：局部边界搜索有效性审计

对每一个被选中的 boundary seed (x_0)，记录其后续 (L) 次局部搜索后代。

定义 3 个指标：

#### 1. Feasible recovery rate

对初始为 infeasible 的 seeds：
[
FRR=\frac{#{\text{在 }L\text{ 次局部查询内找到 feasible 后代}}}{#{\text{被选中的 infeasible seeds}}}
]

#### 2. Useful boundary yield

[
UBY=\frac{#{\text{boundary lineages 中最终有个体进入 external feasible archive}}}{#{\text{被选中的 boundary seeds}}}
]

#### 3. Boundary-induced HV gain

每一代记录 boundary branch 对 external archive 带来的累计 (\Delta HV)。

这一步能直接证明：

* `Uncertain-only` 也许能找到边界，
* 但 **Full** 更能把边界变成“有贡献的可行解”。

---

### 第三层证据：你是否真的完成了“迁移到更有价值可行域”？

你前面最在意这一点，所以实验一定要单独做。

#### Step 3：局部迁移成功率

只看“由 feasible boundary seed 触发的局部迁移事件”。

定义：
[
MIG=\frac{#{\text{迁移后 } \tau \text{ 代内，后代仍留在 external archive 或改善对应 sector}}}{#{\text{所有迁移尝试}}}
]

并且统计：

* 新开启的 feasible sectors 数
* 新发现的 disconnected feasible components 数（可用决策空间聚类近似）
* 每个 component 的 archive contribution

这一步是为了证明你不是“做了一个好看的局部搜索”，而是真的**把种群部分资源迁移到了更值钱的可行域**。

---

### 第四层证据：最终性能提升，而且能归因到边界模块

#### Step 4：最终优化性能

在全部 9 个 DASCMOP-UC 上跑 30 次独立实验：

* Final HV
* Final IGD
* HV-AUC / IGD-AUC（对整个 FE 过程曲线积分）

我建议主表里放：

* Full
* Uncertain-only
* No-local-label
* Random-boundary
* 外部参考算法：NA-EMT / EADMM / 你原版 PRBCCMO

这里要强调一点：

**NA-EMT 和 EADMM 只适合做外部性能参照，不适合证明你的核心机制。**
证明机制成立，必须靠上面三个内部对照版本。

#### 统计检验

* Wilcoxon rank-sum
* Holm 校正
* 再补一个 effect size（如 Cliff’s delta）

这样不是“看曲线觉得好”，而是有统计支撑。

---

## 最后给你一个直接可执行的版本

如果你准备现在就改代码，我建议按下面顺序动手：

1. **先改第 2 条**：把当前 `LocalBoundaryPerturbation` 改成 label-aware operator
2. **再改第 1 条**：把每代 from-scratch 重训改成 3–5 代触发式 warm-start
3. **再改第 4/5 条**：把 4-source candidate pool 压成 2-source；让 `p≈0.5` 主要服务 boundary branch，而不是主种群填充
4. **最后做第 7 条实验**：先做 calibration sanity check，再做 boundary audit，再做最终 HV/IGD

我对改完后的判断是：

* **严格大创新点：1 个**
* **最大的创新点：Pareto-relevant active boundary search with label-aware local search**
* 这比你当前版本更像一个清楚、可证伪、可实验验证的论文机制，而不是一堆 boundary heuristic 的组合。

你下一步最值得做的，是先把我在第 2 条和第 6 条里说的流程写成伪代码。

[1]: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf "https://proceedings.mlr.press/v70/guo17a/guo17a.pdf"
[2]: https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html "https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html"
[3]: https://burrsettles.com/pub/settles.activelearning_20090109.pdf "https://burrsettles.com/pub/settles.activelearning_20090109.pdf"
[4]: https://sci2s.ugr.es/keel/pdf/algorithm/classification-algorithm/moller1990.pdf "https://sci2s.ugr.es/keel/pdf/algorithm/classification-algorithm/moller1990.pdf"
[5]: https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html "https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html"
