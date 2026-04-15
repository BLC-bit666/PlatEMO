在你新增的两个硬约束下，我的最终结论是：

**主线要再收缩一步：不要再碰损失函数、不要再碰模型家族、不要再碰校准；把全部改动压到“边界定义、B 的入选规则、训练数据分布”这三处。**

原因很明确。你自己的来源文件已经把核心创新点定义成“**MLP 学习与 PF 搜索相关的可行/不可行边界，并反过来驱动边界采样**”，而不是双种群本身，也不是边界存档本身。 但 bundle 里的总诊断又显示：**boundary archive 已经比 MLP 更强，训练集 balanced accuracy 与真实 boundary 效果脱钩，当前最常见失败形态是 archive 已经不错、MLP 仍没把 boundary 学出来**。
这说明现在的瓶颈不是“BCE 不够高级”，而是**BCE 被喂错了任务分布**。

---

## 一、这次该坚持什么，不该再动什么

### 1）不改损失函数，这个要求是对的

在你现在的阶段，继续改 loss 会让论文很难解释：一旦引入 pair loss、midpoint loss、class-weight tricks、focal 一类东西，审稿人会自然追问“效果到底来自边界思想，还是来自特制损失”。你的要求反而把问题逼回了最本质的地方：**如果 plain BCE 也能学出边界语义，那创新点才更硬。**

所以这版主线里，我建议**明确排除**：

* pairwise ranking loss
* midpoint loss
* class-weighted BCE
* focal / asymmetric loss
* 温度缩放 / 后处理校准
* Deep Ensembles

最后两项不是因为它们没价值。Guo 等人的工作明确表明现代神经网络常常**概率失校准**，而 temperature scaling 往往有效；Lakshminarayanan 等人的 Deep Ensembles 也确实能给出更好的不确定性估计。([Proceedings of Machine Learning Research][1])
但你这次已经把允许改动的范围收紧到**边界定义、边界存档、训练数据分布**三处，所以它们都应该暂时退出主线。

### 2）你现在真正要学的，不是“全局 feasibility 分类”

你自己的文档已经说得很准：MLP 的目标不是单纯分类，而是尽量让**接近 0.5 的解，对应真实边界附近**。
问题是，在 plain BCE 下，`p≈0.5` 只有在一个前提下才有边界语义：

> **这个区域已经被真实观测到“边界两侧”。**

如果某个扇区从头到尾只见过大量不可行，几乎没见过边界附近可行点，那么 `p≈0.5` 更可能表示“模型犹豫”或“样本稀疏”，而不是“真实边界”。这也是为什么你的 bundle 会出现“train 指标不错，但 boundary 指标不行”的脱钩。

### 3）文献也支持你把“边界”当成主体，而不是附属物

二元/未知约束下，传统依赖 CV 的约束处理会失效或明显变弱；这正是 Huang 的 DRMCMO 和 Li 的 EADMM 都要绕开 CV 引导、改走检测区/双子问题路线的原因。([arXiv][2])
同时，边界本身值得被直接建模。BE-CBO 直接把 unknown constraints 下的最优解视为常常落在 feasible / infeasible boundary 上，并强调复杂边界需要更强的边界建模。([Proceedings of Machine Learning Research][3])
而主动学习文献也提醒：**只靠 uncertainty 往往会挑到 outliers，必须同时考虑 representativeness / density**。([Burr Settles][4])

所以，和你现在约束最一致的路线不是“给 BCE 加技巧”，而是：

> **让 BCE 只看“边界两侧、局部、可信”的数据。**

---

## 二、最终思路：把问题从“分类器设计”改成“边界语义数据设计”

我建议你把整条主线改写成下面这句话：

> **PRBCCMO 不再试图让 MLP 学整个可行域，而是让 plain BCE 在“局部双侧支撑”的样本分布上学习边界附近的 feasible / infeasible 分界。**

这句话会直接决定三件事：

1. 边界解怎么定义
2. 边界存档 B 怎么收
3. MLP 训练集怎么喂

下面我按这三件事展开。

---

## 三、边界解的重新定义

### 1）先引入一个“可信边界扇区”概念

对每个目标空间扇区 `s`，定义它是不是 **trusted boundary sector**：

* 在 `s` 或其邻扇区内，
* 已有**真实评估**过的样本，
* 且至少同时出现过一侧 feasible、另一侧 infeasible。

只有这样的扇区，才允许你把 `|p-0.5|` 解释为“边界接近度”。

这一步非常关键。它等于承认：

* **同一个 MLP 输出**

  * 在 trusted sector 里，可以当“边界代理”
  * 在 untrusted sector 里，只能当“不确定度”，不能当“边界证据”

这样做的好处是，你不用改 loss，也不用改模型，只是**限定了 MLP 在哪里有资格发言**。

### 2）边界相关样本不再定义为“low-margin 样本”，而是“low-margin + 两侧真实支撑样本”

对任一样本 `x`，定义它的局部 opposite-side 距离：

[
d_{\text{opp}}(x)
=================

\min_{x':, y(x')\neq y(x),; s(x')\in {s(x)\cup \mathcal N(s)}}
|x-x'|
]

其中：

* `y(x)` 是**真实标签**，不是模型标签
* `s(x)` 是扇区
* `N(s)` 是邻扇区集合

然后定义边界相关性：

[
C(x)=\exp(-d_{\text{opp}}(x)/\tau_d)
]

解释很简单：

* `C(x)` 大：这个点附近真实见过边界两侧
* `C(x)` 小：这个点即使 low-margin，也更像“空想边界”

所以这次你要明确放弃一句隐含假设：

> **“uncertainty = boundary” 不成立；只有“uncertainty + real opposite-side support” 才近似 boundary。**

这与主动学习里“uncertainty 需要 representativeness/density 修正”的结论是一致的。([Burr Settles][4])

---

## 四、边界存档 B 的入选规则

你现在允许改“边界存档数据的入选要求”，这正是最该下手的地方。

### 1）B 不再收“所有 low-margin 候选”，只收“可信边界候选”

某候选 `x` 可以进入 B，必须满足至少其一：

* 它所在扇区已是 trusted sector；
* 或它本身是一个“补齐两侧证据”的样本，使得这个扇区首次变成 trusted sector。

换句话说：

* **trusted sector 内，MLP 帮你细排**
* **untrusted sector 内，真实两侧证据先于 MLP 发言**

### 2）B 的评分只保留三项，不再叠 novelty

你 bundle 里已经明确建议把路线收缩到三项：边界相关性、轻量防塌缩、采样不再 pure uncertainty。
所以 B 的入选分数我建议固定成：

[
J_B(x)=0.5,M(x)+0.3,(1-C(x))+0.2,Q(x)
]

其中：

* `M(x)=2|p(x)-0.5|`，越小越好
* `C(x)` 是上面的 opposite-side support，越大越好
* `Q(x)` 是 objective relevance，越小越靠近 PF 相关区域

这三项分别对应：

* `M(x)`：模型认为它在边界附近
* `C(x)`：真实数据确认它确实靠近边界两侧
* `Q(x)`：它不是与 PF 搜索无关的边缘噪声

我不建议你再保留 `novelty`。
原因不是 novelty 永远没用，而是你已经有“扇区 + 每扇区 top-k”这一层 objective-space 多样性控制了；再加 novelty，只是在 B 内部做第二层重复惩罚，复杂度上升，但主线没有更清楚。这个判断也和 bundle 的减法建议一致。

### 3）每扇区最多保留 `kappa=3`，但加一个“软性少数类保留”

如果某扇区当前候选同时有 feasible 和 infeasible，但 top-k 全部来自同一侧，则强制检查：

* 最佳 opposite-side 候选的 `J_B`
* 是否只比当前最差已选样本略差

若是，则替换进去。

这不是为了“做类平衡”，而是为了避免 B 退化成单侧档案。
你的 current PRBCCMO 主线本来就已经把 B 定义为“top-k boundary-information samples per sector with soft minority reservation”。
在你当前限制下，这个 soft minority 应该保留，而且要升格为主逻辑。

---

## 五、训练数据分布怎么改，才能在 plain BCE 下逼近边界

这是最核心的一步。

### 1）训练集主干只保留两类：`B + RecentBoundaryOff`

你 current PRBCCMO 已经把训练档案主要来源写成：

* `B`
* `RecentBoundaryOff`
* `P_C` feasible reps
* `P_U` infeasible reps

而且 source 标记也已经做出来了。

在你现在的规则下，我建议进一步收紧成：

[
\mathcal D_t = B_t \cup O^B_t \cup R_C^F \cup R_U^I
]

但语义要改：

* `B_t` 和 `O^B_t` 是**主体**
* `R_C^F`、`R_U^I` 只是**补洞**

也就是说：

* 不是“4 个来源并列”
* 而是“2 个主来源 + 2 个补洞来源”

### 2）`P_C/P_U reps` 不再常规加入，只在“缺一侧”时补一个锚点

对每个扇区 `s`：

* 先看 `B + RecentBoundaryOff` 在 `s` 及其邻扇区内是否已有两侧样本
* 若已有两侧，**不加任何 rep**
* 若缺 feasible，则只补一个最近的 `P_C` feasible rep
* 若缺 infeasible，则只补一个最近的 `P_U` infeasible rep

而且每侧至多补一个。

这样做的结果是：

* 你仍然允许极少量支持点进入训练集，解决“边界存档里可行解太少”的现实问题
* 但你不再让大量主种群样本淹没边界语义

### 3）训练集要按“每扇区定额”裁剪，而不是全局 FIFO

只用 plain BCE 时，最怕的是一个现象：

* 某些扇区有很多不可行点
* 少数扇区只有零星可行点
* 最后模型又退化回“全局 feasibility 分类器”

所以训练档案裁剪必须改成**sector-wise quota**。我建议：

* 每扇区最多保留 2 个 `B`
* 每扇区最多保留 2 个 `RecentBoundaryOff`
* 每扇区每一缺失侧最多保留 1 个 rep

这相当于把训练分布从“总体分布抽样”改成“边界扇区均衡抽样”。

在不改损失函数的情况下，这一步的作用非常大。因为 BCE 最终学到的是**训练分布上的后验**；你不改 loss，就只能改它看到的分布。

### 4）如果你坚持“默认 BCE”，那就要把当前主线里的加权/校准也拿掉

你现有的 clean PRBCCMO 主线里，bundle 已经说明它引入了：

* `BoundWeight`
* calibration buffer
* temperature scaling

这些在你这次新规则下都不该进主线。

也就是说，新版应该进一步简化成：

* 单个 MLP
* warm start 可以保留
* **plain BCE**
* **不加 class weight**
* **不加 bound weight**
* **不做温度缩放**
* 所有“边界语义”完全靠数据组成来实现

这会让结果解释非常干净。

---

## 六、boundary sampling 也必须跟着收缩，否则训练集改了也会被重新污染

虽然你这次强调的是“边界定义、边界存档、训练集分布”，但如果采样链不一起收缩，新的训练集很快又会被旧逻辑污染。

### 1）helper 只允许来自 `P_C ∪ P_U`，不允许来自 `B`

这点其实和你 current clean 版是一致的：helper candidates drawn only from `P_C union P_U`。
这一点要保留，不要回退。

### 2）更进一步：主线里直接删掉 predicted-opposite helper

bundle 已经非常明确：当前建议第一条就是“先删除 predicted-opposite helper，仅保留 real opposite helper；没有 real opposite 就跳过”。

在你现在“不改 loss”的约束下，我认为这条更应该执行。
因为 predicted-opposite helper 本质上还是模型在拿自己的分类结果给自己喂数据，容易形成自证闭环。

所以我建议主线固定成：

* helper 优先级只有一条：**real opposite helper**
* 没有 real opposite helper，**就跳过这个 sector**
* 不再 fallback 到 predicted-opposite

这会减少采样数，但会显著提高边界样本纯度。现在你更需要的是“干净地验证核心创新点”，不是“尽量多出点样本”。

### 3）probe 不再分 precise / discovery，固定一组 β 就够了

bundle 里已经给出一组很干净的 probe 系数：

[
\beta \in {0.25, 0.50, 0.75, 1.05}
]

并明确建议**不要再搞双模式**。
这条我完全同意。

理由是：

* 你现在要的是统一语义
* 不是一边 precise、一边 discovery 再做调度
* 否则采样、B 更新、训练集这三条链又会重新分裂

### 4）probe ranking 也不要再 pure uncertainty

你 current clean 版代码里其实已经把 probe 排序改成了三项式：

* margin
* opposite-side support
* objective relevance

而不再是单纯 `abs(prob-0.5)`。

在你现在的限制下，这个方向应该保留，而且更进一步变成**全链统一语义**：

* B 更新用同一分数
* probe ranking 用同一分数
* trusted-sector 判定仍然用真实两侧支撑

这样整个系统才是一条闭环，而不是三套不同标准。

---

## 七、从已有 PRBCCMO 出发，具体怎么改

你现在的 mainline `PRBCCMO.m` 已经具备了正确骨架：

* `P_C` / `P_U`
* `B`
* `UpdateTrainArchive`
* `anchor-helper-probe`
* helper 只来自 `P_C ∪ P_U`
* 统一 ranking score

这些都在 bundle 里写得很清楚。

所以我建议的**详细改造方式**是：

### 第一步：把训练端进一步“去技巧化”

在 `TrainBoundaryMLP` 一侧，目标只有一个：

* **把 BCE 彻底还原成 plain BCE**

你应该删掉或停用：

* class weighting
* boundary weighting
* pairwise/midpoint 相关遗留
* calibration / temperature scaling

你要保留的只有：

* 一个 MLP
* warm start
* plain BCE

然后把所有边界语义转移到 `UpdateTrainArchive`。

### 第二步：重写 `UpdateTrainArchive` 的组成逻辑

新逻辑应当是：

1. 先取 `B`
2. 再取 `RecentBoundaryOff`
3. 对每个扇区检查是否缺 feasible / infeasible
4. 缺哪边，只补哪边一个 rep
5. 按扇区定额裁剪

这一步做完，训练任务就不再是“整个可行域分类”，而是“边界局部两侧分界”。

### 第三步：重写 `BuildBoundaryMeta / UpdateBoundaryArchive`

新逻辑：

* 先算 trusted sectors
* 非 trusted sector 原则上不进 B，除非它是建立两侧证据的新点
* trusted sector 内按 `J_B` 排序
* 每扇区 top-k=3
* 若双侧都有而 top-k 单侧塌缩，做一次 soft minority swap

### 第四步：主线删除 predicted-opposite helper

把 helper 逻辑收缩成：

* same/neighbor sectors
* `P_C ∪ P_U`
* real opposite only
* no helper → skip sector

### 第五步：删除 PRBCCMO2 / PRBCCMO3 的旁路扩展

这一点 bundle 也已经给出强烈建议：
不要再并行演化 `PRBCCMO2 / PRBCCMO3`，只保留一条 `PRBCCMO` 主线。

具体应当退出主线的东西包括：

* stage / paired-slot / interpolation-only 路线
* pairwise ranking / midpoint
* novelty
* helper-from-B
* precise / discovery 双模式
* 各种动态 score 调度

这些机制的问题不是“完全没用”，而是它们会把论文重心从“MLP 学边界”拖回“手工描边 + 技术补丁”。

---

## 八、Deep Ensembles 现在要不要上

我的答案是：**现在不要。**

理由不是它不好。相反，Deep Ensembles 的确更适合复杂边界不确定性建模，BE-CBO 也明确报告了 ensemble 神经网络在复杂 boundary 上优于 GP。([Proceedings of Machine Learning Research][3])
但你现在的失败主要是：

* boundary archive 已经形成
* MLP 训练效果与真实 boundary 脱钩
* helper/采样中存在自证循环

这说明你当前首先要修的是**任务定义与数据语义**，不是模型容量。

所以顺序应该是：

1. 先把 plain BCE + 数据语义主线做干净
2. 如果这之后 boundary 指标仍然不起来
3. 再把 **唯一的模型家族消融** 设成：MLP vs DE

在那之前上 DE，只会让一个“被错误数据分布驱动的分类器”变成一个“更稳地学错任务的分类器”。

---

## 九、最小实验计划

这次不要再开很多支线。只做 4 个版本：

1. **Base-Plain**

   * 当前 PRBCCMO 主线
   * 但把训练端彻底还原为 plain BCE

2. **+TrustedB**

   * 在 Base-Plain 上加入 trusted-sector 边界定义
   * B 只收可信边界候选

3. **+BoundaryTrain**

   * 再把训练集改成 `B + RecentBoundaryOff + missing-side reps`
   * 扇区定额裁剪

4. **+RealHelperOnly**

   * 再删掉 predicted-opposite helper
   * helper 只保留 real opposite

只看 5 个指标就够：

* `lowmargin_feasible_ratio`
* `lowmargin_mix_score`
* `lowmargin_oppdist`
* `b_mixed_sectors`
* `last_boundary_bal_acc`

这些也是 bundle 已经建议你继续盯住的核心指标。

判定标准也很简单：

* `train_bal_acc` 不是主指标
* 真正想看到的是：

  * `lowmargin_mix_score` 上升
  * `lowmargin_oppdist` 下降
  * `b_mixed_sectors` 上升
  * `last_boundary_bal_acc` 上升

如果这些不动，就不要再加新机制。

---

## 十、一句话收束

**在“禁止改损失函数”的前提下，你最正确的路线不是继续设计更强分类器，而是把 plain BCE 的训练任务强行收缩成“局部双侧支撑下的边界分界任务”。**

落到 PRBCCMO 上，就是四件事：

* **把边界从“low-margin”改成“low-margin + 真实两侧支撑”**
* **把 B 从“收不确定点”改成“收可信边界点”**
* **把训练集从“混合可行性样本”改成“B + RecentBoundaryOff 为主、reps 只补缺侧”**
* **把 helper 收缩为 real-opposite-only，删除 predicted-opposite 自证链**

这条路线最统一、最省改动，也最能验证你的核心创新点。
如果你愿意，我下一条可以直接把这套方案翻成 `UpdateTrainArchive / BuildBoundaryMeta / SelectBoundaryHelper / RankBoundaryProbeCandidates` 的伪代码版本。

[1]: https://proceedings.mlr.press/v70/guo17a.html "https://proceedings.mlr.press/v70/guo17a.html"
[2]: https://arxiv.org/abs/2411.08437 "https://arxiv.org/abs/2411.08437"
[3]: https://proceedings.mlr.press/v235/tian24g.html "https://proceedings.mlr.press/v235/tian24g.html"
[4]: https://burrsettles.com/pub/settles.activelearning_20090109.pdf "https://burrsettles.com/pub/settles.activelearning_20090109.pdf"
