我先给结论。

你现在的瓶颈，**主要确实不在损失函数，而在训练链路**；但更准确地说，不是单独“TrainBoundaryMLP 这几行没训好”，而是 **训练集构造、训练触发时机、以及 MLP 输出如何反过来支配边界存档/不可行解选解** 这三个环节一起错位了。你在文档里已经把创新点定义得很清楚：**MLP 的任务是学“边界”，不是拿“可行概率”替代 CV**，而且你明确说过不希望依赖手工可行/不可行配对。 但当前代码的训练集实际上仍是“每扇区抽一对正负样本，不足就最近邻补对”，这和你的创新叙事是正面冲突的。

我没有看到你当前版本的实验日志和曲线，所以“实验结果”这一部分我只能做**结构性归因**，不能对某一组具体数值异常下断言。

## 1. 当前版本为什么 MLP 拟合边界会差

### 1.1 训练样本太少，而且抽样单位就错了

`BuildBoundaryDataset` 只从 `CollectLocalBoundaryPairs` 拿数据；而 `CollectLocalBoundaryPairs` 对每个 active sector 最多只取 **一条正样本、一条负样本**。若某扇区只有一侧样本，就从另一种群里找一个最近的 opposite sample 来“补成一对”。最后训练集就是 `[AnchorRows; PairRows]`，标签是 `[1;0]`。

这会带来三个后果：

第一，你真正喂给 MLP 的不是“边界带”，而是“每扇区一个代表对”。
第二，你把**连续边界**压扁成了**稀疏点对**。
第三，你实际上又回到了你文档里明确说要放弃的那条路：**手工可行/不可行配对**。

### 1.2 训练证据被过度压缩，历史信息大量丢失

主边界存档 `B` 每个 sector 最多只保留 `kappa=3` 个点。 同时，训练集的 `Core` 只来自 `[B, RecentBoundaryOff]`，而 `RecentBoundaryOff` 只保存**最近一批**边界 offspring。

也就是说，你虽然有双种群和大量不可行解，但真正进入 MLP 训练的数据，被压缩成了：

* 每扇区最多几个边界点
* 外加最近一批 boundary offspring
* 种群里的海量不可行信息几乎都没进训练

所以你现在并没有落实“**大量不可行解 + 少量可行解去确定边界**”，而是在做“**极少量扇区代表点分类**”。

### 1.3 训练触发太钝，而且状态变量基本没真正用起来

当前 retrain 条件是：

* `pair_ready`
* 模型为空
* 距离上次训练超过 `2*Problem.N`
* 或 `recent_mixed_evidence > 0`。

同时 `ModelState` 里虽然存了 `lastTrainSize` 和 `lastDualSectorMask`，但实际触发逻辑只用到了 `lastTrainFE`。

这意味着：

* **训练太早**：刚凑够少量 pair 就开训
* **训练太晚**：边界分布已经变了，但只要没到 FE 间隔，模型不更新
* **训练是否真的退化了**，没有被监控

NA-EMT 至少已经意识到需要用模型准确率触发自适应更新，并把近期样本滚动并回训练集。 你这版在“训练时机”上明显还没到那个成熟度。

### 1.4 每次重训都随机重启，20 个 epoch 根本不稳

`TrainBoundaryMLP` 每次都会重新随机初始化 `W1/W2/b1/b2`，然后只训 20 个 epoch。

这对“边界逐步细化”的任务很伤：

* 上一轮好不容易学到的局部边界形状，下一轮直接清空
* 数据本来就少，随机初始化噪声占比很高
* 20 epoch 对 cold start 来说常常不够，对 warm start 又太浪费

所以你现在的模型很可能不是“越来越会画边界”，而是“每隔一段时间重新猜一次边界”。

### 1.5 MLP 的输出太早、太强地反过来筛训练样本和选解

你现在把 `margin = 2*abs(prob-0.5)` 作为边界度量。 这本身是对的，方向比“直接用概率代替 CV”更符合你的创新点。问题出在：**这个 margin 已经深度进入了 archive 和选解闭环**。

* `SelectTopKPerSector` 用 `SortByBoundaryKey` 选边界存档。
* `GenerateBoundaryOffspring` 只接受 `CandidateMeta.margin < AnchorMargin` 的 child。
* `SelectInfeasibleByBoundaryMeta` 按 `margin → oppSupport → objScore` 排序不可行解。

于是形成了一个危险闭环：

> 旧模型定义谁更像边界 → 存档和 offspring 往那里集中 → 新训练集又来自这些被旧模型筛过的数据 → 新模型进一步强化旧判断。

这就是典型的**自证循环**。一旦第一次边界判断偏了，后面会越来越偏。

---

## 2. 你那两句话，哪句对，哪句要改

### 2.1 “大量不可行解和少量可行解可以训练 MLP 学习边界”

这句话**只对一半**。

对的部分是：
边界确实由“两侧样本”决定，尤其在二元未知约束里，**近边界不可行解**非常重要。项目内文档的核心动机也是“找到对 PF 搜索有意义的可行/不可行边界”。

不对的部分是：
**不是随便大量不可行解都行。** 如果大多数不可行解离边界很远，那么在不改损失函数的前提下，BCE 只会让模型更努力地区分“明显不可行”和“明显可行”，而不是把 `0.5` 压到真实边界上。主动学习和不确定性采样的研究也指出：最不确定样本通常靠近决策边界，但当初始概率模型差、或者数据严重不平衡时，朴素 uncertainty sampling 会偏到多数类或偏斜边界，所以必须做平衡与多样性控制。([Proceedings of Machine Learning Research][1])

所以这句话应改成：

> **大量“近边界”的不可行解 + 少量“近边界”的可行解 + 少量远场锚点，才能训练出真正用于边界定位的 MLP。**

### 2.2 “每个参考向量分区不要顶死是几对”

这句话方向是对的，而且我认为是**关键判断**。

因为不同 sector 的边界复杂度、曲率、混合程度、本代可行/不可行证据量都不同。固定“每扇区 1 对”或“每扇区 2 对”，会同时犯两个错：

* **对复杂扇区喂得太少**：长边界、弯边界、断裂边界都被压成 1 对
* **对稀疏扇区喂得太多**：容易把噪声 pair 当真边界

但这句话还要再补半句：

> 不是“完全不控量”，而是“不要顶死为固定几对，要改成自适应配额，并且只保留近边界证据”。

---

## 3. 我建议的“统一、减法、收敛”式重构主线

我建议你不要再继续围着 `hidden/epoch/lr` 微调，也不要先上更复杂网络。先把主线改成下面这一条：

> **双种群继续供样本；边界存档继续保多样性；MLP 仍用现有损失；但训练集改成“边界带数据集”，训练改成“滚动 warm-start”，选解改成“margin + 真实 opposite 支撑”的二阶段规则。**

### 3.1 把“配对训练集”改成“边界带训练集”

这是第一优先级，也是最关键的一刀。

不要再以 pair 作为训练单位。
训练单位应该回到**单个真实评估样本**，但这些样本要通过“离 opposite-side 最近”的规则进入训练集。

具体做法：

对每个样本 (x)，定义它在本 sector 及相邻 sector 内到 opposite-label 样本的最近距离：

[
d_{\text{opp}}(x)=\min_{x': y(x')\neq y(x),, s(x')\in \mathcal N(s(x))}|x-x'|
]

* (d_{\text{opp}}) 越小，越可能靠近真实边界
* 对可行和不可行两侧都分别排序
* 每个 sector 只保留 (d_{\text{opp}}) 最小的一小段样本进入训练集

这就把“手工凑一对”改成了“用真实评估点构成边界带”。
训练仍然是标准二分类，损失函数完全不用改；你真正改变的是**进入损失函数的数据分布**。这正符合“不改 loss、改训练方式”的原则。NeurIPS 2023 也专门指出，在类不平衡场景下，不一定需要专门改 loss，单靠训练流程和数据管线的调优就可以显著改善表现。([NeurIPS Proceedings][2])

### 3.2 参考向量扇区不再固定“几对”，而是自适应配额

每个 sector 设一个自适应配额 (q_s)，而不是固定 pair 数。建议：

* 下限：只要 sector 有 mixed evidence，至少保留 1 个可行 + 1 个不可行
* 上限：小常数截断，比如每侧最多 3–5 个
* 中间值：由 sector 的 mixed evidence 强弱决定，例如：

  * 本扇区及邻扇区里两侧样本数量
  * 最近几代 boundary offspring 数量
  * 该扇区内 (d_{\text{opp}}) 很小的样本比例

这样做的好处是：

* 复杂 sector 自动多给样本
* 噪声 sector 不会硬凑配额
* 参考向量仍保留多样性约束，主线没变

### 3.3 搜索存档和训练存档要逻辑拆分

当前 `B` 既想做“搜索用的边界精英”，又想做“训练用的数据池”，这两个目标是冲突的。

* 搜索存档 `B_search`：继续小而精，每扇区 `kappa` 个，负责多样性和 boundary offspring 生成
* 训练缓冲 `T_train`：固定总容量、滚动 FIFO，负责给 MLP 提供证据

这不是加第三种群，只是把“边界存档的两种职责”分开。你的文档里也说得很清楚，边界存档本来就只是服务边界拟合的数据容器。

`T_train` 的来源建议统一成：

[
T\leftarrow [B_{\text{search}},\ \text{RecentBoundaryOff},\ \text{本代 mixed sectors 中的 } Population_C,\ Population_U]
]

这样你终于真正用上了“双种群从两侧为边界学习供样本”的创新叙事，而不是像现在这样，大量种群信息在进入 MLP 前就被丢掉了。

### 3.4 重训不要每次从零开始，改成 warm-start

当前每次 `TrainBoundaryMLP` 都随机重启。 这一点我认为必须改。

建议：

* 第一次训练：cold start，训满
* 后续训练：默认 warm-start，沿用上一次 `W1/W2/b1/b2`
* 只在出现明显漂移时再 full reset

这样能把“边界逐步细化”真正积累起来。

同时把训练方式改成：

* shuffle
* balanced mini-batch（按可行/不可行或按 sector 平衡）
* 少量 epoch 高频 fine-tune，而不是低频从零训 20 轮

这仍然不改损失函数，只是把训练流程从“实验脚本”提升到“可持续在线更新”。

### 3.5 训练触发不要只看 FE，要看“边界证据是否变了”

我建议直接废掉“`pair_ready + 2N FE`”这一套主触发。

改成三类事件触发，满足其一就重训：

1. `T_train` 中新增样本占比超过阈值
2. mixed sectors 数量发生扩张
3. 最近边界带验证集上的 balanced accuracy / Brier / 边界样本排序质量明显下降

这里可以借鉴 NA-EMT 的思路：它至少已经意识到 MLP 需要根据模型准确率和新样本自适应更新。 但你不要照搬它“概率即价值”的用法，只借它“滚动数据 + 自适应更新”的机制。

### 3.6 不可行解选解要从“margin-only”改成“margin + opposite 支撑”

你现在的不可行解排序其实已经比“直接用概率代替 CV”更对，因为你用的是 `margin=|p-0.5|`。 这个方向应该保留。

但还差一步：**不能只看 margin，还要看这个 margin 有没有真实 opposite-side 支撑。**

建议加一个硬门槛：

* 只有当样本所在 sector（或邻域 sector）出现 mixed evidence，或 `OppSupport > τ` 时，这个样本才允许进入 MLP-guided 选解候选集
* 否则，即便 `margin` 很小，也先不让它主导选解

原因很简单：
小 margin 可能是真边界，也可能只是模型幻觉。
你现有代码只有 soft ranking，没有 hard gate，所以很容易被幻觉边界带跑。

### 3.7 Boundary offspring 的接受规则也要加“支撑条件”

当前 `GenerateBoundaryOffspring` 只要求：

* `Prob >= 0.5` 来给 child 预测可行/不可行
* `CandidateMeta.margin < AnchorMargin` 才接受 child。

这还不够。

应改成：

* `margin` 变小
* `OppSupport` 不下降，最好上升
* objective relevance 不能明显恶化

也就是说，child 不是“更不确定”就行，而是要“**更像真实边界，并且还在有目标意义的区域里**”。

这一步非常关键，因为它正好把你的创新点和 NA-EMT 这类“概率替代 CV”方法拉开：
NA-EMT 本质上是把“越像可行”当成“越有价值”；而你的方法应该是把“越接近真实边界、越靠近 PF 有意义区域”当成“越有价值”。

---

## 4. 直接落到你这份代码上，最该改哪几个函数

按优先级，我建议你这么动：

### 第一刀：直接替换训练集构造

把：

* `CollectLocalBoundaryPairs`
* `BuildBoundaryDataset`

整体替换成：

* `UpdateTrainingBuffer`
* `CollectSectorBoundaryBand`
* `BuildBoundaryDatasetFromBuffer`

也就是：

* 不再返回 `AnchorRows/PairRows`
* 改成返回一个 sector-balanced 的 boundary-band 数据集

### 第二刀：改训练函数接口

把：

* `Model = TrainBoundaryMLP(Dataset,hidden,epoch,lr)`

改成：

* `Model = TrainBoundaryMLP(Model,Dataset,hidden,epoch,lr,mode)`

其中 `mode ∈ {cold, finetune}`。
默认 `finetune`，只在极端漂移时 `cold`。

### 第三刀：改重训触发

把 `ShouldRetrainBoundaryModel` 从 “pair-ready + FE interval”
改成 “buffer-change + mixed-sector expansion + boundary validation drop”。

### 第四刀：改选解硬门槛

在：

* `GenerateBoundaryOffspring`
* `SelectInfeasibleByBoundaryMeta`

里加入 `OppSupport` / mixed-sector gate，防止 margin 幻觉。

---

## 5. 我对你这版的最终判断

你的核心创新点是成立的，而且方向比“概率替代 CV”更强：

* 你不是在学“可行概率”
* 你是在学“对 PF 搜索有意义的可行/不可行边界” 

但现在这版代码，**实现还停留在“扇区代表点二分类”**，没有真正进入“边界带学习”。这就是为什么 MLP 会显得不准：
不是网络不够大，不是 loss 不够花，而是**你喂给它的根本不是你想让它学的对象**。

最重要的一句话是：

> **先别改 loss，先把训练数据从“固定 pair”改成“自适应边界带”，再把 retrain 从“冷启动重训”改成“滚动 warm-start”，最后给 model-guided 选解加上真实 opposite-side 支撑门槛。**

这三刀下去，你的主线不变，但算法会第一次真正和你的创新叙事对齐。

[1]: https://proceedings.mlr.press/v162/raj22a/raj22a.pdf "https://proceedings.mlr.press/v162/raj22a/raj22a.pdf"
[2]: https://proceedings.neurips.cc/paper_files/paper/2023/file/6ea69f8116b7c01e3c3e43b62e6868fc-Paper-Conference.pdf "https://proceedings.neurips.cc/paper_files/paper/2023/file/6ea69f8116b7c01e3c3e43b62e6868fc-Paper-Conference.pdf"
