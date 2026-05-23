根据你 **2026-05-23** 的最新 bundle，当前版本已经把“**AF 留不下来、GAN 经常根本训不起来**”这一层问题大体修掉了：soft rebuild 后，`AF_retention_ratio` 从 `0.354` 提到 `0.587`，`refresh_count` 提到 `25.59`，`zero_refresh_runs` 从 `30/152` 降到 `2/152`；但同时 `Raw_BHR_cond` 的**均值**虽然上升，**中位数仍然是 0**。这说明瓶颈已经从“有没有边界训练数据”转移到了“**GAN 看到的 AF 到底是不是‘边界可行带’，以及 GAN 是否真的学会把它保留下来**”。

## 1. 当前真正阻碍 GAN 直接生成边界可行样本的问题

我给出的主判断是：

> **当前的主瓶颈不是 AF 数量，而是 AF 的边界语义仍被稀释；随后，标准 GAN 只是在拟合“AF 的总体可行分布”，而不是“AF 中那条很薄的边界可行带”。**

原因分两层。

第一层是 **AF 语义还不够纯**。
虽然 soft rebuild 去掉了硬 `FirstFront` 截断，但在 pair 形成阶段，`CollectCrossPairs_BDG` 仍然先把可行点 `FA` 过滤到 `FirstFront_BDG(FA.objs)`，然后每个参考向量桶里再按 `sum(YF)` 只取前 2 个可行点去找异构近邻。这仍然是**先按收敛性筛，再按边界性配对**，而不是纯粹的边界优先。后面的 `RebuildArchive_BDG` 虽然改成了 soft score，但 `score` 里仍然保留 `conv` 和 `front_rank_penalty`，而 `SelfEnhancePairs_BDG` 又继续按这个 `AF.score` 去挑最优样本反复精修，所以“收敛偏置”还在自增强。  

第二层是 **GAN 学习目标里没有“边界性”这个信号**。
当前代码在 checkpoint 直接用 `AF.decs` 训练标准 GAN；如果已有模型就继续 warm-start，而不是重置。随后 raw 采样时，生成器直接吐出 `nGen` 个样本并真实评估；这里 `AI` 在 raw 生成阶段根本没有参与，函数签名里就是 `SampleBoundaryGAN_BDG(Problem,GAN,AF,~,W,...)`，`AI` 只在 repair/metric 之外被忽略。也就是说，虽然你的方法名义上是 **AF/AI 双边界档案**，但在“直接 raw 生成”这一环，真正进入生成机制的只有 `AF`，而且只是把它当作一个整体可行分布来学。 

这和相关文献的经验是一致的：在 binary/unknown constraints 下，**边界附近往往最有价值，但前提是边界被高精度地建模出来**；如果训练集语义混杂，模型更容易学到“可行区域”而不是“可行边界”。GAN 类方法也一向对“real 样本”的质量高度敏感，训练数据的**收敛性和多样性**会直接决定生成样本的性质。([arXiv][1])

## 2. 这个问题主要来自哪里

我的排序是：

### 主因：AF/AI 的边界语义与训练集质量

这是第一主因，不是指标，也不是单纯网络太小。

最有力的证据是你最新 bundle 里的 `LIRCMOP14_BC run=5`。在 soft rebuild 版本里，这个 run 从 `gen=40` 开始就已经有很大的 AF：`AF_size=61`，后续上升到 `85, 90, 91`；`Ref_cov` 也从 `0.67` 升到接近 `1.0`。但在这些 checkpoint 上，`Raw_feasible_count` 基本是 `19/20` 或 `20/20`，`Raw_hit_count` 却始终是 `0`，`Raw_BHR_cond=0`，同时 `Raw_minAI_dist_med` 长期很大。也就是说：**GAN 不是不会生可行点，而是生出来的是“可行内部点”，不是“边界可行点”**。这几乎直接把“只是刷新不够”这个解释排除了。

### 次因：GAN 学习过程与采样过程

GAN 本身不是第一主因，但现在的学习/采样方式放大了问题：

* 它训练的是**瞬时 AF**，分布随世代漂移；
* 它 warm-start 旧网络，容易把早期偏差带到后期；
* 它每次只 raw 采 `20` 个点，没有任何边界导向筛选；
* `AI` 不参与 raw 生成阶段。

所以一旦 AF 里“边界可行带”只是少数模式，标准 GAN 很容易学成“总体 AF 分布”，而不是那条薄边界。

### 不是主因：Raw_BHR 指标定义

`Raw_BHR_cond` 的定义是：在 raw 可行点里，看有多少点满足“到最近 AI 的归一化目标距离 ≤ 当前 `Gap_obj_med`”。源码里这个逻辑是明确的。

这个指标确实偏严格，但它**不是当前主问题**。因为在上面的 `LIRCMOP14_BC run=5` 中，raw 点已经大量可行，但 `Raw_minAI_dist_med` 一直明显偏大，不是“差一点没过阈值”，而是整体就离 AI 边界不近。所以这里不是指标把结果“压成了 0”，而是 raw 样本确实没有贴边。

## 3. 你们当前诊断还没完全覆盖到的隐藏问题

我认为还有三个。

### 隐藏问题 A：AF/AI 双档案的“AI 半边”没有真正进入 raw 生成机制

这点很关键。
你们的论文叙述是“AF/AI 共同刻画局部边界”，但当前实现里，AI 在 raw 生成阶段并不参与：它不进训练，也不进 raw 采样筛选，只在 repair 和 metric 里发挥作用。于是“AF/AI 双边界档案”在核心创新点里，实际上只落成了“AF 训练 + AI 评估/修复”。这使得你的实现比你的创新表述**弱了一半**。

### 隐藏问题 B：SelfEnhance 在放大当前 score 的偏置

`SelfEnhancePairs_BDG` 每轮都按 `AF.score` 选前 10 个样本再精修。因为现在的 `AF.score` 仍含 `conv` 和 `rankPenalty`，所以 self-enhance 不是中性的“缩边界”，而是在**持续放大当前 score 偏好的语义**。 

### 隐藏问题 C：GAN 在追一个非平稳小分布

这点和 NA-EMT 一类 unknown-constraint 工作的经验相符：当搜索区域漂移时，早期学到的模型可能不再适合后期。你这里虽然没显式建分类器，但 warm-start 的 `netG/netD` 正在追一个不断改写的小型 AF，这很容易让生成器学成“过去若干 checkpoint 的混合分布”，而不是当前局部边界。

## 4. 在不改变核心创新思想前提下，应该如何找原因

我建议只做**一个**最小对照实验，不要同时改很多地方。

### 下一步唯一应该做的最小受控实验

**AF-oracle 对照实验**

做法：

* 保持当前 soft rebuild 版本完全不动；
* 保持 `AF/AI`、`trainGap`、`minTrain`、`repairK`、`nGen`、metrics 全不动；
* 只在 raw 生成处做一个替换：

  * 不用 GAN 生成 raw；
  * 改为从当前 `AF` 中均匀抽样，再在归一化决策空间上加一个很小的高斯扰动，`σ = 0.25 * Gap_dec_med`，裁剪到边界；
  * 得到 `20` 个 raw 样本，**先不 repair**，只记 raw 指标。

比较三项：

* `Raw_BHR_cond`
* `Raw_minAI_dist_med`
* `Raw_FR`

#### 这个实验的解释力非常强

* 如果 **AF-oracle 明显好于 GAN raw**：
  那说明当前 AF 的边界语义已经足够，剩余瓶颈主要在 **GAN 学习/采样**。
* 如果 **AF-oracle 也几乎没有边界命中**：
  那说明剩余瓶颈还在 **AF 语义本身**，不是 GAN 先坏。

这个实验的优点是：
它完全不改你的核心思想，也不引入新模型；只是把“GAN 这一步”换成一个**AF 上界对照**。它会非常干净地告诉你：问题到底在 archive，还是在 generator。

## 5. 核心创新点现在是否成立

我的判断是：

> **目前还不能说这个创新点已经被验证成立，但也不能说它被证伪了。**

更准确地说：

* 你已经证明了：
  **AF/AI 局部边界档案是可以被构建并稳定维护的。**
* 你还没有证明：
  **只用 AF 训练的标准 GAN 可以稳定直接生成边界附近可行样本。**

也就是说，当前证据支持的是**“边界档案”这半条创新链**，还不支持**“直接边界生成”这半条创新链**。

## 6. 基于当前版本代码的最小修正方案

在做完上面的 AF-oracle 实验之后，我建议的**统一、减法式**修正版只有三处改动。

### 改动 1：把 `CollectCrossPairs_BDG` 的前置收敛过滤去掉

当前最伤的是这里：

* `FF = FirstFront_BDG(FA.objs); FA = FA(FF);`
* 每个 reference bucket 再只取 `top2` 收敛最好的 feasible 点

建议：

* 直接删除这两层预筛；
* 改成：在每个 reference bucket 内，对所有 feasible 候选按“最近 infeasible 距离”排序，再取前若干个去 refine。

含义很简单：

* pair 形成阶段只做“边界性”判断；
* 不要在 pair 形成前先把它变成“收敛性”问题。

### 改动 2：让 `RebuildArchive_BDG` 更纯粹地 boundary-first

现在 score 还是偏混合。建议改成：

[
score = 0.45 \cdot gap_obj + 0.45 \cdot gap_dec + 0.10 \cdot conv
]

并且把 `front_rank_penalty` 直接去掉。

理由：

* 你的目标是学边界，不是学前沿内部最优可行集；
* `conv` 仍保留 10% 作为 tie-break 即可；
* 这是对 soft rebuild 的进一步净化，不是重构。

### 改动 3：把 GAN 训练集从“瞬时 AF”改成“小型 recent-AF FIFO”

不改 GAN 结构，不改标准 GAN，只改它吃的数据。

做法：

* 增加一个 `AF_train_bank`；
* 每个 checkpoint 从当前 AF 中按 reference bucket 取最贴边的 1~2 个可行点入 bank；
* bank 用 FIFO 保留最近 5 个 checkpoint，总量控制在 100 左右；
* `BoundaryGAN_BDG('train',...)` 改为吃 `AF_train_bank.decs`，不再直接吃瞬时 `AF.decs`。

这一步很小，但作用很大：

* 它保持“GAN 只用 AF 训练”不变；
* 又能抑制分布漂移和 warm-start 的历史污染。

### 同时做一个减法

把 `SelfEnhancePairs_BDG` 从“每代执行”改成“只在 checkpoint 执行”，并把 `numE` 从 `10` 降到 `5`。
原因不是它没用，而是你现在更需要**停止强化错误语义**，并节省 FE。

---

## 一句话版本

现在真正卡住核心创新点的，不是 AF 不够多，而是：

> **AF 仍然不是一条足够纯的“边界可行带”，而标准 GAN 又只会去拟合它看到的总体 AF 分布。**

所以你的下一步不该是继续调 `nearTau`、`minTrain` 或加大网络，而是：

> **先用一个 AF-oracle 对照实验把“archive 问题”和“GAN 问题”切开；然后只做三处最小修改：去掉 pair 阶段的收敛预筛、把 rebuild 进一步 boundary-first、把 GAN 训练集改成 recent-AF FIFO。**

这条路线最符合你现在的目标，也最符合“统一、减法、收敛”。
相关 unknown/binary-constraint 文献对这条判断是支持的：边界附近往往是最有价值的区域，但前提是边界被准确建模；而 GAN 类方法的效果又高度依赖训练数据的质量与分布。([arXiv][1])

[1]: https://arxiv.org/abs/2402.07692?utm_source=chatgpt.com "Boundary Exploration for Bayesian Optimization With Unknown Physical Constraints"
