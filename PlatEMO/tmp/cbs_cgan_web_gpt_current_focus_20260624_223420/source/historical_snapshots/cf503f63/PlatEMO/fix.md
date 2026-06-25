根据一份 **2026-05-23** 的最新 bundle，你们现在已经把“档案留不住、刷新不起来”这个前置问题基本解决了：soft rebuild 之后，`AF_retention_ratio` 从 `0.354 -> 0.587`，`refresh_count` 从 `17.55 -> 25.59`，`zero_refresh_runs` 从 `30 -> 2`；但与此同时，`Raw_BHR_cond` 的均值虽然从 `0.00315 -> 0.01995` 上升，**中位数仍然是 0**，而且只有 `48/152` 个 run 改善。也就是说，当前真正没解决的，不再是“GAN 有没有机会训练”，而是“GAN 学到的到底是不是边界语义”。 

我的结论先给出来：

**当前结果还不能说明核心创新点被否定；但它已经说明：仅靠 soft rebuild 修好 AF 留存，然后继续“用整个 AF 直接训练标准 GAN”，不足以稳定实现‘raw 直接命中边界附近可行解’。**
残余瓶颈更像是：**训练阶段吃进去的 AF，边界纯度仍然不够；而 AI 中携带的“边界紧致性”信息，在进入 GAN 之前被丢掉了。**

---

## 1. 当前真正阻碍 GAN 直接生成边界可行样本的问题是什么

不是 pair 不够，也不是 refresh 不够，而是：

> **GAN 训练时看到的是“扩散后的可行档案 AF”，不是“高纯度的边界可行壳层”。**

这条判断有两层证据。

第一层是你自己的总体数据。
在 `bisectK=2, nearTau=0.20` 的 first diagnostic 里，`mean_Raw_FR=0.465`、`median_Raw_FR≈0.476`，说明 raw 样本里大约一半本来就是可行的；但 `median_Raw_BHR_cond=0`。这已经说明问题并不主要是“GAN 只会生成不可行点”，而是**它经常生成可行但不贴边的点**。soft rebuild 之后，`mean_Raw_FR` 又从 `0.465 -> 0.500`，但 `Raw_BHR_cond` 的中位数还是 0，这个判断更强了：**更多 AF、更多 refresh，只让 raw 更容易变可行，没有让 raw 更容易变边界可行。** 

第二层是当前源码逻辑。
在最新版本里，soft rebuild 改了 `RebuildArchive_BDG` 的留存策略，但 pair sources、GAN、repair、`minTrain`、`nearTau` 都没改；更关键的是，**GAN 训练入口仍然直接吃整个 `AF.decs`**，而不是一个按边界紧致性再筛过的 AF 子集。也就是说，AF/AI 在“建档阶段”是双档案，但一旦进入“学习阶段”，AI 的约束侧信息就不再直接参与训练样本选择了。

所以，当前真正的阻碍不是“GAN 学不会”，而是：

> **你让 GAN 学的是“当前阶段的一团可行分布”，但你希望它输出的是“当前阶段的一层边界可行壳层”。**

这两者不是一回事。

这一点和已有研究的经验是一致的。BE-CBO 明确指出，在 unknown/binary constraints 下，若最优点靠近可行/不可行边界，那么决定效果的关键是**边界建模准确性**，而不是只增加边界探索次数；他们甚至专门用更强的约束模型去提升边界拟合质量。([arXiv][1])
GMOEA 也明确强调，GAN 驱动的进化算法效果高度依赖训练数据质量，而“real samples”既要有**收敛性**，也要有**分布代表性**。([arXiv][2]) 

---

## 2. 这个问题主要来自哪里

如果必须在你给的几类原因里排序，我的判断是：

### 第一主因：训练集质量 / AF 的边界语义纯度

这是主因。

soft rebuild 解决的是“AF 能留下多少”，不是“AF 里留下来的是什么”。
从结果看，AF 留存和 refresh 已经明显改善，但 raw 仍不贴边，这说明：

* 现在的 AF 已经**够多**；
* 但 AF 仍然不是一个足够干净的“边界可行训练集”。

更准确地说，现在的 AF 更像：

> **边界可行点 + 较优但偏内侧的可行点** 的混合体。

而标准 GAN 没有额外边界监督时，天然会向**更大体积、更稳定、更容易拟合**的那部分分布靠拢。对纯可行训练集来说，这通常就是**可行域内部**，不是薄薄的边界带。

### 第二主因：当前 GAN 学习过程对“边界”没有显式压力

这不是说网络太弱，而是说目标不对。

你现在的标准 GAN 目标只是在 AF 上做分布拟合。
如果 AF 本身不是高纯度边界壳层，那么 GAN 最优解就是去拟合 AF 的主质量区域，而不是拟合“靠近 AI 的那一面”。这和你当前 core idea 的一个实现细节冲突了：

* 你在概念上说的是 **AF/AI 双边界档案刻画边界**；
* 但在学习上实际做的是 **只把 AF 当作普通正样本分布来学**。

这就导致 AI 的意义被压缩到了：

* 建档时的 pair / score；
* 诊断时的 Raw_BHR；
* 修复时的辅助信息。

它没有真正进入“让 GAN 学边界”的那一步。

### 第三主因：`Raw_BHR_cond` 指标定义有噪声，但不是主问题

这是次因，不是主因。

现在的 `Raw_BHR_cond` 本质是：

* 先在 raw 中取可行样本；
* 再看这些可行样本到 AI 的目标空间最小距离，是否小于一个**全局** `Gap_obj_med` 阈值；
* 命中比例记为 `Raw_BHR_cond`。

这个定义有两个天然偏差：

1. 它是**全局阈值**，而你的问题是**局部边界**；
2. 它只看**目标空间**，没有看决策空间局部配对。

所以它可能会低估一部分“确实靠近某个局部边界，但没过全局阈值”的样本。

但我不认为这是当前主因。原因很简单：
如果指标偏差是主因，那么 soft rebuild 之后你应该至少看到 `Raw_FR`、`Raw_minAI_dist` 之类指标和 `Raw_BHR_cond` 出现更一致的改善；可你看到的是 **Raw_FR 上来了，Raw_BHR_cond 中位数依旧 0**。这更像“raw 的确更可行了，但仍没有贴近 AI 所代表的边界”，而不只是指标误杀。 

---

## 3. 还有哪些当前诊断没覆盖到的隐藏问题

我认为有三个隐藏问题。

### 隐藏问题 A：你们现在验证的，其实不是原始创新点本身

你们原始创新点是：

> AF/AI 双边界档案刻画局部边界，且只用 AF 训练 GAN，让 GAN 直接生成边界附近可行解。

但当前实现真正测试的是：

> **把整个 AF 当成普通可行训练集，标准 GAN 能不能自己从里面“悟出”边界。**

这两句话差别很大。
前者要求 AF 是**边界语义高纯度**的；后者只要求 AF 是可行的。
因此，当前结果更准确地说是在否定下面这个更弱、更粗糙的命题：

> “只要 AF 留存够多，标准 GAN 训练在整块 AF 上，就会自然生成边界可行解。”

这个命题现在显然不成立。
但它**不等于**你的核心创新点本身就不成立。

### 隐藏问题 B：`Raw_BHR_cond` 会高估“小样本侥幸命中”

`Raw_BHR_cond = hit / raw_feasible_count`。
如果某次 refresh 只有很少几个 raw feasible 点，其中恰好有 1 个靠边，那么 `Raw_BHR_cond` 也可能看起来不错。
所以对于“核心创新点是否成立”的验证，**主指标不该是 `Raw_BHR_cond`，而应该是 `Raw_BHR` 或 `Raw_hit_count`**。
你现在已经发现 `Raw_BHR_cond` 的中位数是 0，这已经很严厉了；但下一步实验更应该把 `Raw_BHR` 作为第一主指标，`Raw_BHR_cond` 作为辅助指标。

### 隐藏问题 C：问题难度差异会掩盖全局统计

BE-CBO 明确提醒：边界最优不是一条普适公理，某些问题的最优会在可行域内部；他们还专门构造了 interior-optimum 的 shifted synthetic tests 来检查这一点。([arXiv][1])
但你这里的主结论并不依赖这个争议，因为你当前失败不是出现在少数特殊问题，而是**在 soft rebuild 之后的总体中位数仍为 0**。这说明问题首先是实现层面的，而不是“是不是所有问题都边界最优”这个理论争论。

---

## 4. 在不改变核心创新思想前提下，应该如何找原因、如何最小修正

我建议你**只做一个最小修正方向**，而且只改训练入口，不改 pair，不改 rebuild，不改 GAN 结构，不改 repair。

### 最小修正原则

保留这些不动：

* AF/AI 双边界档案思想不动；
* 仍然**只用 AF 训练 GAN**；
* 标准 GAN 不变；
* pair source 不变；
* soft rebuild 不变；
* `bisectK=2, nearTau=0.20, trainGap=20, minTrain=20, ganIter=80` 不变；
* sampling / repair 不变。

只改一件事：

> **GAN 不再训练整个 AF，而是训练一个由 AF/AI 共同定义出来的、更高纯度的 AF 子集 `AF_core`。**

注意，这并没有违背你的核心创新点，因为：

* 训练样本仍然是 **AF** 里的可行点；
* AI 只用于**筛选 AF 中哪些点更像“边界可行点”**；
* AI 不直接喂给 GAN。

### 为什么这是最小而正确的修正

因为它精确命中了当前残余瓶颈：

* 不是 archive 数量不够；
* 不是 refresh 不够；
* 不是 pair 不够；
* 不是网络太小；
* 而是 **训练集边界纯度不够**。

你现在最该验证的，不是更大的模型，而是这句：

> **“如果把 AF 变成更高纯度的边界可行训练集，标准 GAN 是否就能显著提高 raw 边界命中？”**

如果答案是“能”，核心创新点就站住了。
如果答案还是“不能”，那才轮到怀疑“标准 GAN + 只用 AF”的这条严格路线本身不够。

---

## 5. 有证据链的原因分析 + 下一步最小实验设计

### 证据链

证据链其实已经很完整了：

1. `bisectK=2` 和 soft rebuild 已经把“训练机会不足”基本缓解。
   retention、refresh、zero-refresh 全都明显改善。

2. 但 `Raw_BHR_cond` 中位数仍是 0。
   所以“AF 留不住”不是残余主因。

3. 同时 `Raw_FR` 并不低。
   first diagnostic 的 `mean_Raw_FR≈0.465`，soft rebuild 后 `mean_Raw_FR≈0.500`。说明 GAN raw 输出里并不缺可行样本；缺的是**边界样本**。 

4. 当前源码训练仍然直接使用全 AF。
   这说明 residual bottleneck 更像“训练集语义不对”，而不是“GAN 根本没机会训练”。

5. 文献也支持：

   * unknown/binary constraints 下，边界建模精度是关键；([arXiv][1])
   * GAN 在进化优化里高度依赖训练数据质量与代表性。([arXiv][2])

### 下一步只做一个最小控制实验

我建议下一步只做一个实验，名字可以叫：

**E1: AF-core training**

#### 实验目的

验证 residual bottleneck 是否就是 **AF 训练集纯度**。

#### 对照组

当前最新 soft rebuild 版本。

#### 实验组

只改 GAN 训练集：

* 从当前 AF/AI 中，为每个配对样本计算边界紧致性
  [
  tight_i = ranknorm(gap_obj_i) + ranknorm(gap_dec_i)
  ]
  越小越靠近边界。

* 按参考向量分桶；

* 每个桶只保留最紧的前 `50%`，或最紧的前 `k=2` 个 AF 点；

* 把这些点组成 `AF_core`；

* GAN 训练从
  `train(AF.decs)`
  改成
  `train(AF_core.decs)`。

其他所有东西都不动。

#### 我建议记录的新量

只增加 3 个诊断量：

* `AF_core_size`
* `AF_core_gap_obj_med`
* `AF_core_gap_dec_med`

#### 成功判据

这一步不要先看 HV/IGD，先看核心创新点是否更成立：

1. **主指标：`Raw_BHR = Raw_hit_count / Raw_count`**
2. 辅指标：`Raw_hit_count`
3. 次指标：`Raw_BHR_cond`
4. 约束：`refresh_count` 不应显著下降

#### 你应该如何解读结果

* 如果 `Raw_BHR`、`Raw_hit_count` 明显提升，而 `refresh_count` 基本不变：
  说明 residual bottleneck 就是 **AF 训练集语义 dilution**。
  也就是说，**核心创新点成立，当前实现只是没有把 AF 变成真正的边界训练集。**

* 如果 `AF_core` 之后 `Raw_BHR` 仍然接近 0：
  那下一步才该怀疑更深一层的问题——
  **“只用 AF 正样本、标准 GAN、无显式边界压力” 这条学习路线本身不够。**

这时才轮到下一层修改；但现在还不应该走到那一步。

---

## 6. 修正后的算法思路，以及在当前版本基础上怎么改

### 修正后的算法思路

我建议把当前算法的表述从：

> AF/AI 建边界，GAN 用 AF 训练并直接生成边界可行样本

收紧成更准确、也更容易成立的版本：

> **AF/AI 先定义“局部边界紧致度”，再从 AF 中抽取高纯度边界可行子集 `AF_core`；标准 GAN 只在 `AF_core` 上训练，从而直接生成局部边界附近的可行样本。**

这仍然是你的原创新点，只是把“AF 训练”从**整块 AF**改成**高纯度 AF 子集**。

### 在当前版本基础上的修改方式

只动一处主链：

#### 当前

```matlab
GAN = BoundaryGAN_BDG('train', AF.decs, lower, upper, zDim, ganIter, GAN);
```

#### 修改后

```matlab
AF_core = SelectAFTrainCore_BDG(AF, AI, W);
GAN = BoundaryGAN_BDG('train', AF_core.decs, lower, upper, zDim, ganIter, GAN);
```

### 新增函数 `SelectAFTrainCore_BDG`

功能非常简单：

1. 输入 `AF, AI, W`
2. 计算每个 AF/AI pair 的：

   * `gap_obj_i`
   * `gap_dec_i`
3. 计算 `tight_i`
4. 参考向量分桶
5. 每桶保留最紧的一小部分
6. 输出 `AF_core`

### 明确不要改的东西

为了保持“统一、减法、收敛”，这一轮不要动：

* `CollectFlipPairs_BDG`
* `CollectCrossPairs_BDG`
* `RebuildArchive_BDG`
* `bisectK`
* `nearTau`
* `GAN` 结构
* `ganIter`
* `repair`
* `nGen`
* `minTrain`

你现在要的不是更复杂，而是更干净地回答一个问题：

> **问题到底出在“AF 不够多”，还是“AF 不够边界”？**

现在证据已经足够表明：
前者基本修好了，后者还没有。

---

## 最后一条判断

如果只围绕“核心创新点是否成立”来回答，我的判断是：

**目前还不能说这个创新点不成立；但当前实现只证明了一件事：
“soft rebuild 让 AF 更稳定”成立，
“整块 AF 直接训练标准 GAN 就能稳定生成边界可行 raw 样本”不成立。**

所以，下一步最该做的不是重构算法，而是做一个最小、干净、单变量的验证：

> **把 GAN 的训练集从“全部 AF”改成“由 AF/AI 定义的高纯度 AF_core”，然后只看 Raw_BHR / Raw_hit_count 是否上升。**

这一步如果成功，你的创新点就真正开始站住。
这一步如果失败，再去讨论更深层的学习机制问题，才是合理顺序。

[1]: https://arxiv.org/abs/2402.07692?utm_source=chatgpt.com "Boundary Exploration for Bayesian Optimization With Unknown Physical Constraints"
[2]: https://arxiv.org/abs/1910.04966?utm_source=chatgpt.com "Evolutionary Multiobjective Optimization Driven by Generative Adversarial Networks (GANs)"
