## 结论

当前结果**不能证明核心创新点已经成立**，但也**不能直接否定核心创新点**。更准确的判断是：

> **AF/AI 双边界档案的“可构建性”已经基本成立；但“只用 AF 训练标准 GAN 后，GAN raw 样本能稳定命中边界可行带”尚未成立。**

soft rebuild 后，`AF_retention_ratio` 从 0.354 提高到 0.587，`refresh_count` 从 17.55 提高到 25.59，`zero_refresh_runs` 从 30 降到 2，说明原先“AF 留不住、GAN 无法训练”的问题已经显著缓解。与此同时，`Raw_FR` 也从 0.465 提到 0.500，修复率下降，说明 GAN 生成可行样本的能力略有提升。可是 `Raw_BHR_cond` 的中位数仍为 0，且只在 48/152 个 run 中改善，这说明剩余瓶颈已经从**档案规模/刷新机会**转移到**AF 到 GAN raw 边界命中的分布迁移失败**。 

这和 `nearTau` 扫描的结论一致：`nearTau=0.30` 只让 pair 数和 FE 开销略增，但 `Raw_BHR_med` 仍为 0。因此，问题不再是“多找一些 pair”就能解决，而是“找到并保留下来的 AF 是否真的是 GAN 可学习的边界可行训练带”。

---

## 1. 当前真正阻碍 GAN 直接生成边界可行样本的问题

当前真正的问题不是单纯的 GAN 参数，也不是单纯的 AF 数量，而是：

> **训练给 GAN 的 `AF.decs` 仍然不是一个足够“薄、纯、局部一致”的边界可行分布。**

soft rebuild 解决的是“AF 是否能留下来”，但没有证明“留下来的全部 AF 都适合直接作为 GAN 训练集”。

从你的代码和结果看，当前 GAN 在每个 `trainGap` 处直接执行：

```matlab
GAN = BoundaryGAN_BDG('train',AF.decs,...)
```

也就是说，**整个 AF 都被当作真实边界分布**。但 soft rebuild 后的 AF 是“边界候选档案”，不是“边界训练带”。二者不能等同。

证据有三条。

第一，soft rebuild 后 `Raw_FR` 明显改善，但 `Raw_BHR_cond` 中位数仍为 0。也就是说，GAN 更会生成可行解了，但没有更会生成边界可行解。这个现象通常意味着生成分布偏向**可行域内部**，而不是贴着可行/不可行交界。

第二，一些 run 中 AF 已经较大、刷新频繁、Ref 覆盖也不低，但 raw 样本仍然 `Raw_BHR=0`。例如 `LIRCMOP13_BC run8` 中，AF 在第 40 代后已经达到 66 以上，且多次 `Raw_FR=1`，但 `Raw_hit_count` 长期为 0，`Raw_minAI_dist_med` 明显大于边界 gap。这不是“GAN 完全不会生成可行解”，而是“GAN 生成的可行解离 AI 边界侧太远”。

第三，GMOEA 类 GAN-MOEA 工作明确指出，GAN 能否有效生成 offspring，关键取决于 real samples 的分布能否同时表达目标分布的多样性和收敛性；换到你这个问题里，就是 AF 必须真正表达“边界可行带”，而不是只表达“一批可行好解”。

因此，当前最大阻碍不是“GAN 不能用于这个方向”，而是：

> **当前训练集 AF 的边界语义不足以强迫标准 GAN 学出边界。**

---

## 2. 问题主要来自哪里

### 2.1 AF/AI 边界语义：已经改善，但仍不够“训练化”

soft rebuild 后 AF/AI 的留存和刷新明显改善，这说明 AF/AI 作为边界档案的构造方向是有效的。

但当前 AF 仍有两个语义问题。

第一，**pair 形成阶段仍然带有收敛/前沿偏置**。虽然 rebuild 已经 soft，但 cross-pair 的形成逻辑里仍然倾向先找可行前沿点，再按目标空间方向取少数代表点。这会让 AF 偏向“当前较好的可行点”，而不是“最贴近可行/不可行分界的点”。

第二，**AF 的用途混合了两个目标**：

* 算法档案目标：保持当前有价值的边界候选，兼顾收敛、多样性、局部边界；
* GAN 训练目标：提供尽可能薄、纯、局部一致的边界可行样本分布。

当前直接用全量 `AF.decs` 训练 GAN，等价于把这两个目标混在一起。soft rebuild 后 AF 变大，反而可能把更多“边界附近但不够贴边”的样本送进 GAN。

这解释了为什么 retention 和 refresh 上升，但 Raw_BHR_cond 没有同步上升。

### 2.2 训练集质量：这是当前第一主因

当前 GAN 的 real samples 只有 AF。这个设计符合核心创新点，但前提是 AF 本身必须是**边界训练带**。

现在的问题是：AF 是“保留档案”，不是“训练子集”。一个档案为了多样性和阶段性搜索可以宽一点；但 GAN 一旦学习这个宽分布，就会生成这个宽分布的密度中心。对于单侧可行样本，密度中心通常更靠可行内部，而不是边界。

所以，当前剩余问题的第一主因是：

> **训练集没有从 AF 中进一步筛出“最贴边 AF 子集”。**

这不是增加新模块，而是把“档案”和“训练集”解耦：
AF 仍然是 AF，但 GAN 不应无差别吃掉全部 AF。

### 2.3 GAN 学习过程：是第二主因

标准 GAN 的目标只是让生成分布逼近 real distribution。它不会自动知道“边界”是什么。BE-CBO 的结论也说明，在 unknown/binary constraint 场景下，若目标通常靠近可行/不可行边界，那么边界建模精度非常关键；只要边界刻画不准，搜索就会偏离边界。 

在你的算法里，GAN 没有使用 AI，不知道 AF 到 AI 的距离，也没有边界距离损失。由于硬约束要求使用标准 GAN，这本身可以接受，但它意味着：

> **边界压力必须完全来自 AF 训练集本身。**

如果 AF 是宽的、混合的、多模态的，标准 GAN 只会学习这个混合分布，不会自动贴边。

此外，当前训练过程还有一个潜在不稳定点：每次训练复用网络权重，但 Adam 动量状态在训练函数内重新初始化。这不一定是主因，但它可能让在线微调不稳定。现在不建议先改这一点，因为它会把诊断变复杂；但应记录 `lossD/lossG/D(real)/D(fake)`，判断是否存在明显欠拟合、过拟合或塌缩。

### 2.4 Raw_BHR 指标：可能偏严，但不是主因

`Raw_BHR_cond` 中位数为 0 可能有一部分来自指标严格性，因为它要求 raw feasible 样本在目标空间上足够接近 AI 边界侧。

但当前数据中存在大量“Raw_FR 高、Raw_BHR 仍为 0”的情况，而且 `Raw_minAI_dist_med` 往往明显大于当前边界 gap。这说明不是“差一点没过阈值”，而是 raw 样本确实没有贴边。

所以指标需要补充诊断，但不应把当前失败主要归因于指标。

建议以后保留严格版 `Raw_BHR_cond@1x`，同时新增：

* `Raw_BHR_cond@2x`
* `Raw_BHR_cond@3x`
* `Raw_minAI_dist_dec`
* `Raw_minAF_dist_dec`

如果 `@2x/@3x` 明显升高，说明是近边界但阈值过严；如果仍然接近 0，则说明 raw 样本确实远离边界。

### 2.5 源码中还存在的隐藏问题

当前还有四个没有完全覆盖的隐藏问题。

第一，**全 AF 训练会稀释边界性**。
这是当前最重要的隐藏问题。soft rebuild 后 AF 越大，越可能包含不同厚度、不同局部阶段、不同参考向量区域的样本。直接全量训练会把 GAN 推向“平均可行分布”。

第二，**目标空间 BHR 与决策空间训练之间存在度量错位**。
GAN 学的是决策变量分布，但 Raw_BHR 主要在目标空间衡量 raw 到 AI 的距离。由于多对一映射存在，目标空间近邻和决策空间近邻并不总一致。因此必须同时记录 decision-space 距离，否则无法判断是“生成点离边界远”，还是“目标空间指标误判”。

第三，**self-enhancement 可能放大已有偏置**。
当前 self-enhancement 按 AF score 选前若干样本反复二分，这会强化当前 score 偏好的边界片段。如果 score 中仍含收敛项和 rank 项，它可能强化“好可行片段”，而不是“最贴边片段”。

第四，**raw_count=20 对稀有 hit 的估计不稳定**。
20 个 raw 样本太少，可能低估 BHR；但这不能解释全局中位数长期为 0。它只说明下一步诊断时应设置一个不注入种群的 `probeRawCount=200`，用于估计真实 raw boundary hit 能力。

---

## 3. 核心创新点现在应如何判断

可以拆成两部分判断。

### 3.1 AF/AI 双档案刻画局部边界：基本成立

soft rebuild 后，AF 留存、刷新机会、zero-refresh 都显著改善。说明“用 AF/AI 双档案维护可行/不可行局部边界”这部分是有工程可行性的。

这也符合 binary constraint 文献的方向。DRMCMO 说明，在 CV 弱化或不可用时，动态检测区域和邻域配对是处理 CMOP/BC 的关键方向；BE-CBO 也强调 unknown/binary constraint 下边界附近搜索的重要性。 

### 3.2 只用 AF 训练 GAN 直接生成边界可行样本：尚未成立

当前证据显示：

* `Raw_FR` 有提升；
* `Repair_rate` 有下降；
* 但 `Raw_BHR_cond` 中位数仍为 0；
* 改善只出现在 48/152 个 run。

所以现阶段只能说：

> **GAN 已经部分学会生成可行样本，但尚未稳定学会生成边界可行样本。**

这不是一个最终否定，而是说明当前版本还缺一个必要环节：

> **从 AF 档案中抽取“边界训练子集”。**

---

## 4. 下一步应该如何找原因

不要直接上 AF_train_bank、换 GAN、增强 repair 或引入分类器。那些都会改变问题性质。

下一步只做一个受控诊断：

> **判断失败到底来自 AF 语义，还是 GAN 学习。**

### 4.1 诊断实验：AF replay / AF jitter / GAN raw 三者对照

在每个 GAN refresh 点，额外做一个诊断，不改变算法注入逻辑。

#### A. AF replay

直接用当前 AF 自身计算 BHR 指标，不需要额外 FE。

目的：检查“AF 自己在当前 Raw_BHR 定义下是否算边界”。

如果 AF replay 的 BHR 都很低，说明问题在：

* BHR 指标定义；
* AF/AI pair 语义；
* AF/AI 的目标空间归一化；
* 或 AF 本身不是指标意义下的边界。

这时不能怪 GAN。

#### B. AF jitter

从 AF 中采样，加入很小的决策空间扰动：

[
x' = x_{AF} + \epsilon,\quad \epsilon \sim \mathcal{N}(0, \sigma^2)
]

建议：

[
\sigma = 0.25 \times Gap_dec_med
]

然后真实评估，不 repair，不注入种群，只记录。

目的：检查“AF 附近是否存在可学习的边界可行带”。

解释规则：

* 如果 AF jitter BHR 较高，但 GAN raw BHR 低：GAN 学习/采样过程有问题；
* 如果 AF jitter 也低：AF 太薄、太碎、太不稳定，或者 BHR 指标过严；
* 如果 AF replay 高、AF jitter 低：边界带极窄，标准 GAN 很难 raw 命中，需要训练集进一步变薄或改采样，但不能先怪 archive。

#### C. GAN raw probe

当前 raw count 是 20。诊断时额外采样 200 个 raw，只用于指标，不注入、不 repair。

目的：排除“20 个样本太少导致 hit 没采到”的偶然性。

记录：

* `Probe_Raw_FR`
* `Probe_Raw_BHR_cond@1x`
* `Probe_Raw_BHR_cond@2x`
* `Probe_Raw_BHR_cond@3x`
* `Probe_Raw_minAI_dist_obj_med`
* `Probe_Raw_minAI_dist_dec_med`
* `Probe_Raw_minAF_dist_dec_med`

这一组诊断能直接定位：

| 现象                       | 解释                |
| ------------------------ | ----------------- |
| AF replay 低              | AF/AI 或 BHR 指标不一致 |
| AF replay 高、AF jitter 低  | 边界带太薄，扰动即偏离       |
| AF jitter 高、GAN raw 低    | GAN 学习/采样失败       |
| GAN @2x/@3x 高但 @1x 低     | 指标过严或 raw 近边界但不够贴 |
| GAN Raw_FR 高、minAI_obj 高 | GAN 学到可行内部        |
| GAN Raw_FR 低、repair 高    | GAN 支持集漂到不可行侧     |

这是目前最关键的实验。它不会改变算法主逻辑，也不会引入新技巧。

---

## 5. 下一步最小修正方案

如果上述诊断确认 AF replay/AF jitter 有边界信号，而 GAN raw 仍然不命中，那么最小修正不是换 GAN，而是：

> **不要用全量 AF 训练 GAN；改用 AF 中最贴边的训练子集。**

我建议把当前版本改成：

### CCMO-GAN-BDG-softTrain

核心只改一处：

```matlab
GAN = BoundaryGAN_BDG('train',AF.decs,...)
```

改成逻辑上：

```matlab
AFTrain = SelectBoundaryTrainingSet_BDG(AF,AI,W,minTrain);
GAN = BoundaryGAN_BDG('train',AFTrain.decs,...)
```

注意，这不是新增模型，也不是新档案。
AF/AI 仍然是原来的双边界档案；只是 GAN 训练时从 AF 中取一个更贴边的 view。

### 5.1 AFTrain 的选择规则

对每个 AF/AI pair 计算：

[
g_i^{dec} = |x_i^F - x_i^I|
]

[
g_i^{obj} = |f(x_i^F)-f(x_i^I)|
]

然后定义训练分数：

[
s_i^{train}=0.5\cdot rank(g_i^{dec})+0.5\cdot rank(g_i^{obj})
]

这里建议用 rank-normalization，而不是 min-max normalization，避免极端 gap 主导。

然后：

1. 每个参考向量区域先保留 `perRefTrain=1` 个最小 `s_train` 的 AF；
2. 如果总数小于 `minTrain=20`，从全局 next-best 补齐；
3. 最多保留 `trainCap=60` 个；
4. 不额外评估，不改变 AF/AI，不改变 repair。

这样做的含义是：

* AF 仍是算法档案；
* AFTrain 是 GAN 训练带；
* GAN 仍然只看可行样本；
* AI 不进入 GAN 训练，只用于判断 AF 中哪些样本更贴边；
* 标准 GAN 约束不变。

这符合你的核心创新约束。

### 5.2 为什么这是当前最小修正

它只改“训练样本选择”，不改：

* CCMO 主体；
* AF/AI 档案构造；
* soft rebuild；
* GAN 结构；
* GAN 损失；
* repair；
* injection 位置；
* `bisectK=2`；
* `nearTau=0.20`。

因此它是一个干净消融：

> 如果 `AFTrain` 后 Raw_BHR_cond 明显提升，就说明核心创新点的正确表述应是：
> **GAN 应训练于 AF 中的边界可行训练带，而不是全量 AF 档案。**

这比加入 classifier、WGAN、cGAN、AF_train_bank 更干净。

---

## 6. 当前版本中具体应如何修改

### 6.1 `CCMO_GAN_BDG.m`

当前：

```matlab
if size(AF.decs,1) >= minTrain
    GAN = BoundaryGAN_BDG('train',AF.decs,...);
    [OffspringG,GanDiag] = BoundaryGAN_BDG('sample',...);
end
```

改为：

1. 先构造 `AFTrain`；
2. 用 `AFTrain.decs` 训练；
3. sampling 仍然用原始 AF/AI repair。

逻辑：

```matlab
AFTrain = SelectBoundaryTrainingSet_BDG(AF,AI,W,minTrain,trainCap,perRefTrain);

if size(AFTrain.decs,1) >= minTrain
    GAN = BoundaryGAN_BDG('train',AFTrain.decs,...);
    [OffspringG,GanDiag] = BoundaryGAN_BDG('sample',Problem,GAN,AF,AI,W,nGen,repairK);
end
```

注意：`sample` 仍然传原始 AF/AI，而不是 AFTrain。
原因是 AF/AI 是完整边界档案，repair anchor 和指标统计仍需要完整档案。

### 6.2 新增轻量函数 `SelectBoundaryTrainingSet_BDG`

这个函数不评估新点，只做索引选择。

输入：

* `AF`
* `AI`
* `W`
* `minTrain`
* `trainCap=60`
* `perRefTrain=1`

输出：

* `AFTrain.decs`
* `AFTrain.objs`
* `AFTrain.ref`
* `AFTrain.score`

选择规则：

1. 计算每个 pair 的 `gap_dec_i`；
2. 计算每个 pair 的 `gap_obj_i`；
3. rank-normalize；
4. `trainScore = 0.5*rankGapDec + 0.5*rankGapObj`；
5. 每个 ref 取最小的 `perRefTrain` 个；
6. 不足 `minTrain` 时全局补齐；
7. 超过 `trainCap` 时保留 score 最小的 `trainCap` 个。

这个函数本质上是**从 AF 中抽薄边界带**。

### 6.3 `BuildMetricsRow_BDG` 增加训练集诊断

新增字段：

* `AFTrain_size`
* `AFTrain_ref_cov`
* `AFTrain_gap_obj_med`
* `AFTrain_gap_dec_med`
* `AFTrain_score_med`
* `AFTrain_front1_ratio`

这样你能直接看：

* GAN 训练集是否比 AF 更贴边；
* 训练集是否过窄；
* 训练集是否仍覆盖足够参考向量区域。

### 6.4 `BoundaryGAN_BDG.m` 暂时不改

不要先改网络结构、损失函数、学习率、训练轮数。

原因：

* 你现在要验证的是训练集语义；
* 一旦同时改 GAN，就无法判断 Raw_BHR 改善来自哪里；
* GMOEA 已经说明 GAN 的效果高度依赖 real sample 分布，而不是单纯堆网络。

### 6.5 `UpdateBoundaryArchive_BDG.m` 暂时不再大改

soft rebuild 已经解决 retention/refresh。
目前不应再继续改 rebuild，否则会把诊断复杂化。

只建议加诊断，不建议改 pair source。

---

## 7. 最小实验设计

### 实验目标

验证：

> **当前 Raw_BHR_cond 为 0 是因为 GAN 吃了“全 AF 宽档案”，还是因为标准 GAN 根本无法从 AF 学到边界。**

### 实验组

只做两组主实验，加一个诊断记录。

#### G0：当前 soft rebuild full-AF 训练

保持当前版本：

```matlab
GAN train data = AF.decs
```

这是 baseline。

#### G1：soft rebuild + thin-AF training

只改训练输入：

```matlab
GAN train data = AFTrain.decs
```

其他全部不变。

### 诊断 probe

在 G0 和 G1 中都记录：

* `AF_replay_BHR@1x/@2x/@3x`
* `AF_jitter_BHR@1x/@2x/@3x`
* `Probe_Raw_BHR_cond@1x/@2x/@3x`
* `Probe_Raw_FR`
* `Raw_minAI_dist_obj_med`
* `Raw_minAI_dist_dec_med`
* `Raw_minAF_dist_dec_med`
* `lossD/lossG/Dreal/Dfake`

### 优先测试问题

不要一开始全量扩大。先选代表问题：

* `DASCMOP1_BC`：soft rebuild 后 BHR 有一定改善的正例；
* `DASCMOP9_BC`：raw 常严重依赖 repair 的负例；
* `LIRCMOP10_BC`：有部分 raw hit 的中间例；
* `LIRCMOP12_BC`：历史 AF 不足代表；
* `LIRCMOP13_BC`、`LIRCMOP14_BC`：AF 大、Raw_FR 高、Raw_BHR 仍低的关键例。

每个问题 8 runs。
如果 G1 在这些问题上明确改善，再扩到 152 runs。

### 成功判据

不要只看 HV/IGD。当前阶段只看核心动机指标。

最低成功标准：

1. `Raw_BHR_cond@1x` 中位数从 0 变为非 0；
2. `Raw_BHR_cond@2x/@3x` 明显提高；
3. 改善 run 数从 48/152 明显增加；
4. `Raw_FR` 不出现大幅下降；
5. `Repair_rate` 继续下降或至少不升高；
6. `Probe_Raw_minAI_dist_obj_med / Gap_obj_med` 明显下降。

如果 G1 只提升 Raw_FR，不提升 Raw_BHR_cond，则说明 thin training 仍不够；GAN 仍在学可行内部。

如果 AF jitter 能 hit，G1 不能 hit，则主因转向 GAN 学习过程。

如果 AF replay 和 AF jitter 都不能 hit，则主因不是 GAN，而是 AF/AI 边界语义或 BHR 指标。

---

## 8. 暂时不要做的修改

为了保持统一和收敛，下面这些现在不要做。

### 不要先加 AF_train_bank

它可能有用，但会引入时间平滑和历史样本，混淆“当前 AF 是否可学”的判断。当前 bundle 的解释也已经指出，直接加 bank 可能混淆残余失效模式。

### 不要先换 WGAN-GP / cGAN / classifier

这会改变核心创新点。尤其是 classifier 会把方法变成“分类器辅助边界生成”，而不是“AF 直接训练标准 GAN”。

BE-CBO 里的深度集成分类器确实证明了边界建模很重要，但那是另一条方法线，不应作为你当前最小验证版本的第一修正。

### 不要先增强 repair

repair 增强会掩盖 raw generation 是否成立。当前研究目标是验证 raw 是否能命中边界，不是把不可行 raw 拉回来。

### 不要继续调 nearTau

你已经验证过 `nearTau=0.30` 没有改善 Raw_BHR，只增加 pair 和 FE 开销。

---

## 最终判断

当前核心创新点的状态是：

> **AF/AI 边界档案：成立趋势明显。**
> **GAN 直接生成边界可行解：当前证据不足，尚未成立。**

真正阻碍 GAN raw 边界命中的主要问题不是 AF 数量，也不是刷新次数，而是：

> **全量 AF 作为 GAN 训练集太宽、太混合，导致标准 GAN 学到可行分布中心，而不是边界可行带。**

下一步最小修正应是：

> **保持 AF/AI、soft rebuild、标准 GAN、repair 和 CCMO 主体都不变，只把 GAN 训练输入从全量 AF 改为“AF 中最贴边的训练子集 AFTrain”。**

这一步最干净，也最能验证你的核心命题：

> 如果 thin-AF training 后 Raw_BHR_cond 明显提高，核心创新点就可以被重新表述为：
> **“利用 AF/AI 构造边界可行训练带，并用标准 GAN 直接生成边界附近可行解。”**

这比“用全量 AF 训练 GAN”更准确，也更有发表上的方法边界。
