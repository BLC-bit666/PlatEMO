你这个纠偏是对的。
我上一版里把“**局部两侧支撑**”过度具体化成了“**每扇区一正一负**”，这是不对的。

从你最新 compact5 里的源码看，当前训练集裁剪确实先走 `ReserveCoreSideCoverageBySector`，它会在每个 sector 的 core 里优先保留第一条 feasible 和第一条 infeasible；而 `CanTrainBoundaryModel` 又要求 `Trusted` 和 `PairCnt` 都至少达到 `max(2,M)`。这等于把“按扇区双侧齐备”做成了硬门槛。对“小可行域、可行解极少”的二元约束问题，这个门确实过严。 

更合理的说法应该是：

> **边界学习需要的是 pair-level 的双侧证据，不是 sector-level 的类别平衡。**

边界搜索文献本身也更接近这个表述。SDBES 关心的是样本是否覆盖真实边界、是否靠近真实边界，并用 `coverage / number of covering pairs / precision / efficiency` 来评价，而不是要求每个局部格子里都类平衡。它强调的单位更像“boundary pair / boundary coverage”，不是“每个分区一正一负”。

所以你提出的两点，我现在的最终判断如下。

---

## 1）“每扇区一正一负”不是必须条件；真正该保的是“少量可行锚点 + 其局部 opposite infeasible 壳层”

你说得对：
对于这类约束问题，天然就是**可行少、不可行多**。尤其后期很多代找不到新可行解，这不是异常，而是问题本性。CCMO 那篇也明确提到，小可行域 CMOP 本来就会让 MOEA 很难找到足够多的 feasible solutions。

因此，训练 MLP 的 core 不应该再围绕“每个 sector 都凑一正一负”来设计，而应该改成：

> **以稀缺的 feasible 作为锚点（anchor），围绕每个 anchor 只收它附近的少量 opposite-side infeasible，形成 pair-centric core。**

也就是说，训练的最小单元不是 sector，而是：

[
(x_f,;x_i)
]

其中 `x_f` 是少量可行锚点，`x_i` 是它附近的 opposite infeasible。

这才和“MLP 拟合边界”这件事真正对齐。
因为边界不是靠“每个扇区都平衡”定义的，而是靠“**某个可行点附近，确实存在与之相邻的不可行点**”定义的。

### 你应该把 current logic 改成什么

#### 先删掉这个硬假设

* 不再要求 `ReserveCoreSideCoverageBySector`
* 不再要求 `CountCorePairedSectors` 这种“按扇区统计 paired sector 数”
* sectors 只保留一个作用：**控制锚点分布的多样性**
* sectors 不再承担“训练是否允许启动”的职责

#### 新的 core 定义

把 core 改成下面这个结构：

1. `F_anchor`：所有可行锚点
   来源优先级：

   * `B` 里的 feasible
   * `RecentBoundaryOff` 里的 feasible
   * 如果全局 feasible anchor 太少，再从 `P_C` 补最新的 feasible reps
     这里的补是**全局补**，不是“每个 sector 补”。

2. 对每个 `x_f ∈ F_anchor`，找 `k_neg` 个最近的 opposite infeasible
   来源优先级：

   * `B` 里的 infeasible
   * `RecentBoundaryOff` 里的 infeasible
   * 若还不够，再从 `P_U` 里找最近 infeasible

3. 训练集 core = 所有 `x_f` 以及这些被配对到的 `x_i`

我建议最先用很克制的参数：

* `k_neg = 2`
* `MinAnchor = max(2,M)`
* `MinPair = max(4,2*M)`

这就够了，不要再复杂。

### 为什么这种定义更对

因为“大量不可行 + 少量可行”能否确定边界，关键不在全局比例，而在于：

* 这些少量可行是不是**边界锚点**
* 每个锚点附近有没有**真实 opposite infeasible**

只要这两件事成立，哪怕全局上是 `1 : 20`、`1 : 50`，依然可以学边界。
反过来，如果你有很多可行，但它们都在深可行域内部，也一样学不出边界。

所以你这次的修正，核心不是“不要双侧”，而是：

> **双侧支撑要从 sector-level 下降到 anchor-level / pair-level。**

---

## 2）训练集极端不平衡时，plain BCE 的 MLP 会不会偏向不可行？

**会，有明显风险。**
不是一定“无脑全判不可行”，但如果你把原始极度失衡的数据直接整批喂给 unweighted BCE，再用 `0.5` 当阈值，它通常会明显偏向多数类。类不平衡会伤害深网分类性能；Buda 等人的系统研究发现，过采样通常优于欠采样，而且阈值/先验也会影响最终判别。Chawla 的 SMOTE 原始工作也表明，少数类过采样与多数类适度欠采样的组合，往往优于只做欠采样。

但这里你有一个非常重要的约束：**不能改损失函数。**
那就只能走**数据分布**这一条，而且这正好是你想要的。

我的最终建议是：

> **要做平衡，但只做“重采样平衡”，不做“合成扰动平衡”。**

---

## 3）最终方案：保留 plain BCE，不改 loss；只改“core 定义 + batch 分布”

### A. archive / core 层：不追求平衡，只追求“锚点 + opposite shell”

先明确区分两层：

* **边界存档 B**：允许真实不平衡
* **训练 batch**：必须重新平衡

这很关键。
`B` 没必要长成 1:1，它只需要真实保存 boundary evidence。
真正需要平衡的是**送进 BCE 的训练分布**。

### B. batch 层：做 replay oversampling + informative undersampling

这是我最推荐的主线，而且完全符合你的约束。

#### 具体做法

每次训练 MLP，不要直接拿整个 `TrainArchive.Dec / Label` 整批训练。
而是在每个 epoch 先构造一个**重采样后的 epoch set**：

1. `PosIdx`：所有 feasible anchor 样本
2. `NegIdx`：所有与 anchor 配对的 opposite infeasible

然后：

* 对 `PosIdx` 做**重复采样**，直到正类数达到目标规模
  这就是最朴素的 random oversampling / replay
* 对 `NegIdx` 不要全拿，只拿与当前 anchor 配对的那部分，必要时再抽一点
  这就是 informative undersampling

我建议第一版直接设成：

* epoch 内 `pos : neg = 1 : 1`

这样做的好处非常大：

1. 不改损失函数，仍然是 plain BCE
2. 不改模型结构，仍然是当前 MLP
3. `0.5` 的语义更接近“边界中性面”
4. MLP 不会被全局 `infeasible >> feasible` 的先验直接拖走

这其实就是“改训练数据分布，不改训练目标函数”。
完全符合你的硬约束。

### C. 为什么我不建议你先做 feasible 扰动增强

你问得很准：要不要通过可行解扰动来平衡？

我的回答是：

**主线里先不要。**

原因不是 augmentation 永远没用，而是你现在这个任务太特殊：

* 你的 feasible 本来就稀少
* 而且它们大多正是最靠近边界的那一批
* 这意味着对 feasible 做小扰动，**极容易跨过真实边界**
* 一旦你不重新做真实约束评估，就会直接制造错标样本

而 SMOTE / synthetic oversampling 这类方法，文献里一直都强调要小心 boundary overlap 和 noisy / borderline examples；很多 SMOTE 变体之所以要先做 candidate selection 或 noise filtering，正是为了减少边界附近的重叠和噪声。

你的场景恰好是**边界学习**，所以“对边界附近 feasible 直接扰动并继承原标签”是最危险的版本。
这会伤到你最想证明的东西。

所以我的结论很明确：

* **可以过采样**
* 但优先用**重复采样 replay**
* 不要先用**几何扰动 synthetic feasible**
* 更不要用“扰动后默认还可行”的隐含标签继承

只有一种例外：
如果你愿意对扰动后的样本再次做**真实约束评估**，那它就不是“合成样本”了，而是新增真实样本；但这已经不是单纯数据增强，而是额外采样预算。主线里先别走这条。

---

## 4）把 current PRBCCMO 改成什么样

下面是我现在认为最正确、最小的改法。

### 第一步：把“按扇区保双侧”改成“按锚点保 opposite pair”

当前 `SelectArchiveRowsBySectorQuota -> ReserveCoreSideCoverageBySector` 这条逻辑，建议直接退出主线。它现在会按 sector 强行优先保一条 feasible 和一条 infeasible。 

改成一个新的选择器，思路如下：

```matlab
function Pick = ReserveAnchorPairs(TrainArchive,MaxTrain,kNeg)
    % 1) 先拿全部 feasible anchors（source 1/2 优先，3 号 source 仅做全局补足）
    % 2) 对每个 anchor 找最近的 kNeg 个 opposite infeasible
    % 3) 返回这些 unique 样本的并集
end
```

关键点：

* 不再逐 sector 保正负
* 只保 global anchors + local opposite shell

### 第二步：训练启动门改成“anchor 数 + pair 数”

当前 `CanTrainBoundaryModel` 还在看 `Trusted` 和 `PairCnt` 的扇区数量，这正是你指出的问题根源之一。

把它改成：

```matlab
Flag = (NumFeasibleAnchors >= max(2,M)) && ...
       (NumOppositePairs   >= max(4,2*M));
```

不要再要求：

* `nnz(Trusted) >= max(2,M)`
* `paired sectors >= max(2,M)`

这两个门对于“小可行域 + 稀缺 feasible”过于乐观，现实里经常过不了。

### 第三步：`P_C / P_U reps` 的角色改掉

这里也顺着你的质疑一起改：

* `P_C reps`：只做**全局 feasible anchor 补足**
* `P_U reps`：默认不进训练主干；只有某个 anchor 真的找不到 opposite infeasible 时才兜底

也就是说：

* source 3 保留，但只补全局 feasible 数
* source 4 基本退出主线

因为在你的问题上，缺的通常不是 infeasible，而是有意义的 feasible anchors。

### 第四步：训练时做 batch-level 重采样

`TrainBoundaryMLP` 的 BCE 保持不变。
只在进入 epoch 前改 `X, Y`：

```matlab
Pos = find(Y==1);      % feasible anchors
Neg = find(Y==0);      % paired opposite infeasible

PosEpoch = Pos(randi(numel(Pos), targetPos, 1));   % replay oversampling
NegEpoch = Neg(randperm(numel(Neg), targetNeg));   % informative undersampling

Idx = [PosEpoch; NegEpoch];
Idx = Idx(randperm(numel(Idx)));

Xe = X(Idx,:);
Ye = Y(Idx);
```

第一版最建议：

* `targetPos = targetNeg`
* 也就是 `1:1`

这样最有利于让 `p≈0.5` 和“边界中性面”绑定。
如果假阳性明显偏多，再退到 `1:2`。

---

## 5）验证指标也要从“sector”改成“pair”

因为你已经指出 sector 不是正确粒度，所以验证也要一起改。

我建议以后少看：

* `paired_sector_count`
* `trusted_sector_count`

多看这 5 个：

* `anchor_count`：训练 core 里 unique feasible anchors 数
* `pair_count`：unique opposite pairs 数
* `mean_pair_dist`：pair 平均距离
* `lowmargin_pair_hit`：low-margin 点是否真的落在某个 anchor-pair 附近
* `seg_cross_dist`：pair 连线上的真实 crossing 与 `p=0.5` crossing 的距离

最后这个 `seg_cross_dist` 最关键。
因为它直接回答你的核心创新点：

> **MLP 到底是不是在拟合边界，而不是在做全局 feasibility classification。**

---

## 最后收束成一句话

你这次指出的问题是对的：
**“每扇区一正一负”不是边界学习的必要条件。**

真正该保留的不是 sector-balance，而是：

> **少量可行锚点 + 每个锚点附近的少量 opposite infeasible pair。**

而在“禁止修改损失函数”的前提下，最正确的数据级改法不是 feasible 扰动增强，而是：

> **archive 保持真实不平衡，training batch 做 replay oversampling 的正类重采样 + boundary-near negative 的信息化欠采样。**

这条线最统一，也最能把“MLP 拟合边界”这个创新点真正验证出来。

我这次是按你最新的 compact5 bundle 和当前仍可访问的论文材料重新思考的；更早的部分 full bundle 已经过期，如果后面要我按旧版源码逐函数对照改法，需要把那些过期文件再传一次。
