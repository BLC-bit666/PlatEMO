# PairGuide

当前保留两个算法入口：

- `PairGuide`：当前主线；
- `CBS_RegionWGAN_GP`：上一任主线，保留其原有算法与名称。

两个公开类彼此独立，不做名称映射、兼容别名、继承或配置委托。PairGuide 由 `PairGuideCore` 实现；上一任算法由 `CBS_RegionWGAN_GP_Core` 实现。

PairGuide 保留既有双种群框架：P1 仍按约束规则搜索，P2 仍按无约束规则搜索，完整评价子代仍通过原有 `Union` 和环境选择更新两套种群。CGAN 的条件仍是参考向量方向 `w` 与二元端点标签 `s in {0,1}`；生成器直接输出决策解，不输出 P1 上的改进量。

PairGuide 的目标是让每个条件对应一条唯一、窄的可行—不可行 bracket，并通过真实评价反馈使该 bracket 与 P1 双向靠近真实约束边界。首轮实现只修改训练集构造/更新、生成器几何损失、训练日程和 donor 接入方式，不叠加额外网络、权重或问题专用策略。

## 边界档案

每条记录为不可拆分的真实端点对：

```text
id, xf, yf, xi, yi, ref, gap, age, lastFE, active, resumeEligible
```

- `xf/yf` 是真实评价的可行端点及其目标；`xi/yi` 是真实评价的不可行端点及其目标。
- 新建 pair 的可行候选是当前 P1 中 `Fitness < 1` 的可行精英。历史 `xf` 不以当前成员资格或 `Fitness < 1` 作为存活条件。不可行候选来自完整已评价集合 `P1 ∪ P2 ∪ O1 ∪ O2`（包括已评价 PairGuide 子代）及历史仍合法的 `xi`，不限定为 P2 成员。
- 档案 `xi` 必须通过当前 P1 可行精英的全局支配过滤：不存在当前可行精英支配 `xi`。配对本身也始终满足 `xf` 不支配 `xi`。
- 目标归属使用同一份 `RefScale`；该尺度由当前可行精英目标、当前不可行候选目标和历史不可行端点目标共同构造。
- `refDivisor=1`，即请求 `N` 个参考向量，并使用 `UniformPoint` 实际返回的数量。`pairArchivePerRef=1`，即每个参考方向最多保留一个 active pair；训练前必须满足 `numel(unique(Data.ref)) == Data.count`。
- `ref` 由 `yf` 决定。`xi` 可以来自邻近分区，但必须落在 `xf` 方向的角度邻域内；邻域总数固定为 5，含自身和角度最近的另外 4 个方向。`pairNeighborRefCount=5` 表示总数 5，不是 `1+5`。
- 对每个可行精英，在目标合法且位于上述邻域的不可行候选中选择归一化决策距离最近者，不要求端点互为最近邻。同一参考方向的候选 pair 按 `(gap, xi方向角度排名, Fitness(xf), -lastFE, id)` 字典序保留唯一一对，不使用加权和。
- 历史 `xf` 只允许被更新后 P1 中的真实可行点替换，不要求替代者 `Fitness < 1`，但必须严格缩短合法 gap；未进入新 P1 的 Union/反馈可行点不得替换。已有 active pair 的参考方向不得新建竞争 pair，避免最终裁剪绕过严格替换规则；新 pair 只填补当前空缺方向。带 `matchedPairID` 的不可行子代仍只能更新其原 `xi`，普通真实不可行点可以按相同合法性与严格 gap 缩短规则收紧局部 `xi`。
- 只要端点、目标关系、邻域和 `ref` 生命周期仍合法，历史 pair 就保持 active 并进入训练，不要求当前 P1 再次局部支持其 `xf`。失去合法性或生命周期连续性的记录才 inactive，连续 10 代后删除；以后重建必须使用新 ID。
- 原始生成点 `G` 不是档案端点；只有真实完整评价后的子代才能更新档案。

## PairGuide CGAN

训练样本为归一化决策空间中的绝对端点：

```text
[xf_norm, w, s=1]
[xi_norm, w, s=0]
```

一个 mini-batch 含 `B=32` 个完整 pair（64 个 endpoint）。每个 pair 使用同一个噪声 `z` 生成两侧：

```text
xhat_f = G(z,w,1)
xhat_i = G(z,w,0)
```

Critic 继续使用标准 WGAN-GP 损失，`GP=10`。生成器损失固定为：

```text
L_G        = L_adv + lambda_geo * (L_anchor + L_pair)
lambda_geo = 1
L_anchor   = sum(||xhat_f-xf||^2 + ||xhat_i-xi||^2) / (2*B*D)
L_pair     = sum(||(xhat_i-xhat_f)-(xi-xf)||^2) / (B*D)
```

CGAN 真正学习的就是每个 pair 的真实可行端点 `xf` 和真实不可行端点 `xi`；真实反馈持续压缩二者的 gap，因此这些端点对本身构成逐步收窄的边界带训练代理。端点锚定同时约束绝对位置和同条件输出厚度，pair 差分项保持可行—不可行方向。旧 energy-distance relation loss、mode-seeking loss、`pairRelationWeight`、`pairLatentWeight` 和 `pairLatentThreshold` 均不再使用；首轮也不增加独立厚度、覆盖、法向或自适应权重。

固定训练配置：

| 项目 | 值 |
| --- | ---: |
| G/C 隐藏层 | `[32,32]` |
| `zDim` | 6 |
| 学习率 | `1e-4` |
| `GP` | 10 |
| Critic : Generator | `5 : 1` |
| mini-batch | 32 对（64 端点） |
| 首次训练 | 500 Epoch |
| 后续重训 | 20 Epoch |
| 训练/推理 sigma | `1 / 0.3` |
| 最小 active pair 数 | 32 |
| warm-start | G、C 及 Adam 状态全部保留 |

每个 Epoch 对全部 active pair 随机打乱并无放回访问一次，尾批保留。由于每方向最多一对，pair 数就是 active reference 数，不再另设独立的 `pairMinRegions` 门槛。500/20 是当前锁定的首轮验证值，不代表已经证明为最优。

训练诊断记录端点 RMSE、pair 差分 RMSE、同条件厚度和 critic gap。

## 生成、筛选与接入

设本代 PairGuide 名额为 `Q`（`N=100`、20% 配额时 `Q=20`）：

1. 在所有 active pair 上分层均匀分配并随机打乱 500 个 `[w,0]` 查询。`buildQueryContexts` 同时返回 `refs` 和 `pairIds`；pair ID 只用于归属，不进入 CGAN 条件。
2. raw `G` 不做目标或约束评价，不计 FE、不进入 `SOLUTION`/`Union`，也不因接近 `Archive.xi`、种群或档案而被删除。
3. 每个 raw `G` 只匹配其 ID 对应的 active pair。在归一化决策空间定义：

   ```text
   center = (xf+xi)/2
   radius = ||xi-xf||/2
   ```

   仅保留满足 `||G-center|| <= radius` 的闭球内候选。
4. 全部球内 raw 均保留并维持 500 个 raw 的原始随机顺序；不按 pair 去重，也不做距离、占用率或 gap 排序。最多对序列前 `2Q` 个候选调用真实 `CalObj`，但不调用约束、不创建 `SOLUTION`、不进入种群。
5. objective-only 候选必须满足：`G` 不支配配对 `xi`、配对 `xf` 不支配 `G`，且归一化目标逐维位于 `yf` 与 `yi` 张成的闭合 Pareto 走廊内（容差 `1e-12`）。这里明确不对 `G` 使用“不能被任意当前可行精英支配”的全局过滤；全局可行精英过滤只用于档案不可行端点 `xi`。
6. 通过走廊过滤后最多保留 `Q` 个 donor。对每个 donor，从当前 P1 可行 `Fitness < 1` 精英中，在总数 5 的参考邻域内选择距离历史 `xf` 最近的父代 `c`；没有当前可用父代只跳过本次接入，不删除历史训练 pair。
7. 直接生成无变异二分子代 `u = CalDec(0.5*(c+G))`，不做第二次几何筛选，也不使用 `h`、`Fg`、vacancy-weighted maximin、DE 差分或多项式变异。
8. 只在最终二分子代产生后，检查其是否与当前 P1 或当代已选子代重复。最终子代接受完整目标与约束评价，进入正常 `Union`；不足 `Q` 的名额由普通 DE 补齐。

## 评价成本

| 对象 | 目标评价 | 约束评价 | 计入严格总 FE | 进入 `Union` |
| --- | --- | --- | --- | --- |
| 500 个 raw `G` | 否 | 否 | 否 | 否 |
| 最多 `2Q` 个 shortlist `G` | 是 | 否 | 是，另记 `ObjFE` | 否 |
| 最多 `Q` 个二分子代 `u` | 是 | 是 | 是 | 是 |
| 配额不足时的普通 DE | 是 | 是 | 是 | 是 |

目标函数调用不能作为零成本隐藏。运行统计分别记录 objective-only 的 `ObjFE` 与最终完整评价/约束查询数，同时二者都服从严格总预算。六个 `LIRCMOP*_BC` 问题的 `CalObj`、`CalCon` 和 `Evaluation` 必须保持一致，确保 objective-only 筛选调用的是真实目标公式。

P1 原有个体不会因 PairGuide 配额被提前删除；PairGuide 只替代一部分普通 DE 试验机会，所有完整评价子代仍由既有双种群环境选择决定去留。

## 最小回归

```matlab
test_CBS_pair_guide
test_CBS_pair_guide_training_observation
test_CBS_platemo_compliance
test_CBS_region_wgan_mainline
test_CBS_mainline_fingerprint
```

PairGuide 回归应覆盖：档案 `xi` 的全局支配合法性、合法历史 `xf` 持续训练、新 P1 可行点严格替代、总数为 5 的参考邻域、每参考方向唯一 pair、pair 直径闭球、Pareto 走廊、objective-only FE 记账，以及六个 BC 问题的 `CalObj` 一致性。
