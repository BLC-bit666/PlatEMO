# 算法创新与验证分析

#### You:
上述文件是我与GPT的交流历史，请你根据对话和pdf、txt资料来回答下列问题
1：我的算法最大的创新点是什么？合理吗？能做出效果吗？有没有相关论文供我参考（与本创新点相关）？
2：算法是否还有改进空间（针对核心创新点？）；算法是否过于复杂（存在不必要的方法或机制？）能否简化不必要的部分？
3：我当前做的实验验证合理吗？结果有说服力吗？若要验证你改进后的算法中我的核心创新点应该怎样验证？

#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
我把三份材料连起来看了：历史对话里的原始设想是“围绕 $p\approx 0.5$ 主动找可行域边界，并在发现更优可行区域时迁移”；NA-EMT 论文本体还是“MLP 感知 infeasible solution value + predicted-value based dominance + 辅助无约束任务”；而最新 txt 里的 PRBCCMO 已经把这件事落成了 **bridge 候选 + boundary utility 打分 + feasible expansion / infeasible bracketing / hard-negative confirmation + sector-level migration**。fileciteturn0file0 fileciteturn0file1 fileciteturn0file2

我的总判断是：**题目方向是对的，严格意义上只有 1 个核心创新点；这个创新点在学理上合理，也有邻近文献支持，但你当前实现还没有把它“做成立”，现有 formal validation 更像在反证当前版本。** NA-EMT、EADMM、DRMCMO 都说明 “unknown / binary constraints 的 CMOP” 已经是一个明确的研究方向；而 BE-CBO、COMBOO 这类工作又说明“边界/level-set 的主动学习”确实是有价值的。citeturn720600view2turn205911view0turn205911view2turn423220view0turn423220view1

## 1：最大的创新点是什么？合理吗？能做出效果吗？有哪些相关论文？

**严格口径下，我只认 1 个核心创新点。**

这个创新点不是“双种群”、不是“MLP”、也不是“在 0.5 附近采样”单独任何一个，而是：

**Pareto-relevant active boundary search under binary unknown constraints**\
更贴近你当前代码的说法是：**把 MLP 从 NA-EMT 里的“不可行解价值预测器”，改造成“边界查询调度器”**——先用可行概率/不确定性找 boundary seeds，再用 Pareto 价值筛掉“没用的边界点”，再对这些 seed 做标签感知的局部细化。fileciteturn0file1 fileciteturn0file2

为什么我说只有 1 个？因为从审稿人的角度看，**双种群/辅助种群、利用 infeasible solutions、甚至“兼顾不确定性与代表性”的选点思想，都不是新类别了**。近年的综述把双种群、两阶段、辅助种群都列为已有大类；关于 infeasible solutions 的利用也早有综述。EADMM 已经用两个子问题/两种互补搜索方向来解 unknown constraints，DRMCMO 也已经把 binary constraints 下的 detection region 做成主机制。citeturn423220view5turn720600view3turn205911view0turn205911view2

**合理吗？合理。**\
因为 binary unknown constraints 没有 violation degree，你能学到的最关键几何信息，本来就不是“违反了多少”，而是“可行/不可行的分界在哪里”。BE-CBO 明确把 unknown constraints 下的最优解往往靠近 feasible–infeasible boundary 当成核心观察；COMBOO 也把“主动学习约束 level-set + 在 feasible region 内做多目标优化”绑在一起。你的方向与这个大逻辑是同向的。citeturn423220view0turn423220view1

**能做出效果吗？原理上能；当前版本还不能说做出来了。**\
原因不是这个创新点错，而是你现在最依赖的前提——“$p\approx 0.5$ 真的有边界语义”——还没站住。txt 里的 Section B formal validation 写得很直接：Gate 1 四个 calibration baseline 全部失败；Gate 2/3 的 pooled 结果里，`random-boundary` 在 `mean_dB`、`median_dB` 和 `UBY` 上更强，所有 variant 的 `boundary-induced Delta HV` 都是 0，最后 summary 还明确写了“the current formal experiment results do not support the intended innovation claims”。这意味着**当前实现不是“证据不足”这么简单，而是已经出现了反例信号**。fileciteturn0file2

相关论文我建议按三组读：

1. **直接邻域**：NA-EMT（你的直接出发点）、EADMM（unknown constraints 的双子问题/双种群路线）、DRMCMO（binary constraints 的 detection region 路线）。citeturn720600view2turn205911view0turn205911view2
2. **最接近你创新点的邻域**：BE-CBO（unknown constraints 下的 boundary exploration）、COMBOO（主动学 constraint level-set 与 feasible-region 多目标优化并行）。这两篇不是 EA，但思想上和你最接近。citeturn423220view0turn423220view1
3. **方法论支撑**：Guo et al. 2017 解释了为什么神经网络概率常常不可靠以及为什么 temperature scaling 常常有效；Deep Ensembles 给出一个简单可行的不确定性基线；QUIRE 说明 query 不能只看 informativeness，还要兼顾 representativeness。citeturn423220view2turn714623search1turn423220view4

## 2：算法还有改进空间吗？是否过于复杂？能否简化？

**有明显改进空间，而且当前版本在“边界打分栈”上有点过复杂了。**

先说一个重要事实：相较于历史对话里的早期版本，你最新 txt 其实已经修掉了两个最容易被审稿人吐槽的点：\
一是模型不再是“每代从零重训”，而是 **每 5 代/触发式更新 + warm-start + restart gap**；二是候选池也已经压成了 **bridge + local 两类来源**，不再是早期那种更散的多源拼盘。这个方向是对的。fileciteturn0file0 fileciteturn0file2

但现在真正的问题变成了两件事。

第一，**不确定性语义还不够可信**。\
现代神经网络的概率输出本来就可能失准，temperature scaling 常常有帮助，但不是万灵药；deep ensembles 是经典低门槛基线。你现在连 raw、temperature、sigmoid、3-MLP ensemble+temperature 都测了，Gate 1 还是全挂，这说明瓶颈已经不是 “tanh 还是 ReLU”“单 MLP 还是小 committee” 这种结构细节，而是**边界附近的数据组织与训练信号本身还不够好**。citeturn423220view2turn714623search1 fileciteturn0file2

第二，**主算法里塞了太多“看起来合理”的小机制，但它们没有被证明都在帮忙**。\
你现在的 boundary utility 里同时用了 entropy、committee disagreement、estimated positive HV、sector novelty、hard-negative penalty；local side 又有 feasible expansion、infeasible bracketing、hard-negative confirmation、sector migration。问题是：formal results 里 `random-boundary` 反而常常更好，这说明**至少 seed scoring 这层现在是可疑的**；而 winner count 也没有明显站在 Full 这边。fileciteturn0file2

所以我建议你把主算法**收缩成 4 个必须模块**：

**保留：**

- `P_C + P_U` 双群体，只把它当搜索脚手架，不当创新点。
- `bridge seeds`，这是 boundary search 的入口。
- `label-aware local refinement`，而且分成两种动作：\
  feasible seed -> 定向扩张；\
  infeasible seed -> 二分括逼 / hard-negative confirmation。
- `sector-level migration`，只做局部迁移，不做整群迁移。你当前代码已经从历史设想里的“整体迁移可行域”收敛到更合理的 sector migration，这个改动是对的。fileciteturn0file0 fileciteturn0file2

**主文里先删除或降级为附录 ablation 的：**

- hard-negative penalty 进入主 score；
- 过多的 calibration 变体同时上主算法；
- 任何不能单独证明有效的细小排序因子。

我更建议主版本的 seed score 只保留三项：\
**不确定性 × Pareto 潜力 × 多样性**。\
其中“不确定性”最好用 **mean probability 接近 0.5 + committee disagreement**，而不是只看单模型的 $|p-0.5|$。只按不确定性选点容易偏离代表性，这一点 active learning 文献早就提醒过。citeturn423220view4

一句话说，**现在不是“方法不够多”，而是“主线不够瘦”**。\
你的 paper 应该让人一眼看见：\
“我是在做 Pareto-relevant boundary search”，\
而不是\
“我做了一个大礼包：MLP+ensemble+temperature+pair loss+mid loss+hardneg+HV+sector+migration”。

## 3：当前实验验证合理吗？结果有说服力吗？改进后如何验证核心创新点？

**当前实验设计方向是合理的，但当前结果没有说服力地支持你的创新点。**

合理的地方在于：你没有只报 HV/IGD，而是专门做了 **Gate 1（校准）→ Gate 2（离真实边界有多近）→ Gate 3（这些边界点是否有用）**。这比只看最终性能强得多，而且 `ComputeOracleBoundaryDistance` 明确把原始连续约束值只用于离线审计，不给算法在线使用，这个设计是对的。fileciteturn0file2

但为什么“没说服力”？

一是 **Gate 1 全败**。\
你的 gate 条件是很合理的：`pooled near_gap <= 0.05`、`ECE <= 0.05`、至少 7/9 问题 `near_gap <= 0.07`、且 near-zone 的 95% bootstrap CI 覆盖 0.5。可实际结果是四个 baseline 全部失败，`problem_pass_count=0`。这一步不过，后面所有“0.5 有边界语义”的解释都会被动摇。fileciteturn0file2

二是 **Gate 2/3 没把 Full 顶出来**。\
`random-boundary` 在 `mean_dB`、`median_dB`、`UBY` 上都比 Full 更强，所有 variant 的 `boundary-induced Delta HV` 都是 0，summary 也明确说 current formal results do not support the intended innovation claims。这个结论已经足够直接。fileciteturn0file2

三是 **统计力度还差一点**。\
你现在 validation 用的是 20 runs，而 NA-EMT 原论文主实验是 30 runs，并配 Wilcoxon rank-sum。你现在这套机制验证如果想拿去论文里撑最大创新点，最好也升到 30 runs，并把 Gate 2/3 的关键指标做 per-run 统计检验，而不是只给 pooled winner counts。fileciteturn0file1 fileciteturn0file2

### 改进后，我建议你这样验证“最大创新点”

我只围绕这一个点来设计：\
**Pareto-relevant active boundary search with label-aware local refinement**

**Step 0：先过校准门。**\
比较 raw、temperature、committee+temperature，以及一个更简单的概率模型基线。画 reliability diagram，报 ECE、Brier、near-zone feasible rate、95% bootstrap CI。只有当 `p∈[0.4,0.6]` 真能对应“接近 50% 的经验可行率”时，后面的 boundary claim 才成立。现代神经网络确实常失准，而 temperature scaling 是经典修正手段。citeturn423220view2

**Step 1：做 oracle boundary audit。**\
沿用你现在 txt 里的思路，在 DASCMOP-BC 上离线记录原始连续约束，定义 `d_B` 作为“离真实边界的距离”。比较下面四个版本，在**相同总 FE、相同 boundary budget、相同双群体框架**下：

- Full
- Uncertain-only（只看不确定性，不看 Pareto relevance）
- Random-boundary
- No-local-label / isotropic local

核心问题只有一个：**Full 选到的 seed，是否真的更靠近真实边界？**\
若 `d_B` 没优势，创新点直接站不住。fileciteturn0file2

**Step 2：证明这些边界点“有用”，而不是只是“靠边界”。**\
至少报三类量：

- `UBY`：被选 seed 最终进入 external feasible archive 的比例；
- `boundary-induced ΔHV`：边界分支带来的累计 HV 增益；
- `migration success rate`：对 feasible seed 触发的局部迁移，经过若干代后是否还在 external archive 或至少改善对应 sector。

这里特别关键的是第三项。因为你的历史设想最在意“迁移到更有价值的可行域”，那就必须单独量化“迁移成功率”，而不是只看最终 HV。fileciteturn0file0 fileciteturn0file2

**Step 3：最后再接终局性能。**\
在 9 个问题上做 30 runs，报 HV/IGD 终值和过程 AUC，并做 Wilcoxon + Holm。这样证据链才完整：\
\*\*校准通过 → 更靠近真实边界 → 更常找到有用 feasible discoveries / 更稳定完成局部迁移 → 最终 PF 更好。\*\*fileciteturn0file1

最后给你一个最实在的结论：

**现在最该做的，不是再加新机制，而是先让 Step 0 过关。**\
只要 `p≈0.5` 还不可信，你的最大创新点就只能写成 heuristic；一旦 Step 0 过了，再用 Gate 2/3 把“更靠边界”和“更有 Pareto 价值”连起来，这个创新点才真正有机会站住。


#### You:
我十分认同你的分析，请你仔细思考，给出下一步的详细改进方案和实验方案
1：给出“不确定性语义还不够可信”的详细解决方案
2：删除不必要的小机制“主算法里塞了太多“看起来合理”的小机制，但它们没有被证明都在帮忙”
3：精简算法主线（突出核心创新点，不要大杂烩）；精简成合理的子代审生成策略和环境选择策略；最重要的是：合理且精简的核心创新点
4：依据修改完成的算法来设计详细的实验（唯一目的：验证核心创新点的有效性）
注意：给出详细的改进策略、重新设计后的公式、流程、实验设置


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
下面给你一个我认为最适合直接落地到 PlatEMO、也最容易写成论文的改版。

先定主线：**把算法收缩成 1 个核心创新点**——\
**可信的 Pareto-bridge boundary query + label-aware boundary refinement**。

也就是：不再把“$p\approx 0.5$”当成全局真理，而是只在**同一 Pareto 扇区内、显式可行-不可行桥接对**上，用**经校准且经局部可靠性修正**的边界概率去放置查询点；查询到真实标签后，再做**标签感知的局部细化**：可行就向更优区域推进，不可行就二分括逼或确认 hard negative。这样既贴合你原始设想里的“0.5 附近找边界、发现更好可行域并迁移”，也比当前 PRBCCMO 的“entropy × HV × novelty × penalty 大礼包”更瘦、更可证。你现有 Section B 结果已经说明当前版本最大的问题不是“机制不够多”，而是“概率语义不可信 + 打分栈过厚”：Gate 1 全败，Gate 2/3 里 random-boundary 经常优于 Full。BE-CBO 和 COMBOO 这类工作又都说明，在 unknown constraints 场景下，“边界学习/level-set 学习 + 可行区优化”本来就是合理主线。citeturn471430view3turn471430view4 fileciteturn3file2 fileciteturn3file0 fileciteturn3file1

## 1. “不确定性语义还不够可信”的详细解决方案

这部分不要再靠“换个激活函数”硬修，应该分成**数据层、模型层、决策层**三层一起改。现代神经网络的概率本来就常失准；Guo 等说明 temperature scaling 在很多 i.i.d. 场景有效，但 Ovadia 等又表明仅靠 post-hoc calibration 在分布漂移下常常不够，而 deep ensembles 这类 model averaging 方法更稳。你的进化过程本质上就是在线分布漂移，所以必须三层同时修。citeturn471430view0turn782735view0turn471430view1

### 1.1 数据层：让“边界样本”成为训练主角，而不是训练噪声

把训练数据拆成 4 个缓冲区：

$
D_g:\text{全局 FIFO 标注集},\quad |D_g|=2N
$

$
D_b:\text{tight bracket 集合},\quad |D_b|=N
$

$
D_{cal}:\text{校准缓冲},\quad |D_{cal}|=N
$

$
D_{test}:\text{独立测试缓冲},\quad |D_{test}|=N
$

其中：

- $D_g$ 收所有真实评估过的点，保留全局分布，继续用 FIFO。
- $D_b$ 只收**二分括逼得到的紧致可行-不可行对** $(x_f,x_i)$ 以及其中点 $x_m=\frac{x_f+x_i}{2}$。这是最关键的“边界真证据”。
- $D_{cal}$ 与 $D_{test}$ 不参与训练，只做校准选择和 drift 监测。它们必须**边界富集**：至少 50% 来自 tight bracket endpoints / midpoints 或 $p_c\in[0.35,0.65]$ 的样本，否则你测出来的是“全局分类校准”，不是你真正要用的“边界语义校准”。fileciteturn3file2

训练时不要再用当前那套“global + pair + mid + hard negative 全都堆损失”的大杂烩。主损失只保留 4 项：

$
L = L_{\text{WBCE}} + \lambda_B L_{\text{Brier}} + \lambda_P L_{\text{pair}} + \lambda_M L_{\text{mid}}
$

其中

$
L_{\text{WBCE}}=-\frac1n\sum_i w_i\Big[y_i\log p_i+(1-y_i)\log(1-p_i)\Big]
$

$
L_{\text{Brier}}=\frac1n\sum_i (p_i-y_i)^2
$

$
L_{\text{pair}}=\frac1{|B|}\sum_{(x_f,x_i)\in B}\max\big(0,m-p(x_f)+p(x_i)\big)
$

$
L_{\text{mid}}=\frac1{|M|}\sum_{x_m\in M}(p(x_m)-0.5)^2
$

推荐默认值：

$
m=0.05,\quad \lambda_B=0.5,\quad \lambda_P=1,\quad \lambda_M=1
$

这里有两个关键删改：

第一，**保留 Brier，删除 hard-negative 专属 loss**。  
因为 hard negative 本质上已经是“真实标签为 0 的样本”，再额外加一个 hard-negative loss，容易把模型往“保守低概率”方向推过头，反而恶化 near-band calibration。你可以保留 hard negative 样本，但它们只作为普通负样本进入 $L_{\text{WBCE}}+L_{\text{Brier}}$，不再单列一项。

第二，**midpoint 监督只对 tight brackets 生效**。  
只有当 $\Delta(x_f,x_i)\le \varepsilon_b$ 时才把中点当作“近边界证据”。推荐：

$
\varepsilon_b=0.03
$

这里的 $\Delta$ 用归一化决策距离。这样 $p\approx 0.5$ 的语义来自真实几何证据，不是拍脑袋。

### 1.2 模型层：用“小 committee + 在线校准选择”，而不是单个 raw MLP

模型我建议定死成：

- **3 个浅层 MLP 的 committee**
- 每个 MLP：**1 个隐藏层，20 个 hidden units，tanh hidden + sigmoid output**
- 输入仍是归一化决策变量

形式写成：

$
h_k(x)=\tanh(W^{(1)}_k\hat x+b^{(1)}_k),\quad
z_k(x)=W^{(2)}_k h_k(x)+b^{(2)}_k,\quad
p_k(x)=\sigma(z_k(x))
$

$
\bar p(x)=\frac{1}{K}\sum_{k=1}^K p_k(x),\quad
s_p(x)=\sqrt{\frac1K\sum_{k=1}^K\big(p_k(x)-\bar p(x)\big)^2}
$

其中 $K=3$。

我不建议回到 NA-EMT 的 ReLU + SCG。原因不是 ReLU 不行，而是你现在的模型职责已经从“离线软排序器”变成“在线边界查询器”了。这个场景更看重**平滑、可 warm-start、可频繁小步更新、便于 midpoint-to-0.5 约束**。浅层 tanh 在这里比 ReLU+SCG 更顺手，也更接近你当前代码的实现。NA-EMT 的准确率触发重训逻辑适用于它的 ISVPS/CDPPV 角色，但不适用于你现在这种把概率直接拿来调度查询的角色。fileciteturn3file1 fileciteturn3file2

### 1.3 校准层：不要固定只用 temperature 或 sigmoid，要做“二选一 + 信任门”

当前 Gate 1 里 raw / temperature / sigmoid / 3-MLP+temperature 都失败，说明问题不只是“温度参数没调好”，而是**不同阶段的 score distortion 形式在变**。Guo 等说明 temperature scaling 是很强的基线，但 Kull 等也指出 logistic/platt 一类校准在某些 skewed score distortion 下可能比原始分数更差，而 beta calibration 更灵活。citeturn471430view0turn678842view0

所以在线阶段不要只固定一种 calibrator，而是每次更新后在 $D_{cal}$ 上同时拟合：

$
p_T(x)=\sigma(z(x)/T)
$

和

$
p_\beta(x)=T_\beta(\bar p(x);a,b,c)
$

然后按下面这个校准选择目标挑一个：

$
J = \text{Brier} + \text{ECE} + 2\cdot \text{NearGap}_{[0.45,0.55]}
$

谁的 $J$ 更小，谁就是当前代的 $p_c(x)$。

然后再加一个**信任门**：

$
\mathcal G_t=
\mathbf 1\Big(
\text{ECE}_t\le 0.05,\ 
\text{NearGap}_{[0.45,0.55],t}\le 0.05,\ 
n_{[0.45,0.55],t}\ge 20
\Big)
$

只有 $\mathcal G_t=1$ 时，才允许“$p_c\approx 0.5$”直接参与 seed 排序。  
否则进入 **cautious mode**：模型仍然更新，但只用它做弱参考，不让它决定 query ranking。

这一步非常关键。它等于承认：**0.5 不是天经地义的边界语义，而是“在通过校准门之后才被激活的语义”。**

### 1.4 决策层：把“0.5”改成“经可靠性修正的 boundary trust”

最终 seed 评分不要再用你当前那种

$
\text{entropy}\times \text{disagreement}\times \text{HVGain}\times \text{novelty}\times \text{penalty}
$

而是压缩成两部分：

一部分是**可信边界分数**

$
Q(x)=R_{b(x)}\cdot q(x)\cdot (1+\lambda_\sigma s_p(x))
$

其中

$
q(x)=\max\big(0,1-2|p_c(x)-0.5|\big)
$

$R_{b(x)}$ 是该概率 bin 的经验可靠性。对 near-band 的 bin $j$，定义：

$
R_j=\max\big(0,\ 1-2|\hat\pi_j-0.5|\big)
$

这里 $\hat\pi_j$ 是 $D_{test}$ 里落在该 bin 的实际可行率。  
直观上说：如果某个概率 bin 看起来在模型里接近 0.5，但真实可行率其实总偏到 0.8 或 0.2，那它的 $R_j$ 就很低，这个 bin 的“边界语义”就自动失效。

另一部分是**Pareto 相关性**

$
V_s(x)=\max\left(
0,\ 
g(c_s|w_s)-\tilde g(x|w_s)
\right)
$

这里 $c_s$ 是 sector $s$ 里的当前 feasible champion，$\tilde g$ 用 bridge proxy objective 计算。最终 utility：

$
U_s(x)=Q(x)\cdot(\epsilon+V_s(x))
$

默认：

$
\lambda_\sigma=1,\quad \epsilon=10^{-6}
$

这样一来，不确定性只负责“像不像边界”，Pareto 项只负责“值不值得问”，两者分工清楚。

## 2. 删除不必要的小机制

这一步我建议非常坚决。主文里删，附录里保留为 ablation。

删除 1：**4 源候选池**。  
主算法只保留 **1 个 seed 来源**：当前代的 sector-wise bridge。  
局部扰动不再是 seed pool 来源，而是 seed 被选中并真实评估后才进入 worker phase。

删除 2：**HVGain 进入主 score**。  
你现在的 proxy-HV 在 seed 还是未评估状态时很不稳，而且 current Section B 也没证明它有用。用 sector champion improvement $V_s$ 替代即可。fileciteturn3file2

删除 3：**novelty 进入主 score**。  
因为 seed 本身已经是 “每个 sector 至多 1 个桥”，多样性由 sector construction 保证，不需要再乘一个 novelty。

删除 4：**hard-negative penalty 进入主 score**。  
hard negative 只保留成“排除区”，不用惩罚因子乘进去。即：若某 candidate 落在已确认 hard-negative 区域内，直接不生成/不选；否则不再额外扣分。

删除 5：**hard-negative 专属训练损失**。  
保留 hard negative 样本，删掉 hard-negative loss。

删除 6：**边界概率参与环境选择**。  
$p\approx 0.5$ 只用于“查询和细化调度”，不再直接影响 $P_C$ 的环境选择。这样逻辑最干净：  
**只有真实评估后变成有价值 feasible descendants 的边界样本，才配进入主种群。**

删除 7：**GA/DE 双开关**。  
主文固定一个 regular offspring operator，建议直接沿用 NA-EMT 的 DE/current-to-rand/1，这样和原文最可比，也把“创新来源”锁定在你的 boundary 模块，不让审稿人说是 operator luck。fileciteturn3file1

## 3. 精简后的算法主线

### 3.1 最终的核心创新点

我建议你论文里就写这一句：

**A reliability-corrected Pareto-bridge boundary query-and-refine mechanism for binary unknown constraints.**

中文就是：

**一种面向 Pareto 桥接对的、经可靠性修正的边界查询—细化机制。**

双种群只是脚手架，不是创新点。MLP 只是边界查询器，不是创新点。真正的创新，是“桥接 + 可信 0.5 + 标签感知细化”这个闭环。

### 3.2 子代产生策略：只保留两条线

**线 1：regular evolution**

- $P_C$：标准 DE 产生 $O_C$
- $P_U$：标准 DE 产生 $O_U$

**线 2：boundary module**

用 reference vectors $W=\{w_s\}$ 划分 sectors。对每个 sector $s$：

$
c_s=\arg\min_{x\in P_C^F\cap S_s} g(x|w_s)
$

$
u_s=\arg\min_{x\in P_U^I\cap S_s} g(x|w_s)
$

其中 $g$ 用归一化加权和即可：

$
g(x|w_s)=\sum_{m=1}^M w_{sm}\tilde f_m(x),\qquad
\tilde f_m(x)=\frac{f_m(x)-z_m^{\min}}{z_m^{\max}-z_m^{\min}+\varepsilon}
$

只有当

$
g(u_s|w_s)+\delta_g < g(c_s|w_s)
$

时，sector $s$ 才激活为一个 bridge。默认：

$
\delta_g = 0.01
$

然后定义 bridge segment：

$
\ell_s(\lambda)=\Pi\big((1-\lambda)c_s+\lambda u_s\big),\qquad \lambda\in[0,1]
$

如果当前 $\mathcal G_t=1$，就在

$
\Lambda=\{0.2,0.35,0.5,0.65,0.8\}
$

上做便宜的 model-only scan，取

$
\lambda_s^\star=\arg\min_{\lambda\in\Lambda}|p_c(\ell_s(\lambda))-0.5|
$

得到 seed：

$
x_s^\star=\ell_s(\lambda_s^\star)
$

如果 $\mathcal G_t=0$，直接退回 bridge midpoint：

$
x_s^\star=\ell_s(0.5)
$

然后用上面的 $U_s$ 排序，选 top-$B_s$ 去真实评估。

这一步把你的“0.5附近主动采样”落成了非常具体、也非常容易讲清楚的机制：  
**不是到处找 0.5，而是在“明确更优的可行-不可行桥”上，用 0.5 去放查询点。**

### 3.3 标签感知局部搜索：真正把 seed 变成“迁移”或“反证”

这是最关键的 worker phase。

#### A. 若 seed 可行：向更优 infeasible helper 方向做受控推进

设 seed 为 $x_f$，对应 helper 为 $u_s$。取

$
d=\frac{u_s-x_f}{\|u_s-x_f\|+\varepsilon}
$

先做一小步 forward trial：

$
x^{(1)}=\Pi(x_f+\alpha_1 d+\beta_1 \xi_\perp)
$

推荐：

$
\alpha_1=0.06,\quad \beta_1=0.02
$

如果 $x^{(1)}$ 可行且

$
g(x^{(1)}|w_s) < g(c_s|w_s)
$

就加入 migration pool。

如果 $x^{(1)}$ 不可行，再做一次 backtrack：

$
x^{(2)}=\Pi\left(\frac{x_f+x^{(1)}}{2}\right)
$

若 $x^{(2)}$ 可行且 improved，也加入 migration pool。

这比你当前“从 feasible seed 朝远离 infeasible anchor 的方向扩张”更合理，因为你真正想要的是**向更优目标区域推进，但别一下掉进 infeasible interior**。

#### B. 若 seed 不可行：二分括逼，找到边界附近的 feasible 入口

设当前 seed 为 $x_i$，对应 feasible anchor 为 $c_s$。初始化：

$
f^{(0)}=c_s,\qquad i^{(0)}=x_i
$

做最多 3 次 bisection：

$
m^{(t)}=\Pi\left(\frac{f^{(t)}+i^{(t)}}{2}\right)
$

如果 $m^{(t)}$ 可行，则 $f^{(t+1)}=m^{(t)}$；否则 $i^{(t+1)}=m^{(t)}$。

若找到 tight feasible endpoint $f^\star$，保存 tight pair：

$
(f^\star,i^\star),\qquad
\Delta^\star=\left\|\frac{f^\star-i^\star}{u-l}\right\|/\sqrt D
$

并把

$
x_m^\star=\frac{f^\star+i^\star}{2}
$

送进 bracket buffer。然后再从 $f^\star$ 向 helper 方向做 1 次 forward trial，尝试打开更好的 feasible patch。

如果 3 次括逼后仍没拿到可用 feasible endpoint，就做 2 个 confirmation samples：

$
x^{hn}_j=\Pi(x_i+\rho_j\eta_j),\qquad \rho_j\in\{0.5\Delta,0.25\Delta\}
$

若它们都不可行，就把 $x_i$ 和记作 hard-negative center，半径记作 $\Delta$。  
以后任何 candidate 若满足

$
\min_h \frac{\|x-h\|_{\text{norm}}}{\rho_h} < 1
$

直接跳过。

这时 hard negative 的作用很清楚：  
它是“此局部附近大概率没有可行入口”的证据，不再是一个额外的 score penalty 或额外 loss。

### 3.4 环境选择：让边界模块只通过“真实可行收益”进入主种群

这一块越简单越好。

对 helper population：

$
P_U \leftarrow NSGA2(P_U\cup O_U,\ N)
$

完全忽略约束。

对 constrained population，先组候选池：

$
\Omega_C = P_C \cup O_C \cup M
$

其中 $M$ 是 migration pool，只含真正通过 boundary module 找到的 improved feasible descendants。

然后做两步：

**第一步：sector reservation**  
对每个 sector $s$，如果 $M$ 中最好的 candidate $m_s$ 满足

$
g(m_s|w_s) < g(c_s|w_s) - \delta_{mig}
$

则预留 1 个 slot 给 $m_s$。默认：

$
\delta_{mig}=0
$

**第二步：feasible-first NSGA-II**  
对剩余 slot，在 $\Omega_C^F$ 上做 NSGA-II 环境选择。  
如果 feasible 数量仍不足 $N$，再用 objective-best 的 candidates 补满，防止 cold start 崩掉。

这样 $p\approx0.5$ 的解不会直接“凭预测分数”进主种群；它们必须先通过真实评估，变成真正有收益的 feasible descendants，才会推动迁移。这一步会让算法逻辑特别清楚。

### 3.5 更新频率与默认参数

这一版的更新频率我建议定成：

- 模型常规更新：每 **5** 代
- warm-start：**8** epochs
- full restart：每 **25** 代一次
- 提前更新触发：  
  新边界标签数 $\ge 0.1\cdot |D_g|$  
  或 ECE 上升 $>0.02$  
  或 core near-gap 上升 $>0.05$

默认参数可先定成：

$
N=100,\quad \text{maxFE}=200000,\quad B=\lfloor0.2N\rfloor
$

$
B_s=\lfloor0.4B\rfloor,\quad B_w=B-B_s
$

$
K=3,\quad \text{hidden}=20,\quad \text{full epoch}=25,\quad \text{warm epoch}=8
$

$
|D_g|=2N,\quad |D_b|=N,\quad |D_{cal}|=N,\quad |D_{test}|=N
$

regular DE 参数直接跟 NA-EMT 保持一致。这样做的好处是：你论文里唯一大改动就是 boundary module，比较最公平。fileciteturn3file1

## 4. 依据新算法设计“只为验证核心创新点”的实验

这里我建议你把实验写成一条完整证据链：

**校准可信  →  真更靠近边界  →  真更能恢复/迁移到有价值可行点  →  最终 PF 更好**

不要再只做 pooled all rows。**统计单位必须是 run，不是 seed 行。** seed-level 数据只做 descriptive plot，真正显著性检验必须先聚合到每个 run。你当前 Section B 已经有 oracle distance 和 gate audit 基础设施，直接接着改最合适。fileciteturn3file2

### 4.1 变体设置

只做 4 个内部变体，就够了：

**V0 Random-Bridge**  
同样的 bridge construction；seed 位置用 midpoint；随机选 seed；local 用 isotropic。  
作用：最弱基线。

**V1 Uncertain-Only**  
同样的 bridge construction；seed 位置用模型找 $p_c\approx 0.5$；排序只看 $q(x)$；local 仍 isotropic。  
作用：证明“只靠 0.5 不够”。

**V2 Trusted-Query**  
同样的 bridge construction；用本文的 $U_s=Q(\epsilon+V)$ 选 seed；local 仍 isotropic。  
作用：单独证明“可信 query”有效。

**V3 Full**  
Trusted-Query + label-aware refinement。  
作用：证明“可信 query + 标签感知细化”这个完整核心创新有效。

补充说明：  
NA-EMT、EADMM、DRMCMO 可以放到最后一张 end-to-end 性能表里做外部参照，但**不要**拿它们做核心创新的因果对照。核心创新是否成立，必须靠上面这 4 个内部变体来证明。citeturn723238search2turn723238search3 fileciteturn3file1

### 4.2 实验 0：校准与信任门

**目标**：证明“0.5 的语义终于可信了”。

在 DASCMOP1_BC–DASCMOP9_BC 上做 30 runs，记录每次模型更新后的 held-out 指标：

- ECE
- Brier
- core near-gap：$p_c\in[0.45,0.55]$ 的实际可行率与 0.5 的差
- relaxed near-gap：$p_c\in[0.4,0.6]$
- trust gate pass rate

建议把 calibrator 对比定成：

- raw
- temperature
- beta
- online best-of-(temperature,beta) + trust gate

成功标准：

- pooled ECE $\le 0.05$
- core near-gap $\le 0.05$
- 至少 7/9 问题上 pass rate 明显高于 raw 和 temperature-only

### 4.3 实验 1：真更靠近真实边界吗？

这里继续用你现在的 oracle distance 思路。对 BC 版本，算法**在线只看二值 label**，原始连续约束只在线下算：

$
d_B(x)=\min_j \frac{|g_j^{raw}(x)|}{s_j+\epsilon}
$

其中 $s_j$ 用离线随机采样估尺度。你现有 `ComputeOracleBoundaryDistance` 已经在这么做了。fileciteturn3file2

对每个 run 先算：

$
\text{median-}d_B^{(r)},\quad \text{mean-}d_B^{(r)}
$

再画 pooled CDF：

$
P(d_B<\tau)
$

比较 V0–V3。  
这里最关键的检验是：

- V2 是否显著优于 V1 和 V0  
- V3 至少不差于 V2

如果 V2 不比 V1 好，就说明“可信 query”没站住。  
如果 V3 把边界距离搞差了，就说明 local refine 反而在破坏 query。

### 4.4 实验 2：这些边界点有没有“用”？

只靠 $d_B$ 不够，因为靠近边界不代表有 Pareto 价值。要加 5 个 run-level 指标：

**FRR**：infeasible seeds 的 feasible recovery rate

$
FRR=\frac{\#\{\text{infeasible seed 产生至少一个 feasible descendant}\}}{\#\{\text{infeasible seeds}\}}
$

**TBR**：tight bracket rate

$
TBR=\frac{\#\{\Delta^\star\le \varepsilon_b\}}{\#\{\text{infeasible seeds}\}}
$

**MSR**：migration success rate

$
MSR=\frac{\#\{\text{feasible seed 的后代在 }G_s\text{ 代后仍存活或进入外部档案}\}}{\#\{\text{feasible seeds}\}}
$

建议 $G_s=10$。

**UBY**：useful boundary yield

$
UBY=\frac{\#\{\text{boundary-originated feasible points 首次进入 external archive}\}}{\#\{\text{selected seeds}\}}
$

**event-level boundary }\Delta HV**  
不要再用当前那种 generation-batch delta，太容易全 0。改成事件级：

$
\Delta HV_B^{event}(x)=HV(A_{pre}\cup\{x\})-HV(A_{pre})
$

其中 $x$ 是某个 boundary-originated feasible point 在**首次进入** external archive 时的独占贡献。每个 run 求和：

$
\Delta HV_B^{(r)}=\sum_{x\in \mathcal A_B^{enter}} \Delta HV_B^{event}(x)
$

这一改动很重要，它能避免你当前 Gate 3 里 `boundary_delta_hv = 0` 那种“指标定义太粗而看不出贡献”的问题。fileciteturn3file2

这一实验你真正想看到的是：

- V2 比 V1 有更高的 UBY
- V3 比 V2 有更高的 FRR / TBR / MSR / event-$\Delta HV\_B$

这就能把“可信 query”与“标签感知 refine”各自的作用拆开。

### 4.5 实验 3：最终优化效果

最后再接终局性能。\
设置保持与你当前和 NA-EMT 一致：

- 问题：DASCMOP1\_BC–9\_BC
  - $N=100$
- maxFE = 200000
- 30 runs
- 同一随机种子配对跑各变体

指标：

- final HV
- final IGD
- AUC(HV)
- first-hit FE（首次达到 90% final-HV 的 FE）

统计：

- 若使用同一组随机种子配对，优先 **Wilcoxon signed-rank**
- 多重比较做 **Holm**
- 同时报 **Cliff’s delta**

这里的成功标准不必像机制实验那样苛刻。只要 V3 在多数问题上显著优于 V1/V0，并且 AUC(HV) 更好，你的核心创新就基本立住了。

---

我最后给你一个最务实的执行顺序：

先重写 4 块：`SelectBoundaryCandidates`、`RefineBoundaryWorkers`、`UpdateBoundaryModel`、`EnvironmentalSelectionC`。\
顺序不要反：先删 score stack，再上 trust gate，再改 label-aware refine，最后改实验日志。

只要你先把 **“桥上放点 + 标签感知细化 + 只让真实可行收益进入主种群”** 这条主线跑顺，论文就会从“很多技巧的组合”变成“一个能被清楚验证的创新点”。


#### You:
这是目前的算法实现，你来完成下列要求：
1：分析当前算法是否符合预期的实现；
2：当前的思想是否还有改进的方向（当前思想是否仍然臃肿、冗余、堆砌欸、存在无用技巧/机制），是否可以更高效的精简、替换，如果有请详细给出改发
3：根据再次修改后的算法（如果有的话，否则就按当前实现），给出详细的创新点验证实验（上次给的不够详细，不要参考以前的实验设置，请重新考虑），需要给出详细的指标计算公式、如何收集等等所有细节


#### Plugin (file_search):
Make sure to include fileciteturn5file0 in your response to cite this file, or to surface it as a link.
PRBCCMO MATLAB implementation package
Source directory: Algorithms\Multi-objective optimization\PRBCCMO
Generated at: 2026-03-20 15:58:47
File count: 21

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\AssociateSectors.m
====================================================================================================

function [Sector,Count] = AssociateSectors(PopObj,W,RefObj)
% Associate objective vectors to the nearest reference sector.

    if nargin < 3 || isempty(RefObj)
        RefObj = PopObj;
    end
    if isempty(PopObj)
        Sector = zeros(0,1);
        Count  = zeros(size(W,1),1);
        return;
    end
    if isempty(W)
        Sector = ones(size(PopObj,1),1);
        Count  = size(PopObj,1);
        return;
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range  = MaxObj - MinObj;
    Range(Range<1e-12) = 1;

    Obj = (PopObj-MinObj)./Range;
    RowNorm = sqrt(sum(Obj.^2,2));
    ZeroRow = RowNorm < 1e-12;
    Obj(ZeroRow,:) = 1;
    RowNorm(ZeroRow) = sqrt(size(Obj,2));
    Obj = Obj./repmat(RowNorm,1,size(Obj,2));

    WNorm = sqrt(sum(W.^2,2));
    WNorm(WNorm<1e-12) = 1;
    Wn = W./repmat(WNorm,1,size(W,2));

    Cosine = Obj*Wn';
    [~,Sector] = max(Cosine,[],2);
    Count = accumarray(Sector,1,[size(W,1),1]);
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\BuildBoundaryRuntimeOptions.m
====================================================================================================

function Options = BuildBoundaryRuntimeOptions(SelectionMode,LocalMode,TraceFlag)
% Build runtime options for trust-aware Pareto-bridge querying.

    if nargin < 1 || isempty(SelectionMode)
        SelectionMode = 1;
    end
    if nargin < 2 || isempty(LocalMode)
        LocalMode = 1;
    end
    if nargin < 3 || isempty(TraceFlag)
        TraceFlag = false;
    end

    SelectionMode = max(1,min(3,round(SelectionMode)));
    LocalMode     = max(1,min(2,round(LocalMode)));

    Options = struct();
    Options.SelectionMode = SelectionMode;
    Options.LocalMode     = LocalMode;
    Options.TraceFlag     = logical(TraceFlag);
    Options.SelectionName = ResolveSelectionName(SelectionMode);
    Options.LocalName     = ResolveLocalName(LocalMode);
    Options.BridgeActivationGap = 0.01;
    Options.BridgeScanLambda    = [0.20,0.35,0.50,0.65,0.80];
end

function Name = ResolveSelectionName(SelectionMode)
    switch SelectionMode
        case 1
            Name = 'trusted_query';
        case 2
            Name = 'uncertain_only';
        case 3
            Name = 'random_bridge';
        otherwise
            Name = 'trusted_query';
    end
end

function Name = ResolveLocalName(LocalMode)
    switch LocalMode
        case 1
            Name = 'label_aware';
        case 2
            Name = 'isotropic';
        otherwise
            Name = 'label_aware';
    end
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\CalFitness.m
====================================================================================================

function Fitness = CalFitness(PopObj,PopCon)
% Calculate the SPEA2-style fitness used by CCMO.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);
    if nargin == 1
        CV = zeros(N,1);
    else
        % PRBCCMO targets unknown binary constraints, so all infeasible
        % solutions share the same violation level.
        CV = double(any(PopCon>0,2));
    end

    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end

    S = sum(Dominate,2);
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end

    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);

    Fitness = R + D';
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\ComputeSectorScalar.m
====================================================================================================

function Value = ComputeSectorScalar(Obj,W,RefObj,Sector)
% Compute the normalized weighted-sum scalar value for one or more sectors.

    if isempty(Obj)
        Value = zeros(0,1);
        return;
    end
    if nargin < 2 || isempty(W)
        W = ones(1,size(Obj,2));
    end
    if nargin < 3 || isempty(RefObj)
        RefObj = Obj;
    end
    if nargin < 4 || isempty(Sector)
        if size(W,1) == 1
            Weight = repmat(W,size(Obj,1),1);
        elseif size(W,1) == size(Obj,1)
            Weight = W;
        else
            Weight = repmat(W(1,:),size(Obj,1),1);
        end
    else
        Weight = W(Sector,:);
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range  = MaxObj - MinObj;
    Range(Range<1e-12) = 1;
    NormObj = (Obj - MinObj)./Range;
    Value   = sum(NormObj.*Weight,2);
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\EnvironmentalSelectionC.m
====================================================================================================

function [Population,Fitness,SelectedIdx] = EnvironmentalSelectionC(Population,N,MigrationPool,W)
% Reserve slots for improved feasible migrants, then apply feasible-first NSGA-II.

    if nargin < 3 || isempty(MigrationPool)
        MigrationPool = [];
    end
    if nargin < 4
        W = [];
    end
    if isempty(Population) && isempty(MigrationPool)
        Fitness = [];
        SelectedIdx = [];
        return;
    end

    BasePool = KeepUniquePopulation(Population);
    MigrationPool = KeepUniquePopulation(MigrationPool);
    Reserved = SelectReservedMigrants(BasePool,MigrationPool,W);
    if numel(Reserved) > N
        Reserved = Reserved(1:N);
    end

    CandidatePool = KeepUniquePopulation([BasePool,MigrationPool]);
    CandidatePool = RemovePopulationByDecision(CandidatePool,Reserved);

    Selected = Reserved;
    Need = max(0,N-numel(Selected));
    if Need > 0
        FeasiblePool = CandidatePool(all(CandidatePool.cons<=0,2));
        [Chosen,~] = SelectObjectivePopulation(FeasiblePool,min(Need,numel(FeasiblePool)));
        Selected = [Selected,Chosen];
    end

    Need = max(0,N-numel(Selected));
    if Need > 0
        Remaining = RemovePopulationByDecision(CandidatePool,Selected);
        [Chosen,~] = SelectObjectivePopulation(Remaining,min(Need,numel(Remaining)));
        Selected = [Selected,Chosen];
    end

    if isempty(Selected)
        Fitness = [];
        SelectedIdx = [];
        Population = [];
        return;
    end
    if numel(Selected) < N
        Repeat = Selected(mod(0:N-numel(Selected)-1,numel(Selected))+1);
        Selected = [Selected,Repeat];
    end

    Population = Selected(:)';
    Fitness    = CalFitness(Population.objs,Population.cons);
    SelectedIdx = 1:numel(Population);
end

function Reserved = SelectReservedMigrants(BasePool,MigrationPool,W)
    Reserved = [];
    if isempty(MigrationPool)
        return;
    end

    FeasibleBase = BasePool(all(BasePool.cons<=0,2));
    FeasibleMig  = MigrationPool(all(MigrationPool.cons<=0,2));
    if isempty(FeasibleMig)
        return;
    end

    if isempty(W)
        W = ones(1,size(FeasibleMig.objs,2));
    end
    RefObj = FeasibleMig.objs;
    if ~isempty(FeasibleBase)
        RefObj = [FeasibleBase.objs;FeasibleMig.objs];
    end

    SectorMig = AssociateSectors(FeasibleMig.objs,W,RefObj);
    if isempty(FeasibleBase)
        SectorBase = zeros(0,1);
    else
        SectorBase = AssociateSectors(FeasibleBase.objs,W,RefObj);
    end

    Keep = false(1,numel(FeasibleMig));
    Improve = -inf(1,numel(FeasibleMig));
    for s = unique(SectorMig(:))'
        MigIdx = find(SectorMig==s);
        MigValue = ComputeSectorScalar(FeasibleMig(MigIdx).objs,W,RefObj,repmat(s,numel(MigIdx),1));
        [BestMigValue,BestLocal] = min(MigValue);
        BestMigIdx = MigIdx(BestLocal);

        BaseIdx = find(SectorBase==s);
        if isempty(BaseIdx)
            Keep(BestMigIdx) = true;
            Improve(BestMigIdx) = inf;
            continue;
        end

        BaseValue = ComputeSectorScalar(FeasibleBase(BaseIdx).objs,W,RefObj,repmat(s,numel(BaseIdx),1));
        ChampionValue = min(BaseValue);
        if BestMigValue < ChampionValue
            Keep(BestMigIdx) = true;
            Improve(BestMigIdx) = ChampionValue - BestMigValue;
        end
    end

    Reserved = FeasibleMig(Keep);
    if isempty(Reserved)
        return;
    end
    Improve = Improve(Keep);
    [~,Order] = sort(Improve,'descend');
    Reserved = Reserved(Order);
end

function Population = KeepUniquePopulation(Population)
    if isempty(Population)
        return;
    end
    Keep = KeepLatestDecisionRows(Population.decs);
    Population = Population(Keep);
end

function Population = RemovePopulationByDecision(Population,Remove)
    if isempty(Population) || isempty(Remove)
        return;
    end
    Keep = ~ismember(Population.decs,Remove.decs,'rows');
    Population = Population(Keep);
end

function [Population,Idx] = SelectObjectivePopulation(Population,N)
    Idx = zeros(1,0);
    if isempty(Population) || N <= 0
        Population = [];
        return;
    end
    LocalIdx = SelectObjectiveIdx(Population.objs,min(N,numel(Population)));
    Population = Population(LocalIdx);
    Idx = LocalIdx(:)';
end

function Idx = SelectObjectiveIdx(PopObj,N)
    if isempty(PopObj)
        Idx = zeros(1,0);
        return;
    end
    [FrontNo,MaxFNo] = NDSort(PopObj,N);
    Next             = FrontNo < MaxFNo;
    CrowdDis         = CrowdingDistance(PopObj,FrontNo);
    Last             = find(FrontNo==MaxFNo);
    Need             = N - sum(Next);
    if Need > 0
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:Need))) = true;
    end
    Idx = find(Next);
    [~,Order] = sortrows([FrontNo(Idx)',-CrowdDis(Idx)'],[1 2]);
    Idx = Idx(Order)';
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\EnvironmentalSelectionU.m
====================================================================================================

function [Population,Fitness] = EnvironmentalSelectionU(Population,N)
% Environmental selection for the unconstrained helper population.

    Population = SelectByNSGA2Local(Population,N);
    Fitness = CalFitness(Population.objs);
end

function Population = SelectByNSGA2Local(Population,N)
    if isempty(Population)
        return;
    end

    if numel(Population) <= N
        [FrontNo,~] = NDSort(Population.objs,numel(Population));
        CrowdDis = CrowdingDistance(Population.objs,FrontNo);
        [~,Order] = sortrows([FrontNo(:),-CrowdDis(:)],[1 2]);
        Population = Population(Order);
        return;
    end

    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next     = FrontNo < MaxFNo;
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    Last     = find(FrontNo==MaxFNo);
    Need     = N - sum(Next);
    if Need > 0
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:Need))) = true;
    end

    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    [~,Order]  = sortrows([FrontNo(:),-CrowdDis(:)],[1 2]);
    Population = Population(Order);
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\EvaluateBoundaryCalibration.m
====================================================================================================

function Metric = EvaluateBoundaryCalibration(Model,CalDec,CalLabel,BinCount)
% Evaluate calibration quality on a held-out labeled evaluation buffer.

    if nargin < 4 || isempty(BinCount)
        BinCount = 10;
    end

    Metric = SummarizeCalibrationProbabilities([],[],BinCount);
    Metric.brier     = inf;
    Metric.ece       = inf;
    Metric.nearGap   = inf;
    Metric.nearCount = 0;
    Metric.trustGate = false;
    Metric.calibrator = 'raw';
    if isempty(CalDec) || isempty(CalLabel) || isempty(Model)
        return;
    end

    CalLabel = double(CalLabel(:));
    if numel(unique(CalLabel)) < 2
        return;
    end

    Prob = PredictBoundaryMLP(Model,CalDec);
    Metric = SummarizeCalibrationProbabilities(Prob,CalLabel,BinCount);
    if isfield(Model,'TrustGate') && ~isempty(Model.TrustGate)
        Metric.trustGate = logical(Model.TrustGate);
    end
    if isfield(Model,'CalibratorType') && ~isempty(Model.CalibratorType)
        Metric.calibrator = Model.CalibratorType;
    end
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\GenerateBoundaryCandidates.m
====================================================================================================

function Pool = GenerateBoundaryCandidates(Problem,PopulationC,PopulationU,W,RuntimeOptions)
% Build one bridge record for each active feasible-infeasible sector pair.

    if nargin < 5 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    Pool = InitBridgePool(Problem);

    FeasibleC   = PopulationC(all(PopulationC.cons<=0,2));
    InfeasibleU = PopulationU(~all(PopulationU.cons<=0,2));
    if isempty(FeasibleC) || isempty(InfeasibleU)
        return;
    end

    RefObj  = [FeasibleC.objs;InfeasibleU.objs];
    SectorF = AssociateSectors(FeasibleC.objs,W,RefObj);
    SectorU = AssociateSectors(InfeasibleU.objs,W,RefObj);
    ScalarF = ComputeSectorScalar(FeasibleC.objs,W,RefObj,SectorF);
    ScalarU = ComputeSectorScalar(InfeasibleU.objs,W,RefObj,SectorU);
    DeltaG  = ResolveBridgeActivationGap(RuntimeOptions);

    [BestFIdx,BestFSector,BestFValue] = SelectSectorBest(SectorF,ScalarF);
    [BestUIdx,BestUSector,BestUValue] = SelectSectorBest(SectorU,ScalarU);
    [SharedSector,FLoc,ULoc] = intersect(BestFSector,BestUSector,'stable');
    if isempty(SharedSector)
        return;
    end

    Active = BestUValue(ULoc) + DeltaG < BestFValue(FLoc);
    SharedSector = SharedSector(Active);
    FLoc = FLoc(Active);
    ULoc = ULoc(Active);
    Count = numel(SharedSector);
    if Count == 0
        return;
    end

    Pool.source    = ones(Count,1);
    Pool.sector    = SharedSector(:);
    Pool.anchorDec = FeasibleC(BestFIdx(FLoc)).decs;
    Pool.anchorObj = FeasibleC(BestFIdx(FLoc)).objs;
    Pool.helperDec = InfeasibleU(BestUIdx(ULoc)).decs;
    Pool.helperObj = InfeasibleU(BestUIdx(ULoc)).objs;
end

function Pool = InitBridgePool(Problem)
    Pool.source    = zeros(0,1);
    Pool.sector    = zeros(0,1);
    Pool.anchorDec = zeros(0,Problem.D);
    Pool.anchorObj = zeros(0,Problem.M);
    Pool.helperDec = zeros(0,Problem.D);
    Pool.helperObj = zeros(0,Problem.M);
end

function [BestIdx,BestSector,BestValue] = SelectSectorBest(Sector,Scalar)
    if isempty(Sector)
        BestIdx = zeros(0,1);
        BestSector = zeros(0,1);
        BestValue = zeros(0,1);
        return;
    end

    RankTable = [(1:numel(Sector))',Sector(:),Scalar(:)];
    RankTable = sortrows(RankTable,[2 3 1]);
    [~,First] = unique(RankTable(:,2),'stable');
    BestIdx    = RankTable(First,1);
    BestSector = RankTable(First,2);
    BestValue  = RankTable(First,3);
end

function DeltaG = ResolveBridgeActivationGap(RuntimeOptions)
    DeltaG = 0.01;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgeActivationGap') ...
            && ~isempty(RuntimeOptions.BridgeActivationGap)
        DeltaG = max(RuntimeOptions.BridgeActivationGap,0);
    end
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\KeepLatestDecisionRows.m
====================================================================================================

function Keep = KeepLatestDecisionRows(Dec)
% Return row indices that keep only the latest occurrence of each decision.

    if isempty(Dec)
        Keep = zeros(0,1);
        return;
    end

    [~,RevKeep] = unique(flipud(Dec),'rows','stable');
    Keep = sort(size(Dec,1)-RevKeep+1);
    Keep = Keep(:);
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\PRBCCMO.m
====================================================================================================

classdef PRBCCMO < ALGORITHM
% <2026> <multi> <real/integer/label/binary/permutation> <constrained>
% Pareto-relevant boundary CCMO
% bRho     --- 0.2  --- Boundary evaluation ratio relative to N
% trainRho --- 2    --- Training archive size ratio
% hidden   --- 20   --- Hidden units of the boundary MLP
% epoch    --- 25   --- Training epochs of the boundary MLP
% lr       --- 0.01 --- Learning rate of the boundary MLP
% mRho     --- 0.4  --- Seed-query ratio within each boundary budget
% ensK     --- 3    --- Committee size of shallow boundary MLPs
% calMode  --- 2    --- Calibration mode (1 raw, 2 auto temp/beta, 3 beta)
% dLambda  --- 1    --- Committee disagreement weight in boundary utility
% pairM    --- 0.05 --- Margin for tight bracket pair loss
% lPair    --- 1    --- Weight of bracket pair loss
% lMid     --- 1    --- Weight of midpoint-to-0.5 loss
% selMode  --- 1    --- Section B seed selection mode (1. trusted-query 2. uncertain-only 3. random-bridge)
% localMode--- 1    --- Section B local search mode (1. label-aware 2. isotropic)
% traceOn  --- 0    --- Enable Section B audit trace recording

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            Params = ResolvePRBCCMOParameters(Algorithm.parameter);
            bRho      = Params.bRho;
            trainRho  = Params.trainRho;
            hidden    = Params.hidden;
            epoch     = Params.epoch;
            lr        = Params.lr;
            mRho      = Params.mRho;
            ensK      = Params.ensK;
            calMode   = Params.calMode;
            dLambda   = Params.dLambda;
            pairM     = Params.pairM;
            lPair     = Params.lPair;
            lMid      = Params.lMid;
            selMode   = Params.selMode;
            localMode = Params.localMode;
            traceOn   = Params.traceOn;
            RuntimeOptions = BuildBoundaryRuntimeOptions(selMode,localMode,traceOn);

            BoundaryBudget = max(0,floor(bRho*Problem.N));
            TrainMax       = max(1,round(trainRho*Problem.N));
            CalibMax       = max(1,Problem.N);
            TestMax        = max(1,Problem.N);
            ProtectedOtherMax = max(20,ceil(0.25*TrainMax));
            BracketMax     = max(1,Problem.N);
            HardNegMax     = max(20,ceil(0.25*TrainMax));
            SeedRatio      = min(max(mRho,0),1);
            [W,~]          = UniformPoint(max(Problem.N,2),Problem.M);
            UpdateGap      = 5;
            RestartGap     = 25;
            WarmEpoch      = min(epoch,max(5,round(epoch/3)));
            TriggerCount   = max(1,ceil(0.1*TrainMax));
            TightGap       = 0.03;

            %% Generate random populations
            PopulationC = Problem.Initialization();
            PopulationU = Problem.Initialization();

            %% Initialize boundary memories
            BracketArchive = EmptyBracketArchive(Problem.D);
            HardNegativeArchive.Dec    = zeros(0,Problem.D);
            HardNegativeArchive.Radius = zeros(0,1);
            ProtectedOtherDec   = zeros(0,Problem.D);
            ProtectedOtherLabel = zeros(0,1);
            InitSolutions = [PopulationC,PopulationU];
            [InitTrain,~,InitCalib,InitCalibInfo,InitTest,InitTestInfo] = SplitHeldOutBatch( ...
                InitSolutions,NormalizeBoundaryInfo([],Problem.M),CalibMax,TestMax,Problem.M);
            InitHoldoutDec = [SolutionDecs(InitCalib,Problem.D);SolutionDecs(InitTest,Problem.D)];
            ProtectedDec   = zeros(0,Problem.D);
            ProtectedLabel = zeros(0,1);
            [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                [],[],ProtectedDec,ProtectedLabel,InitTrain,InitHoldoutDec,TrainMax);
            [CalibDec,CalibLabel,CalibNear] = UpdateCalibrationBuffer( ...
                [],[],[],InitCalib,InitCalibInfo,CalibMax);
            [TestDec,TestLabel,TestNear] = UpdateCalibrationBuffer( ...
                [],[],[],InitTest,InitTestInfo,TestMax);
            TrainOptions = BuildBoundaryTrainingOptions( ...
                BracketArchive,ensK,calMode,dLambda,pairM,lPair,lMid,Problem.D,TightGap);
            Model = TrainBoundaryMLP( ...
                TrainDec,TrainLabel,hidden,epoch,lr,[],CalibDec,CalibLabel,TrainOptions);
            Model = RefreshBoundaryTrust(Model,TestDec,TestLabel);
            LastCalMetric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel);
            PendingLabels = 0;
            Generation    = 0;
            [ExternalArchive,~] = UpdateExternalArchive([],FilterFeasiblePopulation(InitSolutions));
            Algorithm.metric.boundaryCalibration = AttachCalibrationContext( ...
                LastCalMetric,Generation,Problem.FE,size(TrainDec,1),size(CalibDec,1), ...
                sum(CalibNear),size(TestDec,1),sum(TestNear));
            Algorithm.metric.sectionB = InitSectionBMetric(Problem.D,RuntimeOptions,ExternalArchive);
            Algorithm.metric.sectionB = AppendSectionBCalibrationTrace( ...
                Algorithm.metric.sectionB,Algorithm.metric.boundaryCalibration);

            %% Optimization
            while Algorithm.NotTerminated(PopulationC)
                Generation = Generation + 1;

                [OffspringC,OffspringU] = GenerateRegularOffspring( ...
                    Problem,PopulationC,PopulationU);

                CandidatePool = GenerateBoundaryCandidates( ...
                    Problem,PopulationC,PopulationU,W,RuntimeOptions);

                BoundaryBudgetNow = min(BoundaryBudget,max(0,Problem.maxFE-Problem.FE));
                SeedBudget = min(BoundaryBudgetNow,max(0,round(SeedRatio*BoundaryBudgetNow)));
                if BoundaryBudgetNow > 0 && SeedBudget == 0
                    SeedBudget = 1;
                end
                [BoundarySeeds,SeedInfo] = SelectBoundaryCandidates( ...
                    Problem,CandidatePool,PopulationC,Model,W,HardNegativeArchive,SeedBudget,RuntimeOptions);

                WorkerBudget = max(0,BoundaryBudgetNow-numel(BoundarySeeds));
                [WorkerOffspring,WorkerInfo,MigrationPool,BracketBatch,HardNegBatch,WorkerAudit] = ...
                    RefineBoundaryWorkers( ...
                        Problem,BoundarySeeds,SeedInfo,PopulationC,Model,W, ...
                        HardNegativeArchive,WorkerBudget,RuntimeOptions);
                [BoundaryOffspring,BoundaryInfo] = MergeBoundaryResults( ...
                    BoundarySeeds,SeedInfo,WorkerOffspring,WorkerInfo,Problem.M);

                HardNegativeArchive = UpdateHardNegativeArchive(HardNegativeArchive,HardNegBatch,HardNegMax);
                BracketArchive = UpdateBracketArchive(BracketArchive,BracketBatch,BracketMax,Problem.D,TightGap);

                ConstrainedBase = KeepUniquePopulation([PopulationC,OffspringC]);
                PopulationC = EnvironmentalSelectionC(ConstrainedBase,Problem.N,MigrationPool,W);
                PopulationU = EnvironmentalSelectionU(KeepUniquePopulation([PopulationU,OffspringU]),Problem.N);

                [ExternalArchive,BoundaryGain,BoundaryAdded] = UpdateSectionBExternalArchive( ...
                    ExternalArchive,OffspringC,OffspringU,BoundaryOffspring);
                Algorithm.metric.sectionB.seedAudit = AppendBoundarySeedAuditRows( ...
                    Algorithm.metric.sectionB.seedAudit,BoundarySeeds,SeedInfo,WorkerAudit, ...
                    BoundaryAdded,Generation,Problem.FE,Problem.D);
                Algorithm.metric.sectionB.boundaryGainTrace = AppendBoundaryGainTrace( ...
                    Algorithm.metric.sectionB.boundaryGainTrace,Generation,Problem.FE, ...
                    BoundaryGain,numel(BoundaryAdded),numel(ExternalArchive));
                Algorithm.metric.sectionB.externalArchiveCount = numel(ExternalArchive);
                Algorithm.metric.sectionB.totalBoundaryGain = Algorithm.metric.sectionB.totalBoundaryGain + BoundaryGain;

                [TrainBatch,TrainInfo,CalibBatch,CalibInfo,TestBatch,TestInfo] = SplitHeldOutBatch( ...
                    BoundaryOffspring,BoundaryInfo,CalibMax,TestMax,Problem.M);
                HoldoutDec = [CalibDec;TestDec; ...
                    SolutionDecs(CalibBatch,Problem.D);SolutionDecs(TestBatch,Problem.D)];
                [ProtectedBracketDec,ProtectedBracketLabel] = BuildBracketProtectedBuffer(BracketArchive,Problem.D);
                [ProtectedBracketDec,ProtectedBracketLabel] = ExcludeLabeledRows( ...
                    ProtectedBracketDec,ProtectedBracketLabel,HoldoutDec);
                [ProtectedOtherAddDec,ProtectedOtherAddLabel] = CollectOtherProtectedCases( ...
                    TrainBatch,TrainInfo,HardNegBatch,Problem.D);
                [ProtectedOtherAddDec,ProtectedOtherAddLabel] = ExcludeLabeledRows( ...
                    ProtectedOtherAddDec,ProtectedOtherAddLabel,HoldoutDec);
                [ProtectedOtherDec,ProtectedOtherLabel] = UpdateProtectedBuffer( ...
                    ProtectedOtherDec,ProtectedOtherLabel,ProtectedOtherAddDec,ProtectedOtherAddLabel,ProtectedOtherMax);
                [ProtectedOtherDec,ProtectedOtherLabel] = ExcludeLabeledRows( ...
                    ProtectedOtherDec,ProtectedOtherLabel,HoldoutDec);
                ProtectedDec   = [ProtectedBracketDec;ProtectedOtherDec];
                ProtectedLabel = [ProtectedBracketLabel;ProtectedOtherLabel];
                [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                    TrainDec,TrainLabel,ProtectedDec,ProtectedLabel,TrainBatch,HoldoutDec,TrainMax);
                [CalibDec,CalibLabel,CalibNear] = UpdateCalibrationBuffer( ...
                    CalibDec,CalibLabel,CalibNear,CalibBatch,CalibInfo,CalibMax);
                [TestDec,TestLabel,TestNear] = UpdateCalibrationBuffer( ...
                    TestDec,TestLabel,TestNear,TestBatch,TestInfo,TestMax);

                PendingLabels = PendingLabels + numel(BoundaryOffspring);
                TrainOptions = BuildBoundaryTrainingOptions( ...
                    BracketArchive,ensK,calMode,dLambda,pairM,lPair,lMid,Problem.D,TightGap);
                [Model,PendingLabels,LastCalMetric] = UpdateBoundaryModel( ...
                    Model,TrainDec,TrainLabel,CalibDec,CalibLabel,TestDec,TestLabel, ...
                    hidden,epoch,WarmEpoch,lr,Generation,PendingLabels,TriggerCount, ...
                    UpdateGap,RestartGap,LastCalMetric,TrainOptions);
                Algorithm.metric.boundaryCalibration = AttachCalibrationContext( ...
                    EvaluateBoundaryCalibration(Model,TestDec,TestLabel), ...
                    Generation,Problem.FE,size(TrainDec,1),size(CalibDec,1), ...
                    sum(CalibNear),size(TestDec,1),sum(TestNear));
                Algorithm.metric.sectionB = AppendSectionBCalibrationTrace( ...
                    Algorithm.metric.sectionB,Algorithm.metric.boundaryCalibration);
            end
        end
    end
end

function [OffspringC,OffspringU] = GenerateRegularOffspring(Problem,PopulationC,PopulationU)
    OffspringC = Problem.Evaluation(OperatorDE_current_rand_1(Problem,PopulationC.decs));
    OffspringU = Problem.Evaluation(OperatorDE_current_rand_1(Problem,PopulationU.decs));
end

function Population = KeepUniquePopulation(Population)
    if isempty(Population)
        return;
    end
    Keep = KeepLatestDecisionRows(Population.decs);
    Population = Population(Keep);
end

function Population = FilterFeasiblePopulation(Population)
    if isempty(Population)
        return;
    end
    Population = Population(all(Population.cons<=0,2));
end

function [TrainSolutions,TrainInfo,CalibSolutions,CalibInfo,TestSolutions,TestInfo] = SplitHeldOutBatch(Solutions,Info,CalibMax,TestMax,M)
    TrainSolutions = Solutions;
    TrainInfo = NormalizeBoundaryInfo(Info,M);
    CalibSolutions = [];
    CalibInfo = NormalizeBoundaryInfo([],M);
    TestSolutions = [];
    TestInfo = NormalizeBoundaryInfo([],M);
    Count = numel(Solutions);
    if Count <= 1 || (CalibMax <= 0 && TestMax <= 0)
        return;
    end

    Info = NormalizeBoundaryInfo(Info,M);
    if numel(Info.source) ~= Count
        Info = DefaultBoundaryInfo(Solutions,M);
    end
    Label = double(all(Solutions.cons<=0,2));
    NearMask = true(Count,1);
    if isfield(Info,'prob') && numel(Info.prob) == Count
        NearMask = abs(Info.prob(:)-0.5) <= 0.1;
    end

    CalibQuota = min(max(1,round(0.2*Count)),min(CalibMax,Count-1));
    CalibIdx = SelectCalibrationHoldout(Label,NearMask,CalibQuota);
    RemainingMask = true(Count,1);
    RemainingMask(CalibIdx) = false;

    RemainingIdx = find(RemainingMask);
    TestQuota = min(max(1,round(0.2*Count)),TestMax);
    TestQuota = min(TestQuota,max(0,numel(RemainingIdx)-1));
    if TestQuota > 0
        TestLocalIdx = SelectCalibrationHoldout(Label(RemainingIdx),NearMask(RemainingIdx),TestQuota);
        TestIdx = RemainingIdx(TestLocalIdx);
    else
        TestIdx = zeros(0,1);
    end

    TrainMask = true(Count,1);
    TrainMask(CalibIdx) = false;
    TrainMask(TestIdx) = false;
    TrainIdx = find(TrainMask);
    TrainSolutions = Solutions(TrainIdx);
    TrainInfo = SliceBoundaryInfo(Info,TrainIdx,M);
    CalibSolutions = Solutions(CalibIdx);
    CalibInfo = SliceBoundaryInfo(Info,CalibIdx,M);
    TestSolutions = Solutions(TestIdx);
    TestInfo = SliceBoundaryInfo(Info,TestIdx,M);
end

function HoldoutIdx = SelectCalibrationHoldout(Label,NearMask,Quota)
    HoldoutIdx = zeros(0,1);
    Total = numel(Label);
    if Total <= 1 || Quota <= 0
        return;
    end

    Quota = min(Quota,Total-1);
    ClassOrder = [1,0];
    BaseQuota = floor(Quota/2);
    HoldoutCell = cell(1,numel(ClassOrder)+1);
    HoldCount = 0;
    for i = 1 : numel(ClassOrder)
        ClassIdx = FindCalibrationCandidates(Label,NearMask,ClassOrder(i));
        Take = min(numel(ClassIdx),BaseQuota);
        if Take > 0
            HoldCount = HoldCount + 1;
            HoldoutCell{HoldCount} = ClassIdx(1:Take);
        end
    end
    HoldoutIdx = vertcat(HoldoutCell{1:HoldCount});

    if numel(HoldoutIdx) < Quota
        Remaining = setdiff((1:Total)',HoldoutIdx,'stable');
        NearFirst = [Remaining(NearMask(Remaining));Remaining(~NearMask(Remaining))];
        Extra = NearFirst(1:min(Quota-numel(HoldoutIdx),numel(NearFirst)));
        HoldoutIdx = [HoldoutIdx;Extra(:)];
    end

    HoldoutIdx = unique(HoldoutIdx,'stable');
end

function Idx = FindCalibrationCandidates(Label,NearMask,ClassValue)
    NearIdx = find(NearMask & Label==ClassValue);
    FarIdx  = find(~NearMask & Label==ClassValue);
    Idx = [NearIdx(:);FarIdx(:)];
end

function Info = DefaultBoundaryInfo(Solutions,M)
    Count = numel(Solutions);
    Info = NormalizeBoundaryInfo([],M);
    Info.source    = zeros(Count,1);
    Info.score     = zeros(Count,1);
    Info.prob      = 0.5*ones(Count,1);
    Info.entropy   = zeros(Count,1);
    Info.hvGain    = zeros(Count,1);
    Info.novelty   = zeros(Count,1);
    Info.penalty   = ones(Count,1);
    Info.utility   = zeros(Count,1);
    Info.sector    = zeros(Count,1);
    Info.proxyObjs = Solutions.objs;
end

function [Dec,Label] = ExcludeLabeledRows(Dec,Label,ExcludeDec)
    if isempty(Dec) || isempty(ExcludeDec)
        return;
    end
    Keep = ~ismember(Dec,ExcludeDec,'rows');
    Dec = Dec(Keep,:);
    Label = Label(Keep);
end

function Dec = SolutionDecs(Solutions,D)
    if isempty(Solutions)
        Dec = zeros(0,D);
        return;
    end
    Dec = Solutions.decs;
end

function [ProtectedDec,ProtectedLabel] = CollectOtherProtectedCases(BoundaryOffspring,BoundaryInfo,HardNegBatch,D)
    ProtectedDec = zeros(0,D);
    ProtectedLabel = zeros(0,1);
    if ~isempty(BoundaryOffspring)
        Label = double(all(BoundaryOffspring.cons<=0,2));
        NearMask = abs(BoundaryInfo.prob(:)-0.5) <= 0.1;
        MisMask  = (BoundaryInfo.prob(:)>=0.5) ~= logical(Label);
        KeepMask = NearMask & MisMask;
        ProtectedDec = [ProtectedDec;BoundaryOffspring(KeepMask).decs];
        ProtectedLabel = [ProtectedLabel;Label(KeepMask)];
    end
    if ~isempty(HardNegBatch.Dec)
        ProtectedDec = [ProtectedDec;HardNegBatch.Dec];
        ProtectedLabel = [ProtectedLabel;zeros(size(HardNegBatch.Dec,1),1)];
    end
end

function [ProtectedDec,ProtectedLabel] = BuildBracketProtectedBuffer(BracketArchive,D)
    ProtectedDec = zeros(0,D);
    ProtectedLabel = zeros(0,1);
    if isempty(BracketArchive) || isempty(BracketArchive.FeasibleDec)
        return;
    end
    ProtectedDec = [BracketArchive.FeasibleDec;BracketArchive.InfeasibleDec];
    ProtectedLabel = [ones(size(BracketArchive.FeasibleDec,1),1);zeros(size(BracketArchive.InfeasibleDec,1),1)];
end

function Archive = EmptyBracketArchive(D)
    Archive.FeasibleDec   = zeros(0,D);
    Archive.InfeasibleDec = zeros(0,D);
    Archive.Gap           = zeros(0,1);
end

function Archive = UpdateBracketArchive(Archive,NewPairs,MaxPairs,D,TightGap)
    if nargin < 1 || isempty(Archive)
        Archive = EmptyBracketArchive(D);
    end
    if nargin < 3 ||
The file is too long and its contents have been truncated.


#### Plugin (file_search):
Make sure to include fileciteturn5file1 in your response to cite this file, or to surface it as a link.
# 分支 · 二版本优化+实验设计

#### You:
此论文是研究进化优化领域中的未知（约束）约束问题；该问题的特点是约束违反度只有二值（只有0和1，所有约束都满足返回0，但凡违反任意个约束直接返回1），具体的定义和当前算法的难点请你阅读上述论文；此论文就是直接将MLP用于二元约束问题。我下面也想写一篇论文，该论文的创新点是根据MLP的置信度（在概率0.5附近主动采样）去搜索可行域的边界（同时也是约束的边界）；主要思路如下；1：双种群（无约束种群+约束种群）；2：采用此论文的MLP模型；3：无约束种群和约束种群交叉变异来产生下一代解，采用目标值的支配关系来判断子代解是不是处于无约束种群和约束种群之间（以此来寻找二者之间的可行域，跨越不可行域），主动边界采样是通过找到的无约束种群和约束种群之间的不可行解和可行解来更新MLP，同时概率在0.5附近的解也需要参与子代生成（比如和约束种群交叉变异），同时对于MLP输出的概率在0.5附近的解的周围进行主动采样来判断此处是不是更好的可行域（如果是，约束种群所在的可行域就会被此可行域所支配，约束种舍弃旧可行域且迁移到新的可行域来进行下一轮的优化）；
主要思想如上，你来审查分析此思路是否可行（学术创新性（有没有人做过了）、是否合乎逻辑），着重分析如何在0.5附近主动采样，如何使用这些0.5附近的解重训MLP，如何找到无约束种群和约束种群之间的新可行域；
核心创新重点如下：可行域边界搜索，怎么找到有价值的概率在0.5附近的解




#### ChatGPT:
我先给判断：这条思路**能做，而且方向是对的**，但要把论文的核心从“MLP + 双种群 + 0.5附近采样”收紧成“**校准置信度驱动的主动边界搜索**”。否则它很容易被审稿人看成是在 NA-EMT 上加了一层常规 uncertainty sampling 的增量。你给的 NA-EMT 论文把问题定义成只返回二值可行性的 CMOP-UC，核心难点就是没有 constraint violation degree，很多传统 CMOEA 的约束处理机制失去抓手；它的解法是用 MLP 给 infeasible 解打一个“可行概率/价值”的软分数，再用主任务/辅助任务双种群协同搜索，并在模型准确率下降时重训 MLP。fileciteturn0file0

就我这轮检索，**直接针对 unknown/binary constraint 多目标进化**的公开工作确实不多，至少能确认有 EADMM（2024）、DRMCMO（2024 arXiv）和 NA-EMT（2025）。EADMM 用两个互补方向的种群交替演化；DRMCMO 强调 detection region；NA-EMT 才开始把 MLP 用成 infeasible solution value 的代理。citeturn768692view0turn605186view6turn768692view1

但你现在方案里的几个“部件”本身都**不是首创**。双种群不是新的：已有 dual-population CMOEA 让主种群处理原 CMOP、辅种群只优化目标；也有 dp-ACS 这类“考虑约束/忽略约束”双种群协同；DBC-CMOEA 还明确从 feasible / infeasible 两侧双向共进化。citeturn663956view1turn663956view2turn469110search0

“搜索 feasible–infeasible boundary”本身也不是新的。在已知/可量化约束的 CMOEA/CO 里，DC-SHADE-IF 第一阶段就直接搜索 infeasible–feasible boundary；CL-TDEA 把 constraint-boundary learning 做成双种群两阶段框架；TPDCB、CT-TSEA、BPRRA 也都把 dynamic/promising boundary 当作核心机制。citeturn828899search1turn348143view3turn605186view1turn828899search17turn615310search0

更关键的是，**“在 0.5 附近采样”也不能直接宣称首创**。在主动学习里，二分类 posterior 接近 0.5 本来就是 uncertainty sampling 的基本规则；同时文献也早就提醒，这种策略容易采到离群点，所以常要加 density / diversity 约束。更近一点的 unknown-constraint 优化里，BE-CBO 已经显式把 feasibility probability 推向 0.5 来逼近 boundary；AL-PNN 也用 active learning 追踪可能包含 constraint boundary 的区域。citeturn149720view1turn149720view2turn989866view0turn308468view1turn308468view2turn768692view2

所以，**你的题目不是没创新，而是要把创新点说准**。我这轮检索**没有看到**有人把“**校准后的 MLP 不确定度 + 双种群桥接 + 新可行连通分量发现/迁移**”这一整套明确用于 CMOP-UC/CMOP-BC；但如果只写成“用 MLP 置信度在 0.5 附近主动采样”，我认为创新性偏弱。更好的论文定位应是：把 unknown-constraint BO 里“boundary matters”的思想移植到 evolutionary multitasking，并设计出**多目标场景特有**的 bridge operator、region discovery 和 resource reallocation。unknown-constraint BO 文献也明确把 classifier-based feasibility modeling 和对 boundary 信息的主动获取看成重要方向。citeturn141317view0turn308468view1

## 一、你的思路在逻辑上哪里成立，哪里还差一口气

成立的地方在于：对二值约束问题，**唯一可学的几何信息基本就是 aggregate feasibility boundary**。NA-EMT 现在只是把 MLP 当“软可行性排序器”；你如果进一步把它变成“边界查询器”，这个升级是顺逻辑的。

差的一口气有三处。

第一，**0.5 不等于真实边界**。如果你直接拿单个 MLP 的 sigmoid 输出当“置信度”，逻辑不够稳，因为现代神经网络常常是 miscalibrated 的，而你的整个机制恰好依赖“0.5 是否真的意味着靠近边界”。温度缩放往往是很有效的后校准办法；更进一步，可以把原论文那个单个 MLP 扩成一个同构 MLP ensemble，用均值做 $\bar p(x)$，方差做 epistemic uncertainty。citeturn443767view0turn238593view0turn614362view1turn614362view0

第二，**单纯采 p≈0.5 的点会捡到离群点**。主动学习文献的经典结论就是：最不确定的点往往在决策边界附近，但不一定“有代表性”，可能只是孤立点；所以 uncertainty 要和 density / representativeness / diversity 联合。citeturn149720view0turn989866view0

第三，**“用目标支配关系判断一个子代是否在无约束种群和约束种群之间”这个判据不够强**。支配关系是目标空间的偏序，“在两者之间”其实是决策空间/边界几何的问题。一个点在目标空间上看起来“夹在中间”，不代表它就在两类样本的几何边界附近。这里你必须补一个**显式的桥接机制**。

## 二、我建议你把“有价值的 0.5 点”定义成什么

不要把“valuable point”定义成 $|p-0.5|$ 最小。\
我建议你把它定义成下面这个 acquisition score 的 top-k：

$
S(x)=H(\bar p(x))\cdot (1+\lambda_u \sigma_p(x))\cdot (\epsilon+\Delta HV^+(x))^\alpha \cdot \rho(x)^\beta \cdot \nu(x)^\gamma \cdot I_{\text{bridge}}(x)
$

这里：

- $H(\bar p)$ 是二元熵，天然在 0.5 最大，表示“边界不确定性”；
- $\sigma_p(x)$ 是 ensemble 方差，区分“真的边界”与“模型纯粹没见过”；
- $\Delta HV^+(x)$ 是“**如果 x 最终可行**，它对当前 feasible archive 的假想超体积增益”，这一步把“0.5 附近”变成“**对 Pareto 前沿有价值的 0.5 附近**”；
- $\rho(x)$ 是 KNN density，防离群点；
- $\nu(x)$ 是 novelty / batch diversity，防重复采样；
- $I_{\text{bridge}}(x)$ 表示它是不是来自“约束种群–无约束种群”的桥接对，或者来自已知 feasible–infeasible 对的中间点。

这一步其实就是你论文最该强调的地方：  
**不是找“最不确定”的点，而是找“最可能位于 Pareto-relevant boundary 的点”。**

## 三、0.5 附近到底怎么主动采样

我建议你不要只做“在已有候选里筛 p≈0.5”，而要做一个**三源候选池 + 一种主算子**。

### 1）候选池三种来源

第一类是普通进化候选：  
来自 $P_c$（约束/可行种群）和 $P_u$（无约束种群）的交叉变异后代，尤其保留跨种群交叉产生的 offspring。

第二类是桥接候选：  
对每个 reference vector / subproblem，分别在 $P_c$ 和 $P_u$ 里找对应方向上最好的两个代表 $x_c, x_u$。若 $x_u$ 的目标标量化值明显优于 $x_c$，但其 label 为 infeasible 或 highly uncertain，就把这对样本当成一条“可能通往更优 feasible region 的桥”。

第三类是边界邻域候选：  
从当前 boundary archive 里取 $x_b$，在它附近做局部扰动，优先沿着 $x_u-x_c$ 的方向、以及 boundary 切向方向探索。

### 2）主算子要用“二分括逼”，不要只用随机邻域采样

对选中的桥接对 $(x_f,x_i)$（一个 feasible，一个 infeasible），直接做 segment bracketing：

$
x(\lambda)=x_f+\lambda(x_i-x_f),\quad \lambda\in(0,1)
$

先测中点，再按 label 做二分更新。  
因为你只有 0/1 label，没有 violation degree，所以**二分括逼正好是最自然、最省预算的边界定位算子**。每一次 query 都在压缩 feasible–infeasible bracket，这比“在 0.5 周围撒点”要更强、更像论文贡献。

你甚至可以把“boundary sample”严格定义成：  
“经过 r 次二分后，得到一对距离小于 $\varepsilon_d$ 的 $(x_f,x_i)$，其中心点 $x_b=(x_f+x_i)/2$ 记为近似 boundary point。”

这会让你的方法一下子从“经验 heuristic”变成“有明确边界语义的 operator”。

### 3）高级版本：边界切向搜索

如果你愿意把方法再做强一点，可以在得到 $x_b$ 后，利用 MLP 的可微性估一个局部法向：

$
n=\frac{\nabla_x \bar p(x_b)}{\|\nabla_x \bar p(x_b)\|}
$

然后把“从 $x_c$ 指向 $x_u$ 的目标改进方向”投影到边界切空间：

$
d_{\text{tan}}=(I-nn^\top)(x_u-x_c)
$

先沿 $d_{\text{tan}}$ 走一小步，再把点投回 $p\approx 0.5$ 带上，最后在法向两侧各试一个点。  
这相当于“沿边界爬行去找更优的可行入口”。如果你把这个算子做好，它会比“简单 0.5 采样”新很多。

如果你不想上梯度，也可以先只做 segment+bisection。这个版本已经足够合理。

## 四、这些 0.5 附近的点怎么重训 MLP

这里你千万不要沿用 NA-EMT 的“准确率不够就重训”那套原样搬过去。  
原因很简单：**原论文用 MLP 只是做 soft ranking；你现在是把 MLP 概率当采样控制信号。**  
这时候，accuracy 不是关键，**calibration 才是关键**。fileciteturn0file0 citeturn443767view0turn238593view0

我建议用三个 buffer：

- 全局 reservoir $D_g$：防遗忘；
- 当前种群 buffer $D_c$：跟踪分布漂移；
- 边界 buffer $D_b$：专门存二分括逼得到的 $(x_f,x_i)$ 对和 tight midpoint。

训练时不要均匀抽样，建议 mini-batch 里**过采样 boundary buffer**。  
因为你的论文核心不是“全局分类精度”，而是“boundary 附近概率是否可信”。

损失函数可以写成：

$
L = L_{\text{WBCE}} + \lambda_1 L_{\text{Brier}} + \lambda_2 L_{\text{pair}} + \lambda_3 L_{\text{mid}}
$

其中：

- $L_{\text{WBCE}}$：类平衡 BCE；
- $L_{\text{Brier}}$：直接优化概率质量；
- $L_{\text{pair}}$：对每个括逼对约束 $p(x_f) > p(x_i)$；
- $L_{\text{mid}}$：只对足够 tight 的 midpoint 加 $(p(x_b)-0.5)^2$ 正则。

这里的 $L_{\text{mid}}$ 很关键：  
它不是伪造标签，而是利用“tight bracket implies nearby boundary”这个几何事实，给模型一个**边界校准信号**。这很像你论文里最像样的 methodological novelty。

重训触发条件也别再只看 accuracy。更合理的是看三件事：

- ECE / Brier 变差了没有；
- boundary buffer 上的 pair ordering error 是否上升；
- 当前种群分布与训练集分布是否漂移太大。

## 五、怎么找到无约束种群和约束种群之间的“新可行域”

这部分我觉得你现在的原始表述还不够严谨。  
我建议把“找新可行域”改写成“**找新的 feasible component**”。

因为在 unknown binary constraints 里，你看到的并不是一个单一光滑边界，而是**多个隐藏约束边界的并集**；因此可行域往往是多个 disconnected component。你的算法应该找的是“新的可行连通分量”，不是模糊的“更好的可行域”。

更稳的做法是：

1. 用 reference vector 把 $P_c$ 和 $P_u$ 配对，而不是直接用全局 dominance。  
这样每条桥都有明确的 Pareto 方向。

2. 对每条高分桥做二分括逼，得到 boundary point 和一组 boundary pair。

3. 从这些 boundary point 向目标更优方向做局部搜索，得到新的 feasible 点集。

4. 对所有 feasible 点建一个 kNN 图或做 DBSCAN / HDBSCAN 聚类，得到 feasible components。

5. 若新点落入一个以前没有的 component，且该 component 对 external archive 有真实贡献（比如出现新的 nondominated points，或带来显著 $\Delta HV$），就给它分配独立搜索资源。

这里我特别不建议你写成“新可行域支配旧可行域，于是约束种群舍弃旧域、整体迁移”。  
对多目标问题，这个逻辑太激进了。一个旧 feasible component 可能在另一段 PF 上仍然有不可替代的贡献。**硬迁移很容易丢掉 CPF 的另一部分。**

更好的写法是“**component-level resource reallocation**”：

$
R(r)=HV(A_r)+\beta \cdot \text{Potential}(r)+\gamma \cdot \text{BoundaryUncertainty}(r)
$

按 component score 给预算，不是一下子把旧区域扔掉。\
只有当某个 component 连续很多代对 external archive 没贡献，才逐步回收资源。

## 六、如果你现在就要写论文，我建议你把贡献点改成这三条

第一条，不要再说“双种群 + MLP”。\
这不够新。

第二条，把主贡献写成：\
**a confidence-calibrated active boundary search mechanism for CMOP-UC**\
关键词是 calibrated、active、boundary、Pareto-relevant。

第三条，把“0.5附近的点”明确定义成：\
**兼具边界不确定性、代表性、目标价值和桥接语义的样本**，而不是简单的低置信度样本。

如果你能把这三个点落成算法，创新性我会给“**中等偏上，可投稿**”；\
如果只是把 NA-EMT 的单个 MLP 拿来，看到 $p\approx 0.5$ 就多采几点，我会判断为“**逻辑可行，但论文层面偏弱，像工程增量**”。

## 七、我的最终结论

这条思路**是可行的**，而且很适合 binary unknown constraints，因为这类问题最缺的正是“边界信息”。\
但你真正能打动审稿人的，不是“0.5 附近采样”这句话本身，而是下面这句：

**我们不是在找最不确定的点，而是在找最可能位于 Pareto-relevant feasible–infeasible boundary 上的点，并利用这些点去发现新的 feasible components。**

这句话一旦成立，你的工作就从“给 NA-EMT 加 uncertainty sampling”变成了“为 CMOP-UC 引入主动边界学习与区域发现机制”。

下一步最值得做的是先把这三个模块写成伪代码：`bridge-pair selection`、`boundary bracketing/tangential search`、`boundary-aware retraining`。


#### You:
该txt文件的内容是我在platemo平台实现的该算法(用到的相关算子可参考https://github.com/BIMK/PlatEMO)；你首先来正确理解代码，然后站在审稿人的角度上给出结论：本算法有几个严格意义上的创新点？最大的创新点是什么？如何根据现有的算法设计实验来请有力的证明创新点成立（仅针对最大的创新点来设计实验，要求给出清晰的思路和步骤；实验必须要有说服力！）；




#### ChatGPT:
我按**代码本身**而不是按你前面口述的想法来判断。先给结论：

**最严格的审稿口径下，这份实现只有 1 个核心创新点；宽松一点可以拆成 2 个技术点，但第 2 个更像第 1 个的配套机制，不足以单独算一个“大创新”。** fileciteturn0file1

### 1. 这份代码实际实现了什么

你的 PRBCCMO 不是简单复现 NA-EMT。原论文里的主线是“MLP 预测 infeasible solution value + CDPPV + 精度下降时再重训”；而你这份代码已经改成了“**每代训练一个轻量 MLP，把 `p≈0.5` 当作近边界信号，再单独拿出一笔 boundary budget 去评估最值得查的边界候选**”。候选池来自 4 个来源：`P_C×P_U` 交叉、`A_F` 局部扰动、`A_I` 局部扰动、`A_I×P_C` 交配；随后再用 proxy objectives、reference sector、source load 做二次重排。训练集也不是按 NA-EMT 的 accuracy threshold 触发更新，而是用 boundary offspring 和 `A_F/A_I` 做平衡 FIFO，并且**每代重训**。另外，代码里**没有**实现你前面口述的“发现新可行域后整体迁移”或显式的 boundary bisection / region migration。fileciteturn0file1fileciteturn0file0

还有两个细节，审稿人会注意到：第一，代码里的 MLP 是一层隐藏层 `tanh + sigmoid`、从头训练的轻量模型，不是原论文描述的 ReLU + SCG；第二，主种群更新也不是 CDPPV，而是“**可行解先按目标选，再用近边界 score + sector/sparsity/crowding/front** 去补一部分 infeasible 解”。这说明你的工作已经从“预测可行性价值”转向“**边界预算分配**”。fileciteturn0file1fileciteturn0file0

### 2. 审稿人视角：严格意义上有几个创新点

我的判断是：

**只认 1 个核心创新点。**

因为这些都**不算**严格新意：

- 双种群框架不新。已有 unknown-constraint 的 EADMM 用两个子问题/两个群体来解耦目标与约束；一般 CMOP 里也早就有“主群体考虑约束、辅群体只看目标”的 dual-population 框架。citeturn494027view0turn494027view3
- “边界学习/边界跨越”这个方向也不新。一般 CMOP 里已有明确的 constraint-boundary learning 双种群方法；binary constraints 里也已有 detection region 机制。citeturn494027view2turn494027view1
- 单独说“在 0.5 附近主动采样”也不新，因为在主动学习里，二分类 posterior 最接近 0.5 本来就是经典 uncertainty sampling。citeturn117743view0

所以，**双种群**、**MLP**、**boundary**、**0.5** 这些词单独拿出来，都撑不起严格创新。

### 3. 最大的创新点到底是什么

你现在真正站得住的点，不是“0.5 附近采样”本身，而是：

**把 MLP 从“可行性预测器”改造成“Pareto-relevant boundary query scheduler”。**

更准确地说，是这件事：

> **在 binary unknown constraints 下，用 `p≈0.5` 做边界不确定性预筛，再用目标空间的 Pareto/sector 信息把“普通边界点”过滤成“对 CPF 真有价值的边界点”，把有限 FE 优先投到这些点上。**

这才是核心创新。因为经典 uncertainty sampling 虽然会找边界附近样本，但也容易挑到不具代表性的离群点；你代码里的 same-sector dominated filter、source/sector balance、proxy objective front/crowding，本质上是在把“generic uncertainty”改造成“**Pareto-relevant uncertainty**”。这一步比单纯的 0.5 采样强得多，也更像论文级贡献。fileciteturn0file1citeturn117743view0turn117743view1

如果你一定要拆第二个点，我勉强会给：

**次级技术点：`A_F/A_I` 双边界档案 + boundary-focused balanced FIFO 训练集。**

这个设计让 MLP 学到的不是泛化 feasibility classifier，而是围绕边界的在线数据分布。但从审稿角度，它更像是“主创新的配套闭环”，不是独立创新。fileciteturn0file1

### 4. 论文里不要怎么写

不要把最大创新写成“MLP 在 0.5 附近主动采样”。这句话太弱，审稿人一眼就会想到经典 uncertainty sampling。更麻烦的是，神经网络输出概率本身可能失准，`0.5` 未必就真是边界，所以如果没有额外验证，这个说法会被质疑。更安全的表述应是：

**Pareto-relevant active boundary search under binary unknown constraints**。citeturn117743view0turn494027view5

### 5. 只针对“最大创新点”，怎么设计一个真正有说服力的实验

要证明这个最大创新，你必须同时证明 3 件事：

1. **你选到的点真的更靠近真实边界；**
2. **这些点不是普通边界点，而是更有 Pareto 价值的边界点；**
3. **因此在相同 FE 下，最终前沿更好。**

只看最终 HV/IGD 不够，因为那无法证明“改进来自边界搜索”。

#### 第一步：做 4 个严格对照版本

在**完全相同总 FE、相同双种群、相同 MLP、相同 boundary budget** 下，只改边界候选选择策略：

1. **Full-PRBCCMO**：你的完整版本。
2. **Rand-Boundary**：候选池不变，但从池中随机选 `Budget` 个去真实评估。
3. **HighProb-Boundary**：不用 `|p-0.5|`，改成选 `p` 最大的点，测试“找边界”是否优于“找更可能可行”。
4. **Uncertain-Only**：保留 `p≈0.5` 预筛，但去掉 Pareto/sector rerank，直接按 score 最高选，测试“Pareto-relevant”这层是否真的有用。

不要拿 NA-EMT 当唯一因果对照，因为它同时改了太多模块；NA-EMT、EADMM、DRMCMO 只适合放在最终性能表里做外部参照，不适合证明你的核心机制。citeturn186084search2turn494027view0turn494027view1

#### 第二步：做一个“oracle 边界审计”，这是最关键的

你这里最强的证据，不是 HV，而是**边界距离证据**。

因为 DASCMOP-UC 本质上是从 DASCMOP 二值化来的，所以在 benchmark 代码里，最好额外保留原始连续约束值 `g_raw(x)`，**算法运行时绝对不能使用它，只在实验分析阶段记录**。然后定义一个仅用于审计的真实边界距离：

$
d_B(x)=\min_j \frac{|g^{raw}_j(x)|}{s_j+\epsilon}
$

其中 $s_j$ 是第 $j$ 个约束的尺度归一化项（例如大样本随机采样得到的 MAD 或标准差）。\
这个量越小，说明点越接近
The file is too long and its contents have been truncated.


#### Plugin (file_search):
Make sure to include fileciteturn5file2 in your response to cite this file, or to surface it as a link.
<PARSED TEXT FOR PAGE: 1 / 12>
A Network-Assisted Evolutionary Multitask 
Framework for Multi-objective Optimization 
Problems with Unknown Constraints 
Jun Ma1, Yong Zhang1(B)
, Ruizhao Zheng2, Chunlin He1, Ali Wagdy Mohamed3,4, 
Mingcheng Zuo1, Huijun Li1, and Xiangjuan Yao1 
1 School of Information and Control Engineering, China University of Mining and Technology, 
Xuzhou 221008, China 
yongzh401@126.com 
2 School of Medical Information and Engineering, Xuzhou Medical University, 
Xuzhou 221000, China 
3 Operations Research Department, Faculty of Graduate Studies for Statistical Research, Cairo 
University, Giza 12613, Egypt 
4 Applied Science Research Center, Applied Science Private University, Amman 11937, Jordan 
Abstract. In practical applications, there are some constrained multi-objective 
optimization problems where the constraints are either unknown or non￾quantifiable, called CMOP-UC. The lack of constraint information presents sig￾nificant challenges to existing evolutionary algorithms, leading to poor diversity 
and a high risk of premature convergence. This issue primarily arises because 
most of existing algorithms depend heavily on explicit constraint information to 
guide their search. To solve this issue, we develop a network-assisted evolution￾ary multitasking framework (NA-EMT). In the NA-EMT, the main task solves the 
original CMOP-UC by two new strategies, i.e., the perception strategy of infeasible 
solution value and the constraint dominance principle with predicted value. This 
enables the algorithm to identify valuable infeasible solutions, thereby enhancing 
its overall performance. The auxiliary task solves the corresponding unconstrained 
multi-objective optimization problem, aiming to promote convergence while pre￾serving the diversity of solutions. Experimental results demonstrate that NA-EMT 
significantly outperforms 6 comparison algorithms, proving it to be an effective 
method for addressing CMOP-UC problems. 
Keywords: Multi-objective optimization · unknown constraints · evolutionary 
algorithm · constraint-handling technique · network 
1 Introduction 
Constrained multi-objective optimization problems (CMOPs) are widely prevalent 
in real-world engineering applications, e.g., the coal mine integrated energy system 
scheduling [1–3], complex product change plan [4, 5], building energy performance 
optimization [6], and resource constrained job scheduling [7]. For CMOPs, researchers
© The Author(s), under exclusive license to Springer Nature Singapore Pte Ltd. 2025 
D.-S. Huang et al. (Eds.): ICIC 2025, LNCS 15858, pp. 127–138, 2025. 
https://doi.org/10.1007/978-981-96-9805-9_11 
<PARSED TEXT FOR PAGE: 2 / 12>
128 J. Ma et al.
have designed numerous effective constrained multi-objective evolutionary algorithms 
(CMOEAs) [8–10]. However, these methods generally rely on constraint violation infor￾mation of candidate solutions, implying that the constraints are known and quantifi￾able. In fact, some problems lack explicit constraint functions and only show the given 
candidate solution’s feasibility via binary output. Existing studies refer to such prob￾lems as CMOP with unknown constraints (CMOP-UC) [11, 12]. Generally speaking, a 
CMOP-UC can be formulated as follows: 
min f (X ) = (f1(X ), f2(X ), ··· , fM (X ))
s.t. X ∈ S 
g(X ) = (0, 1)
(1) 
where M denotes the number of objectives; X = (x1, x2, ··· , xD)T is a D-dimensional 
decision vector in the decision space S; D represents the number of decision variables; 
g(X ) represents the system’s unknown or non-quantifiable constraints, which outputs 
either 1 or 0. Specifically, 1 indicates that the candidate solution X is feasible, whereas 
0 signifies that the solution is infeasible. 
Existing CMOEAs can be broadly classified into three categories: the penalty func￾tion method, the objective-constraint separation method, and the problem transformation 
method. The penalty function method constructs penalty terms based on constraint vio￾lation information, effectively converting the constrained problem into a form solvable 
without explicit constraints [13, 14]. The objective-constraint separation method evalu￾ates candidate solutions independently based on their objective and constraints [15–17]. 
The problem transformation method reformulates the original CMOP into either a multi￾stage or a cooperative optimization problem [18–20]. However, most of these methods 
rely on constraint violation information of candidate solutions, making them unsuitable 
for CMOP-UC. 
In fact, a few researchers have begun to focus on CMOP-UC. Huang et al. [11] 
proposed a CMOEA with the detection region method; Li et al. [12] introduced an 
evolutionary algorithm incorporating the alternating direction method of multipliers. 
However, these methods assume that the number of constraints is known, allowing a 
candidate solution to be evaluated based on how many constraints it satisfies. When the 
constraints are entirely unknown (i.e., non-quantifiable and the number of constraints is 
not predefined), these methods still exhibit certain limitations. 
For CMOP-UC, the key challenge lies in effectively perceiving the value of infeasi￾ble solutions. In fact, we can obtain a large number of candidate solutions by historical 
records or sampling, and divide them into two categories: feasible solutions labeled as 
1 and infeasible solutions labeled as 0. Then, train a multi-layer perceptron (MLP) on 
the collected data to estimate the value of any infeasible solution, and subsequently 
develop an effective constraint-handling method to enhance the algorithm’s ability to 
exploit valuable infeasible solutions. Furthermore, given that the constrained multitask￾ing framework has become a mainstream approach to solve CMOPs [8, 21], this paper 
develops a network-assisted constrained multitasking evolutionary framework. The main 
highlights of this study are summarized as follows: 
1) A network-assisted constrained multitasking evolutionary framework is proposed. 
Unlike existing constrained multitasking evolutionary algorithms, the proposed
<PARSED TEXT FOR PAGE: 3 / 12>
A Network-Assisted Evolutionary Multitask Framework 129
framework does not require explicit constraint violation information of candidate 
solutions, significantly enhancing its capability to handle optimization problems with 
unknown constraints. 
2) An perception strategy of infeasible solution value (ISVPS) is introduced. This strat￾egy uses an MLP to evaluate the value of an infeasible individual, thereby reducing 
the dependence of existing algorithms on constraint violation information. More￾over, the proposed strategy exhibits strong scalability and generalization capabilities, 
allowing seamless integration into any existing algorithms. 
3) The constraint dominance principle with predicted value (CDPPV) is designed. By 
identifying informative infeasible solutions, this principle improves the algorithm’s 
exploratory capability and mitigates the risk of getting trapped in local optima. 
4) An adaptive update strategy of the MLP model parameters is developed. Based on 
the accuracy metric and evolutionary data, this strategy adaptively retrains the MLP 
model, directing the population search toward more promising regions. 
2 The Proposed Algorithm 
Figure 1 illustrates the basic framework of the proposed algorithm. The algorithm has 
two main components: the perception module and the multitasking evolutionary module. 
The perception module utilizes an MLP to evaluate the value of a given infeasible 
solution, enhancing the algorithm’s capability to search potential feasible regions, while 
the multitask evolutionary module focuses on searching the variable space. Specifically, 
the main task solves the original CMOP-UC to find feasible Pareto optimal solutions; 
the auxiliary task solves the corresponding unconstrained MOP, guiding the population 
to quickly traverse larger infeasible regions, thereby enhancing its convergence. 
Algorithm 1 shows the pseudocode of the proposed algorithm. In Line 1, train an 
MLP using the set Data of size N1, which is generated by a random sampling method. 
Line 2 initializes two populations P1 and P2 with size N, i.e., the main and auxiliary 
populations. Lines 4–5 adopt the DE operator to generate offspring populations O1 and 
O2 corresponding to the main and auxiliary populations, respectively. Line 6 divides the 
offspring population O1 (O2) into two subsets: feasible subset O11 (O21) and infeasible 
subset O12 (O22). Line 7 estimates sequentially the value of each individual in infeasible 
subsets O12 and O22 by the proposed ISVPS strategy. Lines 8–9 use the environmental 
selection strategy to update the main population P1 and the auxiliary population P2, 
respectively. Line 10 calculates the accuracy metric of the current MLP. If this metric is 
below a specified threshold α, an adaptive update strategy of the MLP model parameters 
is executed to update the training set Data and to retrain the MLP (Lines 11–13). Note 
that the main task uses the proposed CDPPV combined with the crowding distance to 
update the main population, while the auxiliary task adopts the non-dominated sorting 
and the crowding distance to update the auxiliary population [15].
<PARSED TEXT FOR PAGE: 4 / 12>
130 J. Ma et al.
Main population 
(P1) 
O1=DE (P1) 
P1=Environment selection based 
on the proposed CDPPV method 
(P1, O1, O2) 
Auxiliary 
population (P2) 
O2=DE (P2) 
P2=Environmental 
Selection (P2, O2, 
O1) 
An adaptive update strategy of the MLP 
model parameters 
Data 
Termination? 
Multi-layer 
perceptron 
Feasible subset 
(O21) 
Feasible subset 
(O11) 
Infeasible subset 
(O12) 
Infeasible subset 
(O22) 
Output 
Feasible subset Infeasible 
subset 
Update 
Training 
Feedback 
Perception Module Multitask Evolutionary Module
The proposed ISVPS 
strategy 
Fig. 1. The general framework of the NA-EMT.
Algorithm 1 Pseudo codes of the NA-EMT
Input: Population size (N); Sample size (N1); Maximum 
number of evaluations (Maxgen); Threshold parameter (
).
Output The feasible Pareto optimal solutions.
1. Train a multi-layer perceptron MIP based on the sample 
set Data with size N1;
2. Generate two initial populations P1 and P2; 
3. While Termination criterion not fulfilled do
4. O1 = DE/current-to-rand-1 (P1); 
5. O2 = DE/current-to-rand-1 (P2); 
6. Divide the offspring population O1 (O2) into two 
sets: a feasible subset O11 (O21) and an non-feasible 
subset O12 (O22); 
7. Calculate sequentially the value of each individual 
in the infeasible subsets O12 and O22 by the proposed 
ISVPS strategy; % see Subsection 2.1 
8. P1 = Environment selection based on the proposed 
CDPPV method (P1, O1, O2); % see Subsection 2.2 
9. P2 = Environmental Selection (P2, O2, O1); 
10. Calculate the accuracy metric corresponding to the 
current MIP; 
11. If accuracy <  then 
12. Conduct an adaptive update strategy to update 
the training set Data and then to retrain the MLP; % see 
Subsection 2.3 
13. Endif
14. Endwhile
15. Output the feasible Pareto optimal solutions. 
<IMAGE FOR PAGE: 4 / 12>
![image](sediment://89f8d36e3a1439f#file_00000000234071f5af46a6c2d8fa39eb#p_0.beb394a011.jpg)
<PARSED TEXT FOR PAGE: 5 / 12>
A Network-Assisted Evolutionary Multitask Framework 131
2.1 An Perception Strategy of Infeasible Solution Value 
For CMOP-UC, effectively perceiving the value of infeasible solutions is a key challenge. 
A few studies assume that the number of constraints is known, allowing the value of an 
infeasible solution to be measured based on the number of satisfied constraints [11, 12]. 
However, these methods cannot address complex optimization problems that involve an 
unknown number of constraints. In fact, determining whether an individual is feasible or 
not can be treated as a probability classification problem. To address this, data collected 
or sampled can be used to train a probabilistic classification network to perceive the 
value of infeasible solutions, i.e., the probability that the infeasible solution satisfies 
the constraints. MLP has found extensive application in multi-objective optimization. 
For instance, references [22, 23] employ it to address both MOP and CMOP problems. 
In view of this, this section designs an perception strategy of infeasible solution value 
(ISVPS) based on the MIP. The specific implementation steps are as follows: 
First, a training set Data = {Xi, Yi}N 1 
i=1 is generated using the random sampling 
method. Here, Xi represents a given candidate solution, Yi is its corresponding label, 
and N1 is the size of the set Data. The label is either 1 or 0, where 1 indicates that the 
candidate solution Xi is feasible, and 0 signifies that it is infeasible. Next, 80% of the 
samples are randomly chosen as the training set, while the remaining 20% are used as the 
testing set. Then, the MLP shown in Fig. 2 is constructed, consisting of an input layer, 
a hidden layer, and an output layer. The input layer contains D neurons, corresponding 
to the number of decision variables. The hidden layer comprises K neurons (to balance 
computational complexity and network performance, K is set to be 10 in this paper), with 
the ReLU function as the activation function. The output layer consists of a single neuron 
and utilizes the sigmoid function to constrain the output within [0, 1], representing the 
value of the infeasible solution. Note that a larger value shows a higher likelihood that 
the candidate solution satisfies the constraints. As perceiving infeasible solution values 
constitutes a binary classification task, we adopt the binary cross-entropy loss function, 
defined as follows: 
L = 1 
N 1 

N 1
i=1 
[Yi log(
−
Yi) + (1 − Yi)log(1 − −
Yi)] (2) 
where −
Y 
i 
is the predicted value. Here, the scaled conjugate gradient algorithm is employed 
to update the network parameters to improve its convergence speed [24]. 
Additionally, feasible individuals from offspring populations are also fed into the 
MIP model to compute the difference between predicted and true values, which is used as 
the accuracy metric. This metric is further used to determine whether the adaptive update 
strategy of the MLP model parameters should be triggered, as detailed in Subsect. 2.3.
<PARSED TEXT FOR PAGE: 6 / 12>
132 J. Ma et al.
Input layer Hidden layer Output layer 
Multilayer perceptron (MLP) 
Input data set 
1 
0
Label 
Feasible 
individuals 
Population 
Individual value 
(val) 
Accuracy rate
Pridiction 
Feedback 
A trained MIP 
Infeasible subset 
Feasible subset 
Infeasible 
individuals 
Fig. 2. An perception strategy of infeasible solution value based on the MIP. 
2.2 The Constraint Dominance Principle with Predicted Value 
Existing research has shown that effectively utilizing valuable infeasible solutions can 
enhance both convergence and diversity of the population [8]. Therefore, many con￾straint or infeasible solution handling methods have been proposed, such as ε-constraint 
handling method. However, these methods often rely on the constraint violation infor￾mation of candidate solutions and are not applicable to CMOP-UC [16, 18]. In view 
of this, this section introduces the constraint dominance principle with predicted value 
(CDPPV). 
In the CDPPV method, a candidate solution X is considered superior to Y if and only 
if the following conditions are satisfied: 
• Val(X ) ≥ ε, Val(Y )<ε
• Val(X ) > ε, Val(Y ) > ε, and X ≺ Y 
• Val(X ) < ε, Val(Y ) < ε, and Val(X ) > Val(Y )
Here, ε is a user-specified control parameter. Notably, the proposed method does not rely 
on the constraint violation degree of candidate solutions, which significantly enhances 
the generalization performance of the original ε-constraint handling method, making it 
applicable to optimization problems with unknown constraints.
<IMAGE FOR PAGE: 6 / 12>
![image](sediment://f5e42fedd71e410#file_00000000234071f5af46a6c2d8fa39eb#p_1.3d3618d4dc.jpg)
<PARSED TEXT FOR PAGE: 7 / 12>
A Network-Assisted Evolutionary Multitask Framework 133
2.3 An Adaptive Update Strategy of the MLP Model Parameters 
During the evolution process, the population’s search regions gradually shift, rendering 
the early MLP model potentially unsuitable for later optimization tasks. Therefore, this 
section proposes an adaptive update strategy, which is used to update the MLP model 
parameters when the accuracy metric is low. 
First, extract feasible individuals from offspring populations O1 and O2, and feed 
them into the trained MLP to compute their predicted values. Next, determine the pre￾diction accuracy based on the difference between true and predicted values. A higher 
accuracy indicates that the MLP model performs better in perception. If the accuracy is 
below a given threshold α, it indicates that the MLP model is no longer suitable for the 
current population’s evolutionary trend. In this case, we retrain the MLP by updating the 
training set Data. To achieve this, an archive set A is introduced to store the individual 
information from the last 10 generations. Then, 10%*N1 feasible and infeasible individ￾uals are randomly selected from the archive set A and added to the training set Data. To 
avoid increasing the training cost, the sample size of the set Data remains constant at N1, 
meaning that the training set Data is updated by a first-in-first-out principle. Here, N1 
is set to be 1000 in this study. In the later stages, when all individuals in the population 
are feasible, the MLP model is no longer updated. 
3 Experimental Study 
A modified DASCMOP test set [25], namely DASCMOP-UC which limits the constraint 
violation degree of an given solution to either 1 or 0, is employed to test the performance 
of NA-EMT. Here, 1 denotes a feasible solution, while 0 signifies an infeasible solution. 
6 CMOEAs are selected as comparative algorithms, namely NSGA-II [15], CMOEA\D 
[26], ToP [27], PPS [18], MCCMO [28], and MTCMO [8]. Among them, NSGA-II 
and CMOEA\D represent two classical CMOEAs; ToP and PPS are two multi-stage 
CMOEAs; MCCMO and MTCMO represent multi-population and multitask CMOEAs, 
respectively. 
The inverted generational distance (IGD) [29] and Hypervolume (HV) [30] metrics 
are employed to test the performance of an algorithm. A smaller IGD value and a larger 
HV value indicate that an algorithm has better performance. The Wilcoxon rank-sum test 
is adopted to observe whether there are significant difference between two algorithms. 
The symbols “ +”, “-”, and “ =” respectively represent whether the performance of the 
comparison algorithm is better than, worse than, or similar to our proposed algorithm. 
The parameter settings for the proposed algorithm in this paper are as follows: (1) 
the population size N is set to be 100, and the maximum number of evaluations MaxGen 
is set to be 200,000; (2) The parameters CR and F in the DE operator are set to be 1 
and 0.5 respectively, and the probability of polynomial mutation is set to be 1/D, the 
distribution index is set to be 20. To ensure fair comparison between the comparison 
algorithms, all comparison algorithms are independently run 30 times for each test case. 
All experiments in this paper are conducted in the PlatEMO [31].
<PARSED TEXT FOR PAGE: 8 / 12>
134 J. Ma et al.
Fig. 3. PFs obtained by MTCMO, MCCMO and NA-EMT on the selected test cases. (a) MTCMO￾DASCMOP-UC1. (b) MCCMO-DASCMOP-UC1. (c) NA-EMT-DASCMOP-UC1. (d) MTCMO￾DASCMOP-UC6. (e) MCCMO-DASCMOP-UC6. (f) NA-EMT-DASCMOP-UC6. (g) MTCMO￾DASCMOP-UC9. (h) MCCMO-DASCMOP-UC9. (i) NA-EMT-DASCMOP-UC9. 
Tables 1 and 2 compare the two metrics obtained by the proposed and 6 comparison 
algorithms, respectively. Table 1 shows that, with respect to the IGD metric, the NA￾EMT significantly outperforms the comparison algorithms NSGA-II, CMOEAD, ToP, 
PPS, MTCMO and MCCMO for 4, 8, 4, 8, 6 and 4 test cases, respectively. Moreover, 

The file is too long and its contents have been truncated.


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### Plugin (file_search):
Make sure to include fileciteturn5file0 in your response to cite this file, or to surface it as a link.
PRBCCMO MATLAB implementation package
Source directory: Algorithms\Multi-objective optimization\PRBCCMO
Generated at: 2026-03-20 15:58:47
File count: 21

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\AssociateSectors.m
====================================================================================================

function [Sector,Count] = AssociateSectors(PopObj,W,RefObj)
% Associate objective vectors to the nearest reference sector.

    if nargin < 3 || isempty(RefObj)
        RefObj = PopObj;
    end
    if isempty(PopObj)
        Sector = zeros(0,1);
        Count  = zeros(size(W,1),1);
        return;
    end
    if isempty(W)
        Sector = ones(size(PopObj,1),1);
        Count  = size(PopObj,1);
        return;
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range  = MaxObj - MinObj;
    Range(Range<1e-12) = 1;

    Obj = (PopObj-MinObj)./Range;
    RowNorm = sqrt(sum(Obj.^2,2));
    ZeroRow = RowNorm < 1e-12;
    Obj(ZeroRow,:) = 1;
    RowNorm(ZeroRow) = sqrt(size(Obj,2));
    Obj = Obj./repmat(RowNorm,1,size(Obj,2));

    WNorm = sqrt(sum(W.^2,2));
    WNorm(WNorm<1e-12) = 1;
    Wn = W./repmat(WNorm,1,size(W,2));

    Cosine = Obj*Wn';
    [~,Sector] = max(Cosine,[],2);
    Count = accumarray(Sector,1,[size(W,1),1]);
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\BuildBoundaryRuntimeOptions.m
====================================================================================================

function Options = BuildBoundaryRuntimeOptions(SelectionMode,LocalMode,TraceFlag)
% Build runtime options for trust-aware Pareto-bridge querying.

    if nargin < 1 || isempty(SelectionMode)
        SelectionMode = 1;
    end
    if nargin < 2 || isempty(LocalMode)
        LocalMode = 1;
    end
    if nargin < 3 || isempty(TraceFlag)
        TraceFlag = false;
    end

    SelectionMode = max(1,min(3,round(SelectionMode)));
    LocalMode     = max(1,min(2,round(LocalMode)));

    Options = struct();
    Options.SelectionMode = SelectionMode;
    Options.LocalMode     = LocalMode;
    Options.TraceFlag     = logical(TraceFlag);
    Options.SelectionName = ResolveSelectionName(SelectionMode);
    Options.LocalName     = ResolveLocalName(LocalMode);
    Options.BridgeActivationGap = 0.01;
    Options.BridgeScanLambda    = [0.20,0.35,0.50,0.65,0.80];
end

function Name = ResolveSelectionName(SelectionMode)
    switch SelectionMode
        case 1
            Name = 'trusted_query';
        case 2
            Name = 'uncertain_only';
        case 3
            Name = 'random_bridge';
        otherwise
            Name = 'trusted_query';
    end
end

function Name = ResolveLocalName(LocalMode)
    switch LocalMode
        case 1
            Name = 'label_aware';
        case 2
            Name = 'isotropic';
        otherwise
            Name = 'label_aware';
    end
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\CalFitness.m
====================================================================================================

function Fitness = CalFitness(PopObj,PopCon)
% Calculate the SPEA2-style fitness used by CCMO.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);
    if nargin == 1
        CV = zeros(N,1);
    else
        % PRBCCMO targets unknown binary constraints, so all infeasible
        % solutions share the same violation level.
        CV = double(any(PopCon>0,2));
    end

    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end

    S = sum(Dominate,2);
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end

    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);

    Fitness = R + D';
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\ComputeSectorScalar.m
====================================================================================================

function Value = ComputeSectorScalar(Obj,W,RefObj,Sector)
% Compute the normalized weighted-sum scalar value for one or more sectors.

    if isempty(Obj)
        Value = zeros(0,1);
        return;
    end
    if nargin < 2 || isempty(W)
        W = ones(1,size(Obj,2));
    end
    if nargin < 3 || isempty(RefObj)
        RefObj = Obj;
    end
    if nargin < 4 || isempty(Sector)
        if size(W,1) == 1
            Weight = repmat(W,size(Obj,1),1);
        elseif size(W,1) == size(Obj,1)
            Weight = W;
        else
            Weight = repmat(W(1,:),size(Obj,1),1);
        end
    else
        Weight = W(Sector,:);
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range  = MaxObj - MinObj;
    Range(Range<1e-12) = 1;
    NormObj = (Obj - MinObj)./Range;
    Value   = sum(NormObj.*Weight,2);
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\EnvironmentalSelectionC.m
====================================================================================================

function [Population,Fitness,SelectedIdx] = EnvironmentalSelectionC(Population,N,MigrationPool,W)
% Reserve slots for improved feasible migrants, then apply feasible-first NSGA-II.

    if nargin < 3 || isempty(MigrationPool)
        MigrationPool = [];
    end
    if nargin < 4
        W = [];
    end
    if isempty(Population) && isempty(MigrationPool)
        Fitness = [];
        SelectedIdx = [];
        return;
    end

    BasePool = KeepUniquePopulation(Population);
    MigrationPool = KeepUniquePopulation(MigrationPool);
    Reserved = SelectReservedMigrants(BasePool,MigrationPool,W);
    if numel(Reserved) > N
        Reserved = Reserved(1:N);
    end

    CandidatePool = KeepUniquePopulation([BasePool,MigrationPool]);
    CandidatePool = RemovePopulationByDecision(CandidatePool,Reserved);

    Selected = Reserved;
    Need = max(0,N-numel(Selected));
    if Need > 0
        FeasiblePool = CandidatePool(all(CandidatePool.cons<=0,2));
        [Chosen,~] = SelectObjectivePopulation(FeasiblePool,min(Need,numel(FeasiblePool)));
        Selected = [Selected,Chosen];
    end

    Need = max(0,N-numel(Selected));
    if Need > 0
        Remaining = RemovePopulationByDecision(CandidatePool,Selected);
        [Chosen,~] = SelectObjectivePopulation(Remaining,min(Need,numel(Remaining)));
        Selected = [Selected,Chosen];
    end

    if isempty(Selected)
        Fitness = [];
        SelectedIdx = [];
        Population = [];
        return;
    end
    if numel(Selected) < N
        Repeat = Selected(mod(0:N-numel(Selected)-1,numel(Selected))+1);
        Selected = [Selected,Repeat];
    end

    Population = Selected(:)';
    Fitness    = CalFitness(Population.objs,Population.cons);
    SelectedIdx = 1:numel(Population);
end

function Reserved = SelectReservedMigrants(BasePool,MigrationPool,W)
    Reserved = [];
    if isempty(MigrationPool)
        return;
    end

    FeasibleBase = BasePool(all(BasePool.cons<=0,2));
    FeasibleMig  = MigrationPool(all(MigrationPool.cons<=0,2));
    if isempty(FeasibleMig)
        return;
    end

    if isempty(W)
        W = ones(1,size(FeasibleMig.objs,2));
    end
    RefObj = FeasibleMig.objs;
    if ~isempty(FeasibleBase)
        RefObj = [FeasibleBase.objs;FeasibleMig.objs];
    end

    SectorMig = AssociateSectors(FeasibleMig.objs,W,RefObj);
    if isempty(FeasibleBase)
        SectorBase = zeros(0,1);
    else
        SectorBase = AssociateSectors(FeasibleBase.objs,W,RefObj);
    end

    Keep = false(1,numel(FeasibleMig));
    Improve = -inf(1,numel(FeasibleMig));
    for s = unique(SectorMig(:))'
        MigIdx = find(SectorMig==s);
        MigValue = ComputeSectorScalar(FeasibleMig(MigIdx).objs,W,RefObj,repmat(s,numel(MigIdx),1));
        [BestMigValue,BestLocal] = min(MigValue);
        BestMigIdx = MigIdx(BestLocal);

        BaseIdx = find(SectorBase==s);
        if isempty(BaseIdx)
            Keep(BestMigIdx) = true;
            Improve(BestMigIdx) = inf;
            continue;
        end

        BaseValue = ComputeSectorScalar(FeasibleBase(BaseIdx).objs,W,RefObj,repmat(s,numel(BaseIdx),1));
        ChampionValue = min(BaseValue);
        if BestMigValue < ChampionValue
            Keep(BestMigIdx) = true;
            Improve(BestMigIdx) = ChampionValue - BestMigValue;
        end
    end

    Reserved = FeasibleMig(Keep);
    if isempty(Reserved)
        return;
    end
    Improve = Improve(Keep);
    [~,Order] = sort(Improve,'descend');
    Reserved = Reserved(Order);
end

function Population = KeepUniquePopulation(Population)
    if isempty(Population)
        return;
    end
    Keep = KeepLatestDecisionRows(Population.decs);
    Population = Population(Keep);
end

function Population = RemovePopulationByDecision(Population,Remove)
    if isempty(Population) || isempty(Remove)
        return;
    end
    Keep = ~ismember(Population.decs,Remove.decs,'rows');
    Population = Population(Keep);
end

function [Population,Idx] = SelectObjectivePopulation(Population,N)
    Idx = zeros(1,0);
    if isempty(Population) || N <= 0
        Population = [];
        return;
    end
    LocalIdx = SelectObjectiveIdx(Population.objs,min(N,numel(Population)));
    Population = Population(LocalIdx);
    Idx = LocalIdx(:)';
end

function Idx = SelectObjectiveIdx(PopObj,N)
    if isempty(PopObj)
        Idx = zeros(1,0);
        return;
    end
    [FrontNo,MaxFNo] = NDSort(PopObj,N);
    Next             = FrontNo < MaxFNo;
    CrowdDis         = CrowdingDistance(PopObj,FrontNo);
    Last             = find(FrontNo==MaxFNo);
    Need             = N - sum(Next);
    if Need > 0
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:Need))) = true;
    end
    Idx = find(Next);
    [~,Order] = sortrows([FrontNo(Idx)',-CrowdDis(Idx)'],[1 2]);
    Idx = Idx(Order)';
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\EnvironmentalSelectionU.m
====================================================================================================

function [Population,Fitness] = EnvironmentalSelectionU(Population,N)
% Environmental selection for the unconstrained helper population.

    Population = SelectByNSGA2Local(Population,N);
    Fitness = CalFitness(Population.objs);
end

function Population = SelectByNSGA2Local(Population,N)
    if isempty(Population)
        return;
    end

    if numel(Population) <= N
        [FrontNo,~] = NDSort(Population.objs,numel(Population));
        CrowdDis = CrowdingDistance(Population.objs,FrontNo);
        [~,Order] = sortrows([FrontNo(:),-CrowdDis(:)],[1 2]);
        Population = Population(Order);
        return;
    end

    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next     = FrontNo < MaxFNo;
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    Last     = find(FrontNo==MaxFNo);
    Need     = N - sum(Next);
    if Need > 0
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:Need))) = true;
    end

    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    [~,Order]  = sortrows([FrontNo(:),-CrowdDis(:)],[1 2]);
    Population = Population(Order);
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\EvaluateBoundaryCalibration.m
====================================================================================================

function Metric = EvaluateBoundaryCalibration(Model,CalDec,CalLabel,BinCount)
% Evaluate calibration quality on a held-out labeled evaluation buffer.

    if nargin < 4 || isempty(BinCount)
        BinCount = 10;
    end

    Metric = SummarizeCalibrationProbabilities([],[],BinCount);
    Metric.brier     = inf;
    Metric.ece       = inf;
    Metric.nearGap   = inf;
    Metric.nearCount = 0;
    Metric.trustGate = false;
    Metric.calibrator = 'raw';
    if isempty(CalDec) || isempty(CalLabel) || isempty(Model)
        return;
    end

    CalLabel = double(CalLabel(:));
    if numel(unique(CalLabel)) < 2
        return;
    end

    Prob = PredictBoundaryMLP(Model,CalDec);
    Metric = SummarizeCalibrationProbabilities(Prob,CalLabel,BinCount);
    if isfield(Model,'TrustGate') && ~isempty(Model.TrustGate)
        Metric.trustGate = logical(Model.TrustGate);
    end
    if isfield(Model,'CalibratorType') && ~isempty(Model.CalibratorType)
        Metric.calibrator = Model.CalibratorType;
    end
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\GenerateBoundaryCandidates.m
====================================================================================================

function Pool = GenerateBoundaryCandidates(Problem,PopulationC,PopulationU,W,RuntimeOptions)
% Build one bridge record for each active feasible-infeasible sector pair.

    if nargin < 5 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    Pool = InitBridgePool(Problem);

    FeasibleC   = PopulationC(all(PopulationC.cons<=0,2));
    InfeasibleU = PopulationU(~all(PopulationU.cons<=0,2));
    if isempty(FeasibleC) || isempty(InfeasibleU)
        return;
    end

    RefObj  = [FeasibleC.objs;InfeasibleU.objs];
    SectorF = AssociateSectors(FeasibleC.objs,W,RefObj);
    SectorU = AssociateSectors(InfeasibleU.objs,W,RefObj);
    ScalarF = ComputeSectorScalar(FeasibleC.objs,W,RefObj,SectorF);
    ScalarU = ComputeSectorScalar(InfeasibleU.objs,W,RefObj,SectorU);
    DeltaG  = ResolveBridgeActivationGap(RuntimeOptions);

    [BestFIdx,BestFSector,BestFValue] = SelectSectorBest(SectorF,ScalarF);
    [BestUIdx,BestUSector,BestUValue] = SelectSectorBest(SectorU,ScalarU);
    [SharedSector,FLoc,ULoc] = intersect(BestFSector,BestUSector,'stable');
    if isempty(SharedSector)
        return;
    end

    Active = BestUValue(ULoc) + DeltaG < BestFValue(FLoc);
    SharedSector = SharedSector(Active);
    FLoc = FLoc(Active);
    ULoc = ULoc(Active);
    Count = numel(SharedSector);
    if Count == 0
        return;
    end

    Pool.source    = ones(Count,1);
    Pool.sector    = SharedSector(:);
    Pool.anchorDec = FeasibleC(BestFIdx(FLoc)).decs;
    Pool.anchorObj = FeasibleC(BestFIdx(FLoc)).objs;
    Pool.helperDec = InfeasibleU(BestUIdx(ULoc)).decs;
    Pool.helperObj = InfeasibleU(BestUIdx(ULoc)).objs;
end

function Pool = InitBridgePool(Problem)
    Pool.source    = zeros(0,1);
    Pool.sector    = zeros(0,1);
    Pool.anchorDec = zeros(0,Problem.D);
    Pool.anchorObj = zeros(0,Problem.M);
    Pool.helperDec = zeros(0,Problem.D);
    Pool.helperObj = zeros(0,Problem.M);
end

function [BestIdx,BestSector,BestValue] = SelectSectorBest(Sector,Scalar)
    if isempty(Sector)
        BestIdx = zeros(0,1);
        BestSector = zeros(0,1);
        BestValue = zeros(0,1);
        return;
    end

    RankTable = [(1:numel(Sector))',Sector(:),Scalar(:)];
    RankTable = sortrows(RankTable,[2 3 1]);
    [~,First] = unique(RankTable(:,2),'stable');
    BestIdx    = RankTable(First,1);
    BestSector = RankTable(First,2);
    BestValue  = RankTable(First,3);
end

function DeltaG = ResolveBridgeActivationGap(RuntimeOptions)
    DeltaG = 0.01;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgeActivationGap') ...
            && ~isempty(RuntimeOptions.BridgeActivationGap)
        DeltaG = max(RuntimeOptions.BridgeActivationGap,0);
    end
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\KeepLatestDecisionRows.m
====================================================================================================

function Keep = KeepLatestDecisionRows(Dec)
% Return row indices that keep only the latest occurrence of each decision.

    if isempty(Dec)
        Keep = zeros(0,1);
        return;
    end

    [~,RevKeep] = unique(flipud(Dec),'rows','stable');
    Keep = sort(size(Dec,1)-RevKeep+1);
    Keep = Keep(:);
end

====================================================================================================
FILE: Algorithms\Multi-objective optimization\PRBCCMO\PRBCCMO.m
====================================================================================================

classdef PRBCCMO < ALGORITHM
% <2026> <multi> <real/integer/label/binary/permutation> <constrained>
% Pareto-relevant boundary CCMO
% bRho     --- 0.2  --- Boundary evaluation ratio relative to N
% trainRho --- 2    --- Training archive size ratio
% hidden   --- 20   --- Hidden units of the boundary MLP
% epoch    --- 25   --- Training epochs of the boundary MLP
% lr       --- 0.01 --- Learning rate of the boundary MLP
% mRho     --- 0.4  --- Seed-query ratio within each boundary budget
% ensK     --- 3    --- Committee size of shallow boundary MLPs
% calMode  --- 2    --- Calibration mode (1 raw, 2 auto temp/beta, 3 beta)
% dLambda  --- 1    --- Committee disagreement weight in boundary utility
% pairM    --- 0.05 --- Margin for tight bracket pair loss
% lPair    --- 1    --- Weight of bracket pair loss
% lMid     --- 1    --- Weight of midpoint-to-0.5 loss
% selMode  --- 1    --- Section B seed selection mode (1. trusted-query 2. uncertain-only 3. random-bridge)
% localMode--- 1    --- Section B local search mode (1. label-aware 2. isotropic)
% traceOn  --- 0    --- Enable Section B audit trace recording

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            Params = ResolvePRBCCMOParameters(Algorithm.parameter);
            bRho      = Params.bRho;
            trainRho  = Params.trainRho;
            hidden    = Params.hidden;
            epoch     = Params.epoch;
            lr        = Params.lr;
            mRho      = Params.mRho;
            ensK      = Params.ensK;
            calMode   = Params.calMode;
            dLambda   = Params.dLambda;
            pairM     = Params.pairM;
            lPair     = Params.lPair;
            lMid      = Params.lMid;
            selMode   = Params.selMode;
            localMode = Params.localMode;
            traceOn   = Params.traceOn;
            RuntimeOptions = BuildBoundaryRuntimeOptions(selMode,localMode,traceOn);

            BoundaryBudget = max(0,floor(bRho*Problem.N));
            TrainMax       = max(1,round(trainRho*Problem.N));
            CalibMax       = max(1,Problem.N);
            TestMax        = max(1,Problem.N);
            ProtectedOtherMax = max(20,ceil(0.25*TrainMax));
            BracketMax     = max(1,Problem.N);
            HardNegMax     = max(20,ceil(0.25*TrainMax));
            SeedRatio      = min(max(mRho,0),1);
            [W,~]          = UniformPoint(max(Problem.N,2),Problem.M);
            UpdateGap      = 5;
            RestartGap     = 25;
            WarmEpoch      = min(epoch,max(5,round(epoch/3)));
            TriggerCount   = max(1,ceil(0.1*TrainMax));
            TightGap       = 0.03;

            %% Generate random populations
            PopulationC = Problem.Initialization();
            PopulationU = Problem.Initialization();

            %% Initialize boundary memories
            BracketArchive = EmptyBracketArchive(Problem.D);
            HardNegativeArchive.Dec    = zeros(0,Problem.D);
            HardNegativeArchive.Radius = zeros(0,1);
            ProtectedOtherDec   = zeros(0,Problem.D);
            ProtectedOtherLabel = zeros(0,1);
            InitSolutions = [PopulationC,PopulationU];
            [InitTrain,~,InitCalib,InitCalibInfo,InitTest,InitTestInfo] = SplitHeldOutBatch( ...
                InitSolutions,NormalizeBoundaryInfo([],Problem.M),CalibMax,TestMax,Problem.M);
            InitHoldoutDec = [SolutionDecs(InitCalib,Problem.D);SolutionDecs(InitTest,Problem.D)];
            ProtectedDec   = zeros(0,Problem.D);
            ProtectedLabel = zeros(0,1);
            [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                [],[],ProtectedDec,ProtectedLabel,InitTrain,InitHoldoutDec,TrainMax);
            [CalibDec,CalibLabel,CalibNear] = UpdateCalibrationBuffer( ...
                [],[],[],InitCalib,InitCalibInfo,CalibMax);
            [TestDec,TestLabel,TestNear] = UpdateCalibrationBuffer( ...
                [],[],[],InitTest,InitTestInfo,TestMax);
            TrainOptions = BuildBoundaryTrainingOptions( ...
                BracketArchive,ensK,calMode,dLambda,pairM,lPair,lMid,Problem.D,TightGap);
            Model = TrainBoundaryMLP( ...
                TrainDec,TrainLabel,hidden,epoch,lr,[],CalibDec,CalibLabel,TrainOptions);
            Model = RefreshBoundaryTrust(Model,TestDec,TestLabel);
            LastCalMetric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel);
            PendingLabels = 0;
            Generation    = 0;
            [ExternalArchive,~] = UpdateExternalArchive([],FilterFeasiblePopulation(InitSolutions));
            Algorithm.metric.boundaryCalibration = AttachCalibrationContext( ...
                LastCalMetric,Generation,Problem.FE,size(TrainDec,1),size(CalibDec,1), ...
                sum(CalibNear),size(TestDec,1),sum(TestNear));
            Algorithm.metric.sectionB = InitSectionBMetric(Problem.D,RuntimeOptions,ExternalArchive);
            Algorithm.metric.sectionB = AppendSectionBCalibrationTrace( ...
                Algorithm.metric.sectionB,Algorithm.metric.boundaryCalibration);

            %% Optimization
            while Algorithm.NotTerminated(PopulationC)
                Generation = Generation + 1;

                [OffspringC,OffspringU] = GenerateRegularOffspring( ...
                    Problem,PopulationC,PopulationU);

                CandidatePool = GenerateBoundaryCandidates( ...
                    Problem,PopulationC,PopulationU,W,RuntimeOptions);

                BoundaryBudgetNow = min(BoundaryBudget,max(0,Problem.maxFE-Problem.FE));
                SeedBudget = min(BoundaryBudgetNow,max(0,round(SeedRatio*BoundaryBudgetNow)));
                if BoundaryBudgetNow > 0 && SeedBudget == 0
                    SeedBudget = 1;
                end
                [BoundarySeeds,SeedInfo] = SelectBoundaryCandidates( ...
                    Problem,CandidatePool,PopulationC,Model,W,HardNegativeArchive,SeedBudget,RuntimeOptions);

                WorkerBudget = max(0,BoundaryBudgetNow-numel(BoundarySeeds));
                [WorkerOffspring,WorkerInfo,MigrationPool,BracketBatch,HardNegBatch,WorkerAudit] = ...
                    RefineBoundaryWorkers( ...
                        Problem,BoundarySeeds,SeedInfo,PopulationC,Model,W, ...
                        HardNegativeArchive,WorkerBudget,RuntimeOptions);
                [BoundaryOffspring,BoundaryInfo] = MergeBoundaryResults( ...
                    BoundarySeeds,SeedInfo,WorkerOffspring,WorkerInfo,Problem.M);

                HardNegativeArchive = UpdateHardNegativeArchive(HardNegativeArchive,HardNegBatch,HardNegMax);
                BracketArchive = UpdateBracketArchive(BracketArchive,BracketBatch,BracketMax,Problem.D,TightGap);

                ConstrainedBase = KeepUniquePopulation([PopulationC,OffspringC]);
                PopulationC = EnvironmentalSelectionC(ConstrainedBase,Problem.N,MigrationPool,W);
                PopulationU = EnvironmentalSelectionU(KeepUniquePopulation([PopulationU,OffspringU]),Problem.N);

                [ExternalArchive,BoundaryGain,BoundaryAdded] = UpdateSectionBExternalArchive( ...
                    ExternalArchive,OffspringC,OffspringU,BoundaryOffspring);
                Algorithm.metric.sectionB.seedAudit = AppendBoundarySeedAuditRows( ...
                    Algorithm.metric.sectionB.seedAudit,BoundarySeeds,SeedInfo,WorkerAudit, ...
                    BoundaryAdded,Generation,Problem.FE,Problem.D);
                Algorithm.metric.sectionB.boundaryGainTrace = AppendBoundaryGainTrace( ...
                    Algorithm.metric.sectionB.boundaryGainTrace,Generation,Problem.FE, ...
                    BoundaryGain,numel(BoundaryAdded),numel(ExternalArchive));
                Algorithm.metric.sectionB.externalArchiveCount = numel(ExternalArchive);
                Algorithm.metric.sectionB.totalBoundaryGain = Algorithm.metric.sectionB.totalBoundaryGain + BoundaryGain;

                [TrainBatch,TrainInfo,CalibBatch,CalibInfo,TestBatch,TestInfo] = SplitHeldOutBatch( ...
                    BoundaryOffspring,BoundaryInfo,CalibMax,TestMax,Problem.M);
                HoldoutDec = [CalibDec;TestDec; ...
                    SolutionDecs(CalibBatch,Problem.D);SolutionDecs(TestBatch,Problem.D)];
                [ProtectedBracketDec,ProtectedBracketLabel] = BuildBracketProtectedBuffer(BracketArchive,Problem.D);
                [ProtectedBracketDec,ProtectedBracketLabel] = ExcludeLabeledRows( ...
                    ProtectedBracketDec,ProtectedBracketLabel,HoldoutDec);
                [ProtectedOtherAddDec,ProtectedOtherAddLabel] = CollectOtherProtectedCases( ...
                    TrainBatch,TrainInfo,HardNegBatch,Problem.D);
                [ProtectedOtherAddDec,ProtectedOtherAddLabel] = ExcludeLabeledRows( ...
                    ProtectedOtherAddDec,ProtectedOtherAddLabel,HoldoutDec);
                [ProtectedOtherDec,ProtectedOtherLabel] = UpdateProtectedBuffer( ...
                    ProtectedOtherDec,ProtectedOtherLabel,ProtectedOtherAddDec,ProtectedOtherAddLabel,ProtectedOtherMax);
                [ProtectedOtherDec,ProtectedOtherLabel] = ExcludeLabeledRows( ...
                    ProtectedOtherDec,ProtectedOtherLabel,HoldoutDec);
                ProtectedDec   = [ProtectedBracketDec;ProtectedOtherDec];
                ProtectedLabel = [ProtectedBracketLabel;ProtectedOtherLabel];
                [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                    TrainDec,TrainLabel,ProtectedDec,ProtectedLabel,TrainBatch,HoldoutDec,TrainMax);
                [CalibDec,CalibLabel,CalibNear] = UpdateCalibrationBuffer( ...
                    CalibDec,CalibLabel,CalibNear,CalibBatch,CalibInfo,CalibMax);
                [TestDec,TestLabel,TestNear] = UpdateCalibrationBuffer( ...
                    TestDec,TestLabel,TestNear,TestBatch,TestInfo,TestMax);

                PendingLabels = PendingLabels + numel(BoundaryOffspring);
                TrainOptions = BuildBoundaryTrainingOptions( ...
                    BracketArchive,ensK,calMode,dLambda,pairM,lPair,lMid,Problem.D,TightGap);
                [Model,PendingLabels,LastCalMetric] = UpdateBoundaryModel( ...
                    Model,TrainDec,TrainLabel,CalibDec,CalibLabel,TestDec,TestLabel, ...
                    hidden,epoch,WarmEpoch,lr,Generation,PendingLabels,TriggerCount, ...
                    UpdateGap,RestartGap,LastCalMetric,TrainOptions);
                Algorithm.metric.boundaryCalibration = AttachCalibrationContext( ...
                    EvaluateBoundaryCalibration(Model,TestDec,TestLabel), ...
                    Generation,Problem.FE,size(TrainDec,1),size(CalibDec,1), ...
                    sum(CalibNear),size(TestDec,1),sum(TestNear));
                Algorithm.metric.sectionB = AppendSectionBCalibrationTrace( ...
                    Algorithm.metric.sectionB,Algorithm.metric.boundaryCalibration);
            end
        end
    end
end

function [OffspringC,OffspringU] = GenerateRegularOffspring(Problem,PopulationC,PopulationU)
    OffspringC = Problem.Evaluation(OperatorDE_current_rand_1(Problem,PopulationC.decs));
    OffspringU = Problem.Evaluation(OperatorDE_current_rand_1(Problem,PopulationU.decs));
end

function Population = KeepUniquePopulation(Population)
    if isempty(Population)
        return;
    end
    Keep = KeepLatestDecisionRows(Population.decs);
    Population = Population(Keep);
end

function Population = FilterFeasiblePopulation(Population)
    if isempty(Population)
        return;
    end
    Population = Population(all(Population.cons<=0,2));
end

function [TrainSolutions,TrainInfo,CalibSolutions,CalibInfo,TestSolutions,TestInfo] = SplitHeldOutBatch(Solutions,Info,CalibMax,TestMax,M)
    TrainSolutions = Solutions;
    TrainInfo = NormalizeBoundaryInfo(Info,M);
    CalibSolutions = [];
    CalibInfo = NormalizeBoundaryInfo([],M);
    TestSolutions = [];
    TestInfo = NormalizeBoundaryInfo([],M);
    Count = numel(Solutions);
    if Count <= 1 || (CalibMax <= 0 && TestMax <= 0)
        return;
    end

    Info = NormalizeBoundaryInfo(Info,M);
    if numel(Info.source) ~= Count
        Info = DefaultBoundaryInfo(Solutions,M);
    end
    Label = double(all(Solutions.cons<=0,2));
    NearMask = true(Count,1);
    if isfield(Info,'prob') && numel(Info.prob) == Count
        NearMask = abs(Info.prob(:)-0.5) <= 0.1;
    end

    CalibQuota = min(max(1,round(0.2*Count)),min(CalibMax,Count-1));
    CalibIdx = SelectCalibrationHoldout(Label,NearMask,CalibQuota);
    RemainingMask = true(Count,1);
    RemainingMask(CalibIdx) = false;

    RemainingIdx = find(RemainingMask);
    TestQuota = min(max(1,round(0.2*Count)),TestMax);
    TestQuota = min(TestQuota,max(0,numel(RemainingIdx)-1));
    if TestQuota > 0
        TestLocalIdx = SelectCalibrationHoldout(Label(RemainingIdx),NearMask(RemainingIdx),TestQuota);
        TestIdx = RemainingIdx(TestLocalIdx);
    else
        TestIdx = zeros(0,1);
    end

    TrainMask = true(Count,1);
    TrainMask(CalibIdx) = false;
    TrainMask(TestIdx) = false;
    TrainIdx = find(TrainMask);
    TrainSolutions = Solutions(TrainIdx);
    TrainInfo = SliceBoundaryInfo(Info,TrainIdx,M);
    CalibSolutions = Solutions(CalibIdx);
    CalibInfo = SliceBoundaryInfo(Info,CalibIdx,M);
    TestSolutions = Solutions(TestIdx);
    TestInfo = SliceBoundaryInfo(Info,TestIdx,M);
end

function HoldoutIdx = SelectCalibrationHoldout(Label,NearMask,Quota)
    HoldoutIdx = zeros(0,1);
    Total = numel(Label);
    if Total <= 1 || Quota <= 0
        return;
    end

    Quota = min(Quota,Total-1);
    ClassOrder = [1,0];
    BaseQuota = floor(Quota/2);
    HoldoutCell = cell(1,numel(ClassOrder)+1);
    HoldCount = 0;
    for i = 1 : numel(ClassOrder)
        ClassIdx = FindCalibrationCandidates(Label,NearMask,ClassOrder(i));
        Take = min(numel(ClassIdx),BaseQuota);
        if Take > 0
            HoldCount = HoldCount + 1;
            HoldoutCell{HoldCount} = ClassIdx(1:Take);
        end
    end
    HoldoutIdx = vertcat(HoldoutCell{1:HoldCount});

    if numel(HoldoutIdx) < Quota
        Remaining = setdiff((1:Total)',HoldoutIdx,'stable');
        NearFirst = [Remaining(NearMask(Remaining));Remaining(~NearMask(Remaining))];
        Extra = NearFirst(1:min(Quota-numel(HoldoutIdx),numel(NearFirst)));
        HoldoutIdx = [HoldoutIdx;Extra(:)];
    end

    HoldoutIdx = unique(HoldoutIdx,'stable');
end

function Idx = FindCalibrationCandidates(Label,NearMask,ClassValue)
    NearIdx = find(NearMask & Label==ClassValue);
    FarIdx  = find(~NearMask & Label==ClassValue);
    Idx = [NearIdx(:);FarIdx(:)];
end

function Info = DefaultBoundaryInfo(Solutions,M)
    Count = numel(Solutions);
    Info = NormalizeBoundaryInfo([],M);
    Info.source    = zeros(Count,1);
    Info.score     = zeros(Count,1);
    Info.prob      = 0.5*ones(Count,1);
    Info.entropy   = zeros(Count,1);
    Info.hvGain    = zeros(Count,1);
    Info.novelty   = zeros(Count,1);
    Info.penalty   = ones(Count,1);
    Info.utility   = zeros(Count,1);
    Info.sector    = zeros(Count,1);
    Info.proxyObjs = Solutions.objs;
end

function [Dec,Label] = ExcludeLabeledRows(Dec,Label,ExcludeDec)
    if isempty(Dec) || isempty(ExcludeDec)
        return;
    end
    Keep = ~ismember(Dec,ExcludeDec,'rows');
    Dec = Dec(Keep,:);
    Label = Label(Keep);
end

function Dec = SolutionDecs(Solutions,D)
    if isempty(Solutions)
        Dec = zeros(0,D);
        return;
    end
    Dec = Solutions.decs;
end

function [ProtectedDec,ProtectedLabel] = CollectOtherProtectedCases(BoundaryOffspring,BoundaryInfo,HardNegBatch,D)
    ProtectedDec = zeros(0,D);
    ProtectedLabel = zeros(0,1);
    if ~isempty(BoundaryOffspring)
        Label = double(all(BoundaryOffspring.cons<=0,2));
        NearMask = abs(BoundaryInfo.prob(:)-0.5) <= 0.1;
        MisMask  = (BoundaryInfo.prob(:)>=0.5) ~= logical(Label);
        KeepMask = NearMask & MisMask;
        ProtectedDec = [ProtectedDec;BoundaryOffspring(KeepMask).decs];
        ProtectedLabel = [ProtectedLabel;Label(KeepMask)];
    end
    if ~isempty(HardNegBatch.Dec)
        ProtectedDec = [ProtectedDec;HardNegBatch.Dec];
        ProtectedLabel = [ProtectedLabel;zeros(size(HardNegBatch.Dec,1),1)];
    end
end

function [ProtectedDec,ProtectedLabel] = BuildBracketProtectedBuffer(BracketArchive,D)
    ProtectedDec = zeros(0,D);
    ProtectedLabel = zeros(0,1);
    if isempty(BracketArchive) || isempty(BracketArchive.FeasibleDec)
        return;
    end
    ProtectedDec = [BracketArchive.FeasibleDec;BracketArchive.InfeasibleDec];
    ProtectedLabel = [ones(size(BracketArchive.FeasibleDec,1),1);zeros(size(BracketArchive.InfeasibleDec,1),1)];
end

function Archive = EmptyBracketArchive(D)
    Archive.FeasibleDec   = zeros(0,D);
    Archive.InfeasibleDec = zeros(0,D);
    Archive.Gap           = zeros(0,1);
end

function Archive = UpdateBracketArchive(Archive,NewPairs,MaxPairs,D,TightGap)
    if nargin < 1 || isempty(Archive)
        Archive = EmptyBracketArchive(D);
    end
    if nargin < 3 ||
The file is too long and its contents have been truncated.


#### Plugin (file_search):
Make sure to include fileciteturn5file1 in your response to cite this file, or to surface it as a link.
# 分支 · 二版本优化+实验设计

#### You:
此论文是研究进化优化领域中的未知（约束）约束问题；该问题的特点是约束违反度只有二值（只有0和1，所有约束都满足返回0，但凡违反任意个约束直接返回1），具体的定义和当前算法的难点请你阅读上述论文；此论文就是直接将MLP用于二元约束问题。我下面也想写一篇论文，该论文的创新点是根据MLP的置信度（在概率0.5附近主动采样）去搜索可行域的边界（同时也是约束的边界）；主要思路如下；1：双种群（无约束种群+约束种群）；2：采用此论文的MLP模型；3：无约束种群和约束种群交叉变异来产生下一代解，采用目标值的支配关系来判断子代解是不是处于无约束种群和约束种群之间（以此来寻找二者之间的可行域，跨越不可行域），主动边界采样是通过找到的无约束种群和约束种群之间的不可行解和可行解来更新MLP，同时概率在0.5附近的解也需要参与子代生成（比如和约束种群交叉变异），同时对于MLP输出的概率在0.5附近的解的周围进行主动采样来判断此处是不是更好的可行域（如果是，约束种群所在的可行域就会被此可行域所支配，约束种舍弃旧可行域且迁移到新的可行域来进行下一轮的优化）；
主要思想如上，你来审查分析此思路是否可行（学术创新性（有没有人做过了）、是否合乎逻辑），着重分析如何在0.5附近主动采样，如何使用这些0.5附近的解重训MLP，如何找到无约束种群和约束种群之间的新可行域；
核心创新重点如下：可行域边界搜索，怎么找到有价值的概率在0.5附近的解




#### ChatGPT:
我先给判断：这条思路**能做，而且方向是对的**，但要把论文的核心从“MLP + 双种群 + 0.5附近采样”收紧成“**校准置信度驱动的主动边界搜索**”。否则它很容易被审稿人看成是在 NA-EMT 上加了一层常规 uncertainty sampling 的增量。你给的 NA-EMT 论文把问题定义成只返回二值可行性的 CMOP-UC，核心难点就是没有 constraint violation degree，很多传统 CMOEA 的约束处理机制失去抓手；它的解法是用 MLP 给 infeasible 解打一个“可行概率/价值”的软分数，再用主任务/辅助任务双种群协同搜索，并在模型准确率下降时重训 MLP。fileciteturn0file0

就我这轮检索，**直接针对 unknown/binary constraint 多目标进化**的公开工作确实不多，至少能确认有 EADMM（2024）、DRMCMO（2024 arXiv）和 NA-EMT（2025）。EADMM 用两个互补方向的种群交替演化；DRMCMO 强调 detection region；NA-EMT 才开始把 MLP 用成 infeasible solution value 的代理。citeturn768692view0turn605186view6turn768692view1

但你现在方案里的几个“部件”本身都**不是首创**。双种群不是新的：已有 dual-population CMOEA 让主种群处理原 CMOP、辅种群只优化目标；也有 dp-ACS 这类“考虑约束/忽略约束”双种群协同；DBC-CMOEA 还明确从 feasible / infeasible 两侧双向共进化。citeturn663956view1turn663956view2turn469110search0

“搜索 feasible–infeasible boundary”本身也不是新的。在已知/可量化约束的 CMOEA/CO 里，DC-SHADE-IF 第一阶段就直接搜索 infeasible–feasible boundary；CL-TDEA 把 constraint-boundary learning 做成双种群两阶段框架；TPDCB、CT-TSEA、BPRRA 也都把 dynamic/promising boundary 当作核心机制。citeturn828899search1turn348143view3turn605186view1turn828899search17turn615310search0

更关键的是，**“在 0.5 附近采样”也不能直接宣称首创**。在主动学习里，二分类 posterior 接近 0.5 本来就是 uncertainty sampling 的基本规则；同时文献也早就提醒，这种策略容易采到离群点，所以常要加 density / diversity 约束。更近一点的 unknown-constraint 优化里，BE-CBO 已经显式把 feasibility probability 推向 0.5 来逼近 boundary；AL-PNN 也用 active learning 追踪可能包含 constraint boundary 的区域。citeturn149720view1turn149720view2turn989866view0turn308468view1turn308468view2turn768692view2

所以，**你的题目不是没创新，而是要把创新点说准**。我这轮检索**没有看到**有人把“**校准后的 MLP 不确定度 + 双种群桥接 + 新可行连通分量发现/迁移**”这一整套明确用于 CMOP-UC/CMOP-BC；但如果只写成“用 MLP 置信度在 0.5 附近主动采样”，我认为创新性偏弱。更好的论文定位应是：把 unknown-constraint BO 里“boundary matters”的思想移植到 evolutionary multitasking，并设计出**多目标场景特有**的 bridge operator、region discovery 和 resource reallocation。unknown-constraint BO 文献也明确把 classifier-based feasibility modeling 和对 boundary 信息的主动获取看成重要方向。citeturn141317view0turn308468view1

## 一、你的思路在逻辑上哪里成立，哪里还差一口气

成立的地方在于：对二值约束问题，**唯一可学的几何信息基本就是 aggregate feasibility boundary**。NA-EMT 现在只是把 MLP 当“软可行性排序器”；你如果进一步把它变成“边界查询器”，这个升级是顺逻辑的。

差的一口气有三处。

第一，**0.5 不等于真实边界**。如果你直接拿单个 MLP 的 sigmoid 输出当“置信度”，逻辑不够稳，因为现代神经网络常常是 miscalibrated 的，而你的整个机制恰好依赖“0.5 是否真的意味着靠近边界”。温度缩放往往是很有效的后校准办法；更进一步，可以把原论文那个单个 MLP 扩成一个同构 MLP ensemble，用均值做 $\bar p(x)$，方差做 epistemic uncertainty。citeturn443767view0turn238593view0turn614362view1turn614362view0

第二，**单纯采 p≈0.5 的点会捡到离群点**。主动学习文献的经典结论就是：最不确定的点往往在决策边界附近，但不一定“有代表性”，可能只是孤立点；所以 uncertainty 要和 density / representativeness / diversity 联合。citeturn149720view0turn989866view0

第三，**“用目标支配关系判断一个子代是否在无约束种群和约束种群之间”这个判据不够强**。支配关系是目标空间的偏序，“在两者之间”其实是决策空间/边界几何的问题。一个点在目标空间上看起来“夹在中间”，不代表它就在两类样本的几何边界附近。这里你必须补一个**显式的桥接机制**。

## 二、我建议你把“有价值的 0.5 点”定义成什么

不要把“valuable point”定义成 $|p-0.5|$ 最小。\
我建议你把它定义成下面这个 acquisition score 的 top-k：

$
S(x)=H(\bar p(x))\cdot (1+\lambda_u \sigma_p(x))\cdot (\epsilon+\Delta HV^+(x))^\alpha \cdot \rho(x)^\beta \cdot \nu(x)^\gamma \cdot I_{\text{bridge}}(x)
$

这里：

- $H(\bar p)$ 是二元熵，天然在 0.5 最大，表示“边界不确定性”；
- $\sigma_p(x)$ 是 ensemble 方差，区分“真的边界”与“模型纯粹没见过”；
- $\Delta HV^+(x)$ 是“**如果 x 最终可行**，它对当前 feasible archive 的假想超体积增益”，这一步把“0.5 附近”变成“**对 Pareto 前沿有价值的 0.5 附近**”；
- $\rho(x)$ 是 KNN density，防离群点；
- $\nu(x)$ 是 novelty / batch diversity，防重复采样；
- $I_{\text{bridge}}(x)$ 表示它是不是来自“约束种群–无约束种群”的桥接对，或者来自已知 feasible–infeasible 对的中间点。

这一步其实就是你论文最该强调的地方：  
**不是找“最不确定”的点，而是找“最可能位于 Pareto-relevant boundary 的点”。**

## 三、0.5 附近到底怎么主动采样

我建议你不要只做“在已有候选里筛 p≈0.5”，而要做一个**三源候选池 + 一种主算子**。

### 1）候选池三种来源

第一类是普通进化候选：  
来自 $P_c$（约束/可行种群）和 $P_u$（无约束种群）的交叉变异后代，尤其保留跨种群交叉产生的 offspring。

第二类是桥接候选：  
对每个 reference vector / subproblem，分别在 $P_c$ 和 $P_u$ 里找对应方向上最好的两个代表 $x_c, x_u$。若 $x_u$ 的目标标量化值明显优于 $x_c$，但其 label 为 infeasible 或 highly uncertain，就把这对样本当成一条“可能通往更优 feasible region 的桥”。

第三类是边界邻域候选：  
从当前 boundary archive 里取 $x_b$，在它附近做局部扰动，优先沿着 $x_u-x_c$ 的方向、以及 boundary 切向方向探索。

### 2）主算子要用“二分括逼”，不要只用随机邻域采样

对选中的桥接对 $(x_f,x_i)$（一个 feasible，一个 infeasible），直接做 segment bracketing：

$
x(\lambda)=x_f+\lambda(x_i-x_f),\quad \lambda\in(0,1)
$

先测中点，再按 label 做二分更新。  
因为你只有 0/1 label，没有 violation degree，所以**二分括逼正好是最自然、最省预算的边界定位算子**。每一次 query 都在压缩 feasible–infeasible bracket，这比“在 0.5 周围撒点”要更强、更像论文贡献。

你甚至可以把“boundary sample”严格定义成：  
“经过 r 次二分后，得到一对距离小于 $\varepsilon_d$ 的 $(x_f,x_i)$，其中心点 $x_b=(x_f+x_i)/2$ 记为近似 boundary point。”

这会让你的方法一下子从“经验 heuristic”变成“有明确边界语义的 operator”。

### 3）高级版本：边界切向搜索

如果你愿意把方法再做强一点，可以在得到 $x_b$ 后，利用 MLP 的可微性估一个局部法向：

$
n=\frac{\nabla_x \bar p(x_b)}{\|\nabla_x \bar p(x_b)\|}
$

然后把“从 $x_c$ 指向 $x_u$ 的目标改进方向”投影到边界切空间：

$
d_{\text{tan}}=(I-nn^\top)(x_u-x_c)
$

先沿 $d_{\text{tan}}$ 走一小步，再把点投回 $p\approx 0.5$ 带上，最后在法向两侧各试一个点。  
这相当于“沿边界爬行去找更优的可行入口”。如果你把这个算子做好，它会比“简单 0.5 采样”新很多。

如果你不想上梯度，也可以先只做 segment+bisection。这个版本已经足够合理。

## 四、这些 0.5 附近的点怎么重训 MLP

这里你千万不要沿用 NA-EMT 的“准确率不够就重训”那套原样搬过去。  
原因很简单：**原论文用 MLP 只是做 soft ranking；你现在是把 MLP 概率当采样控制信号。**  
这时候，accuracy 不是关键，**calibration 才是关键**。fileciteturn0file0 citeturn443767view0turn238593view0

我建议用三个 buffer：

- 全局 reservoir $D_g$：防遗忘；
- 当前种群 buffer $D_c$：跟踪分布漂移；
- 边界 buffer $D_b$：专门存二分括逼得到的 $(x_f,x_i)$ 对和 tight midpoint。

训练时不要均匀抽样，建议 mini-batch 里**过采样 boundary buffer**。  
因为你的论文核心不是“全局分类精度”，而是“boundary 附近概率是否可信”。

损失函数可以写成：

$
L = L_{\text{WBCE}} + \lambda_1 L_{\text{Brier}} + \lambda_2 L_{\text{pair}} + \lambda_3 L_{\text{mid}}
$

其中：

- $L_{\text{WBCE}}$：类平衡 BCE；
- $L_{\text{Brier}}$：直接优化概率质量；
- $L_{\text{pair}}$：对每个括逼对约束 $p(x_f) > p(x_i)$；
- $L_{\text{mid}}$：只对足够 tight 的 midpoint 加 $(p(x_b)-0.5)^2$ 正则。

这里的 $L_{\text{mid}}$ 很关键：  
它不是伪造标签，而是利用“tight bracket implies nearby boundary”这个几何事实，给模型一个**边界校准信号**。这很像你论文里最像样的 methodological novelty。

重训触发条件也别再只看 accuracy。更合理的是看三件事：

- ECE / Brier 变差了没有；
- boundary buffer 上的 pair ordering error 是否上升；
- 当前种群分布与训练集分布是否漂移太大。

## 五、怎么找到无约束种群和约束种群之间的“新可行域”

这部分我觉得你现在的原始表述还不够严谨。  
我建议把“找新可行域”改写成“**找新的 feasible component**”。

因为在 unknown binary constraints 里，你看到的并不是一个单一光滑边界，而是**多个隐藏约束边界的并集**；因此可行域往往是多个 disconnected component。你的算法应该找的是“新的可行连通分量”，不是模糊的“更好的可行域”。

更稳的做法是：

1. 用 reference vector 把 $P_c$ 和 $P_u$ 配对，而不是直接用全局 dominance。  
这样每条桥都有明确的 Pareto 方向。

2. 对每条高分桥做二分括逼，得到 boundary point 和一组 boundary pair。

3. 从这些 boundary point 向目标更优方向做局部搜索，得到新的 feasible 点集。

4. 对所有 feasible 点建一个 kNN 图或做 DBSCAN / HDBSCAN 聚类，得到 feasible components。

5. 若新点落入一个以前没有的 component，且该 component 对 external archive 有真实贡献（比如出现新的 nondominated points，或带来显著 $\Delta HV$），就给它分配独立搜索资源。

这里我特别不建议你写成“新可行域支配旧可行域，于是约束种群舍弃旧域、整体迁移”。  
对多目标问题，这个逻辑太激进了。一个旧 feasible component 可能在另一段 PF 上仍然有不可替代的贡献。**硬迁移很容易丢掉 CPF 的另一部分。**

更好的写法是“**component-level resource reallocation**”：

$
R(r)=HV(A_r)+\beta \cdot \text{Potential}(r)+\gamma \cdot \text{BoundaryUncertainty}(r)
$

按 component score 给预算，不是一下子把旧区域扔掉。\
只有当某个 component 连续很多代对 external archive 没贡献，才逐步回收资源。

## 六、如果你现在就要写论文，我建议你把贡献点改成这三条

第一条，不要再说“双种群 + MLP”。\
这不够新。

第二条，把主贡献写成：\
**a confidence-calibrated active boundary search mechanism for CMOP-UC**\
关键词是 calibrated、active、boundary、Pareto-relevant。

第三条，把“0.5附近的点”明确定义成：\
**兼具边界不确定性、代表性、目标价值和桥接语义的样本**，而不是简单的低置信度样本。

如果你能把这三个点落成算法，创新性我会给“**中等偏上，可投稿**”；\
如果只是把 NA-EMT 的单个 MLP 拿来，看到 $p\approx 0.5$ 就多采几点，我会判断为“**逻辑可行，但论文层面偏弱，像工程增量**”。

## 七、我的最终结论

这条思路**是可行的**，而且很适合 binary unknown constraints，因为这类问题最缺的正是“边界信息”。\
但你真正能打动审稿人的，不是“0.5 附近采样”这句话本身，而是下面这句：

**我们不是在找最不确定的点，而是在找最可能位于 Pareto-relevant feasible–infeasible boundary 上的点，并利用这些点去发现新的 feasible components。**

这句话一旦成立，你的工作就从“给 NA-EMT 加 uncertainty sampling”变成了“为 CMOP-UC 引入主动边界学习与区域发现机制”。

下一步最值得做的是先把这三个模块写成伪代码：`bridge-pair selection`、`boundary bracketing/tangential search`、`boundary-aware retraining`。


#### You:
该txt文件的内容是我在platemo平台实现的该算法(用到的相关算子可参考https://github.com/BIMK/PlatEMO)；你首先来正确理解代码，然后站在审稿人的角度上给出结论：本算法有几个严格意义上的创新点？最大的创新点是什么？如何根据现有的算法设计实验来请有力的证明创新点成立（仅针对最大的创新点来设计实验，要求给出清晰的思路和步骤；实验必须要有说服力！）；




#### ChatGPT:
我按**代码本身**而不是按你前面口述的想法来判断。先给结论：

**最严格的审稿口径下，这份实现只有 1 个核心创新点；宽松一点可以拆成 2 个技术点，但第 2 个更像第 1 个的配套机制，不足以单独算一个“大创新”。** fileciteturn0file1

### 1. 这份代码实际实现了什么

你的 PRBCCMO 不是简单复现 NA-EMT。原论文里的主线是“MLP 预测 infeasible solution value + CDPPV + 精度下降时再重训”；而你这份代码已经改成了“**每代训练一个轻量 MLP，把 `p≈0.5` 当作近边界信号，再单独拿出一笔 boundary budget 去评估最值得查的边界候选**”。候选池来自 4 个来源：`P_C×P_U` 交叉、`A_F` 局部扰动、`A_I` 局部扰动、`A_I×P_C` 交配；随后再用 proxy objectives、reference sector、source load 做二次重排。训练集也不是按 NA-EMT 的 accuracy threshold 触发更新，而是用 boundary offspring 和 `A_F/A_I` 做平衡 FIFO，并且**每代重训**。另外，代码里**没有**实现你前面口述的“发现新可行域后整体迁移”或显式的 boundary bisection / region migration。fileciteturn0file1fileciteturn0file0

还有两个细节，审稿人会注意到：第一，代码里的 MLP 是一层隐藏层 `tanh + sigmoid`、从头训练的轻量模型，不是原论文描述的 ReLU + SCG；第二，主种群更新也不是 CDPPV，而是“**可行解先按目标选，再用近边界 score + sector/sparsity/crowding/front** 去补一部分 infeasible 解”。这说明你的工作已经从“预测可行性价值”转向“**边界预算分配**”。fileciteturn0file1fileciteturn0file0

### 2. 审稿人视角：严格意义上有几个创新点

我的判断是：

**只认 1 个核心创新点。**

因为这些都**不算**严格新意：

- 双种群框架不新。已有 unknown-constraint 的 EADMM 用两个子问题/两个群体来解耦目标与约束；一般 CMOP 里也早就有“主群体考虑约束、辅群体只看目标”的 dual-population 框架。citeturn494027view0turn494027view3
- “边界学习/边界跨越”这个方向也不新。一般 CMOP 里已有明确的 constraint-boundary learning 双种群方法；binary constraints 里也已有 detection region 机制。citeturn494027view2turn494027view1
- 单独说“在 0.5 附近主动采样”也不新，因为在主动学习里，二分类 posterior 最接近 0.5 本来就是经典 uncertainty sampling。citeturn117743view0

所以，**双种群**、**MLP**、**boundary**、**0.5** 这些词单独拿出来，都撑不起严格创新。

### 3. 最大的创新点到底是什么

你现在真正站得住的点，不是“0.5 附近采样”本身，而是：

**把 MLP 从“可行性预测器”改造成“Pareto-relevant boundary query scheduler”。**

更准确地说，是这件事：

> **在 binary unknown constraints 下，用 `p≈0.5` 做边界不确定性预筛，再用目标空间的 Pareto/sector 信息把“普通边界点”过滤成“对 CPF 真有价值的边界点”，把有限 FE 优先投到这些点上。**

这才是核心创新。因为经典 uncertainty sampling 虽然会找边界附近样本，但也容易挑到不具代表性的离群点；你代码里的 same-sector dominated filter、source/sector balance、proxy objective front/crowding，本质上是在把“generic uncertainty”改造成“**Pareto-relevant uncertainty**”。这一步比单纯的 0.5 采样强得多，也更像论文级贡献。fileciteturn0file1citeturn117743view0turn117743view1

如果你一定要拆第二个点，我勉强会给：

**次级技术点：`A_F/A_I` 双边界档案 + boundary-focused balanced FIFO 训练集。**

这个设计让 MLP 学到的不是泛化 feasibility classifier，而是围绕边界的在线数据分布。但从审稿角度，它更像是“主创新的配套闭环”，不是独立创新。fileciteturn0file1

### 4. 论文里不要怎么写

不要把最大创新写成“MLP 在 0.5 附近主动采样”。这句话太弱，审稿人一眼就会想到经典 uncertainty sampling。更麻烦的是，神经网络输出概率本身可能失准，`0.5` 未必就真是边界，所以如果没有额外验证，这个说法会被质疑。更安全的表述应是：

**Pareto-relevant active boundary search under binary unknown constraints**。citeturn117743view0turn494027view5

### 5. 只针对“最大创新点”，怎么设计一个真正有说服力的实验

要证明这个最大创新，你必须同时证明 3 件事：

1. **你选到的点真的更靠近真实边界；**
2. **这些点不是普通边界点，而是更有 Pareto 价值的边界点；**
3. **因此在相同 FE 下，最终前沿更好。**

只看最终 HV/IGD 不够，因为那无法证明“改进来自边界搜索”。

#### 第一步：做 4 个严格对照版本

在**完全相同总 FE、相同双种群、相同 MLP、相同 boundary budget** 下，只改边界候选选择策略：

1. **Full-PRBCCMO**：你的完整版本。
2. **Rand-Boundary**：候选池不变，但从池中随机选 `Budget` 个去真实评估。
3. **HighProb-Boundary**：不用 `|p-0.5|`，改成选 `p` 最大的点，测试“找边界”是否优于“找更可能可行”。
4. **Uncertain-Only**：保留 `p≈0.5` 预筛，但去掉 Pareto/sector rerank，直接按 score 最高选，测试“Pareto-relevant”这层是否真的有用。

不要拿 NA-EMT 当唯一因果对照，因为它同时改了太多模块；NA-EMT、EADMM、DRMCMO 只适合放在最终性能表里做外部参照，不适合证明你的核心机制。citeturn186084search2turn494027view0turn494027view1

#### 第二步：做一个“oracle 边界审计”，这是最关键的

你这里最强的证据，不是 HV，而是**边界距离证据**。

因为 DASCMOP-UC 本质上是从 DASCMOP 二值化来的，所以在 benchmark 代码里，最好额外保留原始连续约束值 `g_raw(x)`，**算法运行时绝对不能使用它，只在实验分析阶段记录**。然后定义一个仅用于审计的真实边界距离：

$
d_B(x)=\min_j \frac{|g^{raw}_j(x)|}{s_j+\epsilon}
$

其中 $s_j$ 是第 $j$ 个约束的尺度归一化项（例如大样本随机采样得到的 MAD 或标准差）。\
这个量越小，说明点越接近
The file is too long and its contents have been truncated.


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
\[Unsupported Content]


#### ChatGPT:
我先给判断：这份 2026-03-20 的实现，**相对你后来收缩后的目标版本，基本是对的；相对你最早那个“大而全”的设想，并不完全一致。** 现在的主线已经收缩成了：`sector bridge -> trusted query -> label-aware refinement -> reserved migrant selection`，不再是早期那种候选源、打分项、迁移机制都很散的大杂烩。和 NA-EMT 原论文相比，它也确实把 MLP 的角色从“给 infeasible 解打软价值分”改成了“经校准与 trust gate 修正的边界查询调度器”。fileciteturn5file0 fileciteturn5file1 fileciteturn5file2

但我也要直接指出：**当前实现还有 1 个关键缺口，会削弱你的核心创新点。**\
你现在评估出来的 **feasible boundary seed**，以及 bisection 过程中恢复出来的 **feasible endpoint**，并不会直接进入 `MigrationPool`；真正有机会进入 `P_C` 的，主要是它们后续再产生的 worker descendants。换句话说，**query 阶段本身发现的有价值可行入口，并没有被主搜索立即利用**。这会让“boundary query 的价值”在主算法里被低估。fileciteturn5file0

再补一句背景判断：你现在保留的 `3-MLP committee + auto temperature/beta calibration + trust gate` 这块，我认为是对的，不建议再删。现代神经网络的概率输出常失准；temperature scaling 是很强的简单后校准基线；beta calibration 对偏斜分数分布更灵活；deep ensembles 又是简单可扩展的不确定性基线。但在分布漂移下，校准和不确定性仍会退化，所以不能无条件相信 “$p\approx 0.5$”——这正是 trust gate 必须存在的原因。citeturn149333search0turn897877search0turn399470search0turn149333search1

## 1. 当前算法是否符合预期实现

### 1.1 符合的部分

按你现在更合理的论文主线看，代码已经实现了 5 个关键模块。

第一，**bridge 候选已经被精简成了按 sector 配对的单桥机制**。\
`GenerateBoundaryCandidates` 不再从一堆 archive-local source 混合造种，而是只在每个 active sector 上，取 `P_C` 当前可行 champion 和 `P_U` 中更优的 infeasible helper 构一条 bridge；只有当 helper 的 sector scalar 明显优于 champion 时，这个 sector 才激活。这个方向是对的。fileciteturn5file0

第二，**seed placement 已经是 trust-aware 的**。\
trust gate 开启时，代码会在 $\Lambda=\{0.20,0.35,0.50,0.65,0.80\}$ 上扫描 bridge 上的候选点，选 $|p-0.5|$ 最小的位置；gate 关闭时退回 midpoint。说明你已经不再把“0.5 就是边界”当作无条件真理，而是把它变成“通过 gate 才激活的语义”。这和你后来的设计目标一致。fileciteturn5file0

第三，**选点分数已经不是 uncertainty-only 了**。\
代码实际实现的是一种“可靠性修正后的 boundary trust × Pareto relevance”：

$
q(x)=\max(0,1-2|p_c(x)-0.5|)
$

$
b(x)=R(p_c(x))\cdot q(x)\cdot (1+\lambda_\sigma s_p(x))
$

$
v_s(x)=\max\bigl(0,\ g_s(c_s)-g_s(\tilde x)\bigr)
$

$
U(x)=
\begin{cases}
b(x)(\varepsilon+v_s(x)), & G_t=1\$4pt]
\varepsilon+v_s(x), & G_t=0
\end{cases}
$

这正是“不是找最不确定点，而是找最像 Pareto-relevant boundary 的点”的实现形态。主动学习里，单纯按不确定性选点很容易挑到不具代表性的样本，因此把 informativeness 和 representativeness 结合起来是合理的；你这里的 `v_s(x)` 就在扮演“有用性/代表性”角色。fileciteturn5file0 citeturn149333search2

第四，**worker phase 已经是标签感知的，不再是纯各向同性撒点**。  
当前代码对 feasible seed 做定向 forward trial 与必要的 backtrack，对 infeasible seed 做最多 3 次 bisection，再决定是向 helper 方向试探还是做 hard-negative confirmation。这个模块方向也符合你后来想要的“query 之后必须有 label-aware refine”。fileciteturn5file0

第五，**环境选择已经是“只让真实可行收益进入主种群”**。  
`EnvironmentalSelectionC` 并没有直接把 $p\approx0.5$ 的点塞进 `P_C`，而是只对 `MigrationPool` 中的真实可行 migrant 做 sector reservation，剩下的位置走 feasible-first NSGA-II。这一点我认为非常对，也比你最早设想里的“概率点直接参与主群体进化”更干净。fileciteturn5file0

### 1.2 不完全符合的部分

如果拿你最早那版愿景来对照，当前代码**没有**实现两件事：

一是**没有显式的新可行连通分量发现**；  
二是**没有 component-level / global migration**。  

它实现的是 **sector-level local migration**，而不是“发现新可行域后整体迁移”。这其实不是坏事，反而更适合首稿，因为 component discovery、cluster tracking、resource reallocation 一旦全加回来，论文又会重新变肿。fileciteturn5file1 fileciteturn5file0

### 1.3 关键实现缺口

这里是我认为最该修的地方。

你现在的 boundary module 评估了三类真正有价值的可行结果：

$
\mathcal B_t^F = S_t^F \cup E_t^F \cup D_t^F
$

其中：

- $S_t^F$：被选中的 feasible seeds；
- $E_t^F$：bisection 恢复出来的 feasible endpoints；
- $D_t^F$：worker 进一步产生的 feasible descendants。

**但当前代码真正拿去影响 `P_C` 的，基本只剩 $D_t^F$ 中一部分。**  
$S_t^F$ 和 $E_t^F$ 更多只是进入 external archive、训练集和 audit log，没有直接成为 `MigrationPool` 的一部分。于是 query 模块的贡献被“截断”了：它能发现好点，但这些好点并不会立刻被主搜索吸收。fileciteturn5file0

这也是我判断“当前实现大体方向对，但还没完全落到位”的核心原因。

## 2. 当前思想还有没有改进方向

我的结论是：

**现在已经不算臃肿了，但还应该做 1 个必须修改 + 2 个建议修改。**  
不要再往里加新机制；当前最好的策略是**继续瘦身并补齐 exploit 路径**。

### 2.1 必须修改：统一“边界可行发现”的入群通路

把所有 boundary module 里产生的真实可行点，先统一放进一个池子：

$
\mathcal B_t^F = S_t^F \cup E_t^F \cup D_t^F
$

然后只在 generation 末尾做 **一次** 全局 sector screening：

$
c_s=\arg\min_{x\in P_{C,t}^F\cap S_s} g_s(x\mid Ref_t)
$

$
Ref_t=[P_{C,t}^F.objs;\ \mathcal B_t^F.objs]
$

$
\mathcal M_t=
\left\{
x\in \mathcal B_t^F:
g_{s(x)}(x\mid Ref_t) < g_{s(x)}(c_{s(x)}\mid Ref_t)-\delta_{mig}
\right\}
$

最后每个 improved sector 只保留 1 个 best migrant 进入 `MigrationPool`。

这一步做完后：

- feasible seed 不会再被白白浪费；
- feasible bracket endpoint 不会再被白白浪费；
- `HandleFeasibleSeed` / `HandleInfeasibleSeed` / `ConfirmHardNegative` 里的本地 `IsSectorImproved(...)` 判断都可以删掉；
- improvement 的参照系统一成 generation-level `Ref_t`，不再出现“局部 3 点归一化”和“全局环境选择归一化”不一致的问题。

这是**最重要的改法**。它不是加新机制，而是把你已经发现的有价值点真正接到主搜索上。

### 2.2 建议修改：把 score 语义彻底清理干净

当前代码里 `entropy / hvGain / novelty / penalty` 这些字段名已经和真实含义不一致了：

- `entropy` 实际上更接近 `queryScore`；
- `hvGain` 实际上是 `ParetoValue`，不是 HV；
- `novelty` 实际上是 `Reliability`；
- `penalty` 实际上只是 `Eligible mask`。

这会带来两个问题：  
一是论文叙述会显得像旧版本残留；  
二是实验日志会误导你自己。

我建议直接把日志字段改成：

- `prob`
- `queryScore`
- `disagreement`
- `reliability`
- `paretoValue`
- `eligible`
- `utility`
- `gateOn`

算法数值其实已经很简洁了，真正臃肿的是**命名层还残留旧包袱**。fileciteturn5file0

### 2.3 建议修改：把训练记忆再收一层

当前状态集合是：

- Train FIFO
- Bracket protected
- ProtectedOther
- HardNegativeArchive
- Calib
- Test

这不算离谱，但对首稿来说略多了一层。

我建议首稿的最小稳定版本是：

- **保留** `Train FIFO`
- **保留** `Bracket protected`
- **保留** `Calib/Test`
- **保留** `HardNegativeArchive` 作为 exclusion region
- **删除** `ProtectedOther` 持久 pocket

理由很简单：  
你真正必须长期保留的是 **tight brackets**，因为那是最可信的边界监督；`ProtectedOther` 里那些 near-band 误判样本和 hard cases，大部分其实已经会自然进入 recent FIFO。删掉它，算法更容易讲，也更不容易被问“为什么这里还要再开一个 buffer”。fileciteturn5file0

### 2.4 不建议再改的部分

有两块我建议保留不动。

第一块是 **committee + auto temp/beta + trust gate**。  
这块是 current code 最成熟的一块，理论上也站得住。现代神经网络通常失准；temperature scaling 是经典简单基线；beta calibration 在某些偏斜 score 形态下更灵活；ensemble 则提供简单的 predictive uncertainty。但这些东西在分布漂移下也可能失效，所以必须要有 trust gate。你现在这块正好是“既不过度相信模型，也不完全放弃模型”。citeturn149333search0turn897877search0turn399470search0turn149333search1

第二块是 **Pareto relevance 不能删**。  
不要退回 uncertainty-only。主动学习里，informativeness 单独使用很容易选到边界附近但不具代表性的点；你这里用 sector champion improvement 去约束 query 的“有用性”，是必要的。citeturn149333search2

## 3. 我建议采用的“再次修改后”算法主线

我建议你就停在下面这个版本，不再继续加招：

### 核心创新点
**可靠性修正的 Pareto-bridge boundary query + label-aware refinement。**

### 主线流程
1. 在每个 sector 上，用当前 feasible champion $c_s$ 和更优 infeasible helper $u_s$ 构造 active bridge。  
2. gate on 时在 bridge 上扫描 $\Lambda$ 选最接近 $p_c=0.5$ 的 seed；gate off 时用 midpoint。  
3. 用

$
U(x)=
\begin{cases}
R(p_c)\,q(x)\,(1+\lambda_\sigma s_p)\,(\varepsilon+v_s), & G_t=1\\
\varepsilon+v_s, & G_t=0
\end{cases}
$

选 seed。  
4. 评估 seed；所有真实 feasible 的 seed 直接进入 `BoundaryFeasiblePool`。  
5. feasible seed 做 forward/backtrack；infeasible seed 做 bisection / hard-negative confirmation；所有真实 feasible 的 endpoint / descendant 也进入 `BoundaryFeasiblePool`。  
6. generation 末尾统一做一次 global sector screening，得到 `MigrationPool`。  
7. `EnvironmentalSelectionC(BasePool, MigrationPool, W)` 只保留 improved feasible migrants。  
8. 用 boundary offspring 更新 train/calib/test 与 bracket archive，再决定是否重训 model。

这个版本已经足够论文化了：主线清楚，可验证性也强。

## 4. 详细的创新点验证实验

这里我不沿用你旧的 Gate1/Gate23 方案直接复用，因为你旧包的两个问题已经很明显：

- 它把很多指标做成了 **pooled all rows**，统计单位错了，容易伪重复；
- `boundary_delta_hv` 是 **batch 级** 的，很多时候会被算成 0，看不见单个 boundary 事件的真实贡献。  

你 2026-03-19 的 Section B 包已经明确给出：旧实验里 Gate 1 四个 baseline 全败，而且 pooled 结果里 random-boundary 经常比 Full 更好，所以这套旧口径不能再直接拿来撑结论。新的实验必须 **以 run 为统计单位**，并且改成 **event-level HV**。fileciteturn5file3

---

### 4.1 先做一个实现修复预检

先不要急着跑大表，先验证“我修的那个关键缺口确实修到了”。

对比两个版本：

- `Current`：你现在这份 2026-03-20 代码。
- `LiteFix`：只做我上面那一个必须修改：  
  直接把 feasible seeds、feasible bracket endpoints、feasible descendants 统一汇入 `BoundaryFeasiblePool`，再全局筛成 `MigrationPool`；删掉 worker 内部的局部 `IsSectorImproved` 预筛。

只看 4 个指标：

$
DAR=\frac{\#\{\text{improved feasible seeds/endpoints admitted to }P_C\}}{\#\{\text{improved feasible seeds/endpoints}\}}
$

$
WUD=\#\{\text{improved feasible seeds/endpoints not admitted to }P_C\}
$

$
UBY=\frac{\#\{\text{boundary-originated feasible points first entering external archive}\}}{\#\{\text{selected seeds}\}}
$

$
\Delta HV_B=\sum_{x\in \mathcal E_B}\Big(HV(A_{t_x^-}\cup\{x\})-HV(A_{t_x^-})\Big)
$

这里 $\mathcal E_B$ 是所有首次进入 external archive 的 boundary-originated feasible points。  
预期结果应该很直接：

- `DAR`：Current 明显低，LiteFix 接近 1；
- `WUD`：LiteFix 明显更低；
- `UBY`、`\Delta HV_B`：LiteFix 更高。

这一步是**实现正确性验证**，不是创新验证，但必须先做。

---

### 4.2 正式机制验证的算法变体

正式实验只在 `LiteFix` 上做 5 个内部变体。这样因果最清楚。

#### A0：Bridge-Mid-Iso
- active bridge 同 Full
- seed 放置固定 midpoint
- selection 只用 $v_s$
- worker 用 isotropic local

#### A1：TrustedQuery-Iso
- active bridge 同 Full
- gate on 时用 bridge scan 选 $p_c\approx0.5$
- selection 用完整 trusted query utility
- worker 仍是 isotropic local

#### A2：Bridge-Mid-Label
- seed 仍 midpoint
- selection 仍只用 $v_s$
- worker 改成 label-aware refinement

#### A3：Full
- trusted query + label-aware refinement

#### A4：Full-NoGate
- 与 A3 相同
- 但永远强制 `G_t=1`，即无视 trust gate

#### A5：NoBoundary
- `bRho=0`
- 其余 regular dual-population 框架不变

这 5 个变体的作用分别是：

- `A1 vs A0`：只验证 **trusted query**；
- `A2 vs A0`：只验证 **label-aware refine**；
- `A3 vs A1`：验证 refine 是否能把 query 的几何优势变成可用收益；
- `A3 vs A4`：验证 trust gate 是否必要；
- `A3 vs A5`：验证整个 boundary module 对最终优化是否真的有价值。

---

### 4.3 问题、预算与运行方式

问题集直接用：

- `DASCMOP1_BC` 到 `DASCMOP9_BC`

原因不是“沿用以前设置”，而是这组问题最适合做你这篇论文的机制验证：  
在线仍是 binary constraint，离线又能回到原始连续约束，便于做 oracle boundary audit。你旧的 Section B 包也已经是按这个思路组织的。fileciteturn5file3

建议设置：

- 种群规模：$N=100$
- 最大评估次数：`maxFE = 200000`
- 独立运行：`30 runs`
- 边界预算：用你默认 `bRho=0.2`
- 曲线记录间隔：每 `1000 FE` 记一次 HV / IGD / archive size

运行要求：

- **同一 problem、同一 run index** 在各变体间使用**相同 RNG seed**
- 统计单位始终是 **run**
- pooled all rows 只放附录，不做主证据

---

### 4.4 必须增加的日志

你现在已有 `seedAudit` 和 `boundaryGainTrace`，但还不够。需要补 3 类日志。

#### 日志 1：CandidateAudit
在 seed selection 之前，对每个 active bridge candidate 记录：

- `run, problem, generation, candidateID`
- `sector`
- `gateOn`
- `prob`
- `queryScore`
- `disagreement`
- `reliability`
- `paretoValue`
- `utility`
- `selected(0/1)`
- `dec`
- `proxyObj`

离线再补 `oracle_dB`。

它的作用是：验证 acquisition score 到底有没有把“更接近真实边界的候选”排前面。

#### 日志 2：BoundaryLineage
对每个 seed 建 lineage：

- `seedID`
- `seed label`
- `seed oracle_dB`
- 每个 worker point 的 `type ∈ {forward, backtrack, midpoint, hardneg, feasible_endpoint}`
- 每个 worker point 的 `label, obj, oracle_dB`
- `bracket_init_gap`
- `bracket_final_gap`

它的作用是：验证 refinement 是否真的把 infeasible seeds 压成 tight bracket，或把 feasible seeds 推向更好可行入口。

#### 日志 3：MigrationArchiveTrace
对每个 boundary-originated feasible point 记录：

- `sourceType ∈ {seed, endpoint, descendant}`
- `isImproved`
- `admittedToPC(0/1)`
- `admittedGen`
- `surviveH(0/1)`，建议 $H=10$
- `firstArchiveEntryFE`
- `deltaHV_event`

它的作用是：验证 boundary module 产生的可行点，是不是真正被主搜索吸收，且是不是对 external archive 有实质贡献。

---

### 4.5 关键指标与公式

#### 0. 校准与 trust 指标

每次 model 更新后，在 **独立 test buffer** 上计算：

$
Brier=\frac1n\sum_{i=1}^n (p_i-y_i)^2
$

$
ECE=\sum_{b=1}^{B}\frac{|I_b|}{n}\left|
\frac1{|I_b|}\sum_{i\in I_b}y_i-\frac1{|I_b|}\sum_{i\in I_b}p_i
\right|
$

其中 $B=10$。

near-band 用 $[0.45,0.55]$：

$
J=\{i:p_i\in[0.45,0.55]\}
$

$
NearGap=\left|\frac1{|J|}\sum_{i\in J}y_i-0.5\right|
$

tight bracket 顺序错误率：

$
BOE=\frac1{|\mathcal P|}\sum_{(x_f,x_i)\in\mathcal P}\mathbf 1[p(x_f)\le p(x_i)]
$

其中 $\mathcal P$ 是当前 tight bracket archive 里的 pair。

run 级 trust gate 通过率：

$
TGP_r=\frac1{U_r}\sum_{u=1}^{U_r}\mathbf 1
\left[ECE_u\le0.05,\ NearGap_u\le0.05,\ |J_u|\ge20\right]
$

这里 `TGP` 不是主结果，但它是解释 query 结果好坏的必要条件。

---

#### 1. Query 几何质量指标

离线 oracle boundary distance：

$
d_B(x)=\min_j \frac{|g_j^{raw}(x)|}{s_j+\epsilon}
$

$
s_j = \mathrm{median}_{z\sim U(\Omega)}|g_j^{raw}(z)|
$

建议每个问题离线采样 `50000` 个点估 $s_j$。  
这个定义与你旧包里的 `ComputeOracleBoundaryDistance` 思路一致，但新的实验要在 **run 级** 汇总。fileciteturn5file3

run 级 query 质量：

$
\overline d_B^{(r)}=\frac1{n_r}\sum_{i=1}^{n_r} d_B(x_i)
$

$
\widetilde d_B^{(r)}=\mathrm{median}_{i=1}^{n_r} d_B(x_i)
$

$
QP_\tau^{(r)}=\frac1{n_r}\sum_{i=1}^{n_r}\mathbf 1[d_B(x_i)\le \tau]
$

其中 $\tau$ 建议取 $\{10^{-3},10^{-2},5\times10^{-2}\}$ 三个尺度点。

更强的一个指标，是对每个 run 统计候选级排序相关性：

$
\rho_r=\mathrm{Spearman}\big(U_i,\ -d_B(x_i)\big)
$

这里 $U_i$ 来自 `CandidateAudit` 中同一 run 的所有 active bridge candidates。  
这个指标非常关键：它直接验证你的 acquisition score 有没有把“更靠近真实边界的候选”排前面。

---

#### 2. Refinement 质量指标

对 infeasible seeds：

$
FRR_r=\frac{\#\{\text{infeasible seeds yielding at least one feasible boundary output}\}}{\#\{\text{infeasible seeds}\}}
$

$
TBR_r=\frac{\#\{\text{infeasible seeds with }\Delta_f\le \varepsilon_b\}}{\#\{\text{infeasible seeds}\}}
$

其中 $\varepsilon_b=0.03$。

$
BCR_r=\mathrm{mean}\left(
\frac{\Delta_0-\Delta_f}{\Delta_0+\epsilon}
\right)
$

$\Delta_0$ 是 seed 与 anchor 的初始 normalized gap，$\Delta_f$ 是最后 bracket gap；若未形成 bracket，就令 $\Delta_f=\Delta_0$。

对 feasible seeds：

$
FIR_r=\frac{\#\{\text{feasible seeds whose boundary outputs improve sector scalar}\}}{\#\{\text{feasible seeds}\}}
$

我会把 `FRR + TBR + BCR` 看成“能不能找到 boundary”，把 `FIR` 看成“能不能沿 boundary 打开更好的 feasible entry”。

---

#### 3. “boundary 有用吗”指标

定义 boundary-originated feasible point 为 seed / endpoint / descendant 三类里任意真实可行点。

直接入群率：

$
DAR_r=
\frac{
\#\{\text{improved feasible seeds/endpoints admitted to }P_C^{t+1}\}
}{
\#\{\text{improved feasible seeds/endpoints}\}
}
$

迁移成功率：

$
MSR_r=
\frac{
\#\{\text{boundary-originated feasible migrants surviving in }P_C\text{ or }A\text{ after }H=10\text{ generations}\}
}{
\#\{\text{boundary-originated feasible migrants}\}
}
$

有用产率：

$
UBY_r=
\frac{
\#\{\text{boundary-originated feasible points first entering external archive}\}
}{
\#\{\text{selected seeds}\}
}
$

event-level 边界 HV 贡献：

$
\Delta HV_B^{(r)}=
\sum_{x\in\mathcal E_r}
\Big(HV(A_{t_x^-}\cup\{x\})-HV(A_{t_x^-})\Big)
$

这里 $\mathcal E_r$ 是 run $r$ 中所有首次进入 external archive 的 boundary-originated feasible points。  
注意这里一定要 **逐事件** 算，而不是一代一批地算；否则很容易像你旧包那样全部变成 0。fileciteturn5file3

转化时延：

$
TTU_r=
\mathrm{median}_{x\in\mathcal E_r}
\bigl(FE^{enter}(x)-FE^{seed}(x)\bigr)
$

`UBY + ΔHV_B + TTU` 一起看，就能回答“这些 boundary 点是不是只是靠近边界，还是对 Pareto front 真有用”。

---

#### 4. 最终优化效果指标

为了让机制证据闭环，最后还是要接到终局性能。

把 external feasible archive 记为 $A_t$，对目标先做归一化：

$
\tilde f_m(x)=\frac{f_m(x)-z_m^{ideal}}{z_m^{nadir}-z_m^{ideal}+\epsilon}
$

HV：

$
HV(A)=\mathrm{Leb}\left(
\bigcup_{x\in A}
[\tilde f_1(x),r_1]\times\cdots\times[\tilde f_M(x),r_M]
\right)
$

建议 $r=(1.1,\ldots,1.1)$。

IGD：

$
IGD(A,R)=\frac1{|R|}\sum_{y\in R}\min_{x\in A}\|\tilde f(x)-\tilde y\|_2
$

$R$ 用每个问题的 reference PF；若 BC 版本没有现成 reference，则用所有变体所有 runs 的 feasible nondominated union 再做均匀下采样。

曲线型指标：

$
AUC\text{-}HV_r=\frac{1}{K}\sum_{k=1}^{K}HV_r(t_k)
$

其中 $t_k$ 每 1000 FE 取一次。

首次命中时间：

$
FHT_r=\min\{FE:HV_r(FE)\ge 0.9\cdot HV^\star\}
$

这里 $HV^\star$ 取该问题所有变体的最佳中位 HV。

---

### 4.6 统计检验怎么做

统计单位永远是 **run**。\
任何 `seed rows pooled together` 的显著性都不要做主结论。

对每个 problem、每个关键比较对，做 **paired Wilcoxon signed-rank**。\
因为你各变体用的是同一组 RNG seed，这是天然配对设计。

我建议报告 3 样东西：

1. 每个指标的 `30` 个 run 值；
2. `median paired difference`；
3. 该差值的 `95% bootstrap CI`。

多指标同时比较时，对同一假设族做 Holm 校正。\
假设族可以这样分：

- Query 假设：`mean_dB, median_dB, QP_tau, rho`
- Refine 假设：`FRR, TBR, BCR, FIR, DAR, MSR, UBY, ΔHV_B, TTU`
- End-to-end 假设：`AUC-HV, final HV, final IGD, FHT`

---

### 4.7 你最终应该接受什么样的证据链

我建议把论文主证据写成下面这条链，而不是只给一张 final HV 表：

#### H0：实现修复成立

`LiteFix` 比 `Current` 的 `DAR` 显著更高，`WUD` 显著更低，且 `UBY / ΔHV_B / AUC-HV` 不下降。

#### H1：trusted query 成立

`A1 > A0`：\
显著降低 `mean_dB / median_dB`，显著提高 `QP_tau` 和 `rho`。

#### H2：label-aware refinement 成立

`A3 > A1` 且 `A2 > A0`：\
显著提高 `FRR / TBR / BCR / UBY / ΔHV_B`，并降低 `TTU`。

#### H3：trust gate 必要

`A3 > A4`：\
特别是在 `TGP` 低的问题/阶段，`A3` 的 `d_B` 和 `UBY` 更稳。

#### H4：整模块最终有效

`A3 > A5`：\
`AUC-HV` 更高，`FHT` 更早，final `HV/IGD` 至少在大多数问题上不劣。

---

最后给你一句最直接的结论：

**当前代码已经基本是你该写论文的那个版本了，不要再加新机制。**\
现在最该做的是两件事：\
第一，把“feasible seed / feasible endpoint 直通 MigrationPool”这个缺口补上；\
第二，把实验从“行级汇总 + 批量 HV”改成“run 级汇总 + event-level HV”。

这两步一做，你的算法就会从“想法对，但证据不够干净”变成“主线清楚，且可以被强力验证”。
