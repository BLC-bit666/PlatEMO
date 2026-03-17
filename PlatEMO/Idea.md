# 分支 · 论文创新点分析

#### You:
此论文是研究进化优化领域中的未知（约束）约束问题；该问题的特点是约束违反度只有二值（只有0和1，所有约束都满足返回0，但凡违反任意个约束直接返回1），具体的定义和当前算法的难点请你阅读上述论文；此论文就是直接将MLP用于二元约束问题。我下面也想写一篇论文，该论文的创新点是根据MLP的置信度（在概率0.5附近主动采样）去搜索可行域的边界（同时也是约束的边界）；主要思路如下；1：双种群（无约束种群+约束种群）；2：采用此论文的MLP模型；3：无约束种群和约束种群交叉变异来产生下一代解，采用目标值的支配关系来判断子代解是不是处于无约束种群和约束种群之间（以此来寻找二者之间的可行域，跨越不可行域），主动边界采样是通过找到的无约束种群和约束种群之间的不可行解和可行解来更新MLP，同时概率在0.5附近的解也需要参与子代生成（比如和约束种群交叉变异），同时对于MLP输出的概率在0.5附近的解的周围进行主动采样来判断此处是不是更好的可行域（如果是，约束种群所在的可行域就会被此可行域所支配，约束种舍弃旧可行域且迁移到新的可行域来进行下一轮的优化）；
主要思想如上，你来审查分析此思路是否可行（学术创新性（有没有人做过了）、是否合乎逻辑），着重分析如何在0.5附近主动采样，如何使用这些0.5附近的解重训MLP，如何找到无约束种群和约束种群之间的新可行域；
核心创新重点如下：可行域边界搜索，怎么找到有价值的概率在0.5附近的解

#### ChatGPT:
我先给判断：这条思路**能做，而且方向是对的**，但要把论文的核心从“MLP + 双种群 + 0.5附近采样”收紧成“**校准置信度驱动的主动边界搜索**”。否则它很容易被审稿人看成是在现有不确定性采样框架上做常规增量。

就我这轮检索，**直接针对 unknown/binary constraint 多目标进化**的公开工作确实不多，至少能确认有 EADMM（2024）和 DRMCMO（2024 arXiv）。EADMM 用两个互补方向的种群交替演化；DRMCMO 强调 detection region。citeturn768692view0turn605186view6

但你现在方案里的几个“部件”本身都**不是首创**。双种群不是新的：已有 dual-population CMOEA 让主种群处理原 CMOP、辅种群只优化目标；也有 dp-ACS 这类“考虑约束/忽略约束”双种群协同；DBC-CMOEA 还明确从 feasible / infeasible 两侧双向共进化。citeturn663956view1turn663956view2turn469110search0

“搜索 feasible–infeasible boundary”本身也不是新的。在已知/可量化约束的 CMOEA/CO 里，DC-SHADE-IF 第一阶段就直接搜索 infeasible–feasible boundary；CL-TDEA 把 constraint-boundary learning 做成双种群两阶段框架；TPDCB、CT-TSEA、BPRRA 也都把 dynamic/promising boundary 当作核心机制。citeturn828899search1turn348143view3turn605186view1turn828899search17turn615310search0

更关键的是，**“在 0.5 附近采样”也不能直接宣称首创**。在主动学习里，二分类 posterior 接近 0.5 本来就是 uncertainty sampling 的基本规则；同时文献也早就提醒，这种策略容易采到离群点，所以常要加 density / diversity 约束。更近一点的 unknown-constraint 优化里，BE-CBO 已经显式把 feasibility probability 推向 0.5 来逼近 boundary；AL-PNN 也用 active learning 追踪可能包含 constraint boundary 的区域。citeturn149720view1turn149720view2turn989866view0turn308468view1turn308468view2turn768692view2

所以，**你的题目不是没创新，而是要把创新点说准**。我这轮检索**没有看到**有人把“**校准后的 MLP 不确定度 + 双种群桥接 + 新可行连通分量发现/迁移**”这一整套明确用于 CMOP-UC/CMOP-BC；但如果只写成“用 MLP 置信度在 0.5 附近主动采样”，我认为创新性偏弱。更好的论文定位应是：把 unknown-constraint BO 里“boundary matters”的思想移植到 evolutionary multitasking，并设计出**多目标场景特有**的 bridge operator、region discovery 和 resource reallocation。unknown-constraint BO 文献也明确把 classifier-based feasibility modeling 和对 boundary 信息的主动获取看成重要方向。citeturn141317view0turn308468view1

## 一、你的思路在逻辑上哪里成立，哪里还差一口气

成立的地方在于：对二值约束问题，**唯一可学的几何信息基本就是 aggregate feasibility boundary**。如果你把 MLP 从“软可行性排序器”进一步变成“边界查询器”，这个升级是顺逻辑的。

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

这里不要把“准确率不够就重训”的旧策略原样搬过去。  
原因很简单：**原论文用 MLP 只是做 soft ranking；你现在是把 MLP 概率当采样控制信号。**  
这时候，accuracy 不是关键，**calibration 才是关键**。 citeturn443767view0turn238593view0

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
如果只是把单个 MLP 拿来，看到 $p\approx 0.5$ 就多采几点，我会判断为“**逻辑可行，但论文层面偏弱，像工程增量**”。

## 七、我的最终结论

这条思路**是可行的**，而且很适合 binary unknown constraints，因为这类问题最缺的正是“边界信息”。\
但你真正能打动审稿人的，不是“0.5 附近采样”这句话本身，而是下面这句：

**我们不是在找最不确定的点，而是在找最可能位于 Pareto-relevant feasible–infeasible boundary 上的点，并利用这些点去发现新的 feasible components。**

这句话一旦成立，你的工作就从“简单 uncertainty sampling”变成了“为 CMOP-UC 引入主动边界学习与区域发现机制”。

下一步最值得做的是先把这三个模块写成伪代码：`bridge-pair selection`、`boundary bracketing/tangential search`、`boundary-aware retraining`。


#### You:
你分析的我仅仅认同一部分；1：对于当前阶段我期望采用单个 MLP 而不是 MLP ensemble（先用简单机制验证核心创新点，若有效果再考虑引入复杂机制）；2：对于p≈0.5 的点，我们可以只使用其目标值支配信息、拥挤度信息来排除无价值的点，我们目前所考虑的主要就是在目标空间中的可行域关系；3：对于“0.5 附近到底怎么主动采样”，你提的策略我很认可，然是我们要先验证创新点（主要是第一类是普通进化候选（p≈0.5的解也要参与和约束种群进行交叉变异）和第三类是边界邻域候选）至于二分括逼，我们可以在验证完创新点后作为更强的策略（暂时先不加入）；4：对于“这些 0.5 附近的点怎么重训 MLP”我们当前同样也不引入过于复杂的策略；总之，我的当前阶段的任务只有一个：怎样验证创新点成立？其余的增强策略我们都先不考虑；请你重新分析


#### ChatGPT:
可以。按你现在收缩后的设定，我的结论比上次更明确：

**这条路非常适合作为“第一阶段创新点验证”**，而且验证时就应该**故意保持简单**：继续沿用单个 MLP 与双种群框架，只改“怎样使用 $p\approx 0.5$ 的解”。

同时，文献位置也支持你现在这种“先做最小可验证版本”的思路：EADMM 已经把 unknown constraints 做成两个互补方向的种群；已有 dual-population / multitask CMOEA 也早就在做主种群考虑约束、辅助种群忽略约束；DRMCMO 则专门针对 binary constraints 做 detection region；MTCMO 还强调利用 infeasible solutions、并让辅助任务的 boundary 动态收缩。也就是说，**双种群本身不是你的创新点**，而应该是“在 CMOP-UC/BC 中，用单个概率分类器的 near-boundary cue 做 Pareto-relevant 边界候选搜索”。在我这轮检索范围内，我没有看到与你现在这套“单 MLP + near-0.5 候选 + 交叉/邻域双用法”完全相同的公开组合，但各个部件单独都已有先例。citeturn890911view0turn941617view1turn941617view2turn847630view5

再说你坚持的 single MLP：**完全可以**。因为你当前阶段不是要证明“概率校准做得最好”，而是要验证一个更小、更干净的命题。主动学习里，对二分类概率模型，选择 posterior 最接近 0.5 的样本，本来就是最经典的 uncertainty sampling；同时文献也明确提醒，这类策略可能会捡到离群点，所以再加一个 representativeness / density / diversity 的过滤是合理的。你现在用“目标支配关系 + 拥挤度”去过滤 near-0.5 解，虽然不是传统意义上的 density weighting，但作为**Pareto 相关性过滤**是逻辑自洽的轻量版本。citeturn258359view1turn258359view3

另外，你的总方向也不是拍脑袋。AL-PNN 这类工作已经在约束可行性分类里用 active learning 去追踪“可能包含 constraint boundary 的 interesting region”；BE-CBO 也明确把“准确建模并高效探索 feasible–infeasible boundary”当作发现更优设计的关键。它们不是你的同类算法，但都说明：\*\*unknown/binary constraint 场景里，边界导向搜索是合理且重要的科学假设。\*\*citeturn847630view1turn847630view2

---

## 逐点回应你的四点

### 1. 单个 MLP，不上 ensemble

我认可，而且这反而有利于**隔离变量**。

当前阶段最好的做法不是“把模型做强”，而是“把创新点做干净”：

- MLP 结构、损失、更新触发、训练集维护，尽量保持固定；
- 只新增：near-0.5 解的选择与使用方式。

这样一来，后面如果结果变好，审稿人很难说“是因为你换了更强的 surrogate”，而更容易承认“是边界候选机制起作用了”。

我只建议你在论文措辞上，少用“置信度 confidence”，多用：

- **boundary score**
- **uncertainty cue**
- **near-boundary candidate**

因为你现在验证的是“可用于搜索的边界信号”，不是“严格校准的后验概率”。

### 2. 只用目标值支配 + 拥挤度去排除无价值点

我也认可，但这里有一个**表述边界**要收紧。

如果你主要靠目标空间关系来筛 near-0.5 点，那么你当前验证的其实不是“恢复真实决策空间中的约束边界几何”，而是：

**找到对 Pareto 搜索有价值的 boundary candidates。**

所以论文里更稳妥的表述，不是“发现新的 feasible region boundary”，而是：

- “发现新的 **feasible objective sectors**”
- “发现新的 **Pareto-relevant feasible boundary segments**”
- “发现新的 **promising feasible sectors**”

这会更符合你现在算法真正做的事。

### 3. 当前只做两件事：

- near-0.5 解参与与约束种群的交叉变异
- near-0.5 解周围做边界邻域采样

这也够了。\
因为这两件事分别验证两种不同作用：

第一种验证的是：\
**near-0.5 解能不能当“桥”去产生更好的 feasible offspring。**

第二种验证的是：\
**near-0.5 解附近是不是真的“边界富集区”，值得局部 probing。**

只要这两条都成立，你的创新点就已经站住了。二分括逼、ensemble、复杂重训都可以作为第二篇或扩展版的增强策略。

### 4. 当前不引入复杂重训

这同样是对的。

现在最优实验设计不是“重训更聪明”，而是“让 MLP 的训练机制保持不变”。\
最简单的方案就是：

- retraining 触发保持固定；
- boundary operator 新产生的样本照常进入 archive / FIFO 更新；
- 不引入新的损失项，不改网络结构。

这样可以把所有收益都归因到“near-0.5 候选的利用方式”，而不是归因到模型训练技巧。

---

## 你现在真正要验证的“最小命题”

我建议你把当前论文的核心命题收成下面这一句：

> 在 CMOP-UC 中，单个 MLP 输出的 near-0.5 候选，经过目标支配与拥挤度过滤后，比随机候选或高预测值候选更容易产生 **Pareto-useful feasible offspring**；把这些候选用于与约束种群交叉和局部边界邻域采样，在**相同评估预算**下能更快发现新的 feasible objective sectors，并改善整体优化性能。

这个命题一旦成立，你的创新点就成立。

---

## 怎么验证这个创新点成立

我建议你按**三层证据链**来做，而不是只看最终 HV/IGD。

### 第一层：机制成立

先证明 near-0.5 点本身确实更“边界有用”。

你需要比较三类 seed：

1. **U-seed**：$|p-0.5|$ 最小的一批点
2. **H-seed**：预测值最高的一批点
3. **R-seed**：随机选的一批点

三组都用**相同数量**、**相同预算**、**相同后续 operator**。

你要看的不是最终 PF，而是这几个 seed-level 指标：

#### (a) Seed Success Rate, SSR

一个 seed 如果在固定小预算内（比如 1 次与 $P_c$ 交叉 + 1 次邻域采样）产生了至少一个“有价值”的 feasible 子代，就记成功。

“有价值”建议定义成二选一：

- feasible 且进入环境选择后存活；
- feasible 且落入此前未被占据的 objective-space sector。

如果 U-seed 的 SSR 明显高于 H-seed 和 R-seed，你的核心机制就成立了。

#### (b) Feasible Yield, FY

每单位 boundary budget 产生的 feasible offspring 数量。

$
FY = \frac{\#\text{feasible offspring from boundary operators}}{\#\text{boundary evaluations}}
$

#### (c) New-Sector Yield, NSY
每单位 boundary budget 发现的**新可行目标区段**数量。

这里我建议你不要在当前阶段硬说“新可行域”，而是把 objective space 离散化：
- 用 reference vectors，或
- 用统一网格 / bins

某个 feasible 点如果落到此前未被 $P_c$ 占据的 bin / niche，就记为发现一个新 sector。

这会非常符合你“主要看目标空间中的可行域关系”的设定。

---

### 第二层：边界假设成立
再证明 near-0.5 点确实比别的点更接近“feasible–infeasible 交界”。

这一步**不参与算法本身**，只做**事后诊断**，所以不会污染主实验。

做法很简单：  
对每类 seed（U/H/R），在它周围取一小圈诊断扰动点，检查这些扰动点的 feasibility label：

- 如果局部邻域里同时出现 feasible 和 infeasible，说明 seed 靠近边界；
- 如果邻域几乎全 feasible 或全 infeasible，说明 seed 更像区域内部点。

于是你可以定义一个：

#### Boundary Hit Rate, BHR
$
BHR = \frac{\#\text{seeds whose local neighborhood contains both labels}}{\#\text{seeds}}
$

如果 U-seed 的 BHR 显著高于 H-seed 和 R-seed，你就能非常直接地证明：

**单个 MLP 的 near-0.5 输出，确实在这个任务里携带了 boundary information。**

这一步非常关键。  
因为如果没有 BHR 这类机制诊断，审稿人很可能会说：

> “你性能变好，可能只是因为多做了局部搜索，未必真是 boundary idea 起作用。”

---

### 第三层：优化收益成立
最后才看整体优化是不是变好。

这里直接沿用当前未知约束测试协议即可：
- DASCMOP-UC1–9；
- 30 次独立运行；
- IGD / HV；
- Wilcoxon；
- Friedman 排名。

但我建议你加一个比最终 HV/IGD 更重要的指标：

#### Early-stage performance
因为你的创新点本质是“更快找到有价值的 feasible sector”，所以应该重点报告：
- 10% 预算时的 HV / IGD
- 20% 预算时的 HV / IGD
- HV-AUC / IGD-AUC（整条演化曲线下面积）
- Evaluations to first feasible
- Evaluations to first new sector
- Evaluations to $k$ 个 feasible sectors

如果你的方法只是最终结果略好，但前期没有更快发现 feasible sector，那“边界搜索”这个故事就不够强。  
反过来，如果早期优势明显，哪怕最终 HV 只是持平偏好，创新点也成立。

---

## 你最该做的消融，不是“多不多模块”，而是“因果对照是否干净”

我建议至少做下面 6 个版本，而且**总评估预算必须完全一致**。

### A0：CCMO 原始基线
什么都不加。

### A1：Uncertain-Mating Only
只把 near-0.5 点用于与 $P_c$ 交叉。

### A2：Neighborhood-Probing Only
只在 near-0.5 点附近做局部采样。

### A3：Full-Minimal
A1 + A2，同时启用。  
这就是你当前阶段的完整创新版。

### C1：Random-Control
把 A3 里的 near-0.5 选点，替换成随机选点。  
预算、交叉次数、邻域次数完全一样。

### C2：High-p Control
把 A3 里的 near-0.5 选点，替换成预测值最高的点。  
这一步非常重要，因为它能回答一个很尖锐的问题：

> 你为什么不直接选“最可能可行”的点，非要选 0.5 附近？

如果 A3 明显优于 C2，你的 boundary story 就真正站住了。

### 可选的 C3：Uncertainty w/o Pareto Filter
只按 near-0.5 选，不加支配与拥挤度过滤。

如果 A3 > C3，说明你的“目标空间价值过滤”确实有贡献；  
如果 A3 ≈ C3，则说明 near-0.5 信号本身已经足够强，过滤机制未必需要写得太重。

---

## 一个最小可实现的算法版本

为了避免阈值不稳，我不建议用固定区间 $0.45 \le p \le 0.55$。  
更简单也更稳的是：

$
u(x)=1-2|p(x)-0.5|
$

在当前候选池里，直接取 $u(x)$ 最大的前 $q\%$ 作为 uncertain pool。\
这仍然是单 MLP、零额外复杂度。

然后：

1. 把 uncertain pool 与当前 $P_c$ 合并，只按**目标值**做非支配排序；
2. 同 front 内按 crowding distance 排；
3. 选前 $m$ 个 uncertain seeds。

对每个 seed $x$，分配固定预算：

- 以概率 $\eta$：和 $P_c$ 中一个匹配的 feasible parent 交叉；
- 以概率 $1-\eta$：在 $x$ 周围做一次局部扰动。

这里 parent 的匹配，你既然强调目标空间关系，那就用最简单的：

- 在归一化目标空间中选最近的 feasible parent，或
- 选同一 reference vector / niche 的 feasible parent。

邻域采样也别搞复杂：

- 直接围绕 $x$ 做 polynomial mutation / Gaussian perturbation；
- 或者做一个很轻的 directional perturbation：朝匹配 feasible parent 的方向偏一点点，再加局部噪声。

这已经足够验证创新点了，不需要二分、不需要 bracket、不需要 ensemble。

---

## 论文里“创新点成立”的判据，我建议你这样写

不是“最终 HV 提升了，所以创新成立”，而是同时满足下面三条：

### 判据 1：边界性

U-seed 的 BHR 显著高于随机和 high-p 对照。

### 判据 2：有用性

U-seed 的 SSR / FY / NSY 显著高于随机和 high-p 对照。

### 判据 3：优化性

在相同总预算下，A3 在早期可行解发现速度、new-sector discovery、以及 IGD/HV 或其 AUC 上优于 A0、C1、C2。

只要这三条成立，你就可以很有底气地说：

**single-MLP 产生的 near-boundary cue 在 CMOP-UC 中是有用的，且这种 usefulness 可以通过目标空间过滤与边界导向 operator 被转化成真实优化收益。**

---

## 我对你当前版本的最终判断

我现在比上次更支持你这个简化版，因为它更像一个**干净的科学实验**：

- 不去争 ensemble 是否更强；
- 不去争 bisection 是否更精确；
- 不去争复杂重训是否更稳；

你只证明一件事：

**“0.5 附近的点，是不是值得专门拿来找 Pareto-relevant feasible boundary sectors？”**

如果答案是“是”，而且你能用上面那套三层证据链证明出来，那你的创新点就成立了。

这个阶段我给你的最核心建议只有一句：

**把论文从“更复杂的算法设计”改成“更干净的创新点验证”。**

如果你愿意，我下一条可以直接帮你把这套内容整理成论文里的“Method + Research Questions + Ablation Design”三段式写法。


#### You:
我计划在CCMO算法上进行测试（以他为基准来验证我的创新点）；其次请你分析near-0.5的点来源于（最初：从约束种群域无约束种群的交叉变异产生的子代中选择p接近0.5的解）然后排除被当前约束种群支配的子代，然后再真实评估，如果是不可行解则（1：在其附近局部搜索来找其余p=0.5的解；2：与约束种群交叉变异产生下一代的子代）；如果是可行解则加入约束种群并在其附近进行局部搜索（主要是多样性，用来搜新发现的可行域的前沿边界）；只要是p接近0.5，无论是不是可行解都参与参与MLP的下次重训（当然在存档中我们要控制参与训练的可行解与不可行解的比例）。你认为我的分析是否正确？请按照此思路为我设计从CCMO基础上逐步改造和验证的策略和过程


#### You:
同时在此情况之下，环境选择应该怎样设计，本代的子代是哪几部分组成？1：（从约束种群域无约束种群的交叉变异产生的子代+上一代p接近0.5的解附近局部搜索来找的其余p=0.5的解+上一代p接近0.5的解与约束种群交叉变异产生下一代的子代）是第一种；2：（从约束种群域无约束种群的交叉变异产生的子代+真实评估本代p接近0.5的解后再在附近局部搜索来找的其余p=0.5的解再真实评估+真实评估本代p接近0.5的解后再与约束种群交叉变异产生子代再进行真实评估）是第二种；你任务本代子代是什么？在本代子代中如何进行环境选择（对于子代中的可行解，毫无疑问要选择不被当前约束种群支配的；对于不可行解应该怎么选？P接近0.5的吗）？请你仔细分析



#### ChatGPT:
我先给结论：

**第一版请用“第一种”，不要用“第二种”做主流程。**\
也就是：**本代真正参与环境选择的子代，应当由“本代开始时就已经确定的父代/边界档案”一次性生成，然后统一真实评估，再统一做一次环境选择。** 不要在同一代里先评估本代的 near-0.5 种子，再立刻围绕这些“刚发现的种子”继续局部搜索、继续交叉、继续评估。那样会把一代变成“同代内两阶段 memetic/异步搜索”，预算难控，环境选择也会变得不清楚。CCMO 这类双种群框架的核心协作点本来就在环境选择阶段的信息交换。citeturn541209view4

更重要的是，这样设计正好对应 CCMO 的已知优缺点：CCMO 在 UPF 和 CPF 有较强重叠时表现好，因为辅助种群能帮助主种群穿越 infeasible barrier；但当 UPF 与 CPF 完全分离时，CCMO 的互补作用会明显下降。综述还明确指出，对 Type IV 这类完全分离的问题，“UPF 和 CPF 之间的解”对找到 CPF 很有效。你的 near-0.5 机制，本质上就是在主动保留和扩展这类“位于两者之间的桥梁解”。citeturn541209view4turn524584search9

---

## 一、你问的“本代子代是什么”，我建议这样定义

你这里其实要分清两个概念：

- **候选子代池**：先由各种算子生成、还没真实评估的候选；
- **本代真实子代**：本代真正拿去做 oracle 评估、并进入环境选择的那一批。

你现在最合适的做法是：

### 1）候选子代池

在第 $t$ 代开始时，你手里已有：

- 当前约束种群 $P_C^t$
- 当前无约束种群 $P_U^t$
- 上一代已经真实评估过的 near-0.5 档案\
  $A\_F^{t-1}$：feasible near-0.5\
  $A\_I^{t-1}$：infeasible near-0.5

然后生成 5 类候选：

$
Q^t = Q_C^t \cup Q_U^t \cup Q_X^t \cup Q_{LS}^t \cup Q_M^t
$

其中：

- $Q_C^t$：CCMO 原始的约束种群常规 offspring
- $Q_U^t$：CCMO 原始的无约束种群常规 offspring
- $Q_X^t$：**本代** $P_C^t$ 与 $P_U^t$ 交叉/变异得到的 cross-pop candidates
- $Q_{LS}^t$：**上一代** near-0.5 档案 $A_F^{t-1}\cup A_I^{t-1}$ 周围的局部搜索 candidates
- $Q_M^t$：**上一代** infeasible near-0.5 档案 $A_I^{t-1}$ 与当前 $P_C^t$ 交叉得到的 candidates

所以，如果你问“边界模块的本代子代是哪几部分组成”，答案就是你说的**第一种**：  
**cross-pop offspring + 上一代 near-0.5 周围的局部 offspring + 上一代 infeasible near-0.5 与 $P_C$ 交叉得到的 offspring。**

但如果你问“整个算法本代子代是什么”，那答案应当是：

**原 CCMO 的常规子代 + 边界模块子代。**

也就是说，不是只保留你列的三部分，而是：

$
O^t = O_C^t \cup O_U^t \cup O_B^t
$

其中

$
O_B^t \subseteq Q_X^t \cup Q_{LS}^t \cup Q_M^t
$

这里 $O_B^t$ 是从边界候选池里筛出来、并真实评估的那一部分。

---

## 二、为什么我不建议你用“第二种”

因为第二种相当于：

1. 本代先发现一批 near-0.5 种子  
2. 立刻对这些种子做局部搜索/再交叉  
3. 本代再追加一轮真实评估  
4. 再统一环境选择

这会带来 3 个问题。

第一，**一代的语义变了**。  
它不再是“父代 $\to$ 子代 $\to$ 选择”，而是“父代 $\to$ 第一批子代 $\to$ 第二批子代 $\to$ 选择”。这已经更像两阶段 memetic 搜索。

第二，**预算不干净**。  
same-generation 的追加评估数量取决于你本代发现了多少 near-0.5 种子，这会让不同代的真实评估开销波动很大，不利于和 CCMO、随机控制组做公平比较。

第三，**环境选择会变得含糊**。  
你会被迫回答：  
“是先对第一批子代做一次局部选择，再合并第二批？还是最后一次性选？”  
这会让你的第一版验证变复杂。

所以我的建议很明确：

**本代 newly discovered near-0.5 seeds，只负责进入 $A_F^t/A_I^t$ 和重训数据；它们的局部扩展和边界交叉，延迟到下一代 $t+1$ 再做。**

这就是最干净、最可验证的版本。

---

## 三、near-0.5 候选怎么从候选池里筛成“本代真实子代”

因为你不是要把所有 $Q_X,Q_{LS},Q_M$ 都拿去真实评估，而是先用单个 MLP 做 cheap screening。

定义边界分数：

$
u(x)=1-2|p(x)-0.5|
$

$u(x)$ 越大，越接近 0.5。

但这里不能只按 $u(x)$ 选。主动学习里，不确定性采样本来就常用“最不确定样本”，但同时也常配一个后续 diversity / representativeness 过滤，否则容易捡到不具代表性的点。现在很多 hybrid AL 也是先做 uncertainty top-k，再做 diversity subset。你的“支配关系 + 拥挤度/sector 稀疏性”正好可以扮演这个第二层过滤器。citeturn119213search6turn119213search12turn119213search3

所以边界候选的筛法，我建议是：

### 第一步：按 $u(x)$ 取 top-k
从 $Q_X^t \cup Q_{LS}^t \cup Q_M^t$ 中先取最接近 0.5 的一批。

### 第二步：做 objective-space 过滤
不要用“被当前 $P_C$ 任意支配就删”的全局硬规则。  
用**sector-aware feasible filter**：

把目标空间归一化后分 reference sectors / niches。  
对候选 $x$，若存在当前 feasible 参考点 $y$ 使得：

- $y \prec x$
- 且 $x,y$ 落在同一 sector
- 且该 sector 已经不稀疏

则丢弃 $x$。

否则保留。

这样做的原因是：  
CCMO 弱点恰恰在于 UPF 与 CPF 分离时，中间桥梁区域难找；而综述明确指出，这类问题上“UPF 和 CPF 之间的解”是有效的。若你做全局 dominance 一刀切，很容易把“全局上略差、但在稀疏 sector 里是入口”的桥点过早删掉。citeturn541209view4

### 第三步：再按 crowding / sector sparsity 排序
优先保留稀疏 sector 里的点，而不是让所有 near-0.5 都挤在同一片区域。

### 第四步：只对最终选中的这批做真实评估
这一批才叫本代真正的边界 offspring $O_B^t$。

---

## 四、环境选择到底怎么做

这里最关键的一句是：

**环境选择不是“对可行子代选、对不可行子代也一起硬排”。而是“先选 feasible，再决定是否需要少量 infeasible bridge points 填充/存档”。**

我建议你把**种群选择**和**边界档案选择**分开。

---

### A. 约束种群 $P_C^{t+1}$ 的环境选择

先构造联合集合：

$
R_C^t = P_C^t \cup O_C^t \cup O_B^t
$

为了第一版干净，我建议不要把所有 $O_U^t$ 原封不动扔进 $P_C$ 的环境选择里；因为你的边界模块已经通过 $Q_X^t$ 把 $P_U$ 的信息引进来了，再把原始 $O_U$ 全塞进来会加噪声。  
也就是说，$P_U\to P_C$ 的知识转移，第一版主要走 **cross-pop offspring** 这条路。

然后把 $R_C^t$ 分成：

- $R_C^{F}$：真实评估为 feasible
- $R_C^{I}$：真实评估为 infeasible

#### 第 1 层：先选 feasible
这一步不是“只保留不被当前 $P_C$ 支配的子代”，而是：

**在整个 feasible 联合集合 $R_C^F$ 上做 non-dominated sorting + crowding/reference niching。**

也就是：
- 先按 Pareto front 分层；
- 不够装满整层时，用 crowding distance 或 reference-vector niching 选。

如果 $|R_C^F|\ge N$，那 $P_C^{t+1}$ 直接全部由 feasible 组成，**不保留任何 infeasible 到 $P_C$**。

这点我建议你一定写清楚：

> **当 feasible 足够时，$P_C$ 不需要保留 infeasible。**  
> infeasible 的主要去处应是边界档案 $A_I$，不是强行塞进 $P_C$。

#### 第 2 层：只有当 feasible 不足时，才用 infeasible 填空
若 $|R_C^F|<N$，令已选 feasible 集为 $S_F$，再从 infeasible 里补足剩余位置。

这时 infeasible 的选择规则不要用“最高 p”。
我建议用下面这个**词典序**：

1. 先删除被 $S_F$ 在**同一 sector** 中支配的 infeasible；
2. 对剩余 infeasible，按 $u(x)=1-2|p(x)-0.5|$ 从大到小排；
3. 若并列，则优先 sector 稀疏的；
4. 再并列，看 crowding distance；
5. 再并列，看 objective-space 上的非支配层。

所以答案是：

**对 $P_C$ 中要保留的 infeasible，确实优先选 $p$ 接近 0.5 的，但不是单独只看 $p\approx0.5$，而是“near-0.5 + 不被当前已选 feasible 在同一 sector 压死 + 具有目标空间多样性”。**

这和已有 CMOEA 思路是一致的：  
近年的方法会用 feasible search boundary 保留边界附近的 infeasible，也有方法在目标空间若干 subregions 中专门维护分布良好的 infeasible，以帮助跨越 infeasible region 和维持 diversity。citeturn771312view1turn771312view2

---

### B. infeasible 不应该主要在 $P_C$ 里选，而应该主要在 $A_I$ 里选

这是你现在最容易混淆的点。

**对于你的算法来说，infeasible near-0.5 的主战场不是 $P_C$，而是边界档案 $A_I$。**

我建议：

$
A_I^{t+1} = \text{Select}\left(A_I^t \cup \{x\in O_B^t \mid x \text{ infeasible and near-}0.5\}\right)
$

其选择规则与上面类似，但更强调 boundary role：

1. 不被当前 selected feasible set 在同一 sector 完全压死；
2. $u(x)$ 大；
3. sector 稀疏；
4. crowding 大。

这和 CEDE 的思路很接近：它专门提出 archive 去保存“potentially valuable infeasible solutions”，避免它们在进化中被直接丢失；另外也有方法明确把 feasible search boundary 内的 infeasible 当成 promising candidates。citeturn771312view0turn771312view1

所以你如果问：

> “不可行解应该怎么选？P 接近 0.5 的吗？”

我的回答是：

- **对 $A_I$**：是，核心就是选 $p\approx0.5$ 的；
- **对 $P_C$**：只有在 feasible 不足时，才从这些 $p\approx0.5$ 的 infeasible 中选少量填空；
- **绝不建议**把大量 infeasible 直接长期塞进 $P_C$。

---

### C. feasible near-0.5 解怎么处理

你前面说“如果是可行解则加入约束种群并在其附近局部搜索”，我认可，但我要加一层区分：

- 对 $P_C$：它必须经过 $R_C^F$ 上的正常环境选择，不是强插入；
- 对边界可行档案 $A_F$：只要是 near-0.5 且真实评估为 feasible，就可以优先保留。

也就是说：

$
A_F^{t+1} = \text{Select}\left(A_F^t \cup \{x\in O_B^t \mid x \text{ feasible and near-}0.5\}\right)
$

在 $A_F$ 里，这些点是“可行侧边界种子”；  
在 $P_C$ 里，它们只是“feasible 候选”。

这样分开后逻辑会很顺：

- $P_C$：负责最终 constrained search；
- $A_F/A_I$：负责下一代边界算子的亲本/中心。

---

### D. 无约束种群 $P_U^{t+1}$ 的环境选择

这个最简单，尽量保持 CCMO 原味：

$
R_U^t = P_U^t \cup O_U^t \cup O_X^t
$

其中 $O_X^t$ 是真实评估后的 cross-pop offspring。  
对 $R_U^t$ 直接做**不考虑约束的 non-dominated sorting + crowding** 即可。

我不建议把 $O_{LS}^t$ 和 $O_M^t$ 也扔给 $P_U$，因为这两个已经是边界特化后的 offspring，主要服务 $P_C$ 侧的 boundary search，不应该再污染 $P_U$ 的“纯 helper problem”角色。

---

## 五、把它写成一套你可以直接实现的“本代流程”

我给你一个最适合当前论文阶段的版本。

在第 $t$ 代开始时，已知：
- $P_C^t$, $P_U^t$
- $A_F^{t-1}$, $A_I^{t-1}$
- 当前单 MLP $M_t$

### 第一步：生成候选池
$
Q^t = Q_C^t \cup Q_U^t \cup Q_X^t \cup Q_{LS}^t \cup Q_M^t
$

其中
- $Q_C^t$: 原 CCMO 常规约束 offspring
- $Q_U^t$: 原 CCMO 常规无约束 offspring
- $Q_X^t$: $P_C^t \times P_U^t$
- $Q_{LS}^t$: 以 $A_F^{t-1}\cup A_I^{t-1}$ 为中心的局部扰动
- $Q_M^t$: $A_I^{t-1} \times P_C^t$

### 第二步：MLP 预筛
对 $Q_X^t \cup Q_{LS}^t \cup Q_M^t$ 计算 $u(x)$。

先取 uncertainty top-k，再做：
- sector-aware feasible filter
- crowding / 稀疏性排序

选出固定预算的一批去真实评估，记为 $O_B^t$。

### 第三步：真实 offspring 定义
$
O^t = O_C^t \cup O_U^t \cup O_B^t
$

这就是**本代真正参与环境选择的子代**。

### 第四步：更新 $P_C$
$
R_C^t = P_C^t \cup O_C^t \cup O_B^t
$

- feasible 先选满；
- 若未满，再从 infeasible 中选 near-0.5 + sector-sparse 的桥点补足。

### 第五步：更新 $P_U$
$
R_U^t = P_U^t \cup O_U^t \cup O_X^t
$

直接 objective-only 的 ND + crowding。

### 第六步：更新边界档案

- infeasible near-0.5 $\to A_I^t$
- feasible near-0.5 $\to A_F^t$

  它们**不在本代继续扩展**，而是留到下一代再做 $Q_{LS}^{t+1}$ 和 $Q\_M^{t+1}$。

### 第七步：重训 MLP

本代真实评估过、且 near-0.5 的样本进入边界训练集；\
再与 core archive 混合、做类别平衡后重训。\
未真实评估的 predicted near-0.5 不能进训练集。

---

## 六、你现在最需要记住的三句话

第一句：

**本代 offspring 不是“所有生成出来的候选”，而是“本代真正做了真实评估的那一批”。**

第二句：

**第一版应该选第一种代际结构，不要同代内递归扩展 newly found near-0.5 seeds。**

第三句：

**不可行解不是简单按最高 p 选，而是按“near-0.5 + 同 sector 不被已选 feasible 压死 + 多样性”选；而且它们主要应该进 $A\_I$，不是长期塞进 $P\_C$。**

如果你愿意，我下一条可以直接把这套流程写成“伪代码版 Algorithm 1 + P\_C / P\_U / A\_I / A\_F 的环境选择规则”。
