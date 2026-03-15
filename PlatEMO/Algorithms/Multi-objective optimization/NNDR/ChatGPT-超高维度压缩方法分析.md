# 最终确定的最小可用版本（MVP）

## 1. 背景

本文方案面向连续型大规模多目标优化问题，尤其是 LSMOP、LMF、GLSMOP 这一类具有**高维、分组、linkage、变量贡献不均衡**等特点的 benchmark 家族。目标不是再堆一个复杂网络，而是把高维决策向量 `x in R^D` 的搜索，压缩成一个低维表示，同时保留后期精修能力。

本文最终确定的版本坚持三个原则：

1. **只保留一个核心创新点**：把 NNDREAMO 中“直接搜索共享生成器参数”的思路，进一步压缩为“只搜索低维 latent”。
2. **不引入不必要模块**：最小可用版不把 context 当主输入，不做在线训练，不堆叠多头结构。
3. **保留 coarse-to-fine 思想**：前期在低维流形上搜索，后期只释放少量残差自由度做精修。

---

## 2. 要解决的问题

设原始问题为

$
\min_{x \in [L,U]^D} F(x) = (f_1(x), \dots, f_m(x)), \qquad D \gg 1000.
$

希望构造一个低维搜索变量 `y`，满足：

- 搜索维度与 `D` 尽可能解耦；
- 能利用变量的顺序、分组和局部 linkage 结构；
- 不需要监督样本；
- 不依赖在线训练；
- 能在后期对少量关键坐标做局部精修。

换言之，我们要解决的不是“再设计一个更复杂的神经网络”，而是：

> **如何用一个足够小、足够稳、足够可解释的低维对象，生成完整高维解，并在多目标搜索过程中持续可用。**

---

## 3. 灵感来源

### 3.1 来自 NNDREAMO 的部分

NNDREAMO 最值得继承的不是它那一个具体 FCN 的结构，而是三件事：

1. **共享逐变量生成**：用同一个映射逐变量地产生完整解；
2. **搜索对象与 `D` 解耦**：搜索小维度连续参数，而不是直接搜索高维决策向量；
3. **两阶段 refinement**：先在低维空间粗搜，后期回到更细粒度空间精修。fileciteturn8file1 fileciteturn8file2

另外，NNDREAMO 的一个关键经验是：它依赖变量/条目特征与最优变量值之间存在相关性；论文中的扰动实验表明，一旦这种相关性被破坏，性能会明显下降。它同时刻意采用了非常轻的网络，并指出更复杂网络并未显著带来更好效果。fileciteturn8file0 fileciteturn8file2

### 3.2 来自 coordinate network / INR 的部分

对连续变量问题，逐坐标生成天然对应 coordinate decoder 的建模方式。它允许我们把“变量编号 / 位置 / 分块信息”转为输入特征，再由共享生成器输出 `x_i`。

### 3.3 来自低维嵌入方法的部分

REMBO 一类方法提醒我们：论文必须回答“是不是简单低维嵌入就够了”。因此，最终方案虽然采用神经生成器，但必须保留一个强的**非神经低维嵌入对照**。另一方面，Fourier features 提供了对 coordinate MLP spectral bias 的直接修正，FiLM 提供了简洁的条件化方式，而 D'OH 则说明“固定随机投影/decoder + 运行时 latent 优化”本身就是一条合理而且学术上干净的路线。citeturn0search3turn0search1turn0search2turn3search0

---

## 4. 对两个约束的最终判断

## 4.1 约束一：变量顺序 / 坐标特征必须有一定语义

**判断：合理，而且必须写成显式适用条件。**

本方法默认变量不是完全无结构的“任意打乱都一样”的向量，而是至少满足下列条件之一：

- 变量顺序本身有意义；
- 变量存在稳定分组；
- 相邻变量或同组变量更可能具有相似作用；
- benchmark 本身按子组件、linkage、distance grouping 等方式构造变量结构。

因此，这个方法的适用范围应明确表述为：

> **面向 LSMOP / LMF / GLSMOP 这类“变量存在组结构或顺序结构”的连续型大规模多目标优化问题。**

而不是宣称它对任意 permutation-invariant 的高维问题都成立。这个边界并不是弱点，反而会让论文叙事更学术、更诚实。citeturn6search3turn4view2turn5view2

## 4.2 约束二：必须保留非神经强 baseline

**判断：完全合理，而且是必须项。**

否则审稿人很容易质疑：

> 你的提升究竟来自“神经生成器”，还是仅仅来自“把高维搜索换成了低维搜索”？

因此，实验中至少要有一类**线性 / 随机低维嵌入**对照。最推荐的对照不是只在 related work 里提 REMBO，而是直接给出一个同框架下的线性版本：

$
x = B z + M_A \alpha,
$

其中 `B` 是固定随机基或固定分块线性基，外层仍使用同一个多目标进化框架。这个对照最干净，也最能回答“神经非线性是否必要”。citeturn0search3

---

## 5. 最终确定的最小可用版本

本文最终确定的最小可用版本为：

$
x_i = G(\phi_i; z_g, z_{b(i)}) + [M_A \alpha]_i, \qquad i = 1, \dots, D.
$

其中：

- `G`：**冻结的共享坐标解码器**；
- `\phi_i`：第 `i` 个变量的坐标特征；
- `z_g`：global latent，控制整体形状；
- `z_{b(i)}`：block latent，控制第 `i` 个变量所在块的局部形状；
- `M_A \alpha`：稀疏残差项，只对 active set `A` 生效。

这个版本刻意不包含：

- `context` 作为 decoder 主输入；
- 在线训练 decoder；
- 更细一级的 `z_f`；
- SIREN 作为默认主干；
- 大型 hypernetwork。

原因很简单：**MVP 的目标不是追求最强表达力，而是先验证“theta -> z 的二级压缩 + coarse-to-fine refinement”这一个核心创新是否站得住。**

---

## 6. 特征设计

每个变量 `x_i` 使用如下特征：

$
\phi_i = [u_i,\; p_i,\; c_i,\; FF(u_i),\; \xi_i,\; \psi_i^{\text{struct}}].
$

含义如下：

- `u_i`：全局归一化位置，`u_i = (i-1)/(D-1)`；
- `p_i`：块内归一化位置；
- `c_i`：所在块的中心位置或块编号归一化值；
- `FF(u_i)`：Fourier positional encoding；
- `\xi_i`：很小的固定扰动，用于打破完全重复特征；
- `\psi_i^{struct}`：可选结构特征，如已知的物理坐标、图谱特征、工程分组标签等。

### 最小默认特征

如果问题没有额外结构信息，则只使用：

$
\phi_i = [u_i,\; p_i,\; c_i,\; FF(u_i),\; \xi_i].
$

### 分块方式

- 若 benchmark 或问题本身提供自然分组，则直接使用该分组；
- 否则，将 `1,...,D` 等分成 `B` 个连续块。

这个处理与 LSMOP/LMF/GLSMOP 的 benchmark 结构是相容的，但不假设知道真实最优分组。Fourier positional encoding 放在这里，是因为标准坐标 MLP 有明显 spectral bias，而 Fourier features 能更好表达高频变化。citeturn0search1turn6search3turn4view2

---

## 7. 网络结构

### 7.1 共享坐标解码器

采用一个**小型、冻结、共享**的 coordinate decoder：

- hidden layers：2 层；
- width：32；
- activation：`tanh`；
- 输出通过边界映射进入 `[L_i, U_i]`。

记

$
h_i^{(0)} = P\phi_i.
$

然后对第 `\ell` 层，使用最小化的 FiLM 调制：

$
[\gamma_b^{(\ell)}, \beta_b^{(\ell)}] = R_{\ell}[z_g; z_b],
$

$
h_i^{(\ell+1)} = \tanh\big(\gamma_{b(i)}^{(\ell)} \odot (W_{\ell} h_i^{(\ell)} + b_{\ell}) + \beta_{b(i)}^{(\ell)}\big).
$

最后输出粗解：

$
\tilde{x}_i = w_o^\top h_i^{(2)} + b_o,
$

$
x_i^{\text{coarse}} = L_i + \frac{\tanh(\tilde{x}_i)+1}{2}(U_i-L_i).
$

最终解为：

$
x_i = x_i^{\text{coarse}} + [M_A \alpha]_i.
$

### 7.2 为什么用 FiLM，而不是完整 hypernetwork

因为本文的目标是“干净、最小可用”。

FiLM 只需要一个很小的映射 `R_\ell`，就可以让 latent 影响解码器的响应，而不用让 latent 直接生成整个网络的全部参数。这样既保留了“latent 控制共享生成器”的核心思想，又避免把方法重新做重。FiLM 本身就是一种简洁的 feature-wise affine modulation；D'OH 也支持“固定随机投影/decoder + 运行时 latent 优化”这条轻量路线。citeturn0search2turn3search0

### 7.3 默认参数

建议的最小默认配置：

- Fourier 频率数：`K = 6`；
- 块数：`B = 16`；
- global latent 维数：`d_g = 8`；
- 每个 block latent 维数：`d_b = 2`；
- Stage 1 搜索维数：`8 + 16*2 = 40`；
- Stage 2 active residual 维数：`K_active = 64`。

这组参数的目的，是让 Stage 1 保持在一个真正小的低维空间中搜索。

---

## 8. 训练方法

## 8.1 单任务主版本：不训练

最小可用版采取如下策略：

- `P, W_\ell, b_\ell, w_o, R_\ell` 初始化一次后**全部冻结**；
- 不做 online gradient training；
- 真正被搜索的只有低维变量：

$
y = [z_g, z_1, \dots, z_B]
$

以及第二阶段中的

$
y = [z_g, z_1, \dots, z_B, \alpha].
$

### 原因

1. 这是对 NNDREAMO “不依赖预训练”的继承；
2. 在黑箱多目标设定下，在线训练很容易把生成器塑造成“当前种群附近的坏流形”；
3. 对 MVP 而言，冻结 decoder 更能证明真正的贡献来自 `theta -> z` 的压缩，而不是“训练技巧”。NNDREAMO 本身也明确把“不需预训练、直接在低维连续空间中搜索”作为其动机和优势之一。fileciteturn8file4 fileciteturn8file9

## 8.2 外层优化器

外层直接使用标准连续型多目标进化框架即可，推荐：

- `NSGA-II + SBX + Polynomial Mutation`。

理由：

- 简洁；
- 易于在 PlatEMO 中复现；
- 与 NNDREAMO 的“在低维连续空间中搜索”叙事一致；
- 便于做后续消融。NNDREAMO 在其低维阶段同样使用了连续遗传算子，在多目标场景下用非支配排序和 crowding distance 做选择。fileciteturn8file1

---

## 9. 整体流程

### 阶段 0：预处理

1. 输入问题维数 `D` 与边界 `[L,U]`；
2. 构造 `B` 个连续 block；
3. 为所有变量构造 `\phi_i`；
4. 初始化冻结的 decoder 参数；
5. 初始化 reduced population：`Y = { [z_g, z_1, ..., z_B] }`。

### 阶段 1：低维 coarse search

对种群中每个 reduced vector：

1. 通过 `R_\ell` 计算 FiLM 调制参数；
2. 对每个坐标 `i` 调用共享 decoder，生成 `x_i^{coarse}`；
3. 拼接完整解 `x`；
4. 调用 `Problem.Evaluation(x)`；
5. 用 NSGA-II 进行环境选择。

这一阶段只搜索 `z_g + z_b`，不开放 residual。

### 阶段 2：稀疏残差精修

当满足以下条件之一时进入第二阶段：

- `FE / FEmax >= 0.6`；
- 或连续 `T` 代无明显改进。

此时：

1. 取当前精英集 `E`（如第一非支配前沿，必要时按 crowding 截断）；
2. 计算每个 block 的方差分数：

$
s_b = \frac{1}{|I_b|}\sum_{i \in I_b} \mathrm{Var}_{x \in E}(x_i);
$

3. 选出 top-`Q` 个活跃 block；
4. 在这些 block 内，按坐标方差选出 top-`K_active` 个 active coordinates，得到 `A`；
5. 打开 residual 变量 `\alpha`，继续搜索

$
y = [z_g, z_1, \dots, z_B, \alpha].
$

### 残差约束

为避免 residual 破坏 coarse generator，使用 trust-region：

$
|\alpha_j| \le \rho_j, \qquad \rho_j = \kappa \cdot IQR_j(E),
$

其中 `\kappa = 0.5` 为默认值。

### 输出

输出最终非支配解集，以及对应的 reduced vectors。

---

## 10. 为什么这个版本是“最小可用”

因为它只做了四件必要的事：

1. **共享坐标生成器**：把高维解写成逐变量共享生成；
2. **global + block latent**：让搜索对象远小于 `D`；
3. **冻结 decoder**：保证 `z` 才是真正被搜索的压缩对象；
4. **sparse residual**：保留后期精修能力。

它没有引入：

- context decoder；
- archive encoder；
- 在线训练；
- 多尺度 latent 层级；
- SIREN 主干；
- 复杂 controller。

因此，这个版本最适合当作论文中的**基准主方法**：核心逻辑单一、实现代价低、消融边界清楚。

---

## 11. 后续实验建议

## 11.1 必做比较方法

1. **原空间基线**：直接在 `R^D` 上运行 NSGA-II / MOEA/D；
2. **线性低维解码器基线**：

$
x = Bz + M_A\alpha;
$

3. **NNDREAMO-style 对照**：保持共享 decoder 思想，但直接搜索 decoder 参数而不是搜索 latent；
4. **代表性 LSMOP 算法**：选择 benchmark 论文中常见的代表算法进行比较。LMF 论文本身就在 LMF1-LMF12 上使用 IGD/HV，并比较了 MOEA/DVA、WOF、LSMOF、LMOCSO、DGEA 等代表性 LMOEA；这类 benchmark-paper baselines 很适合直接沿用。citeturn8view1

## 11.2 必做消融

1. `Fourier features` on / off；
2. `global + block latent` vs `global-only latent`；
3. `residual` on / off；
4. `FiLM modulation` vs `latent 直接拼接输入`；
5. `冻结 decoder` vs `慢更新 decoder`；
6. block 数 `B` 的敏感性分析。

## 11.3 必做压力测试

1. **变量置乱实验**：随机打乱变量顺序，验证方法对坐标语义的依赖程度；
2. **无额外结构特征** vs **加入结构特征**；
3. **不同维度 D** 下的扩展性测试；
4. **不同 FE 预算** 下的稳健性测试。

## 11.4 后续增强方向

以下内容不应放入 MVP，但可以作为第二阶段工作：

1. 增加 fine-block latent `z_f`；
2. 将 context 仅作为 search controller，而不是 decoder 输入；
3. 在存在大量相关历史任务时，尝试离线预训练 decoder；
4. 将 SIREN 作为增强版或消融版主干。SIREN 对高频细节表达很强，但 MVP 默认仍以 Fourier-MLP 为主，更稳、更简洁。citeturn3search2turn0search1

---

## 12. 一句话结论

对于 LSMOP、LMF、GLSMOP 这类具有变量顺序 / 分组 / linkage 结构的大规模连续多目标优化问题，**最干净、最清晰、最学术的最小可用版本**不是“context-heavy 的训练模型”，而是：

> **冻结的共享坐标解码器 + global/block latent 低维搜索 + sparse residual 两阶段精修。**

这条主线真正的核心创新只有一个：

> **把 NNDREAMO 中“搜索共享生成器参数”的思路，再压缩为“只搜索低维 latent”，并保留 coarse-to-fine refinement。** fileciteturn8file1 citeturn3search0

---

## 参考来源（用于论文写作时展开）

1. NNDREAMO / NNDREA：*Neural Network-Based Dimensionality Reduction for Large-Scale Binary Optimization With Millions of Variables*。
2. LSMOP：*Test Problems for Large-Scale Multiobjective and Many-Objective Optimization*。
3. LMF：*Evolutionary Large-Scale Multiobjective Optimization: Benchmarks and Algorithms*。
4. GLSMOP：*Test Suites and Performance of Algorithms in Large-Scale Multiobjective Evolutionary Optimization*。
5. Fourier Features：*Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*。
6. FiLM：*FiLM: Visual Reasoning with a General Conditioning Layer*。
7. REMBO：*Bayesian Optimization in a Billion Dimensions via Random Embeddings*。
8. D'OH：*Decoder-Only Random Hypernetworks for Implicit Neural Representations*。

---

你下一步最值得做的，是把这份 MVP 直接落成“方法节 + 伪代码 + PlatEMO 接口定义”。
