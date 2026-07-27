# EADMM 复现说明

本目录实现 Li 等人论文 *Evolutionary Alternating Direction Method of Multipliers for Constrained Multi-Objective Optimization with Unknown Constraints*（IEEE TEVC，DOI `10.1109/TEVC.2024.3425629`）报告的三个实例：

- `EADMMNSGAII.m`：EADMM/NSGA-II；
- `EADMMIBEA.m`：EADMM/IBEA；
- `EADMMMOEAD.m`：EADMM/MOEA-D。

论文把 EADMM 定义为可嵌入不同骨架的框架，而不是一个唯一环境选择器，因此不设置会静默选择骨架的统一入口。

## 论文与补充材料明确规定并已实现的部分

- 维护两个大小均为 `N` 的独立初始种群：`P` 求解原约束问题，`Pbar` 忽略约束求解对应无约束问题。
- NSGA-II 实例的 `P` 使用论文修改的约束支配：可行优于不可行；不可行解中违反约束数量较少者优先；违反数量相同时按目标 Pareto 支配；`Pbar` 使用忽略约束的 vanilla NSGA-II。二元锦标赛的 `k=2`。
- IBEA 实例只允许可行的来访解进入主种群候选池；若本次没有可行来访解，全部父代原样保留。辅助种群使用 vanilla IBEA，`kappa=0.05`。
- MOEA/D 实例使用论文给出的四条替换规则和 modified Tchebycheff；两个种群共享权重与邻域，辅助种群使用 vanilla MOEA/D。邻域大小为 `T=N/10`；代码对非论文的小种群测试仅设置下限 2。
- SBX 参数为 `pc=1, eta_c=20`，多项式变异为 `pm=1/D, eta_m=20`，直接复用 PlatEMO `OperatorGA`/`OperatorGAhalf`。
- Module 3 按五个连续步骤执行：交叉更新两个种群；从主任务后代建立临时档案；围绕每个档案解求解差异子问题；将每个局部搜索结果反馈给两个种群。
- 临时档案条件中的两个存在量词分别计算：主任务后代只需支配 `P` 中至少一个解，并支配 `Pbar` 中至少一个解；两个被支配解无需相同。
- `rho=(gamma/|P|)*ell`，其中 `gamma` 是 `P` 中可行解数，`ell` 是约束数。
- 补充材料 Appendix B 已恢复于 [COLA Laboratory](https://colalab.ai/supplementary/supp_admm.pdf)。其中还给出 31 次独立运行，以及各 UC 问题的 `N/maxFE`：`m=2/3` 时 `N=100`，`m=5/10` 时 `N=150`；每个具体问题的 `maxFE` 见其 Table I；RL 为 `N=50,maxFE=2500`，WDS 为 `N=50,maxFE=1500`。

## 原文公式冲突及实现采用的勘误

论文式 (13) 印成

`argmin sum_j 1(g_j(x)==0) + rho*||x-xhat||^2`。

论文同时定义 `g_j(x)==0` 表示满足约束。若照印刷公式最小化，就会主动减少满足的约束数，与该节“寻找可行解”、Remark 11/12 及 `rho` 的解释方向相反。因此实现采用与全文算法意图一致的勘误形式：

`sum_j 1(g_j(x) is violated) + rho*||x-xhat||^2`，

在 PlatEMO 中即 `sum(~(cons<=0))`。这是对原文内部矛盾的必要修正；不能把本实现称为不含勘误的公式字面复刻。

## 论文未公开、实现必须补齐的部分

1. 论文只说明式 (13) 使用 MATLAB R2021a 的 Genetic Algorithm (`ga`)，正文和补充材料都没有给覆盖选项。实现按该版本常见默认值显式固定：`PopulationSize` 在 `D<=5` 时为 50、否则为 200，`MaxGenerations=100*D`、5% 精英、随机均匀创建、stochastic-uniform 选择、scattered crossover、adaptive-feasible mutation及对应停止阈值。无法证明作者私有实现没有覆盖这些默认值。
2. 式 (13) 的适应度只查询约束标签和距离。实现通过公开的 `Problem.CalCon` 完成 GA 内部约束查询，不把这些查询计为外层多目标目标函数 FE；每个中心完成完整 GA 后，仅将最终 `xcheck` 通过 `Problem.Evaluation` 一次并计入 FE。这避免一次局部搜索耗尽论文 RL/WDS 的全部外层预算，也符合目标函数只在 Step 5 环境选择时才需要的流程，但论文没有明示其计数器口径。
3. 因此，目标问题必须在 `CalCon` 中提供可单独查询的约束列。每个 `cons` 列视为一个约束且只读取是否满足 `<=0`。若问题把全部约束聚合成单个 0/1 列，算法只能区分总体可行/不可行，无法恢复论文所用的“违反约束数量”；若问题只重写 `Evaluation` 而没有正确重写 `CalCon`，则不满足本算法接口要求。
4. 原文没有公开作者源码；数组方向、并列解顺序、随机数消耗顺序及两个初始种群的独立采样按 PlatEMO 约定实现，不保证与作者私有代码逐随机数一致。

## 骨架特有但论文未展开的解释

1. IBEA 的“只考虑可行解”解释为候选池 `Parents union feasible(Incoming)`。这同时满足固定种群大小、无可行后代时全部父代保留，以及论文所称“仅作轻量修改”。
2. MOEA/D 中每个 `Q(k)`、`Qbar(k)` 保留产生它的第 `k` 个子问题索引。Module 3 的交叉更新作用于 `B(k,:)`，局部解继承其档案中心的索引。若更新全种群会改变论文给出的 `O(mTN)` 复杂度；若重新匹配最近权重则会新增论文没有的启发式。
3. 约束 MOEA/D 按论文公式对每个候选使用当前 `P union {candidate}` 计算理想点；无约束侧沿用 vanilla MOEA/D 的历史理想点。权重由 `UniformPoint` 产生，其零分量按 PlatEMO 约定截为 `1e-6`。
4. `UniformPoint` 可能把请求的 `Problem.N` 调整为可生成的均匀权重数。例如当前 PlatEMO 请求 `N=150` 时，`M=5/10` 分别得到 126/110 个权重。PlatEMO 原生 MOEA/D 也采用这一约定，但它与补充材料表中 `N=150` 的字面规模存在差异；作者未公开其权重生成器，不能无依据改用另一种精确数量方案。
5. 正文统一报告 SBX/多项式变异参数，却没有在 MOEA/D 小节再次指定繁殖算子；实现采用 `OperatorGAhalf(...,{1,20,1,20})`，不擅自改用 DE。
6. 三个正式入口不公开局部 GA 种群或代数参数，因为补充材料明确声明 EADMM 不引入骨架以外的额外参数。`maxFE` 必须至少容纳两个大小为 `N` 的初始种群，否则入口显式拒绝该配置。每代生成两个完整的 `N` 后代批次，并为至多 `N` 个临时档案中心各生成一个局部解，所以一代新增 `2N+|Archive|`、最多 `3N` 次外层 FE。实现遵循 PlatEMO 的代际算法约定：完整执行已经开始的一代，再由下一次 `Algorithm.NotTerminated` 统一终止和保存；因此实际 FE 最多超过 `maxFE` 达 `3N-1`，但不会新增论文未定义的半代，也不会截断论文要求逐一处理的临时档案。论文及补充材料没有公开作者的末代预算处理方式。
7. IBEA 指标计算沿用 PlatEMO 公式，但在某个目标完全相同或指标尺度全零时把相应除数设为 1，避免退化种群产生 `NaN`；非退化情形与 vanilla IBEA 相同。
