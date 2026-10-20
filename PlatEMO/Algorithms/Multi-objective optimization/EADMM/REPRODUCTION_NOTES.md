# EADMM 复现说明

本目录实现 Li 等人论文 *Evolutionary Alternating Direction Method of Multipliers for Constrained Multi-Objective Optimization with Unknown Constraints*（IEEE TEVC，DOI `10.1109/TEVC.2024.3425629`）报告的三个实例：

- `EADMMNSGAII.m`：EADMM/NSGA-II；
- `EADMMIBEA.m`：EADMM/IBEA；
- `EADMMMOEAD.m`：EADMM/MOEA-D。

论文把 EADMM 定义为可嵌入不同骨架的框架，而不是一个唯一环境选择器，因此不设置会静默选择骨架的统一入口。

## 论文与补充材料明确规定并已实现的部分

- 维护两个大小均为 `N` 的独立初始种群：`P` 求解原约束问题，`Pbar` 忽略约束求解对应无约束问题。
- NSGA-II 实例的 `P` 使用论文修改的约束支配：可行优于不可行；不可行解中违反约束数量较少者优先；违反数量相同时按目标 Pareto 支配；`Pbar` 使用忽略约束的 vanilla NSGA-II。二元锦标赛的 `k=2`。
- IBEA 实例对全部可行父代和可行来访解建立可行池，池内执行IBEA；不足N时才从旧不可行父代补齐。若本次没有可行来访解，全部父代原样保留。辅助种群使用 vanilla IBEA，`kappa=0.05`。
- MOEA/D 实例使用论文给出的四条替换规则和 modified Tchebycheff；两个种群共享权重与邻域，辅助种群使用 vanilla MOEA/D。论文邻域大小为 `T=N/10`。保持实际人口规模所采用的MUD生成器及小种群边界处理属于本地实现约定，详见下文。
- SBX 参数为 `pc=1, eta_c=20`，多项式变异为 `pm=1/D, eta_m=20`，直接复用 PlatEMO `OperatorGA`/`OperatorGAhalf`。
- Module 3 按五个连续步骤执行：交叉更新两个种群；从主任务后代建立临时档案；围绕每个档案解求解差异子问题；将每个局部搜索结果反馈给两个种群。
- 临时档案条件中的两个存在量词分别计算：主任务后代只需支配 `P` 中至少一个解，并支配 `Pbar` 中至少一个解；两个被支配解无需相同。
- `rho=(gamma/|P|)*ell`，其中 `gamma` 是 `P` 中可行解数，`ell` 是问题接口公开的二值约束信号数；聚合约束问题只公开一个总体 0/1 标签，因此 `ell=1`。
- 补充材料 Appendix B 已恢复于 [COLA Laboratory](https://colalab.ai/supplementary/supp_admm.pdf)。其中还给出 31 次独立运行，以及各 UC 问题的 `N/maxFE`：`m=2/3` 时 `N=100`，`m=5/10` 时 `N=150`；每个具体问题的 `maxFE` 见其 Table I；RL 为 `N=50,maxFE=2500`，WDS 为 `N=50,maxFE=1500`。
- 截至 2026-10-19，按完整标题、DOI、算法名和作者检索作者实验室、GitHub、Zenodo、Figshare、OSF 及 PlatEMO，未发现作者公开代码；补充文档是目前找到的最高优先级附加材料。
- 2026-10-20补查8个相关仓库的默认分支文件树，仍未找到目标EADMM实现。取得的作者[OptimalStoping_NeurIPS2023代码](https://github.com/ShuangLiOptimize/OptimalStoping_NeurIPS2023)含相同月球着陆控制器，但只有奖励目标，其步数/episode聚合也不同；[实验室CBOB代码](https://github.com/COLA-Laboratory/CBOB)的目标、约束和控制器均存在差异，不能替代EADMM评价器。EADMM补充材料明确最大500步、第二目标为墙钟耗时。详细对照在实验目录`evidence/rl_companion_code_comparison.json`。

## 原文公式冲突及实现采用的勘误

论文式 (13) 印成

`argmin sum_j 1(g_j(x)==0) + rho*||x-xhat||^2`。

论文同时定义 `g_j(x)==0` 表示满足约束。若照印刷公式最小化，就会主动减少满足的约束数，与该节“寻找可行解”、Remark 11/12 及 `rho` 的解释方向相反。因此实现采用与全文算法意图一致的勘误形式：

`sum_j 1(g_j(x) is violated) + rho*||x-xhat||^2`，

在 PlatEMO 中即 `sum(~(cons<=0))`。这是对原文内部矛盾的必要修正；不能把本实现称为不含勘误的公式字面复刻。

## 论文未公开、实现必须补齐的部分

1. 论文只说明式 (13) 使用 MATLAB R2021a 的 Genetic Algorithm (`ga`)，正文和补充材料都没有给覆盖选项。正文第7页复杂度分析明确定义了局部GA种群规模tilde N，但未给数值，因此按用户“仅调论文出现但未报告取值的参数”要求，将它公开为`localN`；0保留此前推定默认（D<=5为50，否则200）。其他选项仍固定为`MaxGenerations=100*D`、5%精英、随机均匀创建、stochastic-uniform选择、scattered crossover、adaptive-feasible mutation及对应停止阈值，不能一并作为自由参数。无法证明作者私有实现没有覆盖这些默认值。
   已补核对正文第6页及公开TeX：式(13)的搜索域是完整Omega，中心通过平方距离惩罚进入目标；Remark 10--12未说明把档案中心注入GA初始种群，也未给局部边界半径。因此不能仅凭“around the center”把增加初始中心或缩小边界视为论文明示的修正。当前随机初始化仍是未获作者代码确认的固定约定，检查证据见实验目录`evidence/eadmm_local_initialization_source_check.json`。
2. 式 (13) 的适应度只查询约束标签和距离。实现通过公开的 `Problem.CalCon` 完成 GA 内部约束查询，不把这些查询计为外层多目标目标函数 FE；每个中心完成完整 GA 后，仅将最终 `xcheck` 直接交给 `Problem.Evaluation`，由平台在内部执行一次 `CalDec` 并计入一次 FE。这避免一次局部搜索耗尽论文 RL/WDS 的全部外层预算，也符合目标函数只在 Step 5 环境选择时才需要的流程，但论文没有明示其计数器口径。
3. 原论文实验需要逐约束二值向量，且合成问题包含补充材料式(40)的噪声列。算法可直接接收这些列。总体单标签 `_BC` 问题只属于另行适配实验，不能代替原论文 UC 问题；不得反推或补造其隐藏约束。
4. 原文没有公开作者源码；数组方向、并列解顺序、随机数消耗顺序及两个初始种群的独立采样按 PlatEMO 约定实现，不保证与作者私有代码逐随机数一致。

## 骨架特有但论文未展开的解释

1. 2026-10-20 修正了 IBEA 把不可行父代加入指标竞争、从而淘汰唯一可行后代的问题。可行池不足N时的旧不可行父代补齐仍属论文未明确的约定；没有可行新解的例外保持原文要求。
2. MOEA/D 中每个 `Q(k)`、`Qbar(k)` 保留产生它的第 `k` 个子问题索引。Module 3 的交叉更新作用于 `B(k,:)`，局部解继承其档案中心的索引。若更新全种群会改变论文给出的 `O(mTN)` 复杂度；若重新匹配最近权重则会新增论文没有的启发式。
3. 约束 MOEA/D 按论文公式对每个候选使用当前 `P union {candidate}` 计算理想点；无约束侧沿用 vanilla MOEA/D 的历史理想点。权重零分量截为 `1e-6` 后重新归一化，避免 modified Tchebycheff 除零。
4. PlatEMO 默认 NBI 均匀点可能把正式请求的 `N=100/150` 改为 91、126 或 110，与补充材料声明的实际人口规模冲突。本实现改用平台已有的 MUD 生成器，精确生成 `N` 个单纯形权重。作者未公开权重生成器，因此 MUD 是满足补充材料字面人口规模的必要复现选择，不宣称是作者私有代码的已知实现。
5. 正文统一报告 SBX/多项式变异参数，却没有在 MOEA/D 小节再次指定繁殖算子；实现采用 `OperatorGAhalf(...,{1,20,1,20})`，不擅自改用 DE。
6. `localN`是NSGA-II、MOEA/D入口的第1个参数，IBEA入口的第2个参数（第1个kappa仍为0.05）；不公开GA代数或其他停止参数。`maxFE`必须至少容纳两个大小为N的初始种群。每代生成两个完整的N后代批次，并为至多N个临时档案中心各生成一个局部解，一代新增`2N+|Archive|`、最多3N次外层FE。实现完整执行已经开始的一代，因此实际FE最多超过maxFE达`3N-1`；论文未公开末代预算处理方式。补充材料“不引入额外参数”的表述不否定正文明确出现的GA固有种群规模。
7. IBEA 指标计算逐式沿用 PlatEMO 的 vanilla `CalFitness`，不额外加入零目标范围或零指标尺度保护；退化种群的数值行为也保持骨架一致。

## 论文结果复现

完整核验范围、参数限制与独立种子见 `Experiments/PaperReproduction_20261020/PROTOCOL.md`。当前GA计数口径、权重生成等尚未得到作者源码验证，不因结果接近就宣称这些选择与作者实现相同。
