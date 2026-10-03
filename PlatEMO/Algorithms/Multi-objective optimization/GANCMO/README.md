# GANCMO

本轮双IGD实验已结束，新增三组各10次完整搜索，均未满足两题最终IGD与过程IGD同时优于纯DE。10代组和角度坐标组在7题出现严重种子退化；最近父代名额替换改善过程面积，但8题最终IGD变差。0.002窄配对组因长期未训练且G=0停止。见[精简实验报告](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_IGD_20260929/REPORT.md)。
工作算法已恢复本轮开始时的M20源码，并与其冻结清单逐文件一致；所有实验源码、完整结果及中止记录保留。未将新机制或数值提升为已达标默认，未完成四题统一验收。

直接 CGAN 子代研究版本，算法名保持 GANCMO。当前主线只允许 K=1，`GANCMOSettings` 拒绝其他 K。
条件统一为`[w/norm(w), s]`：参考方向严格M维，真实侧标签s严格为一维0/1，总计M+1维。
工作源码不再包含joint、onehot、带符号方向或额外方向缩放分支；历史实验仍使用各自冻结源码。
本轮M维实验仅验证7/8，N100、D30、200000FE、五配对种子、一个10进程池，各worker单计算线程。
[M维实验报告](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_MCondition_20260929/REPORT.md)已完成四组共40次完整搜索，选定两题共用研究配置`m_unit192_exp_u1500_t20_v1`：
G为192单层sine、D为16单层LeakyReLU；初训4000、续训固定1500次G更新，间隔20代或数据变化指标≥0.2，D:G=2:1；每训练事件学习率从1e-3指数降至1e-6，保留网络和Adam状态。
本轮M=2时条件3维、总参数9423。7/8最终IGD较DE下降2.01%/1.09%，十次运行后25%每代均有实际G；累计真实方向均值46.4/52.6（分母100），存档全新方向仍为0，离边尾部及全过程IGD加速未达成。
研究运行需显式加载[选定设置](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_MCondition_20260929/m_unit192_exp_u1500_t20_v1.json)，并使用对应[冻结源码清单](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_K1_20260929/cases/m_unit192_exp_u1500_t20_v1/config.json)。
工作算法源码与该冻结源码一致；数值设置未提升为通用默认，不宣称全部目标或四题统一验收通过。40代同结构对照虽省约37%耗时，但7题一个种子明显退化，不采纳。

此前精简实验完成九组各10次完整搜索；旧研究配置`simple_linear_noise002_d2_v1`采用joint条件、线性G、D16，
初训4000、最低8/4对、G:D=1:2、噪声0.02、配对目标间距0.007、候选距F上限1e-4。
其10149参数、最终IGD较DE下降6.08%/5.19%等结果来自G共享进入P1/P2的旧版本，仅作为历史记录。
本轮已重新运行P1-only joint对照，不能把上述旧性能归给当前代码。

[精简实验完整报告](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_Simple_20260929/REPORT.md)包含六个问题的证据、训练细节、全部实际G及最差点、真实覆盖、后期使用和IGD曲线。
[历史研究设置](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_Simple_20260929/simple_linear_noise002_d2_v1.json)须配合对应[冻结源码清单](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_K1_20260929/cases/simple_linear_noise002_d2_v1/config.json)复现。
工作源码已统一objective_gap配对，并移除G末尾tanh、采样时裁剪变量范围；历史tanh结果必须使用各自冻结源码。
2026-09-29后续调整：CGAN直接子代仅参与P1环境选择；DE1、DE2仍共享给两边。
上述IGD及覆盖结果来自调整前的冻结源码，不代表这一环境选择改动后的性能。

## 历史四题候选

较早四题统一实验仍没有同时达到贴边、充分输出、补缺和四题IGD要求的配置；`affine_fixed_narrow`是当时的窄带研究候选。
[历史报告](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_K1_20260929/REPORT.md)与[历史设置](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/GANCMO_K1_20260929/final_k1_comparison_v1/candidate_settings.json)保留原结果。下表仅描述该冻结tanh版本。

四题均为 N=100、D=30、maxFE=200000、seed=1:5；一次使用10个进程worker，各设置单计算线程。
5/6 与 7/8 的配置、源码及运行支持哈希完全一致。

| 项目 | 研究候选设置 |
|---|---|
| 生成器 / critic | 无隐藏层仿射层+tanh / 64×64 |
| 条件 / 损失 | joint；条件WGAN-GP，加同方向同真实侧最近端点损失，权重1000 |
| 首训 / 续训 | 8000次G更新 / round(100+400×进度)次G更新，每G更新1次critic |
| 频率 / 学习率 | 间隔10代或变化0.2；每次前3/4为0.001，末1/4为0.00001 |
| 数据门槛 | 至少1对；训练目标间距≤0.01；objective_gap活跃优先级 |
| 候选 | 查询500，噪声0.1，归一化决策距离F≤0.0005，每个支持F目标方向最多1个 |
| 方向 / 批次 | 固定原点0、跨度1；每次平衡批次覆盖两侧全部条件方向 |

7/8后25%每代均有实际G，真实累计覆盖分别为6–20和12–22个方向，旧窄带K1仅1–2个。
5/6仍稀疏，6的全部五次后25%无G；四题均未补足当前P1/P2/完整存档联合空缺方向。
最终IGD相对DE：5降低12.70%，6降低3.64%，7升高4.17%，8降低0.76%。过程IGD仅5改善。
这些结果不构成四题全部达标或统计显著性的证明，少量输出也不算贴边成功。

## 主线与资格

P1以普通DE和CGAN直接子代混合，计划G比例按扣除初始化的FE进度从10%线性增至90%；
P2始终普通DE。候选不足按原方式补DE，生成点直接送评并占用P1名额，所有真实评价计入FE。
不存在DE桥接、旧BoundaryStep收缩、边界投影或免费重采。
环境选择为P1从`[P1,DE1,G,DE2]`按约束选优，P2从`[P2,DE1,DE2]`仅按目标选优。
G在P1存活后可参与下一代DE1繁殖；这些DE子代仍可进入P2，未按祖先来源隔离。

F必须属于完整可行非支配前沿；I不得被该可行前沿支配。存档使用选择前的完整联合池，
加入历史端点，持续更新、收紧及删除失格对；反馈包括全部已送评G，不只可行或存活G。
每方向仅一个活跃配对，存档总容量500。窄训练门槛不改变宽对的原资格。

训练使用两侧真实标签，查询仅请求s=1，请求标签不作为真实可行标签。
训练流、查询流独立持续推进，结束后恢复搜索流。真实分界、CPF和IGD只用于离线分析。

## 设置与记录

- `generatorHidden=[]` 在当前工作源码中使用学习到的仿射映射，采样时裁剪到变量范围，仍输入噪声与条件；没有直接设置为真实F或插入端点。历史冻结版本含tanh。
- `conditionEncoding` 固定unit、`conditionScale`固定1，其他编码或缩放会报错。方向单位化保持原射线，最后一位标签不变；M维方向不等于完整目标点的径向位置。
- `generatorActivation`支持普通LeakyReLU与sine；本轮选定sine192研究配置的`generatorFrequency=256`仅设置首层方向权重的随机初始化尺度，全部权重仍参与梯度学习，不使用方向查表或端点系数注入。
- `endpointWeight` 加入同方向、同真实侧最近端点距离损失，不调用目标或约束。
- `maxTrainingObjectiveGap` 限制真实活跃对的目标间距；`maxCandidateDistanceF` 限制候选到同方向真实F的归一化决策距离。
- `candidateTargetCap` 默认Inf；设为1时按已评价支持F的固定目标方向保留最近的原始候选，真实G方向须评价后统计。
- `fullDirectionBatch` 默认false；开启后每个批次至少覆盖两侧全部条件方向，不改变真实标签及平衡抽样原则。
- `referenceScaling` 默认adaptive；objective使用固定原点0、跨度1，统一存档匹配、K1激活和训练/查询方向。
  资格公式保持，但方向关联和邻域匹配会改变。当前固定坐标仅验证四个同量纲问题，不自动推广到任意单位或平移。
- `learningRateSchedule`的cold_tail使用总更新进度，event_tail在每次训练前3/4恢复基本率、末1/4细化；event_exp在每次训练中指数下降至基本率乘learningRateDecay。训练步数均按G更新计，D更新由criticSteps决定。
- `activePairPriority` 支持quality、gap、objective_gap。当前objective_gap实验将新配对、端点收紧、活跃保留统一为目标间距优先，距离相同时参考决策间距；不改变端点Pareto资格、5方向邻域及K1。旧冻结版本仅在活跃保留时使用目标间距，复现必须区分源码版本。
- `minimumPairs` 与 `minimumPairsContinue` 分别控制初始化与续训；没有合格数据时不虚构配对或继续输出。

`RecordTrace`记录所有真实G、原始查询/合格/方向去重后的候选数、真实目标方向、生成前P1/P2及完整存档计数、
P1/P2存活标记、当代空缺和历史访问状态。覆盖同时记录当前尺度及固定尺度，观察不消耗搜索随机数和FE。
工作源码按SOLUTION对象身份统计当代G存活，避免把决策向量相同的独立DE子代误记为G。
末批有G与终代有G分开报告。[指标口径](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_K1_20260929/METRICS.md)说明距离近似、覆盖分母和资格审计范围。

## 验证与复现

历史四题实验新增50次完整K1搜索；最新7/8精简实验另新增90次，复用既有DE与affine对照。新数据在`Data/GANCMO_K1_20260929`，
每个case保留设置、源码、支持脚本、输入/输出哈希及原始轨迹。新源码不回写旧冻结配置。
[运行器](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_K1_20260929/run_case.py)先核对哈希；已有结果不得覆盖。

相关检查覆盖直接送评与FE、K=1限制、端点学习、固定方向、原资格删除、候选方向筛选及记录开关逐值回放。
`Tests/test_GANCMO_p1_only.m`验证G可在P1存活、无法直接进入P2及末批预算；
[本次变更与检查](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_Simple_20260929/P1_ONLY_CHANGE.md)区分旧性能结果与新选择规则。
实际性能依据完整实验，接口检查不替代边界或IGD验收。`test.m`及GUI既有无关改动均保留。
当前状态及验证日志见[STATE.md](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Experiments/GANCMO_K1_20260929/STATE.md)，本轮修改前源码位于`Experiments/GANCMO_K1_20260929/source_before`。

参考：Tian等，A Coevolutionary Framework for Constrained Multiobjective Optimization Problems，IEEE TEVC 25(1):102–116，2021；
Gulrajani等，Improved Training of Wasserstein GANs，NeurIPS 2017。PlatEMO原版权及引用要求继续适用。
