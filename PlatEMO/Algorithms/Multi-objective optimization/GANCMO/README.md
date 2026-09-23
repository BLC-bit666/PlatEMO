# GANCMO

唯一主线为 **双种群协同进化 + 真实边界配对 + 条件 WGAN 引导 + coverage α=0.04 资源控制**。直接运行 `GANCMO` 即使用该方案，无候选机制、消融模式或实验参数切换。

## 完整流程

1. 初始化 P1/P2 和参考向量。P1 考虑约束，P2 忽略约束；用真实评价结果构建初始 F/I 配对存档。
2. 每代开始时，使用当前 P1/P2、存档和资源状态判断 CGAN 是否仍处于 active period，并检查训练触发条件。
3. 未停止时首训/续训或复用 WGAN-GP，生成500个未评价候选；盒内候选按到真实 F 的距离选取，另给未覆盖且盒外方向最多 `floor(.2*Q)` 个探索名额，每方向最多1点。缺额先退回盒内，再由普通DE补齐，得到本代 q。
4. 随后两群分别生成 25% GA、75% DE 子代。P1 的部分 DE 槽立即消费本代 q，按决策距离贪心匹配；P2 只使用普通 GA/DE。
5. 对 P1 引导槽，先作中点桥接并执行原 DE 交叉、修复与变异；再由 q 定位最近真实 F/I 线段，将子代收缩到该线段中点周围。只评价最终子代，q 不直接评价或进入种群。
6. P1/P2 各自从自身与两群子代中作环境选择；用选择前完整联合种群更新真实端点存档，保留最多500对，每方向一个 active 对。
7. 由前沿身份差异 S 与参考方向覆盖率 C 更新单一状态 z，得到下一代名额 Q 与停止比例 tau。若下一完整批次越过截止 FE，永久停用 CGAN，继续普通双种群搜索至总预算耗尽。

## 统一控制公式

同代 P1/P2 按决策去重后，U 是忽略约束的第一非支配前沿，F 是可行第一非支配前沿。交并集以个体身份为准；方向在存档使用的同一目标尺度下分配。

```text
S = 1 - |U∩F| / |U∪F|
C = min(U覆盖方向数, F覆盖方向数) / |W|
z0 = 20
z ← (1 - .04*C)*z + .04*C*(10 + 20*S)     # C=0时保持
Q = clip(round(z), 10, 30)
tau = clip(.2 + .5*(z-10)/20, .2, .7)
cutoffFE = floor(tau*maxFE/(2N) + 1e-10)*(2N)
停止条件：当前FE + 2N > cutoffFE，且永久锁存
```

同一状态分配生成名额与可用时间，无额外步幅参数；名义 Q 单代最多变化1。初始 Q=20 对应 tau=.45。实际引导数可能低于 Q；“五分之一”是未覆盖且盒外候选的计划上限，不是实际占比保证。

## 已确定的内部设置

| 模块 | 设置 |
| --- | --- |
| 存档 | 上限500对；每方向1个 active 对；五方向邻域；完整可行第一前沿；I不得被该前沿支配 |
| 网络 | zDim=12，G=[8,8]，critic=[16,16]，条件[w,s]，G使用tanh输出 |
| WGAN-GP | GP=10，critic:G=5:1，Adam lr=.001、β=(0,.9)，高斯σ=.1 |
| 训练 | 至少8个 active 对；首训1000/续训20次G更新；batch≤64且为偶数 |
| 续训触发 | 端点/条件变化≥.2，或距上次训练≥10代；不足8对时保留已有模型 |
| 筛选 | 500个候选、侧别各250；去重距离1e-6；合法性/配对改善容差1e-12 |

这些是既定设计常数，不是新增的公开调参接口；不得把整体算法称为无参数。具体端点变化度与归一化见 `GANCMOModel.m`，执行契约见根目录 `Agent.md`。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `GANCMO.m` | 唯一入口与代际流程，负责永久停用 |
| `GANCMOControl.m` | 单一资源状态、名额与截止FE |
| `GANCMOArchive.m` | 真实端点收紧、重配对、容量竞争和训练数据 |
| `GANCMOModel.m` | 固定条件 WGAN-GP 的训练、续训、采样与数值失败回退 |
| `GANCMOCandidates.m` | 500个条件查询及盒内/盒外筛选 |
| `GANCMOOffspring.m` | 原随机流下的GA/DE及q的一一消费 |
| `GANCMOBoundaryStep.m` | q定位的真实F/I局部收缩 |
| `GANCMOFitness.m` | SPEA2强度与密度适应度 |
| `GANCMOSelection.m` | 双种群约束/无约束环境选择 |
| `GANCMOReference.m` | 一致目标尺度下的参考方向分配 |

## 运行

从 PlatEMO 根目录调用标准入口：

```matlab
platemo('algorithm',@GANCMO,'problem',@LIRCMOP5_BC, ...
        'N',100,'maxFE',200000,'save',1);
```

若需固定随机种子，使用直接实例化方式；`platemo` 会自行执行 `rng('shuffle')`：

```matlab
addpath(genpath('Algorithms'),genpath('Problems'),genpath('Metrics'));
rng(1,'twister');
P = LIRCMOP5_BC('N',100,'maxFE',200000,'maxRuntime',Inf);
A = GANCMO('save',1,'outputFcn',@(varargin)[]);
A.Solve(P);
```

依赖 PlatEMO、Deep Learning Toolbox 和 Statistics and Machine Learning Toolbox（`pdist2`）。此次数值等价性核验环境为 MATLAB R2025b、CPU单计算线程。原训练随机状态复制行为被显式保留；无需运行研究诊断。不要把 `Data` / `Deliverables` 的冻结源码加入正常运行路径。

## 整理记录

淘汰机制、旧 CBS 入口、参数扫描器与旧研究测试已从活动算法目录删除；实验数据及整理前源码保留。7个基准问题中仅删除失去用途的计时钩子，目标/约束公式未改。根目录 `test.m` 已同步新入口并隔离旧结果缓存，本轮未启动该批量实验。

验证、耗时与完整删除清单见 [本轮报告](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideFinalMainline_20260919/REPORT.md>)；选型证据见 [既有实验报告](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideMinimalFull_20260919/REPORT.md>)。结果只支持已核验条件，不将重构包装成新的算法收益。

参考：Tian 等，A Coevolutionary Framework for Constrained Multiobjective Optimization Problems，IEEE TEVC 25(1):102–116，2021；Gulrajani 等，Improved Training of Wasserstein GANs，NeurIPS 2017。PlatEMO 原代码版权及引用要求继续适用。
