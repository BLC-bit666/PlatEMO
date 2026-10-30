# MTCMO：原GA保留，新增DE

统一入口参数 `type`：1为GA（默认），2为DE。已有算子机制保持原样，仅补充缺少的算子模式；不将原作者设计的GA/DE一律替换成通用算子。

type=1保留原二元锦标赛与 `OperatorGAhalf`。type=2新增通用目录原生默认 `OperatorDE`：当前种群前半作为基向量，原锦标赛父代分成两组提供差分向量。每个种群仍产生floor(N/2)个子代，环境选择与任务交换不变。

新增模式只调用 `Algorithms/Utility functions` 原生算子，不覆盖默认参数；原生函数路径在入口核对。原算法自带算子的路径和实现保留。实数GA为SBX加多项式变异；`OperatorGAhalf`每对父代产生一个子代，用于保留原子代数量。

```matlab
A = MTCMO();                 % 默认type=1
A = MTCMO('parameter',{1}); % GA模式
A = MTCMO('parameter',{2}); % DE模式
```

原MTCMO对应type=1；type=2应标为MTCMO的DE扩展版。 统一参数接口不意味着比较原算法时所有算法都应采用同一type。扩展模式的结果应单独标注，不能混入原模式的已有结果。

本次仅扩展算子入口，种群、环境选择、约束处理、存档等其余机制保留。原有整代FE终止方式保持不变，不额外统一各算法的每代评价量。修订与验证见[记录](../../../Experiments/Comparators_PreserveOperators_20261031/REPORT.md)。
