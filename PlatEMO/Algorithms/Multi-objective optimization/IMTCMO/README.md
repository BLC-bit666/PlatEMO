# IMTCMO：原DE保留，新增GA

统一入口参数 `type`：1为GA（默认），2为DE。已有算子机制保持原样，仅补充缺少的算子模式；不将原作者设计的GA/DE一律替换成通用算子。

type=2完整保留原DE组合：每个种群一半调用 `OperatorDE_rand_1`，另一半调用 `OperatorDE_pbest_1_main`，包含原pbest选择、F/CR采样与变异。type=1仅新增通用目录原生默认 `OperatorGAhalf`，使用已有邻居配对的Mate1/Mate2，每个种群仍产生N个子代。

新增模式只调用 `Algorithms/Utility functions` 原生算子，不覆盖默认参数；原生函数路径在入口核对。原算法自带算子的路径和实现保留。实数GA为SBX加多项式变异；`OperatorGAhalf`每对父代产生一个子代，用于保留原子代数量。

```matlab
A = IMTCMO();                 % 默认type=1
A = IMTCMO('parameter',{1}); % GA模式
A = IMTCMO('parameter',{2}); % DE模式
```

原IMTCMO对应type=2；type=1应标为IMTCMO的GA扩展版。 统一参数接口不意味着比较原算法时所有算法都应采用同一type。扩展模式的结果应单独标注，不能混入原模式的已有结果。

本次仅扩展算子入口，种群、环境选择、约束处理、存档等其余机制保留。原有整代FE终止方式保持不变，不额外统一各算法的每代评价量。修订与验证见[记录](../../../Experiments/Comparators_PreserveOperators_20261031/REPORT.md)。
