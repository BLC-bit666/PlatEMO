# DRMCMO：原GA与DE分支均保留

统一入口参数 `type`：1为GA（默认），2为DE。已有算子机制保持原样，仅补充缺少的算子模式；不将原作者设计的GA/DE一律替换成通用算子。

type=1保留原 `OperatorGA` 及其父代配对；type=2保留原DE父代选择，以及以相同随机调用在默认参数和 `{0.5,0.5,0.5,0.75}` 之间切换的机制。没有将原DE改成单一默认参数。

新增模式只调用 `Algorithms/Utility functions` 原生算子，不覆盖默认参数；原生函数路径在入口核对。原算法自带算子的路径和实现保留。实数GA为SBX加多项式变异；`OperatorGAhalf`每对父代产生一个子代，用于保留原子代数量。

```matlab
A = DRMCMO();                 % 默认type=1
A = DRMCMO('parameter',{1}); % GA模式
A = DRMCMO('parameter',{2}); % DE模式
```

原源码默认GA；历史test.m曾明确传入type=2，对应原DE分支。当前test.m改为按算法逐行配置type，不再单独强制DRMCMO使用DE。 统一参数接口不意味着比较原算法时所有算法都应采用同一type。扩展模式的结果应单独标注，不能混入原模式的已有结果。

本次仅扩展算子入口，种群、环境选择、约束处理、存档等其余机制保留。原有整代FE终止方式保持不变，不额外统一各算法的每代评价量。修订与验证见[记录](../../../Experiments/Comparators_PreserveOperators_20261031/REPORT.md)。
