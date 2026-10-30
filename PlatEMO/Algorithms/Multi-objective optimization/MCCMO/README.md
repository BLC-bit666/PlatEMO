# MCCMO：原GA与DE分支均保留

统一入口参数 `type`：1为GA（默认），2为DE。已有算子机制保持原样，仅补充缺少的算子模式；不将原作者设计的GA/DE一律替换成通用算子。

type=1选择原GA分支（`OperatorGAhalf` / `OperatorGA`），type=2选择原DE分支。原父代、算子参数、子代数量及各分支的控制流程保留；仅将按问题类名自动分支改成显式type选择。

新增模式只调用 `Algorithms/Utility functions` 原生算子，不覆盖默认参数；原生函数路径在入口核对。原算法自带算子的路径和实现保留。实数GA为SBX加多项式变异；`OperatorGAhalf`每对父代产生一个子代，用于保留原子代数量。

```matlab
A = MCCMO();                 % 默认type=1
A = MCCMO('parameter',{1}); % GA模式
A = MCCMO('parameter',{2}); % DE模式
```

复现原按问题切换的配置时：LIRCMOP、DASCMOP、DOC、RWMOP选type=2；其他问题（包括ZXH_CF）选type=1。 统一参数接口不意味着比较原算法时所有算法都应采用同一type。扩展模式的结果应单独标注，不能混入原模式的已有结果。

本次仅扩展算子入口，种群、环境选择、约束处理、存档等其余机制保留。原有整代FE终止方式保持不变，不额外统一各算法的每代评价量。修订与验证见[记录](../../../Experiments/Comparators_PreserveOperators_20261031/REPORT.md)。
