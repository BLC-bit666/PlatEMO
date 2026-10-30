# NCTP 原作者 MATLAB 源码

- 原始论文：J.-P. Li, Y. Wang, S. Yang, Z. Cai. A Comparative Study of Constraint-Handling Techniques in Evolutionary Constrained Multiobjective Optimization. CEC 2016, 4175–4182. DOI: [10.1109/CEC.2016.7744320](https://doi.org/10.1109/CEC.2016.7744320)。
- [作者实验室论文与代码页](https://intleo.csu.edu.cn/publication.html)。
- [原始代码 ZIP](https://intleo.csu.edu.cn/codes/A-comparative-study-of-CHTs-in-CMOPs.zip)。
- 本地原包：`original.zip`；完整性及文件校验记录：`SOURCE_MANIFEST.json`。

原包保留 ATM、CDP、SP 三种方法的实验程序、108 个 `.m` 文件、参考点文本和作者自带的备份文件。这些文件属于同一套 NCTP 的实验材料，不代表 108 个测试问题。

优先从下面的原始定义阅读整套问题：

| 原目录（均在 `private/original/A comparative study of CHMs in CMOPs/` 下） | 基准范围 | 关键文件 |
| --- | --- | --- |
| `CDP(type-I)` | NCTP1–6 | `fit_cv.m`：目标与约束；`main.m`：变量范围、维数和编号 |
| `CDP(type-II)` | NCTP7–12 | 同上 |
| `CDP(type-III)` | NCTP13–18 | 同上 |

各组 `main.m` 中的 12 个 case 对应该组 6 个问题的 30 维与 10 维配置。问题通过 `fit_cv(P,problem)` 计算，没有独立的 `NCTP1.m` 至 `NCTP18.m` 类文件。

原代码保持不变。2026-10-28 已另行提供当前 PlatEMO 的 18 个常规类与 18 个 `_BC` 类，见 [适配说明](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Problems/Multi-objective optimization/NCTP/README.md>)。适配保留作者目标、边界和约束，稳定化 g=0 目标表达，并将批次约束归一化留在算法层之外。第三方 Python 移植与作者包的部分参数不一致，本目录及新适配均以作者下载包为来源。
