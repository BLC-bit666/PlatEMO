# NCTP：当前 PlatEMO 适配

包含 `NCTP1`–`NCTP18`；对应二元版为相邻目录 `NCTP_BC` 中的 `NCTP1_BC`–`NCTP18_BC`。均为 `PROBLEM` 子类，第二行标签可被当前 PlatEMO GUI 识别；`NCTPBase` 是共享实现，不列为独立测试问题。

来源：J.-P. Li, Y. Wang, S. Yang, Z. Cai, *A Comparative Study of Constraint-Handling Techniques in Evolutionary Constrained Multiobjective Optimization*, CEC 2016, 4175–4182，[DOI](https://doi.org/10.1109/CEC.2016.7744320)。[作者 ZIP](https://intleo.csu.edu.cn/codes/A-comparative-study-of-CHTs-in-CMOPs.zip) 于 2026-10-28 重新下载核对，SHA-256 为 `9b8c5267eb990ce77b3e10a5f812d5de706cc51150a968b20c8bdcb84e051f37`，与已有原包一致。

原始文件保留在 [NCTP 作者归档](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Problems/MATLAB source archives/NCTP/README.md>)，未修改作者代码，未推定作者包另有开源许可证。

## 保留的定义与适配差异

- M=2；默认 D=30；作者的 D=10、30 均已验证。实现支持 D≥3 的同公式扩展，其他维度不冒称作者实验设置。全部实数变量范围为 `[0,5]`。
- NCTP1–6 对应 `CDP(type-I)`，7–12 对应 `CDP(type-II)`，13–18 对应 `CDP(type-III)`。作者每组 case 1–6 为 30 维，7–12 为相同六题的 10 维。
- 距离为 Rosenbrock 链：`g=sum(100*(x3:D-x2:D-1.^2).^2+(1-x2:D-1).^2)`。目标为 `f1=x1, f2=g-sqrt(x1)+z`，其中 z 为 `[-0.5×6,1×6,2,2.5,4,2,2.5,4]`。
- 采用上述稳定表达，避免作者 `g*(1-sqrt(f1)/g)` 在 g=0 时产生 NaN；g>0 时与原式代数等价。
- 常规版 `CalCon` 返回原始有符号约束，满足全部 `C<=0` 时可行。每组六题的前三题有两个约束、后三题有一个约束。约束幅度/指数依次为 `(0.2,0.5)`、`(0.75,0.5)`、`(2,6)`；第二约束写成 `f2+0.73*f1-[4,4,6]`，保留作者各组目标平移后的等价定义。
- 作者的按批次最大违反量归一化属于实验算法处理，没有放入问题定义；因此同一点单独与批量计算时的约束相同。
- `_BC` 继承常规版，唯一变化为 `C=double(any(C>0,2))`：N×1，0 可行、1 不可行。目标、边界、维数、参考前沿、FE 计数一致。

## 参考前沿

`GetOptimum` 返回确定性的**数值 CPF 参考集**，不是直接复制作者的粗参考文件，也不宣称解析精确 CPF。方法：参数化旋转坐标中的约束边界，加入解析 UPF 与端点候选，过滤不可行及不可达点，再保留非支配点。二元版与常规版使用相同参考点，不消耗搜索 RNG 或 FE。

NCTP1–6 的 CPF 与 UPF 完全分离；7–12 相交但非完全重合；13–18 的 CPF=UPF。第一约束有平方根正弦尖点，目标点反算为决策变量再计算时，机器舍入误差可被平方根放大。本次参考点反算最大约束残差为 `6.12e-8`；验证参考集采用 `1e-6` 数值容差，**实际问题的可行性判断仍严格使用原约束的 `>0`，没有添加容差**。

## 使用与验证

```matlab
P = NCTP1('D',30,'N',100,'maxFE',50000);
B = NCTP1_BC('D',30,'N',100,'maxFE',50000);
```

当前 PlatEMO 正常启动会加入问题目录。独立脚本需要把 `NCTP`、`NCTP_BC` 及平台基础目录加入路径。

运行 [Tests/test_NCTP.m](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Tests/test_NCTP.m>) 可复核：36 类、两种维数；作者目标/归一化 CV 对照；g=0；批量一致性；边界修复；FE/RNG；常规与二元等价；参考集；72 次 NSGA-II 短程兼容性运行。该验证不证明优化算法在 NCTP 上的最终性能。

结果：[nctp_validation.json](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Deliverables/CPF_UPF_Benchmark_Screening_20261028/NCTP_and_extended_audit/nctp_validation.json>)。未修改实验入口 `test.m` 或 GANCMO。
