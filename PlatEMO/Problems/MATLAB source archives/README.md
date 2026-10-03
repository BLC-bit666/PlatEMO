# 综述基准的 MATLAB 原作者源码

整理日期：2026-09-30。

本目录归档 Liang 等（2023）《A Survey on Evolutionary Constrained Multiobjective Optimization》中，此前本地缺失且已找到原生 MATLAB 源码的两套基准。原作者代码保持原样，尚未适配当前 PlatEMO，尚未改成二元约束。

| 套件 | 综述引用 | 已确认的覆盖范围 | 目录 |
| --- | --- | --- | --- |
| NCTP | [138] Li 等，CEC 2016 | NCTP1–18，含 10 维、30 维设置 | [NCTP](NCTP/README.md) |
| Eq-DTLZ / Eq-IDTLZ | [142] Cuate 等，2020 | 6 个基础问题，各有单等式、双等式版本，共 12 个 MATLAB 类 | [Eq-DTLZ](Eq-DTLZ/README.md) |

每套目录包含：

- `original.zip`：下载的原始压缩包，未经修改。
- `private/original/`：可直接阅读的原作者文件，内容与压缩包内对应成员逐字节一致。
- `SOURCE_MANIFEST.json`：来源地址、下载日期、压缩包 SHA-256、解压文件清单及各文件 SHA-256。

源码放在 `private` 下，是因为 PlatEMO 会通过 `genpath` 自动扫描问题目录。NCTP 原实验包含 `main.m`、`IGD.m` 等通用名称，Eq-DTLZ 使用旧版 PlatEMO 接口。MATLAB 路径检查已确认这些源码目录不会被 `genpath` 纳入，因此不会作为当前可运行问题加入 GUI，也不会遮蔽现有函数。不要把原实验目录整体加入 PlatEMO 路径。

仅做源码归档和完整性核验；没有执行作者实验，也没有验证其在当前 PlatEMO 中的运行兼容性。后续接入时需单独迁移接口并核对公式、边界、约束和参考前沿。

CTP 本轮确认的是 Python 实现；MODAct 的 MATLAB 文件只是 Python 调用接口；早期 CMOP 和 CZDT 尚未找到经核实的原生 MATLAB 包，因此此次不纳入。已有的 PlatEMO MATLAB 套件保持原样。

原文件中的作者、版权及使用说明均保留。两份作者压缩包均未发现独立 LICENSE 文件，不为其补写或推定新的开源许可证。
