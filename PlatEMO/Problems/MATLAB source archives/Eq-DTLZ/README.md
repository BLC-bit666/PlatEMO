# Eq-DTLZ / Eq-IDTLZ 原作者 MATLAB 源码

- 原始论文：O. Cuate, L. Uribe, A. Lara, O. Schütze. A Benchmark for Equality Constrained Multi-objective Optimization. Swarm and Evolutionary Computation 52, 100619, 2020. DOI: [10.1016/j.swevo.2019.100619](https://doi.org/10.1016/j.swevo.2019.100619)。
- [作者配套数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6994572/)，DOI: [10.1016/j.dib.2020.105130](https://doi.org/10.1016/j.dib.2020.105130)。
- [作者补充材料 ZIP](https://ars.els-cdn.com/content/image/1-s2.0-S235234092030024X-mmc1.zip)。
- 本地完整原包：`original.zip`；来源与校验记录：`SOURCE_MANIFEST.json`。

从作者附件的 `DataInBrief/EqDTLZ/` 提取全部 12 个 MATLAB 类到 `private/original/DataInBrief/EqDTLZ/`，保留原文件字节和版权注释。原附件的图像等其他材料仍保存在原 ZIP 中，未重复解压。

| 基础问题 | 单等式版本 | 双等式版本 |
| --- | --- | --- |
| Eq-DTLZ1 | `Eq1_DTLZ1.m` | `Eq2_DTLZ1.m` |
| Eq-DTLZ2 | `Eq1_DTLZ2.m` | `Eq2_DTLZ2.m` |
| Eq-DTLZ3 | `Eq1_DTLZ3.m` | `Eq2_DTLZ3.m` |
| Eq-DTLZ4 | `Eq1_DTLZ4.m` | `Eq2_DTLZ4.m` |
| Eq-IDTLZ1 | `Eq1_IDTLZ1.m` | `Eq2_IDTLZ1.m` |
| Eq-IDTLZ2 | `Eq1_IDTLZ2.m` | `Eq2_IDTLZ2.m` |

综述称 Eq-DTLZ 有 8 个问题；作者数据论文及此附件提供的是上述 6 个基础问题的 12 个版本，不能将其表述为已补齐综述中的 8 个基础问题。

源码基于 PlatEMO 2.0，使用 `obj.Global`、旧构造函数和 `PF` 接口；与当前版本尚未做兼容性迁移。原约束实现保留了 `1e-4` 等式容差，没有改为严格零等式或二元标签。接口、参考前沿与指标适用性需在正式接入时一起验证。
