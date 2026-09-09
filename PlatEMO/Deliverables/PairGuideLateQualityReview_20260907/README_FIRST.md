# PairGuide 后期生成质量：独立分析资料包

本包用于诊断当前单点 CGAN 为什么后期跟不上 P1 搜索。最新目标是产生有搜索价值的候选，不要求直接生成可行边界点。请以 [PROMPT_TO_GPT.md](PROMPT_TO_GPT.md) 为本次分析要求，历史规则不覆盖它。

## 阅读顺序

1. [分析 prompt](PROMPT_TO_GPT.md) 和 [证据说明](review_notes/EVIDENCE_GUIDE.md)。
2. [本轮报告](experiments/PairGuideSinglePoint_20260907/RESULTS.md)、最终配置、首用表格、完整运行对照与条件质量表。
3. `code/Algorithms/Multi-objective optimization/PairGuide/PairGuide.m` 及 CBS-CGAN/Core 下的 `PairGuideCore.m`、`PairBoundaryArchive_RC.m`、`PairBoundaryWGAN_RC.m`。当前算法入口是 PairGuide，其他旧算法不是当前主线。
4. `experiments/PairGuideSinglePoint_20260907/full_run/analysis/figures/` 的全部 48 张 run 1 分布图；结合 `numeric_evidence/` 检查数值。
5. 按需要查历史资料，避免混用旧配对损失和当前单点损失。

## 包内内容

- `code/`：当前实现、相关 PlatEMO 依赖、问题/指标、实验运行/审计/绘图脚本和相关测试。保留源码版权声明。
- `experiments/PairGuideSinglePoint_20260907/`：本轮 121 个首用案例、36 次完整运行、8 个留出/续训检查的全部现有非 MAT 文本证据与 PNG；含冻结源码和实验前后版本。
- `experiments/PairGuideFilteredArchive_5to8_R3_20260907/`：修正存档后的旧配对 CGAN 12 次正式实验及 52 张图。
- `experiments/PairGuideTrainingDiagnosis_20260907/`：上述旧配对版本的固定数据诊断、48 个诊断结果及 3 张图。其重建损失、逆 gap 权重和 one-hot 结论不能直接套用当前主线。
- `experiments/PairGuideDominanceDrift_20260907/`：更早的存档资格/训练漂移诊断及 2 张图，只作历史追溯。
- `numeric_evidence/`：从既有缓存导出的 JSON/CSV，包括 12 次 CGAN 的训练事件/逐代计数、24 个 run 1 绘图状态、23 个有效生成查询、4 题首末模型及 8 个留出模型的数值。没有重新训练或调用目标/约束函数。
- `tools/`：导出、无损去重、构包及便携复现实验入口。
- `MANIFEST.csv`：包内文件大小和 SHA-256；`SOURCE_PROVENANCE.csv`：来源与文档链接改写记录；`BUNDLE_VALIDATION.json`：最终验证结果。

本包保留 117 张相关 PNG 原图，没有缩小或重编码。排除所有 `.mat`、`.fig`、系统文件、缓存和已有 ZIP；`.mat.csv` / `.mat.checkpoints.csv` 是纯文本 CSV，沿用原始文件名。本包不是整个 PlatEMO 仓库，也不重复收录更早、已退役且与本轮机制判断无直接关系的全量实验。

## 复现边界

MATLAB 环境需要 Deep Learning Toolbox，以及代码实际调用的统计/并行工具箱。原实验 MATLAB 版本及参数见 `numeric_evidence/full_protocol.json`。原始脚本保留当时绝对路径，供审计；跨机器运行请用 `tools/reproduce_current.m`，它从解压目录定位代码，并使用新目录保存结果。

```matlab
addpath('/解压路径/PairGuideLateQualityReview_20260907/tools');
reproduce_current('first_use'); % 一题一个种子的首训首用实验
% reproduce_current('full',1); % 明确需要时才运行 36 次完整搜索
```

不要对整个资料包执行 `addpath(genpath(...))`；历史快照会与当前函数重名。只启用 `code/` 的运行目录。没有 MAT 时不能直接恢复所有旧搜索状态或原封不动执行所有历史重放脚本；已提供主要数值证据与重新运行入口，不宣称打包等同于重新通过全部实验。

`numeric_evidence` 中较大的重复数组/状态使用 `{"$ref":"shared/某文件.json"}` 引用同目录下的共享文件，所有值均可无损还原。`tools/compact_evidence.py` 的 `resolve()` 可展开引用，且已逐文件验证还原前后一致。JSON `null` 表示 MATLAB 原来的 NaN/Inf 或缺测，不是 0。
