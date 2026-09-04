# CBS-PairGuide 外部评审资料包

本资料包用于让外部 GPT 讨论两个仍未解决的问题：CGAN 生成解应如何接入双种群，以及为什么当前原始生成解在目标空间呈厚状点云而非期望的边界/分界线。

建议阅读顺序：

1. `PROMPT_TO_GPT.txt`：可直接复制给 GPT 的提问。
2. `docs/ALGORITHM_OVERVIEW.md`：当前算法、数据流、网络、训练和使用机制。
3. `docs/FIGURE_GUIDE.md`：图片符号、目录和阅读限制。
4. `docs/EXPERIMENT_INDEX.md`：实验协议、当前结论及历史关系。
5. `code/`：当前工作树中的算法、实验脚本、测试和六个实验问题。
6. `experiments/`：CSV/TXT/MD 汇总和代表性 PNG 图；不含 MAT 原始结果。

## 本附件的精简范围

- 这是供外部 GPT 直接上传、读取的精简附件；算法代码、实验脚本、全部统计汇总和核心提问均完整保留。
- 删除了两份约 20 MB 的逐训练事件明细 `training_diagnostics.csv`；对应的 `task_metrics.csv`、`arm_summary.csv`、`problem_summary.csv` 与结论文件仍在。
- 图片只保留评审这两个问题所需的代表集：首次训练 400/1000 Epoch 的 run01、全部 3×10 放大细节图、sigma 对照 run01、LIRCMOP14 三维补充图和两张用户参考图。
- `figure_index.csv` 等索引仍记录原实验的完整图集，因此其中少数未选中的图片路径不会出现在本精简附件中。

## 当前研究基准

- 当前研究主线：`CBS_RegionWGAN_GP_PairGuide`。
- 当前默认约束种群 P1 子代：25% GA、55% 普通 DE、20% PairGuide。
- 无约束种群 P2 子代：25% GA、75% 普通 DE。
- 当前暂定训练配置：首次 500 Epoch、后续 10 Epoch、5 Critic : 1 Generator。
- 500 个原始 CGAN 候选不计入搜索 FE、不进入双种群；只有由其引导产生的子代接受真实评价。
- 最新配额实验不支持把全局名额从 20 提高到 30、40 或 50，因此当前保留 20。

## 范围说明

- `CBS_RegionWGAN_GP` 是仍保留的早期生产路径，供机制对照。
- PairGuide 的 DE20、Quota30、Quota40、Quota50 分支均已收录。
- 已从当前仓库删除的 A0–A2、E0–E8、Random20、GA20、FullCGAN、Screening 等历史分支没有恢复或打包。
- v1/v3 首次 Epoch 实验只保留分析摘要；最新 v4 保留完整非 MAT 摘要与代表性图片。
- 资料包不是完整 PlatEMO 副本，`code/platemo_dependencies/` 仅保留理解关键调用所需的接口与算子。
- 已排除所有 `*.mat`、`.DS_Store`、运行日志、崩溃转储、缓存和体积较大的逐训练事件明细。
