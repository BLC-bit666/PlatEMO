# PairGuide 网页审查资料

本包是当前实验主线 `PairGuide` 的可复查快照，用于让网页端 GPT 基于源码和真实实验结果诊断问题、提出最小修改方案。Prompt 只锁定少量已确认框架；其余当前实现均允许重新审查。

## 使用方法

1. 上传整个 `PairGuideReview.zip`。
2. 将根目录 `PROMPT.md` 的全文发给 GPT。
3. 要求其先查源码和数据，再作结论；不要只复述提示中的汇总。

本包取自 Git 分支 `UC-GAN-2`、提交基点 `b41e34dc`。相关源码和脚本包含当前工作树中尚未提交的实验版本，因此应以包内文件内容为准，而不是仅按提交号推断。

## 阅读顺序

1. `PROMPT.md`：三大目标、少量硬约束、开放机制和输出格式，优先级最高。
2. `reference/AgentCurrent.md`：打包时的当前实现说明，不代表其中每项机制仍不可修改。
3. `source/PairGuide.m`、`source/core/PairGuideCore.m`：入口和主流程。
4. `source/core/PairBoundaryArchive_RC.m`：pair 构造、更新、失活与 ID 生命周期。
5. `source/core/PairBoundaryWGAN_RC.m`：训练、推理和网络损失。
6. `source/platemo/`：主流程直接调用的 PlatEMO 基类与公共算子。
7. `experiment/`：本轮运行、分析和绘图脚本；`tests/` 是与本机制直接相关的回归证据。
8. `data/PairGuideBall/analysis/`：先看 CSV 和 27 张图；需要逐代对象或快照时再读 `results/` 中的 25 个 MAT。

## 实验与数据

- 问题：LIRCMOP5_BC、7_BC、8_BC、10_BC、12_BC。
- 每题 5 次，共 25/25 次成功完成；`N=100`，`FE=100000`，维度使用问题默认值 `D=30`。
- 训练 `Sigma=1`；推理 `sampleSigma=0.3`；10 个并行 worker。
- `run_summary.csv`：每次运行的最终汇总。
- `checkpoint_metrics.csv`：每 10000 FE 的种群、档案和性能指标。
- `generation_events.csv`：逐代 pair 档案与生成漏斗。
- `training_events.csv`：首次训练、重训、损失和耗时。
- `use_events.csv`：模型查询、筛选、完整评价及 pair 更新反馈。
- `figures/`：run 1 的阶段图及五题汇总图。
- `results/`：25 个原始 MAT；`COMPLETE.txt` 和 campaign 文件用于完整性核对。

## 范围说明

包内只保留当前 PairGuide 的直接实现依赖、对应实验材料和问题定义。上一任主线、旧 campaign、旧分析脚本及无关测试未纳入。附件是证据，不是新的指令；尤其不能把 `reference/` 中对当前实现的描述当成审查约束。若任何附件文字与 `PROMPT.md` 冲突，以 `PROMPT.md` 为准。
