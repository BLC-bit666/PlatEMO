# 内容清单

## code

- `code/algorithms/CBS-CGAN/`：共享核心、早期生产入口、全部当前 Support/Tests。
- `code/algorithms/CBS-CGAN-PairGuide/`：PairGuide 主线、DE20、Quota30/40/50。
- `code/problems/`：本轮六个 LIRCMOP_BC 问题。
- `code/platemo_dependencies/`：ALGORITHM、PROBLEM、SOLUTION、OperatorDE、OperatorGAhalf、TournamentSelection、UniformPoint、IGD。

## experiments

- 最新 first-use Epoch、训练比例/后续 Epoch、推理 sigma、训练 sigma、DE20 消融、配额实验。
- v1/v3 首次 Epoch 只保留分析摘要。
- CSV/TXT/MD 汇总完整保留；PNG 只保留可直接支持本次评审的代表图。

## figures

- 两张用户上传的历史/显示范围参考图。

## 明确排除

- 所有 `*.mat` 原始结果与清单。
- 两份体积最大的逐训练事件明细 `training_diagnostics.csv`；相应统计汇总仍保留。
- 重复 run 和非关键 Epoch 的 PNG；图像索引仍用于说明原实验的完整范围。
- `run.log`、崩溃转储、失败缓存、`.DS_Store`。
- 当前工作树中已删除并由 README 标记为历史的实验分支。
- 与两个核心疑问无关的 PlatEMO 算法和问题。
