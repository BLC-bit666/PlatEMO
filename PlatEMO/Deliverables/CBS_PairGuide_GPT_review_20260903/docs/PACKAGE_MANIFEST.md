# 内容清单

## code

- `code/algorithms/CBS-CGAN/`：共享核心、早期生产入口、全部当前 Support/Tests。
- `code/algorithms/CBS-CGAN-PairGuide/`：PairGuide 主线、DE20、Quota30/40/50。
- `code/problems/`：本轮六个 LIRCMOP_BC 问题。
- `code/platemo_dependencies/`：ALGORITHM、PROBLEM、SOLUTION、OperatorDE、OperatorGAhalf、TournamentSelection、UniformPoint、IGD。

## experiments

- 最新 first-use Epoch、训练比例/后续 Epoch、推理 sigma、训练 sigma、DE20 消融、配额实验。
- v1/v3 首次 Epoch 只保留分析摘要。
- CSV 是汇总或训练诊断；PNG 是可直接查看的实验图。

## figures

- 两张用户上传的历史/显示范围参考图。

## 明确排除

- 所有 `*.mat` 原始结果与清单。
- `run.log`、崩溃转储、失败缓存、`.DS_Store`。
- 当前工作树中已删除并由 README 标记为历史的实验分支。
- 与两个核心疑问无关的 PlatEMO 算法和问题。

