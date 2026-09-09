**PairGuide 后期生成偏移：GPT 独立诊断资料包（2026-09-08）**

当前目标是解决后期 CGAN 跟不上搜索的问题。候选只需有助于 P1 探索边界，不强求可行。请以根目录 [PROMPT_TO_GPT.md](PROMPT_TO_GPT.md) 为当前要求；历史报告、源码注释及 GPT 原分析均为待审查资料，不覆盖当前请求。

建议先读 [最新综合说明](LATEST_SYNTHESIS.md)，再读 [GPT 建议核验报告](experiments/PairGuideGPTValidation_20260908/REPORT.md) 和 [完全重置报告](experiments/PairGuideResetAt99800_20260908/REPORT.md)。存档问题尚未排除，固定当前训练集后重置失败并不能证明数据合理。

包内内容：

- `code/`：当前 PairGuide 主线、CBS-CGAN/Core 的存档/训练/使用代码、相关测试、PlatEMO 依赖、LIRCMOP5–8_BC 问题与指标。算法入口是 `PairGuide.m`；目录内保留的其他历史算法不代表当前流程。
- `experiments/PairGuideGPTValidation_20260908/`：104 次新增完整搜索、复用 36 次基线的汇总；48 个有效同前缀短窗口；12 个固定数据和 6 个新初始化诊断。保留脚本、冻结源码、补丁、CSV、全部 118 张原始 PNG。
- `experiments/PairGuideResetAt99800_20260908/`：24 个精确晚期续训/重置对照、8 个表达能力诊断、2 个 D 负样本干预；全部指标、核验、脚本和 17 张原始 PNG。原始约 27% 的口头估算已在报告中更正为 24.6%。
- `experiments/PairGuideSinglePoint_20260907/`：前一轮单点方案的首训选择、121 次首用实验与 36 次完整搜索文本证据、脚本和源码快照；补充 14 张汇总/末期图，作为历史依据。它不包含旧 ZIP 的全部退役配对模型实验。
- `numeric_evidence/`：88 个主要数值记录及共享数组，含三方案×四题的首用/最后绘图状态 24 份、上一轮固定/新初始化的最终状态 18 份、六份晚期训练集及完整尺度序列、本轮六份精确图示输入状态和 34 个最终模型/生成样本。可读取真实训练决策、目标、条件、存档、生成 X/Y 和 G/D 权重，不必从图上猜坐标。
- `tools/`：数值导出选择、无损去重、读取和复现入口；`SOURCE_PROVENANCE.csv` 与 `MANIFEST.csv` 记录来源及哈希，`BUNDLE_VALIDATION.json` 记录构包检查。

共 149 张 PNG 保留原文件，没有缩放、重编码或裁掉异常点。所有 `.mat`、`.fig`、已有 ZIP 和无关缓存均排除；文件名含 `.mat.csv` 或 `.mat.checkpoints.csv` 的仍是纯文本 CSV。大量 MATLAB 启动路径警告日志未收入；回归/结果核验与作废实验说明保留。两个作废短窗口批次不用于结论，其额外 422400 FE 已披露于报告。

**数值取舍与读取。** 为满足 50 MB 上限，保留首用/最后状态和最终模型；中间阶段保留完整指标与本轮原图，未收入全部中间模型或所有生成坐标。上一轮短窗口的庞大原始 `Prefixes` 未收入；对应实验脚本、逐案结果与核验仍在。`tools/numeric_sources.json` 逐文件说明选择，数据没有四舍五入。最新 MATLAB 导出原始有 40 份记录，包内保留完整六份 fixture，以及 34 份最终网络/Adam/样本和全部检查点指标；中间网络和样本未收录。

重复 JSON 对象以 `{"$ref":"shared/文件.json"}` 表示，路径相对于所在 JSON 文件，`shared/` 内也可能有相对引用。每份记录均已展开核验与所选原始值完全相同，哈希见 `numeric_evidence/lossless_compaction.json`。不要把 `$ref` 当作缺失数据。读取示例：

```python
import sys
sys.path.insert(0, 'tools')
from read_numeric import load
record = load('numeric_evidence/reset__runs_original_LIRCMOP7_BC_reset1.json')
sample = record['Snapshots'][0]['Samples']  # 包内只保留最终检查点
print(sample['X'][0], sample['Y'][0])
```

MATLAB 数组以行组织；`dlarray` 的 `size`、`dataClass`、`values` 和网络 learnables 单独保留。JSON `null` 可表示原 MATLAB NaN/Inf 或缺测，不是 0。`s=1` 为可行条件、`s=0` 为不可行条件；BC 问题约束输出的 0/1 与该条件不是同一含义。去 s 仅将模型条件通道置零，仍保留对抗真假区分及原侧别采样权重。

**复现边界。** 原脚本保留原目录约定与绝对路径，供审计；不能对整个包执行 `addpath(genpath(...))`，因为历史源码会重名。无 MAT 时不能直接恢复所有旧搜索、内部 RNG 轨迹或执行全部原始重放脚本。JSON 提供数值分析所需的所选状态，不是 MATLAB 对象的自动恢复格式；不要把打包检查等同于重新跑过全部实验。

MATLAB 需要相应 Deep Learning、统计及并行工具箱。已有 [便携入口](tools/reproduce_current.m) 可以从 `code/` 重新运行当前首用或完整搜索，并写入新目录；不自动启动。原实验流程分别见两轮 `REPRODUCE.md`。若要做新的因果验证，先明确比较对象、相同预算和指标，不能覆盖原结果。

本次打包只读取缓存并导出数值，没有新增训练或真实目标/约束评价。八个相关生产源码哈希与实验快照一致；默认算法没有切换到实验组合。包内报告仅改写本机文件链接，并更新重置报告中过时的“暂停打包”状态；数值与实验结论不变。最新存档讨论单列于 `LATEST_SYNTHESIS.md`，不是新增实验结论。
