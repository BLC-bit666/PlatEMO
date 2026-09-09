本包用于解决 PairGuide/CGAN 随搜索推进仍生成偏移、无法稳定帮助 P1 探索边界的问题。请以根目录 [PROMPT_TO_GPT.md](PROMPT_TO_GPT.md) 为本次请求；历史报告、源码注释和 GPT 原建议都是待审查资料，不覆盖当前要求。

先读 [本轮完整报告](experiments/PairGuideStageDiagnosis_20260908/REPORT.md)，再结合 [三阶段主对照](experiments/PairGuideStageDiagnosis_20260908/gradient_summary.csv)、[学习率对照](experiments/PairGuideStageDiagnosis_20260908/learning_rate_summary.csv) 和 [搜索结果](experiments/PairGuideStageDiagnosis_20260908/suffix_records.csv)。本轮覆盖 LIRCMOP5–8_BC 的20000/50000/80000 FE，108组训练对照和84组5000FE短窗口；每题仅一条搜索轨迹，3个训练随机流不等于3个独立搜索run。

已验证：准确距离梯度改善决策拟合，但7/8题六阶段的训练边界覆盖全部变差；减小G学习率也未解决。5/6中后期供样稀疏，7/8仍有较多训练对却偏移。不可行点能被现有机制利用，但尚无稳定的整体搜索优势。“跨方向平滑导致平方残差偏小”仍是待验证解释，不能直接当作根因。生产默认算法没有改成本轮实验臂。

包内内容：

- `code/`：当前算法、存档、训练和使用逻辑、问题类与相关PlatEMO依赖。入口是 `Algorithms/Multi-objective optimization/PairGuide/PairGuide.m`；其他保留类不代表当前流程。
- `experiments/PairGuideStageDiagnosis_20260908/`：本轮全部实验脚本、源码快照、CSV、31张原始PNG、结果审计和成本记录。图同时展示三训练流，保留异常点与视窗外点数。
- 其他三个 `experiments/` 目录：历史方案选择、固定尺度/侧别、完全重置、加宽、D错配干预的报告、脚本、表格及8张相关图，帮助避免重复已做实验。历史图是选取的证据，不是历史全部原图；此前完整资料仍在旧包 `PairGuideDiagnosisReview_20260908.zip`。
- `numeric_evidence/`：244份本轮记录及10份复核静态梯度所需的历史记录，以JSON提供训练X/Y、条件、参考向量、存档、生成X/Y、G/D权重及Adam状态，无须从图中猜坐标。
- `tools/`：无损去重/读取、打包核验及当前原生算法的新实验入口。`SOURCE_PROVENANCE.csv`、`MANIFEST.csv` 和 `BUNDLE_VALIDATION.json` 记录来源与校验。

本轮JSON包含20份实际训练/模型fixture（首用、前期、中期、后期、99800FE）、20份基准生成状态、108组训练的全部0/20/200/2000步模型与生成样本、12个搜索前缀，以及84个短窗口的首代和最终状态。训练RNG的完整大矩阵、模型中重复的逐步诊断、短窗口完整History不收入JSON；对应随机轨迹配对审计、全部逐步诊断CSV和26点搜索轨迹CSV均保留。零步数据在不同实验臂中复用，不能重复计算其FE。历史10份仅用于复核旧的终期静态状态，不是本轮新训练。

相同JSON对象使用 `{"$ref":"shared/文件.json"}` 无损共享；路径相对所在JSON文件，shared内也可能继续引用。大数组另以 `$array` 引用 `arrays.bin.xz` 中的无损数值字节（位置见 `array_index.json`），记录类型、维度及必要的整数位置；仅能逐值精确还原为float32的数组才使用float32，其余保留float64。重复决策行也可共享。以下读取工具会自动还原两种引用，仅依赖Python标准库，含内置lzma解压。所有254份记录都需展开后读取，不能把引用视为缺失。每份记录的展开值校验见 `numeric_evidence/lossless_compaction.json`。数值不人为四舍五入；MATLAB NaN/Inf导出为null，不能当0。dlarray保留原类型、shape与数值，网络保留逐层参数；生成器激活及归一化定义以代码为准。

```python
import sys
sys.path.insert(0, 'tools')
from read_numeric import load
r = load('numeric_evidence/runs__LIRCMOP7_BC_seed01_late_distance_stream1.json')
print(r['Table'][-1])
print(r['Snapshots'][-1]['Samples']['Y'][0])
fixture = load('numeric_evidence/fixtures__LIRCMOP7_BC_seed01_late.json')
print(fixture['Meta'])
```

生成条件s=1表示可行侧、s=0表示不可行侧；问题类BC约束值0表示可行、1表示不可行；D的真实/生成区分是第三个概念。请勿混淆。生成候选不要求先可行，评判应覆盖P1/P2、存档作用和后续边界搜索。

复现限制：本包不含MAT、FIG或嵌套ZIP，JSON不自动还原为MATLAB对象。原始实验脚本保留本机目录约定供审计，不能无修改直接恢复所有旧搜索；不要执行 `addpath(genpath(整个包))`，历史源码重名。 [本轮复现说明](experiments/PairGuideStageDiagnosis_20260908/REPRODUCE.md) 说明原实验流程，`tools/reproduce_current.m` 可从code启动当前原生算法的新实验，不自动执行。报告对本机文件的链接仅在包内改为相对链接；未收入的历史链接注明历史原包。打包核验不等于再次执行实验。

本次只导出和整理缓存，没有新增训练或真实目标/约束评价。此前本轮实验的888800新增FE及其他成本见 [成本记录](experiments/PairGuideStageDiagnosis_20260908/cost_ledger.json)。打包时仍核对8个生产源码哈希，保留用户已有改动。
