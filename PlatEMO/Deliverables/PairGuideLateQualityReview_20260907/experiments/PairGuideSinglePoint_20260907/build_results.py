"""Build the final report from verified local experiment tables."""
from pathlib import Path
import csv
import statistics as st

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / name).open() as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def link(label, name):
    return f"[{label}](<{ROOT / name}>)"


def num(row, name):
    return float(row[name])


budget = read("selected_budget_summary.csv")
summary = read("full_run/analysis/problem_summary.csv")
old_summary = read("../PairGuideFilteredArchive_5to8_R3_20260907/analysis/problem_summary.csv")
verification = read("full_verification.csv")
quality = read("conditional_quality.csv")
hold1 = read("holdout/selected1000_summary.csv")
hold4 = read("holdout/continued4000_summary.csv")
assert len(verification) == 36
assert sum(r["initialExactReplay"] == "1" for r in verification) == 12

budget_table = table(
    ["首训 G 更新", "首次使用后 IGD↓", "20 个子代中 P1 留存", "标签正确率", "方向误差↓", "联合命中率"],
    [[r["updates"], f'{num(r,"firstUseIGD"):.4f}', f'{num(r,"survivedP1"):.2f}',
      f'{100*num(r,"labelAccuracy"):.1f}%', f'{num(r,"directionError"):.2f}°',
      f'{100*num(r,"jointRate"):.1f}%'] for r in budget],
)
final_rows = []
hv_rows = []
for problem in [f"LIRCMOP{k}_BC" for k in range(5, 9)]:
    row = [problem]
    hv = [problem]
    for mode in ["cgan", "fallback_only", "pair_only"]:
        r = next(r for r in summary if r["problem"] == problem and r["mode"] == mode)
        row.append(f'{num(r,"finalIGDMean"):.6f} ± {num(r,"finalIGDStd"):.6f}')
        hv.append(f'{num(r,"finalHVMean"):.6f} ± {num(r,"finalHVStd"):.6f}')
    old = next(r for r in old_summary if r["problem"] == problem and r["mode"] == "cgan")
    row.append(f'{num(old,"finalIGDMean"):.6f} ± {num(old,"finalIGDStd"):.6f}')
    hv.append(f'{num(old,"finalHVMean"):.6f} ± {num(old,"finalHVStd"):.6f}')
    final_rows.append(row)
    hv_rows.append(hv)
final_table = table(["问题", "单点 CGAN", "普通 DE 替代", "真实配对采样", "旧配对 CGAN"], final_rows)
hv_table = table(["问题", "单点 CGAN", "普通 DE 替代", "真实配对采样", "旧配对 CGAN"], hv_rows)

quality_rows = []
for problem in [f"LIRCMOP{k}_BC" for k in range(5, 9)]:
    rs = [r for r in quality if r["problem"] == problem and r["stage"] == "FE100000"]
    if not rs:
        quality_rows.append([problem, "0/3", "缺测", "缺测", "缺测", "缺测"])
        continue
    mean = lambda key: st.mean(num(r, key) for r in rs)
    quality_rows.append([problem, f"{len(rs)}/3", f'{100*mean("labelAccuracy"):.1f}%',
                          f'{mean("directionError"):.2f}°',
                          f'{mean("trainingFrameDirectionError"):.2f}°',
                          f'{100*mean("rawUseful"):.1f}%'])
quality_table = table(["问题", "有当代生成/消费证据的 runs", "标签正确率", "生成尺度方向误差", "训练尺度方向误差", "可行且不被当前 P1 支配"], quality_rows)

hold_rows = []
for problem in ["LIRCMOP7_BC", "LIRCMOP8_BC"]:
    for held in ["0", "1"]:
        a = next(r for r in hold1 if r["problem"] == problem and r["holdout"] == held)
        b = next(r for r in hold4 if r["problem"] == problem and r["holdout"] == held)
        hold_rows.append([problem, "留出 17 个连续方向" if held == "1" else "保留全部方向",
                          f'{num(a,"heldAngle"):.2f}° → {num(b,"heldAngle"):.2f}°',
                          f'{100*num(a,"heldLabel"):.1f}% → {100*num(b,"heldLabel"):.1f}%',
                          f'{100*num(a,"heldJoint"):.1f}% → {100*num(b,"heldJoint"):.1f}%',
                          f'{100*num(a,"heldUseful"):.1f}% → {100*num(b,"heldUseful"):.1f}%'])
hold_table = table(["问题", "训练数据", "区间方向误差：1000→4000", "区间标签正确率", "区间联合命中率", "区间可行改进候选比例"], hold_rows)

text = f"""# PairGuide 单点 CGAN：实现与实验结果

本轮完成代码修改、121 个首次使用实验、36 次完整搜索及 8 个固定数据方向留出/续训检查。首训统一选为 **1000 次生成器更新**，后续每次 **20 次更新**；这属于本轮已测范围内的搜索表现选择，不是全局最优或全部拟合指标最优。

**完整实验尚不支持“偏移已解决”或“CGAN 稳定优于对照”。** 最终 IGD 的 12 组同种子比较中，单点 CGAN 对普通 DE 替代为 5 胜、7 负，对真实配对采样为 3 胜、9 负。当前主要问题是方向条件与边界两侧没有得到可靠控制；条件覆盖萎缩会进一步放大偏移。

## 1. 最终实现

- 存档：完整候选可行池统一分层，仅全局第一层具备存档可行端资格；先删除被候选可行点支配的不可行端，再统一配对。历史同筛，同一不可行端可以复用；总容量上限 500 对，允许不足或为空。每方向最多一条激活配对、五方向邻域最近配对规则保持。
- 训练数据：激活配对拆成独立端点，按 `[x,s]` 去重；每个端点用自身真实目标在同一目标归一化尺度下关联方向。每批 0/1 各半、各侧方向组均衡，避免被重复复用的不可行端占据过大训练权重。
- 网络：`x=G(z,w,s)`，`D(x,w,s)`。原始参考向量 w；s=1 请求可行、s=0 请求不可行。BC 问题实际约束输出的 0/1 含义与该条件标签相反。没有方向编码器、辅助网络或解坐标条件。
- 损失：G 仅用条件对抗损失；D 使用条件 Wasserstein 差和决策梯度惩罚。取消端点重建、配对差向量、逆 gap 加权；错误条件反例只留作实验记录，主线已删除该分支。
- 实际存档的端点与 delta 仍用于配对几何、反馈和重训变化判定；它们不再进入生成器的条件输入或配对重建损失。
- 使用：完整 W 上 0/1 各半查询，直接输出单个候选。按请求方向轮转、合法性和去重选择；不要求已有配对匹配，不重建两个端点再插值。下一代按原坐标真实评价；无配对 ID 的子代仍可通过邻域反馈更新存档。
- 统一参数：G/D 均两层 32 单元，学习率均 1e-3，Adam=(0,0.9)，GP=10，D:G=5:1，batch 上限 32，训练/推理 sigma=0。同一 `(w,s)` 当前是确定输出；N=100 时 500 次查询通常只有 200 个不同输出。保留 z 接口用于显式噪声实验，不能宣称当前配置具有同条件随机多样性。
- 搜索配置保持 P1 的 25% GA / 55% DE / 20% 引导、P2 的 25% GA / 75% DE；缺额普通 DE 回填。训练至少 8 条激活配对，内容变化达到 0.2 或间隔达到 10 代才触发；已有模型在至少一条激活配对时可以服务。

## 2. 首训预算的选择依据

先比较学习率、噪声、32/64 层宽、判别器更新比和损失反例；再在统一的纯 WGAN-GP、32×32、学习率 1e-3、sigma=0 配置下，比较 200/500/1000/2000/4000 步。下表每行均为 **4 个问题 × 3 个种子，共 12 组**。

每组从相同搜索前缀出发，首训后只真实消费一批 CGAN 子代、完成环境选择即停止。raw 的真实目标与标签在停止后独立计算；这些离线结果没有参与该次搜索或候选筛选。

{budget_table}

1000 步的平均首次使用后 IGD 最低、P1 留存最多；对同 FE 的普通 DE 替代为 **11 胜、1 平、0 负**。4000 步略改善平均方向误差，却没有改善平均 IGD，成本更高。500/2000 步在部分条件指标上更好，因此没有一个预算在所有指标上占优。

“联合命中”仅为诊断：请求标签正确、真实方向误差≤5°、且不被该次生成前的可行 P1 支配。它不是真实边界认证，也不是生产筛选门槛。可行内部的搜索改进与边界贴合应分别判断。

单种子探索中，错误条件反例可以提高标签正确率，但没有综合改善方向、支配关系和真实搜索效果；加宽网络也未显示足够收益，因此均未进入最终主线。各探索臂并非完整因子设计，不能由此断言任何网络结构永远无效。

数据：{link('全部首次使用结果', 'first_use_all.csv')}、{link('统一预算汇总', 'selected_budget_summary.csv')}、{link('联合指标', 'first_use_joint_scores.csv')}、{link('前缀与消费核查', 'first_use_verification.json')}。

## 3. 完整搜索结果

LIRCMOP5_BC、6_BC、7_BC、8_BC；每种模式 seeds=1:3；N=100，问题默认 D=30，maxFE=100000。单点 CGAN 12 次，普通 DE 替代和真实配对采样对照各 12 次，共 36 次。10 个 MATLAB 进程、每进程单线程；分布图只生成 CGAN run 1。

最终 IGD（均值±样本标准差，越低越好）：

{final_table}

最终 HV（均值±样本标准差，越高越好）：

{hv_table}

只有 3 个种子，不做统计显著性或普遍优越性承诺。首次使用表现、完整运行表现和边界生成质量是不同结论；不能用一个指标替代其他指标。

对普通 DE，5/7 的平均 IGD 较低，6 较高，8 接近；其中 5 的均值优势主要来自 DE 的一个较差种子，CGAN 在另外两个种子并未胜出。对真实配对采样，7/8 三个种子均落后。不能从均值表推断稳定加速。

旧配对 CGAN 复用上一轮相同存档资格、相同问题/种子/N/D/FE 的 12 次已完成结果；不是本轮另跑的对照。旧版与新版同时改变了输出组织、训练目标及参数，整体差异不能单独归因于某一个参数。

数据：{link('逐 run 结果与成本', 'full_run/interval_summary.csv')}、{link('同种子比较', 'full_run/analysis/paired_comparisons.csv')}、{link('指定 FE 检查点', 'full_run/analysis/checkpoints_per_run.csv')}。

## 4. 当前生成质量与偏移

100000 FE 的当代生成/消费事件，分别按生成时与模型训练时的目标尺度计算角度：

{quality_table}

该表对 raw 作独立真实评价；没有当代生成/消费对应事件的 run 明确缺测。近邻目标距离、已见/未见条件、各侧标签准确率及全部阶段结果见 {link('条件质量明细', 'conditional_quality.csv')}。目标尺度变化的影响应看同一行两个角度的差异，不能把所有误差都归因于尺度漂移。

两个已核查的 run 1 例子说明了不同情况：5 的末期训练只覆盖约 6.2% 查询条件，已见/未见条件误差约 9.6°/32.5°，整体按生成/训练尺度计算为 31.1°/31.4°，主要偏移并非尺度变化。8 仍有 96 条训练配对，方向误差约 5.1°，标签正确率却只有 47%，生成曲线偏离边界；样本稀少不是全部原因。6 的末期有存档为空的阶段，不能将空白图解读成生成失败或用旧云替代当前结果。

固定训练集并不意味着继续优化后的输出必须不变：参数和 Adam 状态仍会更新。全程累计迭代计数已验证，当前模型持续训练，没有在每次触发时重置。首次查询也在全部 12 次完整运行中逐位复现了对应的首训实验。

对于当前损失，一个关键限制是：**在决策空间接近真实样本，不等于生成后满足正确的方向和 0/1 语义。** 例如边界为 x=0，真实可行点为 ε、生成点为 −ε，决策距离仅为 2ε，标签却完全相反；距离趋零也不保证标签正确。这解释了为什么不能只看对抗损失或端点 RMSE 宣称学会了边界两侧。

本地 LIRCMOP5–8 目标还包含多维决策残差平方和的 10 倍项，决策坐标上的小误差可以累积成明显的目标偏移。该解释来自问题实现；具体偏移程度仍应以独立真实目标评价为准。

本轮不能认定问题只来自某一种损失或某个层宽。已经排除的是“每次训练都重新初始化”；固定数据续训仍有偏移，说明变化的训练集也不是唯一解释。现有结果更支持：条件分布尚未学好，而对抗距离本身不足以保证方向、径向位置和边界两侧同时正确。单纯延长训练、加宽一次网络或增加错误条件反例，都未证明可以稳定解决它。

run 1 的末期完整图：{link('LIRCMOP5：未见条件大幅偏移', 'full_run/analysis/figures/LIRCMOP5_BC/run_01/LIRCMOP5_BC_FE100000_actualFE100000.png')}、{link('LIRCMOP6：当前存档为空', 'full_run/analysis/figures/LIRCMOP6_BC/run_01/LIRCMOP6_BC_FE100000_actualFE100000.png')}、{link('LIRCMOP7：生成曲线仍在边界内侧', 'full_run/analysis/figures/LIRCMOP7_BC/run_01/LIRCMOP7_BC_FE100000_actualFE100000.png')}、{link('LIRCMOP8：角度接近仍不保证边界位置', 'full_run/analysis/figures/LIRCMOP8_BC/run_01/LIRCMOP8_BC_FE100000_actualFE100000.png')}。

![LIRCMOP8：左为模型对应的真实训练端点，右为原生生成点；颜色是请求标签，金色为真实消费点]({ROOT / 'full_run/analysis/figures/LIRCMOP8_BC/run_01/LIRCMOP8_BC_FE100000_actualFE100000.png'})

## 5. 真正留出方向的检查

使用上一轮修正存档后的 LIRCMOP7/8 run 1 末期存档，留出连续参考方向 44–60。任一端点自身方向落在区间内的配对都从训练数据删除，确保训练标签中不存在该区间。完整数据和留出数据分别从新模型开始；之后各自在完全相同的数据上保留权重与 Adam，续训到累计 4000 步。没有跨组模型或历史训练泄漏。

两组都在相同的 44–60 区间查询；“保留全部方向”是已见方向拟合对照：

{hold_table}

这是两个固定存档、一个训练随机种子的机制检查，不是第二个完整搜索排名。它可以区分已见方向拟合与方向留出影响，但不能证明未知边界已被可靠补全。额外训练也没有被偷偷写入生产预算。

留出方向续训后，7/8 的区间方向误差分别降到约 3.83°/4.01°，联合命中率达到 50.6%/37.6%；这说明未见方向并非完全不能生成。与此同时，区间标签正确率仍约 50%，上述有效命中主要来自请求不可行的一侧，尚未证明能可靠控制边界两侧。不可行候选可能通过存档反馈产生搜索价值，不能只用可行率否定它；是否带来价值需结合完整运行的真实反馈与搜索结果。

数据：{link('1000 步留出对照', 'holdout/selected1000_summary.csv')}、{link('续训至 4000 步', 'holdout/continued4000_summary.csv')}。

生图：{link('LIRCMOP7 留出与续训', 'holdout/LIRCMOP7_BC_direction_holdout.png')}、{link('LIRCMOP8 留出与续训', 'holdout/LIRCMOP8_BC_direction_holdout.png')}。这些图只使用已缓存的真实目标，没有额外 CalObj/CalCon 调用。

## 6. 验证、成本与复现边界

- 六项相关回归通过：单点机制、容量与反馈、全局第一层与完整 Union、6001 FE 精确观察重放、PlatEMO 兼容、上一任算法独立性。六个修改后的核心/运行/生图文件 Code Analyzer 为 0 条消息，相关差异通过空白检查。
- 36 次完整运行均完成 100000 FE；每次搜索账单为 CalObjRows=200000、CalConRows=100000，额外 objective-only/constraint-only FE 为 0。BC 的 CalCon 内部再次调用 CalObj，所以每次完整评价有两次目标函数行访问。
- 搜索完成后的汇总曾因旧配对诊断对 NaN 使用逻辑运算而失败，已修正离线脚本并从原始结果重新汇总，没有重跑或更改搜索。离线绘图 v7 同时修正缺测说明，并以真实消费点作为金色标记。两项离线源码变更单独留档，完整搜索源码快照保持原样。
- 所有已消费 CGAN 子代来自保存的原生候选；消费时合法去重可回填普通 DE，未修改候选坐标。0/1 查询各 250 条，未绑定现有配对。全程存档资格、容量与累计更新计数已核对。
- 首次使用离线评估、完整运行条件审计、生图和留出检查分别记账；没有把额外标签回传训练或搜索。墙钟时间受并行竞争影响，不能将其当作独立机器上的严格速度比。
- 早期 44 个首训记录曾误把实验臂名称写入 sourceHash，已保留为 cacheKey，并明确标记“未记录精确运行源码哈希”；保留对应阶段源码快照，没有事后伪造精确哈希。其数值结果不变，其余 77 个首训记录有真实运行哈希。全部首训记录另存解析后的完整训练参数，避免新默认值改变旧配置含义。最终完整实验使用独立冻结源码与协议，且首次生成逐位复现。
- 已知上一任算法数值指纹检查的历史偏差没有作为本轮修复对象；本轮检查通过不代表它已修复。本轮没有提交、推送或更换分支，没有覆盖旧实验。

证据：{link('最终回归日志', 'implementation/final_regression.log')}、{link('静态检查', 'implementation/final_static.log')}、{link('完整运行核查', 'full_verification.csv')}、{link('条件审计独立账单', 'conditional_quality_offline_cost.csv')}、{link('最终配置', 'selected_configuration.json')}、{link('冻结源码清单', 'full_source_manifest.json')}、{link('首训源码来源说明', 'first_source_provenance.csv')}。
"""
(ROOT / "RESULTS.md").write_text(text)
print(ROOT / "RESULTS.md")
