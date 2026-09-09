from pathlib import Path
import csv, json, hashlib, statistics

P = Path(__file__).resolve().parent
ROOT = P.parent.parent

def read(name):
    with (P / name).open() as f:
        return list(csv.DictReader(f))

def link(name, title=None):
    return f'[{title or name}](<{P / name}>)'

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])

V = read('verification.csv')
assert len(V) == 24 and all(r['verified'] == '1' for r in V)
MV = read('mismatch_verification_10000.csv')
assert len(MV) == 2 and all(r['verified'] == '1' for r in MV)
S = [r for r in read('reset_summary.csv') if r['addedUpdates'] == '10000']
assert len(S) == 6
C = read('critic_condition_probe.csv')
assert len(C) == 30
MC = [r for r in read('critic_condition_probe_mismatch_10000.csv') if r['updates'] == '10000']
assert len(MC) == 2
names = {'original':'原方案', 'stable_side':'固定尺度＋0/1', 'stable_no_side':'固定尺度、去 0/1'}
parts = ['**FE99800 完全重置网络的验证结果**',
    '**完全重置后，仍会出现图中这类弯折和偏移。重置不是稳定的解决办法，旧权重或 Adam 历史不能单独解释这个现象。** 该结论针对图中六份固定晚期训练集，不能扩大为“任何重置都无效”或“历史状态没有影响”。',
    '本轮精确复现图中 FE99800 的训练数据、模型和生成点，然后冻结真实点、条件、尺度、网络结构与训练参数。每份数据比较原模型继续训练，以及 3 个全新 G/D 初始化；新模型的四个 Adam 矩估计和两个迭代计数均清零。24 个模型各训练 10000 次 G 更新，记录 1000、2000、5000、10000 步，使用原生采样器并核对完整随机数状态轨迹。训练中没有更新 P1/P2 或存档。',
    '下表为生成点到当前真实训练点集合的平均目标空间欧氏距离，越小表示越靠近该集合；不代表可行率或完整搜索收益。续训与重置均追加 10000 步，重置列是三个独立初始化的均值。']
parts.append(table(['方案','问题','图中原模型','原模型续训','完全重置'],
    [[names[r['arm']],r['problem'],*[f'{float(r[k]):.4f}' for k in ['baselineObjectiveGap','warmObjectiveGap','freshObjectiveGapMean']]] for r in S]))
parts += [f"18 个新初始化中，{sum(int(r['freshBetterThanBaseline']) for r in S)} 个的最终距离小于图中原模型，{sum(int(r['freshBetterThanWarm']) for r in S)} 个优于相同预算续训。六份数据中，重置均值仅在两份上优于图中原模型。覆盖距离也没有一致改善。三个初始化均完整展示，未挑选最好的一次。"+link('reset_summary.csv','逐组结果')+'；'+link('reset_records.csv','全部训练检查点')+'。',
    f"![原方案：完全重置与续训]({P/'figures/original_reset_10000.png'})",
    f"![固定尺度方案：完全重置与续训]({P/'figures/stable_side_reset_10000.png'})",
    f"![去掉 0/1 条件：完全重置与续训]({P/'figures/stable_no_side_reset_10000.png'})",
    '蓝点只表示生成结果；本实验没有再次进行候选筛选，因此不画被选择点的橙色标记。所有图使用相同坐标范围，并标出窗口外点数。',
    f"![固定数据上的训练轨迹]({P/'figures/objectiveGap_trajectories.png'})",
    '更长训练没有带来一致、持续的误差下降。10000 步是当前首训预算的十倍，但仍只是实验预算，不是数学上的收敛证明。',
    '**已经定位到的具体问题：当前 G 没有准确学到方向条件与高维解之间的关系；这种决策误差又被目标函数中的多变量耦合项转换为明显的目标偏移。** 原方案真实训练点的平均方向误差约 0.18°/0.20°；同一数据上，全新初始化 1 训练 10000 步后，已见条件的生成误差仍约 5.89°/5.39°。因此问题不只发生在未见方向，也不能用历史条件漂移解释这一静态对照。',
    '7/8 题的目标函数都包含由其余 29 个变量共同决定、系数为 10 的平方和项。以原方案 7 题为例，相对真实样本的同条件均值，原模型 f1 平均偏移为 −0.1927，其中 x1 项贡献 +0.0128，平方和项贡献 −0.2054。全新初始化 1 训练 10000 步后，平方和项仍贡献约 −0.2116。这解释了向训练边界内部偏移的一个具体来源；它是目标偏差分解，不是对训练失败根因的唯一证明。'+link('objective_offset_decomposition.csv','逐项分解与方向误差')+'。',
    '对网络表达能力也做了小范围诊断：以已见条件的加权决策均值作为输出目标，用同一个 G 做固定数据的全批次 MSE 拟合。原方案 7/8 的代表点 RMSE 在 10000 步后仍为 0.1015/0.0599；把两层宽度从 32 改为 128 后为 0.1006/0.0553，没有统一消除目标偏移。某些实验中决策 RMSE 降低，目标距离反而增大。因此，当前证据不支持简单换成重建损失或加宽网络。这个诊断改变了损失、批次与目标，不能被当作单独的损失消融。代表点仅作输出参照，未作为 G 的坐标条件。'+link('capacity_records.csv','同结构诊断')+'；'+link('capacity_wide_records.csv','宽度诊断')+'。',
    '**判别器存在可复算的方向评分异常，但针对它的最小干预没有解决生成偏移。** 保持真实解不变，只在同侧样本之间交换方向条件，所有方向条件的边际分布完全不变。多组 D 对错配组合给出更高平均评分；这种现象在约 5° 的局部错配中也存在，不仅限于很远的方向。它说明 D 的评分不能直接当作正确方向匹配的证据。'+link('critic_condition_probe.csv','30 个模型的评分诊断')+'。',
    '为检验因果，在原方案 7/8 上固定初始化 600001、数据、G 损失、网络和全部随机数轨迹，只将 D 的负样本改为一半生成点、一半同侧但错配方向的真实点；梯度惩罚也对这个负样本混合执行。两个模型训练到相同的 10000 步。D 学会了降低错配方向的分数，但生成距离没有改善。']
parts.append(table(['问题','原 D 生成距离','干预后距离','距离变化','原方向误差','干预后方向误差'],
    [[r['problem'],f"{float(r['baselineObjectiveGap']):.4f}",f"{float(r['mismatchObjectiveGap']):.4f}",
      f"{100*(float(r['mismatchObjectiveGap'])/float(r['baselineObjectiveGap'])-1):+.1f}%",
      f"{float(r['baselineDirectionError']):.2f}°",f"{float(r['mismatchDirectionError']):.2f}°"] for r in MV]))
parts += ['上述百分比按相同初始化逐一配对计算，准确值为 22.7% 和 24.6%；此前进度说明中第二题约 27% 的口头估算偏大。干预后的覆盖距离有所改善，8 题方向误差下降，但没有消除分布偏移，7 题方向误差明显增大。不能把“D 的错配评分修正了”直接推导为生成质量解决。'+link('mismatch_verification_10000.csv','干预核验')+'；'+link('critic_condition_probe_mismatch_10000.csv','干预后 D 评分')+'；'+link('mismatch_critic.patch','唯一训练改动')+'。',
    f"![判别器负样本干预]({P/'figures/mismatch_10000.png'})",
    '8 题干预后的 D 对生成点平均评分甚至高于真实点约 0.236，但生成偏差依然存在。这是一个已观测到的评分与几何质量不一致的例子；WGAN 评分不具备这种质量校准，不能据此单独判断训练已经成功，也不能把它当成唯一根因的证明。',
    '本轮的判断是：重置猜想已经得到直接检验，完全重置仍能复现同类问题。当前故障集中在固定晚期数据上的条件生成学习，以及决策空间学习误差与目标空间几何之间的关系。尚未证明某一损失项、网络宽度或初始化历史是唯一根因；D 的方向评分异常是已证实缺陷，其简单修正不足以解决问题。没有依据把这些实验改动推广为默认算法。',
    '这批实验衡量生成偏差，不要求每个生成点可行，也不能替代“是否引导 P1 探索新的边界”的完整搜索检验。接下来的工作应围绕固定数据上的条件生成任务和对抗优化展开；避免仅根据训练 RMSE、可行率或一张较好的图决定机制。',
    '核验与复现：'+link('verification.csv','24 组重置核验')+'、'+link('REPRODUCE.md')+'、'+link('experiment_plan.json','实验范围与探索性追加记录')+'、'+link('source_manifest.json','生产源码哈希')+'。新增模型训练共 34 组：24 组重置对照、8 组表达能力诊断、2 组 D 干预。离线完整评价合计 28364 FE，均独立计费且未反馈搜索；绘图、错配评分和目标分解只读取缓存或做网络前向。生产代码与默认参数未改，打包任务继续暂停。']
for name in ['figures/original_reset_10000.png','figures/stable_side_reset_10000.png',
             'figures/stable_no_side_reset_10000.png','figures/objectiveGap_trajectories.png','figures/mismatch_10000.png']:
    assert (P / name).exists()
(P / 'REPORT.md').write_text('\n\n'.join(parts)+'\n')
print('RESET_REPORT_WRITTEN', P / 'REPORT.md')
