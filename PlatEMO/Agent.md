# CBS RegionWGAN-GP 当前状态

更新时间：2026-07-13

## 1. 唯一研发主线

仓库只保留 `UC-GAN-2` 分支和一个可执行算法：

- 算法：`CBS_RegionWGAN_GP`
- 范式：条件 WGAN-GP
- 生成对象：完整决策向量 `x`
- 条件：参考向量 `W(ref,:)`
- 边界定义：目标空间可行区域与不可行区域之间的经验边界
- 在线信息：只使用进化过程中已经真实评价的个体
- 双群体、BMem 和在线 warm-start：保留

已删除：

- `CCMO-GAN-BDG` 整条算法线
- `CBS_CGAN`、`CBS_RegionCGAN` 及 BCE CGAN 实现
- UC、UC2、UC-GAN 等旧本地 Git 分支
- 旧 query、条件 scalar、动态训练、critic prescreen、pointwise、batch-only、MMD 等可执行实验分支
- 对应的旧 runner、测试、图像和重复实验数据

远程 Git 分支没有修改；本轮不提交、不推送。

## 2. 当前固定机制

### 2.1 每代流程

```text
P1/P2
  -> 两组 DE 后代
  -> 当前评价窗口更新 BMem
  -> BuildBoundaryDataset_RC 构造 TrainX/TrainC
  -> 条件 WGAN-GP 在线 warm-start
  -> one-sixth query 生成完整 x
  -> 真实 Evaluation
  -> P1/P2 两次环境选择
  -> query 转化、BMem 进入和最终群体归因
```

### 2.2 BMem 与训练集

- 可行 anchor 来自当前可行解的前 `frontDepth=2` 个目标 front。
- 上一轮 BMem 的可行 anchor 参与当前候选竞争，并只和当前不可行解重新配对；连续入选时 age 会继续累积，因此当前实现不是严格 TTL=1。
- 配对仅在参考向量邻域内进行；每 ref 最多保留 `maxAnchorsPerRef=5`。
- `TrainX=BMem.x_f`，即经验边界的可行侧完整决策向量；当前实现按 BMem 行直接训练，尚未对同一 `ref` 下完全相同的 `x_f` 去重。
- `TrainC=W(BMem.ref,:)`；不再存在 scalar condition 或多种 condition mode。
- 不可行端、pair gap 和目标空间几何只用于 BMem 构造及诊断，不进入 WGAN 损失。

### 2.3 唯一 query

当 populated 与一跳 frontier 同时存在时：

- frontier 数量：`round(nGen/6)`
- populated 数量：其余预算
- `nGen=30` 时为 5 个 frontier + 25 个 populated
- remote 条件预算恒为 0

没有 frontier 时，预算全部回退到 populated refs；没有 populated 支撑时不生成。

### 2.4 WGAN-GP

当前主线参数：

| 参数 | 值 |
|---|---:|
| `ganIter` | 100 |
| `nCritic` | 2 |
| `zDim` | 6 |
| `ganMiniBatch` | 32 |
| `ganLrD/ganLrG` | 1e-4 / 1e-4 |
| gradient penalty | 10 |
| G/C hidden | [32 32] / [32 32] |
| train sigma | 1.0 |
| sample sigma | 0.3 |

每个训练事件为 200 次 critic 更新和 100 次 generator 更新。没有动态迭代、训练触发、critic prescreen、额外生成损失或能量距离。

## 3. 已建立的实验结论

### 3.1 支撑错配已经修正

旧 `random_all_w` 会把大量生成预算投到无训练支撑的 remote refs。已完成的 6 问题 × 3 seeds 对照显示，限制到训练支撑附近可以明显降低真实边界距离和边界宽度。

当前 one-sixth query 在此基础上保留小预算的一跳扩展，同时禁止 remote 外推。它是唯一保留的 query，不再维护 `random_all_w`、`boundary_populated` 或固定 60/40 frontier 的源码分支。

### 3.2 当前厚带首因

正式诊断表明，BMem 本身不是当前厚带的首要瓶颈。主要断点位于：

```text
TrainX/TrainC -> conditional WGAN-GP -> generated x
```

同一个粗 ref 条件中可能存在多个相距很远的完整决策模式。随机 `z` 只提供表达这些模式的自由度，并不自动建立“每个 z 对应哪个模式”的可识别分工。小样本、粗条件、在线非平稳训练和对抗目标的弱约束会导致：

- generator 只覆盖部分决策支持；
- 不同 `z` 落到模式之间或偏离训练支持；
- 小的决策偏差经目标函数映射后被放大为目标空间厚带。

因此“存在 z”不等于“多模态已经被正确学习”。

### 3.3 已拒绝的最小机制

- pointwise same-condition assignment：边界收窄主要来自把不同 `z` 拉向较小决策支持，出现明显 mode collapse，拒绝。
- grouped batch-only：没有复制 pointwise 的严重坍缩，但几何收益不足，未通过 promotion。
- deployment MMD：降低了 generated-to-train 决策距离和宽度，但没有显著改善主要真实边界距离，18/18 problem-run 的同条件多样性护栏恶化，且运行时间显著增加，拒绝。
- coherent BMem（当前窗口归一化、严格 TTL、pair-before-cap、删除 dominance skip）：18/18 正式实验没有改善 BMem distance/width，distance 反而 16/18 变差，拒绝；不替换 legacy BMem。
- categorical-z + MI：Q 的 mode 识别可以学到，但生成支持显著外扩，主要 distance/width 和 populated width 均恶化；已完成部分形成一致强负信号，机制拒绝，不补跑剩余 runs。
- 同条件能量距离：用户明确排除，不设计、不实现、不实验。

这些机制只在保留证据中存在，不能从历史记录恢复为当前分支。

### 3.4 legacy BMem 去重与低价值模态审计

已对 exact-current legacy control 的 6 问题 × 3 seeds、7,492 个 snapshots 做完只读审计：不训练 GAN、不调用真实 `Evaluation`、不改变主线。

- 592,626 个 BMem 行中只有 312,327 个同 snapshot/ref 唯一 `sample_id_f`；280,299 行是额外重复权重，占 47.30%。
- 150,549 个 snapshot/ref 组中，123,170 组含重复，占 81.81%；其中 13,728 组是 5 行只对应 1 个唯一 anchor。
- 重复来源中，81.83% 是 current 与 previous anchor 混合重复，13.67% 是 current-only，4.50% 是 previous-only；说明重复既来自双群体当前窗口，也会由旧 BMem 续存累积。
- 精确去重后，当前 `minGANTrainCount=32` 下可训练 snapshots 从 6,408 降至 4,543；1,865 个当前可训练事件（29.10%）会落到阈值以下。因此去重不是无行为影响的清理，必须单独实验。
- 去重后的 150,549 个组中，严格相邻 snapshot、局部分离、在线质量支配同时成立的可删除组只有 2,864（1.90%），只会删除 3,652/312,327（1.17%）唯一 anchors。
- 低价值模态删除相对纯去重只显著缩小 width（Holm `p=0.0434`），但 distance 不改善（Holm `p=0.375`，18-run 中 5 胜/11 平/2 负）；两个主指标联合门失败，直接聚类删除路线拒绝。

随后完成了“只在训练集中按 `(ref,x_f)` 精确去重”的 6 问题 × 3 paired seeds 正式实验：

- A/B 均为 18/18 `ok`，18/18 严格 `finalFE=maxFE=100000`，remote query 为 0，且未生成图。
- B 共删除 280,809 条精确重复训练行；可训练/生成事件由 A 的 6,396 降为 B 的 4,514，下降 29.42%。
- 主指标均未通过：`bdist50_true` 中位配对差 `+8.36e-4`（5 胜/0 平/13 负，Holm `p=0.4969`）；`bwidth90_10_true` 为 `+8.94e-3`（8/0/10，Holm `p=0.4969`）。
- 同条件决策多样性偏差在 17/18 个 problem-run 上恶化（中位差 `+0.242`，`p=3.27e-4`）；frontier width 也恶化（中位差 `+0.0442`，5/0/13，`p=0.0222`）。
- anchor utilization 在 18/18 上提高（中位差 `+0.238`），墙钟在 18/18 上降低（中位 `-224.3 s`），但这些次级收益不能覆盖两个主几何指标和多样性护栏失败。
- 预注册结论为 `reject`：保留 legacy 逐行训练权重，停止精确去重与低价值聚类清理路线，不增加 seeds，不叠加 mode label、额外损失或阈值补偿。

审计证据位于：

`Data/CBS_RegionGAN_compare/bmem_clusterability_audit_20260712_234214`

精确去重正式实验与预注册分析证据位于：

`Data/CBS_RegionGAN_compare/train_exact_dedup_20260713_001304`

## 4. 尚未证明

- 尚未证明 CGAN 在无训练支撑区域优于同 ref 的 DE。
- 尚未证明 frontier -> populated 转化能稳定改善最终 IGD/HV。
- 尚未完成 exact-current no-GAN 和等 FE extra-DE 的最终因果对照。
- 当前真实边界正式诊断只覆盖二维目标的 LIRCMOP5_BC–10_BC，不能直接外推到 M>2。
- 约 2% 的历史即时生存率不足以单独证明 GAN FE 划算。

## 5. 保留证据

唯一数据入口：

`Data/CBS_RegionGAN_compare/retained_evidence`

内容：

- `current_mainline_off`：最新 6 问题 × 3 seeds 主线汇总、最终群体及 query 转化证据；
- `query_selection`：旧 query 与 one-sixth promotion 的紧凑比较证据；
- `rejected_pointwise_assignment`；
- `rejected_batch_only`；
- `rejected_deployment_mmd`；
- `retention_manifest.csv`：文件大小与 SHA-256。

旧 CSV 中的绝对路径只表示历史来源，不是当前运行依赖。跨实验比较必须同时核对 `provenance.csv` 和 `source_manifest.csv`，不能只看 Git SHA。

## 6. 当前源码入口

- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionWGAN_GP.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionGAN_Base.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/BuildBoundaryDataset_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/RunRegionGAN_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryWGAN_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/RegionQueryTracker_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionWGAN_GP_mainline.m`

主 runner 是非绘图 runner，默认正式范围为：

```text
LIRCMOP5_BC–10_BC
N=100, D=30, maxFE=100000
seeds=1,2,3
```

## 7. 重构与性能约束

本轮重构以固定种子核心轨迹的逐文件 SHA-256 为等价门禁：

- final P1
- final P2
- final feasible ND
- query samples
- query ref conversion
- direct GAN BMem entry

重构后的 10k FE 语义检查与重构前完全一致。

Profiler 显示约 90% 时间在 WGAN 的 `dlfeval`、critic 梯度和网络 forward。曾验证 MATLAB `dlaccelerate` 候选：单次 10k FE 从暖机约 23–25 秒降到 12.60 秒，但改变了上述核心哈希，因此已撤回。严格轨迹等价优先于该加速，不保留数值路径变化。

## 8. 当前验证收口

当前范围内的“同条件 anchor 聚类/低价值模态清理”已完成收口：

| 指导项 | 当前结论 | 后续动作 |
|---|---|---|
| 直接聚类删除 | 联合主指标门失败 | 拒绝 |
| `(ref,x_f)` 精确去重 | 主指标、多样性和 frontier 几何护栏失败 | 拒绝，恢复 legacy 行权重 |
| `50×2 + signature skip` 速度实验 | 机制阶段未 promotion | 不启动，不能用提速收益包装几何失败 |
| no-GAN / extra-DE / 最终 IGD-HV | 用户明确暂停 | 不计入当前阶段欠账 |
| M>2 推广 | 当前范围外 | 不实验 |
| 决策误差映射放大 | 用户已排除 | 不恢复 |

因此当前没有被证据和阶段门允许启动的新实验。主线保持 legacy BMem、legacy 逐行训练、one-sixth query 和 WGAN 100×2。不恢复已拒绝的聚类、去重、categorical-z、pointwise、batch-only、MMD 或额外损失。

后续机制若被提出，仍必须只有一个实验分支，并预先说明：

- 唯一缺陷；
- 参数数目和自适应来源；
- 更新量、墙钟时间和 FE 代价；
- `bdist50_true`、`bwidth90_10_true`、coverage、多样性、survival 和最终优化护栏；
- promotion / hold / reject 规则。

不恢复旧 query、scalar condition、pair-thin、pointwise、batch-only、MMD、能量距离、categorical-z、聚类删除、prescreen、动态训练或更深网络的组合分支。
