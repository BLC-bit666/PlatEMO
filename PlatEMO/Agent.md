# PairGuide 当前主线契约

更新时间：2026-09-04

适用范围：`/Users/lanai/Code/Matlab/PlatEMO/PlatEMO` 当前工作树。本文件只记录已经确认并正在执行的算法，不记录被否决方案、阶段实验结果或过期 campaign。

## 1. 算法身份

- 当前 Git 工作分支：`UC-GAN-2`；本地没有其他 Git 分支。
- 当前主线：`PairGuide`；实现类：`PairGuideCore`。
- 上一任主线保留原名 `CBS_RegionWGAN_GP`，实现类为 `CBS_RegionWGAN_GP_Core`；其行为是 `a1e8e436` 中的 global-critic + `A/h/T` + 50% FE cutoff。
- 两个公开类各自连接自己的实现，不互相继承、委托、替名或充当兼容别名。
- `Deliverables/` 中的审查快照只作冻结证据；`addCBSPaths` 会移除其全部路径，禁止其中旧类参与 MATLAB 类解析。

活动路径只保留两个算法入口：当前 `PairGuide` 与上一任 `CBS_RegionWGAN_GP`。旧实验分支和旧 PairGuide 长名称均不参与 MATLAB 类解析。

## 2. 固定配置

| 项目 | 当前值 |
| --- | ---: |
| P1 GA 份额 | 25% |
| P1 普通 DE 份额 | 55% |
| P1 PairGuide 名额 | 20% |
| P2 GA / 普通 DE | 25% / 75% |
| raw `G` 查询数 | 500 |
| `refDivisor` | 1 |
| 每参考方向 active pair 上限 | 1 |
| pair 邻域总参考方向数 | 5，含自身 |
| 最少 active pair | 32 |
| 首次训练 | 500 Epoch |
| 后续重训 | 20 Epoch |
| mini-batch | 32 个完整 pair，即 64 个端点 |
| Critic : Generator | 5 : 1 |
| G/C 隐藏层 | `[32,32]` |
| `zDim` | 6 |
| 学习率 | `1e-4` |
| WGAN-GP 系数 | 10 |
| 几何损失权重 | 1 |
| 训练 / 推理噪声标准差 | 1 / 1 |
| 重训变化阈值 | 8 个 pair |
| 重训新参考方向阈值 | 2 个方向 |
| inactive 删除阈值 | 连续 10 代 |
| 固定 FE cutoff | 无 |

P1 的整数舍入余数、PairGuide 未就绪或筛选不足的名额全部由普通 DE 吸收。不得用问题类型、运行阶段或人工经验动态改配额。

## 3. 当前算法流程

### 3.1 初始化

1. 用 `UniformPoint(max(2,round(N/refDivisor)),M)` 建立参考向量；以函数实际返回数量为准。
2. 分别调用 `Problem.Initialization()` 初始化 P1 与 P2，各进行真实完整评价。
3. P1 使用 `CalFitness_CBS(objs,cons)`；P2 使用 `CalFitness_CBS(objs)`。
4. pair archive、CGAN、下一代 donor 池均从空状态开始。

### 3.2 每代搜索与选择

1. P1 先消费上一代末生成的 donor 池。完整 `N=100` 代名义产生 25 个 GA、55 个普通 DE、20 个 PairGuide 子代；缺失 PairGuide 子代立即由同一普通 DE 路径回填。
2. P2 独立产生 25 个 GA 与 75 个普通 DE，不读取 pair archive、CGAN 或参考方向引导。
3. 原始生成点和 objective-only shortlist 均不进入种群。共享候选集严格为 `P1 + P2 + O1 + O2`。
4. 从同一共享候选集更新两套种群：P1 使用约束适应度，P2 使用无约束适应度；两者都使用现有 raw-objective SPEA2 密度与字典序截断。
5. 用更新后的 P1 和本代真实评价联合集更新 pair archive。
6. 满足门槛时首次训练或 warm-start 重训 CGAN；可用模型随后查询并筛选 donor，供下一代 P1 使用。
7. 重复以上流程，直到严格总 FE 预算耗尽。PairGuide 没有固定阶段切换或永久关闭点。

下一代使用上一代末 donor 的时序不得改成同代即时使用；环境选择也不得从共享 Union 改成种群专属候选池。

## 4. Pair archive

每条记录不可拆分：

```text
id, xf, yf, xi, yi, ref, gap, rank, fitness,
age, lastFE, active, resumeEligible
```

- `xf/yf` 必须来自真实完整评价的可行点；`xi/yi` 必须来自真实完整评价的不可行点。
- 当前建对可行端只取更新后 P1 中真实可行且 `Fitness < 1` 的精英。
- 不可行候选来自本代完整已评价联合集、已归属 PairGuide 子代和仍合法的历史 `xi`，不要求属于当前 P2。
- `xi` 必须未被任一当前 P1 可行精英支配；同时配对 `xf` 不得支配 `xi`。这是档案合法性，不得与下述 raw `G` 规则混淆。
- `ref` 由 `yf` 在统一 `RefScale` 下确定。`xi` 只允许位于该方向及其角度最近方向组成的总数 5 邻域。
- 对每个可行精英，合法 `xi` 按归一化决策距离最近原则配对；每个参考方向最终按 `(gap, xi 方向角度排名, Fitness(xf), -lastFE, id)` 保留唯一 pair。
- 带 `matchedPairID` 的真实 PairGuide 子代只能更新对应 pair；其他真实评价点可在相同合法性与严格 gap 缩短条件下更新局部 pair。
- 可行端或不可行端只有在归一化 gap 严格缩小时替换。可行端替换同步更新 `ref`；不可行端替换不改变 `ref`。
- 新建、严格收紧或仍获当前 P1 局部支持的合法 pair 为 active。inactive 连续达到 10 代后删除。
- 仅因失去 P1 局部支持而暂停，且端点、合法性与 `ref` 均未改变的精确同 pair，恢复时复用旧 `id/lastFE`。
- 曾变得目标或邻域非法，或无新端点评价却发生 `ref` 重归属时，`resumeEligible=false`；以后重建必须分配新 ID。
- pair ID 只承担反馈归属和训练集变化检测，不进入 CGAN 条件。

## 5. PairGuide CGAN

训练样本使用归一化决策空间绝对端点：

```text
[xf_norm, w, 1]
[xi_norm, w, 0]
```

同一 pair 两侧用同一个 `z`。Critic 使用标准 WGAN-GP。Generator 固定使用：

```text
L_G        = L_adv + lambda_geo * (L_anchor + L_pair)
lambda_geo = 1
L_anchor   = sum(||G(z,w,1)-xf||^2 + ||G(z,w,0)-xi||^2) / (2*B*D)
L_pair     = sum(||(G(z,w,0)-G(z,w,1))-(xi-xf)||^2) / (B*D)
```

- 首次训练要求至少 32 个 active pair。
- 后续仅在训练集成员或端点累计变化至少 8 个，或新增参考方向至少 2 个时重训。
- 每个 Epoch 将全部 active pair 随机打乱并无放回访问一次，完整 pair 不拆分，尾批保留。
- 每个 batch 先执行 5 次 Critic 更新，再执行 1 次 Generator 更新。
- 后续训练保留 G、C、Adam 状态并 warm-start；训练结果立即采用，不设置 checkpoint 接受/拒绝分支。

## 6. donor 生成、筛选与接入

设本代 PairGuide 名额为 `Q`：

1. 在全部 active pair 上尽量均衡分配并随机打乱 500 个 `[w,0]` 查询；查询同时携带 `pairId` 供归属。
2. raw `G` 只由生成器前向得到，不调用目标或约束，不创建 `SOLUTION`，不进入 Union，也不更新 archive。
3. 对每个 pair，从当前 P1 的真实可行 `Fitness < 1` 精英中，只在同一总数 5 的参考邻域内找距离 `xf` 最近的父代 `c`。
4. 在归一化决策空间计算：

   ```text
   r   = ||xi-xf||
   e   = ||c-xf|| + ||G-xi||
   rho = e/(r+eps)
   ```

   仅保留 `rho < 1`，并且每个 pair 只留 rho 最小的一个 raw `G`。
5. 候选按 `(occupancy(ref), rho, -gap, id)` 字典序排序；最多取前 `2Q` 个调用真实 `Problem.CalObj`。
6. objective-only 候选必须满足：`G` 不支配配对 `xi`、配对 `xf` 不支配 `G`，且归一化目标逐维落在 `yf` 与 `yi` 的闭合 Pareto 走廊内。
7. 不对 raw `G` 增加“未被任一当前可行精英支配”的全局过滤。该全局过滤只适用于档案不可行端 `xi`。
8. 通过走廊的前 `Q` 个 donor 与各自父代直接产生 `u = CalDec(0.5*(c+G))`；不做 DE 差分、多项式变异、`A/h/T` 映射或二次筛选。
9. 二分后才检查 `u` 是否与当前 P1 或本代已选 PairGuide 子代重复。保留子代接受完整评价并进入下一次共享 Union；空缺继续由普通 DE 回填。

## 7. 严格评价成本

| 对象 | 目标 | 约束 | 计入总 FE | 进入 Union |
| --- | --- | --- | --- | --- |
| 500 个 raw `G` | 否 | 否 | 否 | 否 |
| 最多 `2Q` 个 shortlist `G` | 是 | 否 | 是，另记 `ObjFE` | 否 |
| 最多 `Q` 个二分子代 `u` | 是 | 是 | 是 | 是 |
| 普通 DE 回填 | 是 | 是 | 是 | 是 |

objective-only 调用不是零成本。它通过 `Problem.FE` 纳入与完整评价相同的严格总预算，但不伪造约束值。六个活动 `LIRCMOP*_BC` 类的 `CalObj`、`CalCon` 与 `Evaluation` 必须保持公式一致。

## 8. 禁止漂移

`PairGuide` 不启用上一任主线的 global critic 排序、500→200 critic 筛选、`A/h/T` 映射、50% FE cutoff、主动边界校准或 late boundary-target。也不保留 DE20、Quota30/40/50、A0/A1/A2、E0—E8、Random20、GA20、FullCGAN 等活动入口。

不得在未获明确授权时新增网络、损失项、动态权重、自适应配额、问题专用规则、额外评价、并行随机路径或不同环境选择。性能优化必须保持选择顺序、随机数消费、FE 记账和数值结果不变。

## 9. 权威文件与验证

当前主线权威实现：

- `Algorithms/Multi-objective optimization/PairGuide/PairGuide.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Core/PairGuideCore.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Core/PairBoundaryArchive_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Core/PairBoundaryWGAN_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Core/EnvironmentalSelection_CBS.m`

上一任主线权威实现：

- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionWGAN_GP.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Core/CBS_RegionWGAN_GP_Core.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Core/RefineBoundaryObservations_RC.m`

最小必跑回归：

```matlab
test_CBS_pair_guide
test_CBS_pair_guide_training_observation
test_CBS_platemo_compliance
test_CBS_region_wgan_mainline
test_CBS_mainline_fingerprint
```

回归必须覆盖：两个入口相互独立、各自确定性指纹、20% 配额与 DE 回填、完整 Epoch 访问、warm-start 触发、matched ID 隔离、合法暂停恢复、非法后新 ID、总数 5 邻域、每方向唯一 pair、`rho < 1`、Pareto 走廊、objective-only FE 和 `rawOracleCount=0`。
