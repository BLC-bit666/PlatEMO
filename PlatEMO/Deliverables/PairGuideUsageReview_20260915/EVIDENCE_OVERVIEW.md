# 当前证据总览（2026-09-15，只读打包）

目标尚未完成。当前默认仍是bridge，Q20/盒外最多4/.5截止。下表全部在当前固定参数下，每次100000FE，比较同骨架DE；负数更好。过程为200–100000FE原生P1 IGD梯形积分/99800；先算同题同种子比，再取几何平均。图的算术均值不是此配对汇总。每行保留全部对应运行，不删坏例。

5种子是开发证据，seed1–5已反复使用；2种子只是初筛，不能跨样本数将表排序成可靠冠军。bridge为已核验的历史同条件复用。DE复用不算新增独立证据。

| 用法 | 每题种子数 | 全程过程变化 | 最终变化 | 后期变化 | P6最终变化 |
|---|---:|---:|---:|---:|---:|
| bridge | 5 | -2.98% | +0.61% | +5.16% | +6.19% |
| boundary_local | 5 | -4.27% | -5.16% | +6.44% | +12.79% |
| boundary_segment | 5 | -2.99% | +13.98% | +28.09% | +5.26% |
| boundary_tube | 2 | +3.07% | -4.29% | +45.13% | +14.99% |
| generator_tangent | 2 | -2.35% | -4.02% | +10.54% | +36.93% |
| generator_tangent_anchor | 2 | -1.20% | -1.04% | +22.49% | +22.58% |
| generator_tangent_anchor_energy | 2 | -2.84% | +1.22% | +5.91% | +29.33% |
| critic_bridge | 2 | -0.36% | +5.78% | +11.86% | +14.65% |
| feasible_ray | 2 | -0.24% | -8.95% | +9.89% | +20.16% |
| feasible_ray_de | 2 | +2.89% | +3.09% | +62.81% | +30.34% |
| feasible_chord | 2 | -1.04% | -6.26% | +4.62% | +22.10% |
| coordinate_bridge | 2 | +2.76% | +1.99% | +52.24% | +13.29% |
| voronoi_bridge | 2 | +3.42% | +0.30% | +93.11% | +20.67% |

boundary_local整体过程−4.27%、最终−5.16%，但P6和后期退化，两个主指标名义95%区间均跨1；这是值得理解的局部有效机制，不是已经全面成功。boundary_segment从2种子扩至5种子后终点优势消失，P7 seed5真实发生大幅退化，保留该例。后续多数算子仍在P6或晚期失败。

## 已尝试的机制及阅读路径

- 当前bridge：p+0.5(q−p)+0.5(a−b)，接原OperatorDE。boundary_local用q定位真实F/I邻域，将桥接后的实际子代压入以(F+I)/2为中心、半径||F−I||/2的开球；boundary_segment/tube改变局部几何。精确实现见当前PairGuideBoundaryStep_RC/PairGuideBoundarySearch_RC及各历史源码。
- generator_tangent：保留G局部输出子空间内的DE分量，使用互补空间q拉力；anchor和anchor_energy改用真实F锚定/保持步长。critic_bridge加入现有D梯度。feasible_ray及feasible_ray_de以F为起点在q或q+DE方向收缩；feasible_chord用q定位两个真实F形成弦。
- coordinate_bridge把平均半强度q拉力改成私有随机流的15/30坐标完整拉力；voronoi_bridge把原桥接真实子代限制在档案F相对I的近邻半空间。两者8次完整筛选均失败；公式不能保证真实可行或接近CPF。
- 仅诊断、没有完整性能结论：LocalReflect（反射及软映射），DirectionUse（DE向q反射、角平分线），LeaderUse（q选真实F引导），TangentPull（保留完整DE+G子空间q拉力），GeneratorOrient（G子空间内反射DE的反向分量）。对应REPORT.md、诊断CSV和原型.m均收录。很多单代IGD持平，不能当成完整运行等效或无望。
- SideDirection为0FE只读模型诊断，不是已完成性能实验。G(z,w,1)−G(z,w,0)与邻近实测F−I的均值余弦P5/P6/P7/P8为+.0167/−.0584/+.3945/+.4973；P6仅3/20同向。不能把该差分当真实法向，也不能由首次四状态解释所有退化。
- LocalReflect缓存的80个已选q真值来自历史已付费离线绘图，仅20个可行；s=1的56个中16个可行，s=0的24个中4个可行。只作离线解释，绝不能反馈候选/训练/搜索。
- 历史UseDesign还试过pull、rotate、bridge、donor及当时配额自适应；旧全程启用/旧网络的优势不可移植到当前契约。ProposalCompletion/GPTValidation等保留早期建议核查及负结果，防止重复；当时能改网络/生成，不代表本轮仍允许。

## 已确定参数的来历（不是本轮优化目标）

SequentialTuning完成2020次独立完整运行，确定z12/G8×8/D16×16与batch上限64/G1000/20；其旧0.95截止已被后续约定替代。QuotaCutoff完成360次，曾选Q30/.5；seed1–5同条件Q20Review显示Q20过程更好，Q20R25的15+5未优于16+4，最终用户指定Q20/16+4/.5。旧报告中“当前”“默认”均按当时日期理解，不覆盖本包FROZEN_CONTRACT。

GPUTraining记录本机Apple GPU无法走当时原生MATLAB GPU训练路径；当前源码没有接入新的GPU训练后端。本轮保持现有执行环境，不讨论训练加速改造。

## 来源和限制

Data目录下保存报告、协议、逐次结果、完整轨迹、诊断、审计、原型与绘图脚本。SourceSnapshots内是tar归档拆出的原版源码；历史before/prior_runtime等也保留。SOURCE_VERSION_COVERAGE.json按旧manifest的path+SHA定位内容；不能用最新共同核心假装历史运行用了当前版本。当前Core支持的实验分支不等于默认已启用。

所有纳入文件按SHA去重；重复路径在FILE_INDEX.json保留映射。FIGURE_INDEX.json列出原图路径/原图哈希/无损显示像素哈希/分辨率。2026-09-14/15的科学图全部保留（非QA预览）；历史图优先IGD、汇总和诊断，然后首次/50k/100k分布全景，较低优先级重复视图按12MB无损图像额度取舍。旧分布图属于各自旧参数，不能冒充当前Q20新用法分布。排除清单明确列出重复PDF、启动日志、冗余完整MAT/JSON状态和早期重复首用图；结果CSV、失败及选择依据不按好坏删减。

多数MAT完整轨迹状态未收入，保留完整数值CSV及既有原生核验；额外纳入16份近期模型/状态MAT，供核对真实q、模型权重与几何输入。不能声称本包含原机每一字节或能脱离MATLAB复现训练。原生审计为当时结果，本次只做打包和数值交叉核对，不进行新训练或优化评价。
