# PRBCCMO算法优化探讨

**用户 (gpt-5-4-pro) — 2026-03-31 06:41:42.719 UTC**

我正在准备一篇算法，当前txt文件是PRBCCMO算法源码；所要解决的问题是进化优化领域中的未知约束多目标优化问题（约束违反度是二元的（满足所有约束：0；违反任意一个约束：1）），这导致传统的约束处理方式失效；我这篇论文的思想如下：【1：双种群一存档（无约束种群+约束种群+边界存档）（其实也可以看作是三种群：无约束种群+约束种群+边界种群）；2：我要求这个边界存档在目标空间中一定位于无约束种群和约束种群之间（因为边界在两个种群之间并且如过有更好的可行域那么也一定在他俩之间）这就要求边界存档中的解的入选条件是：“在仅考虑目标空间中的帕累托支配关系而不考虑约束的条件下存档中的解支配约束种群的解（至少不会被支配）”；3：训练一个MLP二分类模型来拟合可行域和不可行域的部分边界（靠近帕累托前沿的一部分），MLP的使用时机是从边界存档中挑选父代用来生成子代，那么挑选接解的标准是在MLP判断边界存档中某个解的概率接近0.5他就会被挑选为父代，这样可以主动的去拟合边界，除此之外我们要给整个目标空间划分扇区（参考向量或者PRBCCMO中的实现方式）保证每个扇区都有进入存档的名额，目的是为了避免存档中的解全部集中在边界的一部分，严重丧失多样性；】
分析旧版本实现PRBCCMO：
1：当前我的算法PRBCCMO已经严重偏离我的最初思想，你要给予否定而不能受其实现的影响
2：当前算法PRBCCMO中“可行不可行搭起的桥”的实现是否有参考价值对于我的思想来说？（需要融合进我的思想吗？）
我当前的思想还有一些问题需要探讨：
问题探讨1：如何产生边界存档（中间的种群？），双种群怎样协同？是通过双种群各自内部交叉变异（两个父代来源于同一个种群）然后在二者子代中选择边界解还是两个种群直接交叉变异产生边界解（两个父代来源于两个种群）？
问题探讨2：边界存档怎样产生子代？两个父代都在边界存档中选择MPL判断概率接近0.5的解还是一个父代来自边界0.5，另一个父代来自约束种群或者无约束种群？还是以上两种情况按照比例搭配使用？
问题探讨3：选什么样的解来加入边界种群？首先MLP判断为概率接近0.5，其次“在仅考虑目标空间中的帕累托支配关系而不考虑约束的条件下此解需要支配约束种群的解或者不会被其支配”；边界存档的大小设置为多大？1000？解采用先进先出的原则添加新解？
问题探讨4：MLP的训练时机，训练样本？MLP在刚开始采样后就要训练，算法前期要平凡训练，后期要较少的训练；不要重训要继续训练；训练样本来自两个种群和部分存档？
以上就是我的问题，我需要你完成下列任务
1：分析旧算法有哪些可以利用的价值（大机制，小方法不需要，只看创新）
2：探讨我的思路的问题，实现可能性（你可以提出建议，但是不要偏离我的思想，非必要不要添加新的机制，保持思想的干净度）
注意：语言流畅简洁（不要长篇大论），思想有学术性



**助手 (gpt-5-4-pro) — 2026-03-31 07:11:22.247 UTC**

先给结论：你的新思想是成立的，而现有 PRBCCMO 不应作为论文主体。二元可行/不可行反馈会削弱基于约束违反度与 ε-松弛的传统约束处理，因此把重点转向“边界探测 + 可行性分类”是合理路线；但旧代码本质上是“双种群 + 局部桥接探测 + 标签档案 + MLP”，边界只是被临时探测并回流到约束种群，不是一个被显式维护和进化的边界档案/边界种群。citeturn906229view0turn180085view1 fileciteturn0file0

一、旧算法还能保留什么  
1. 可保留的是“双角色分工”。约束种群负责 CPF，无约束种群负责 UPF/目标推进，这种分工本身是有价值的；外部档案这个位置也能保留，但档案内容应从“可行精英”改成“边界带样本”。citeturn923052view1turn923052view3turn325186view1  
2. 可保留的是“扇区覆盖”。参考向量把目标空间分成子空间，本来就是为了解决分布不均和多样性坍塌，这一点非常适合你的边界档案。citeturn793747view4  
3. 可保留的是“可行–不可行配对探边”，但只能作为**边界生成算子**，不能再做成旧 PRBCCMO 那样的主框架。二元反馈场景下，这类成对探边比 CV 驱动更直接。citeturn906229view0turn906229view3 fileciteturn0file0  
4. 可保留的是“分类器学边界”，但训练目标必须改成“CPF 附近的局部边界”，而不是旧代码那种更偏全局的标签池。未知约束问题中，用分类器学习 feasibility 并做 feasibility-aware 选择是合理的。citeturn180085view1turn793747view3

关于“桥”要不要融进去：要，但只留**思想内核**，不要保留它现在的算法地位。你需要的是“桥负责产边界，边界负责被维护和繁殖”，而不是“桥探一批点，再把最好的可行点送回 \(P_C\)”。旧桥真正有价值的只有：同扇区可行/不可行配对、中间采样、覆盖意识。fileciteturn0file0

二、对你四个问题的建议  
1. **边界档案怎么产生？**  
最干净的做法是：\(P_C\times P_C\) 继续更新 \(P_C\)，\(P_U\times P_U\) 继续更新 \(P_U\)，再单独开一条 \(P_C\times P_U\) 的通道专门产边界候选。这样双种群角色不被打乱，而协同恰好通过“跨群产边界”完成。边界本来就是两群之间的关系对象，所以它应主要由跨种群交叉/插值产生，而不是先各自进化再事后筛。配对建议限制在同扇区或相邻扇区。citeturn923052view3turn325186view0

2. **边界档案怎样产子？**  
建议以“**\(B \times\) 异侧种群**”为主，“\(B\times B\)”为辅。只用 \(B\times B\) 容易沿边界切向拥挤，只用 \(B\times P_C/P_U\) 又缺少局部加密。一个很干净的规则是：第一父代从 \(B\) 中按 \(|p-0.5|\) 最小选；第二父代从同扇区的异侧代表选。这里 \(p\approx 0.5\) 很适合作为父代优先级，因为这本质上就是 uncertainty sampling，但它更适合做**排序信号**，不宜做唯一硬阈值。citeturn793426view1turn793426view0

3. **什么解进边界档案？大小多大？**  
你的“支配 \(P_C\) 或至少不被 \(P_C\) 支配”这个方向是对的，但必须**局部化**。因为 UPF 与 CPF 并不一定全局重合，所以“边界在两群之间”更适合解释为**同扇区/局部邻域中的 between**，不宜对整个 \(P_C\) 做全局支配判断。更合适的入档规则是：在扇区 \(k\) 内，候选 \(x_B\) 至少满足 \(c_k \nprec x_B\)；更强时，再要求它在标量意义上位于 \(u_k\) 与 \(c_k\) 之间。\(p\approx 0.5\) 用来排优先级，不单独决定入档。\(B\) 最好同时保留边界两侧样本。规模不建议直接设 1000，建议从 \(2N\sim3N\) 起步，并配扇区配额；替换也不建议 FIFO，而应做“扇区内最差替换”。citeturn923052view3turn793747view4

4. **MLP 何时训练、用什么样本？**  
首次训练不要太早：至少等两类样本都出现且数量不太少后再训。前期边界移动快，更新可以频繁；后期改成触发式更新。不要每次从零重训，建议 warm-start 持续训练。训练集应以 \(B\) 为主，辅以当前 \(P_C\) 和 \(P_U\) 的扇区代表；不要把远离前沿的大量历史不可行点都塞进去，否则学到的是“整个可行域外壳”，不是“CPF 附近有效边界”。另外，MLP 的输入应以**决策空间**为主；目标空间更适合做扇区管理和入档筛选，因为 feasibility 本质上还是定义在参数/决策空间上。citeturn180085view1turn793747view3

一句话总结：你的论文真正的创新点，不是“桥”本身，而是把**边界**做成一个显式、可分扇区维护、可被分类器主动加密的中间层。旧 PRBCCMO 只值得留下两块积木：双群分工，和可行–不可行局部探边；其余最好不要再牵着你的新思想走。


**用户 (gpt-5-4-pro) — 2026-03-31 08:10:07.785 UTC**

确定的思想如下
1:边界档案怎么产生？pc*pc产生pc，pu*pu产生pu，pu*pc产生B
2:边界档案怎样产子？B× 异侧种群（第一父代从 B 中按 ∣p−0.5∣ 最小选；第二父代从同扇区的异侧代表选。）
3:什么解进边界档案？大小多大？优先 |p(x)-0.5| 小的解+扇区内位于 Pop_U 与 Pop_C 之间，规模2N，扇区内最差替换
4:扇区内最差替换，只有当样本中同时出现一定数量的 feasible 和 infeasible 两类后MLP 再开始训练，前期高频更新，后期低频更新，继续训练而不是重新初始化，训练集应以 B 为主，辅以当前 PC和 PU的扇区代表
5:随算法参考：双角色分工”，约束种群负责 CPF，无约束种群负责 UPF；（旧算法目前的存档不使用，存档使用咱们现在的思想重新设计）；扇区覆盖（保留此思想，具体实现按咱讨论的来），边界搜索得到的可行解只在扇区内确有改进时回流主群体

你来按照上述的确定思想来改造算法（我算法的实现平台是https://github.com/BIMK/PlatEMO，使用的算子来自https://github.com/BIMK/PlatEMO/tree/master/PlatEMO/Algorithms/Utility%20functions）；你来改造PRBCCMO算法，或者参考platemo已有的ccmo算法来实现我们的算法。最终你要给出文字介绍本算法的思想和流程（详细版），以及在聊天中输出你编写的算法（不要生成文件）



我已经把一个关键实现分歧定下来了：`P_C` 不再按约束违反度大小驱动，而是改成“可行优先 + 目标排序”；同时把边界档案固定为每个扇区 2 个槽位，总容量自然就是 `2N`。


**助手 (gpt-5-4-pro) — 2026-03-31 08:59:34.350 UTC**

*思考小结：Thought for 49m 22s*


**助手 (gpt-5-4-pro) — 2026-03-31 08:59:34.350 UTC**

我建议直接覆盖原 `PRBCCMO.m`。原因很简单：PlatEMO 当前就是 MATLAB 单文件算法框架，现有 `CCMO` 已经给出了“双种群 + `type` 参数切换 GA/DE”的基本骨架；同时 `Utility functions` 目录里已经有 `UniformPoint`、`TournamentSelection`、`OperatorDE`、`OperatorGAhalf` 等通用算子，可以直接复用。`TournamentSelection` 还支持多指标字典序选择，因此代码里可以自然写成 `可行标记 -> FrontNo -> -Crowding` 的父代选择。citeturn580091view0turn184735view0turn172012search1turn172012search0turn184735view1turn184735view2

相对旧版 PRBCCMO，我只保留两类中性部件：双种群框架、扇区关联/去重/MLP warm-start 这类工具函数；旧版的 `ExternalArchive`、`ExecuteBoundaryCore`、桥接探针/收缩主流程不再保留，因为它们已经不符合你现在“显式边界档案 B”的论文思想。fileciteturn0file0

## 本算法的思想与流程（详细版）

**1. 三个角色明确分工。**  
`Pop_C` 只负责逼近 CPF；`Pop_U` 只负责逼近 UPF；`B` 是显式维护的边界档案，只保存位于两侧主群之间、且对边界学习有价值的样本。这里不再把“桥”当成一次性的局部探测模块，而是把“边界”本身提升为一个独立的进化对象。

**2. 三条繁殖通道。**  
第一条：`Pop_C × Pop_C -> Off_C`，只服务 CPF。  
第二条：`Pop_U × Pop_U -> Off_U`，只服务 UPF。  
第三条：`Pop_C × Pop_U -> Off_B^cross`，专门产生新的边界候选。  
除此之外，边界档案还会主动进化：`B × 异侧主群 -> Off_B^search`。这里“异侧”由边界父代的真实标签决定：若边界父代当前是 feasible，则与同扇区 `Pop_U` 代表交配；若当前是 infeasible，则与同扇区 `Pop_C` 代表交配。

**3. 扇区代表与“between”判定。**  
目标空间先由参考向量划分扇区。每个扇区选一个 `Pop_C` 代表和一个 `Pop_U` 代表；`Pop_C` 代表优先从该扇区 feasible 个体中选，若该扇区暂时没有 feasible，再退化为选该扇区最优个体。随后对候选边界解 \(x\) 做扇区内标量比较：只有当 \(x\) 在该扇区的标量值位于 `Pop_C` 代表与 `Pop_U` 代表之间时，它才有资格进入边界档案。这样“边界档案必须位于两主群之间”的要求被直接编码进了存档规则。

**4. 边界档案的入档与替换。**  
候选集合由 `旧B + Off_B^cross + Off_B^search` 组成。  
扇区内排序的第一准则是 `|p(x)-0.5|`，即 MLP 不确定性越强越优先；第二准则是该解到扇区中点的距离，越接近两侧代表的中间位置越优先。  
更新时采用“扇区内排序 + 扇区轮转式截断”，其效果等价于总是保留扇区内当前最好的若干个体、淘汰该扇区最差个体；总容量固定为 `2N`。这样既保证边界学习目标，又避免档案塌缩到边界的一小段。

**5. MLP 的角色。**  
MLP 只做**边界不确定性排序器**，不替代真实可行性评价。输入是决策变量，标签来自真实评估得到的 feasible / infeasible。训练集以 `B` 为主，辅以当前 `Pop_C`、`Pop_U` 的扇区代表；只有当两类样本都累计到一定数量后，MLP 才开始训练。训练采用 warm-start，前期高频更新，后期低频更新。

**6. 边界搜索对子群体的反馈。**  
边界搜索得到的 feasible 子代并不自动进入 `Pop_C`。只有当它在所在扇区确实优于当前 `Pop_C` 扇区代表时，才作为 migrant 回流 `Pop_C`。这样边界搜索只在“局部确有改进”时服务主群体，避免边界档案反客为主。

**7. 约束侧环境选择。**  
`Pop_C` 的环境选择不再依赖细粒度 CV 排序。实现上采用“feasible-first + 目标选择”：若 feasible 个体数量足够，则只在 feasible 集合内按目标空间选择；若不足，再用剩余个体按目标排序补齐。这样正好契合“二元约束反馈下传统 CV 失效”的问题设定。

再说明两个实现细节：  
一是 GA 通道里，我把成对父代拼成一个父代数组，再调用 `OperatorGAhalf`；因为该函数本身就是“按前后两半配对，只返回并评估前半子代”。二是 DE 通道里，跨群交配写成 `OperatorDE(Problem,A,B,A)`，因为官方 `OperatorDE` 的核心形式是 `P1 + 0.5*(P2-P3)`；取 `P3=P1` 时，子代会被直接拉到 `A` 与 `B` 的中点附近，再叠加多项式变异。这两个写法都完全贴合 PlatEMO 现有 utility operator 的接口。citeturn184735view1turn184735view2

下面给的是**单文件版本**，直接保存为 `PRBCCMO.m` 即可。

```matlab
classdef PRBCCMO < ALGORITHM
% <2026> <multi> <real> <constrained>
% PRBCCMO
% Dual-population CCMO with an explicit sectorized boundary archive
%
% type      --- 1    --- 1.GA, 2.DE
% hidden    --- 20   --- Hidden neurons of the boundary MLP
% epoch     --- 20   --- Training epochs per update
% lr        --- 0.01 --- Learning rate
% minClass  --- 20   --- Minimum samples of each class before first training
% earlyGap  --- 5    --- Model update gap in early stage
% lateGap   --- 15   --- Model update gap in late stage
% switchRho --- 0.5  --- Early/late switch FE ratio
%
% Main design:
%   1) Pop_C x Pop_C -> Pop_C
%   2) Pop_U x Pop_U -> Pop_U
%   3) Pop_C x Pop_U -> boundary candidates
%   4) B x opposite-side main population -> active boundary search
%   5) Boundary archive B keeps sector-wise "between" samples only
%   6) Feasible boundary offspring migrate back to Pop_C only if they
%      truly improve the current sector champion

    methods
        function main(Algorithm,Problem)
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            [type,hidden,epoch,lr,minClass,earlyGap,lateGap,switchRho] = ...
                Algorithm.ParameterSet(1,20,20,0.01,20,5,15,0.5);

            MaxB         = 2*Problem.N;
            MaxTrain     = 6*Problem.N;
            Generation   = 0;

            PopulationC  = Problem.Initialization();
            PopulationU  = Problem.Initialization();
            B            = [];
            Model        = [];
            TrainArchive = InitTrainArchive(Problem.D);

            % Initial boundary seeding
            SeedB        = GenerateCrossBoundaryOffspring(Problem,PopulationC,PopulationU,W,type);
            B            = UpdateBoundaryArchive([B,SeedB],PopulationC,PopulationU,W,Model,MaxB);
            TrainArchive = UpdateTrainArchive(TrainArchive,B,PopulationC,PopulationU,W,MaxTrain);
            if CanTrainBoundaryModel(TrainArchive,minClass)
                Model = TrainBoundaryMLP(TrainArchive.Dec,TrainArchive.Label,hidden,epoch,lr,[]);
            end

            while Algorithm.NotTerminated(PopulationC)
                Generation = Generation + 1;

                % 1) Inner evolution of two main populations
                OffspringC = GenerateWithinOffspring(Problem,PopulationC,type,true);
                OffspringU = GenerateWithinOffspring(Problem,PopulationU,type,false);

                PoolC = KeepUniquePopulation([PopulationC,OffspringC]);
                PoolU = KeepUniquePopulation([PopulationU,OffspringU]);

                % 2) Boundary generation
                CrossB     = GenerateCrossBoundaryOffspring(Problem,PoolC,PoolU,W,type);
                SearchB    = GenerateArchiveBoundaryOffspring(Problem,B,PoolC,PoolU,W,Model,type);
                BoundaryOff = KeepUniquePopulation([CrossB,SearchB]);

                % 3) Boundary -> Pop_C migration (feasible only, sector improvement only)
                Migrants = ExtractBoundaryMigrants(BoundaryOff,PoolC,W);

                % 4) Update explicit boundary archive
                B = UpdateBoundaryArchive([B,BoundaryOff],PoolC,PoolU,W,Model,MaxB);

                % 5) Environmental selection of two main populations
                PopulationC = EnvironmentalSelectionC([PopulationC,OffspringC,Migrants],Problem.N);
                PopulationU = EnvironmentalSelectionU([PopulationU,OffspringU],Problem.N);

                % 6) Warm-start MLP update
                TrainArchive = UpdateTrainArchive(TrainArchive,B,PopulationC,PopulationU,W,MaxTrain);
                if CanTrainBoundaryModel(TrainArchive,minClass)
                    if Problem.FE/max(Problem.maxFE,1) < switchRho
                        Gap = earlyGap;
                    else
                        Gap = lateGap;
                    end
                    if isempty(Model) || mod(Generation,Gap) == 0
                        Model = TrainBoundaryMLP(TrainArchive.Dec,TrainArchive.Label,hidden,epoch,lr,Model);
                    end
                end
            end
        end
    end
end

%% ========== Offspring generation ==========

function Offspring = GenerateWithinOffspring(Problem,Population,type,isConstraintSide)
    if isempty(Population)
        Offspring = [];
        return;
    end
    N = numel(Population);
    if isConstraintSide
        [Flag,FrontNo,CrowdDis] = ConstraintSideIndicator(Population);
        MatingPool = TournamentSelection(2,2*N,Flag,FrontNo,-CrowdDis);
    else
        [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population);
        MatingPool = TournamentSelection(2,2*N,FrontNo,-CrowdDis);
    end

    if type == 1
        Offspring = OperatorGAhalf(Problem,Population(MatingPool));
    else
        Offspring = OperatorDE(Problem,Population,...
            Population(MatingPool(1:N)),Population(MatingPool(N+1:end)));
    end
end

function Offspring = GenerateCrossBoundaryOffspring(Problem,PopulationC,PopulationU,W,type)
    if isempty(PopulationC) || isempty(PopulationU)
        Offspring = [];
        return;
    end
    RefObj = [PopulationC.objs;PopulationU.objs];
    RepC   = BuildSectorRepresentatives(PopulationC,W,RefObj,true);
    RepU   = BuildSectorRepresentatives(PopulationU,W,RefObj,false);
    Shared = find(RepC.has & RepU.has);

    if isempty(Shared)
        Offspring = PopulationC([]);
        return;
    end

    ParentC  = PopulationC(RepC.idx(Shared));
    ParentU  = PopulationU(RepU.idx(Shared));
    Offspring = CrossTwoSets(Problem,ParentC,ParentU,type);
end

function Offspring = GenerateArchiveBoundaryOffspring(Problem,B,PopulationC,PopulationU,W,Model,type)
    if isempty(B) || isempty(PopulationC) || isempty(PopulationU)
        Offspring = [];
        return;
    end

    RefObj  = [B.objs;PopulationC.objs;PopulationU.objs];
    SectorB = AssociateSectorsLocal(B.objs,W,RefObj);
    ScalarB = ComputeSectorScalar(B.objs,W,RefObj,SectorB);
    RepC    = BuildSectorRepresentatives(PopulationC,W,RefObj,true);
    RepU    = BuildSectorRepresentatives(PopulationU,W,RefObj,false);
    [ProbB,~] = PredictBoundaryMLP(Model,B.decs);

    ParentB  = B([]);
    Opposite = PopulationC([]);

    for s = 1 : size(W,1)
        if ~RepC.has(s) || ~RepU.has(s)
            continue;
        end
        idx = find(SectorB == s);
        if isempty(idx)
            continue;
        end

        % Keep only samples still lying between Pop_C and Pop_U in this sector
        idx = idx(IsBetween(ScalarB(idx),RepC.scalar(s),RepU.scalar(s)));
        if isempty(idx)
            continue;
        end

        Mid   = 0.5*(RepC.scalar(s) + RepU.scalar(s));
        Span  = max(abs(RepC.scalar(s) - RepU.scalar(s)),1e-12);
        Score1 = abs(ProbB(idx) - 0.5);
        Score2 = abs(ScalarB(idx) - Mid)./Span;
        [~,order] = sortrows([Score1(:),Score2(:)],[1 2]);
        pick = idx(order(1));

        ParentB(end+1) = B(pick); %#ok<AGROW>
        if all(B(pick).cons <= 0,2)
            Opposite(end+1) = PopulationU(RepU.idx(s)); %#ok<AGROW>
        else
            Opposite(end+1) = PopulationC(RepC.idx(s)); %#ok<AGROW>
        end
    end

    if isempty(ParentB)
        Offspring = B([]);
    else
        Offspring = CrossTwoSets(Problem,ParentB,Opposite,type);
    end
end

function Offspring = CrossTwoSets(Problem,Parent1,Parent2,type)
    if isempty(Parent1) || isempty(Parent2)
        Offspring = [];
        return;
    end
    K = min(numel(Parent1),numel(Parent2));
    Parent1 = Parent1(1:K);
    Parent2 = Parent2(1:K);

    if type == 1
        % Pair Parent1(i) with Parent2(i)
        Offspring = OperatorGAhalf(Problem,[Parent1,Parent2]);
    else
        % DE midpoint pull: Parent1 + 0.5*(Parent2-Parent1)
        Offspring = OperatorDE(Problem,Parent1,Parent2,Parent1);
    end
end

%% ========== Boundary archive ==========

function B = UpdateBoundaryArchive(CandidateB,PopulationC,PopulationU,W,Model,MaxB)
    if isempty(CandidateB)
        B = CandidateB;
        return;
    end

    CandidateB = KeepUniquePopulation(CandidateB);
    RefObj  = [CandidateB.objs;PopulationC.objs;PopulationU.objs];
    SectorB = AssociateSectorsLocal(CandidateB.objs,W,RefObj);
    ScalarB = ComputeSectorScalar(CandidateB.objs,W,RefObj,SectorB);
    RepC    = BuildSectorRepresentatives(PopulationC,W,RefObj,true);
    RepU    = BuildSectorRepresentatives(PopulationU,W,RefObj,false);
    [Prob,~] = PredictBoundaryMLP(Model,CandidateB.decs);

    % Sector-local ranking:
    %   1) |p-0.5| smaller -> better
    %   2) closer to the middle of Pop_C and Pop_U -> better
    Ranked = cell(size(W,1),1);
    for s = 1 : size(W,1)
        if ~RepC.has(s) || ~RepU.has(s)
            continue;
        end
        idx = find(SectorB == s);
        if isempty(idx)
            continue;
        end

        idx = idx(IsBetween(ScalarB(idx),RepC.scalar(s),RepU.scalar(s)));
        if isempty(idx)
            continue;
        end

        Mid   = 0.5*(RepC.scalar(s) + RepU.scalar(s));
        Span  = max(abs(RepC.scalar(s) - RepU.scalar(s)),1e-12);
        Score1 = abs(Prob(idx) - 0.5);
        Score2 = abs(ScalarB(idx) - Mid)./Span;
        [~,order] = sortrows([Score1(:),Score2(:)],[1 2]);
        Ranked{s} = idx(order);
    end

    % Sector-round-robin truncation:
    % equivalent to keeping the best samples inside each sector and
    % removing the worst samples of overloaded sectors.
    Pick  = zeros(0,1);
    Round = 1;
    while numel(Pick) < MaxB
        Changed = false;
        for s = 1 : numel(Ranked)
            if numel(Ranked{s}) >= Round
                Pick(end+1,1) = Ranked{s}(Round); %#ok<AGROW>
                Changed = true;
                if numel(Pick) >= MaxB
                    break;
                end
            end
        end
        if ~Changed
            break;
        end
        Round = Round + 1;
    end

    if isempty(Pick)
        B = CandidateB([]);
    else
        B = CandidateB(unique(Pick,'stable'));
    end
end

function Migrants = ExtractBoundaryMigrants(BoundaryOff,PopulationC,W)
    if isempty(BoundaryOff)
        Migrants = [];
        return;
    end

    BoundaryOff = FilterFeasiblePopulation(BoundaryOff);
    if isempty(BoundaryOff)
        Migrants = BoundaryOff;
        return;
    end

    RefObj   = [BoundaryOff.objs;PopulationC.objs];
    SectorOff = AssociateSectorsLocal(BoundaryOff.objs,W,RefObj);
    ScalarOff = ComputeSectorScalar(BoundaryOff.objs,W,RefObj,SectorOff);
    RepC      = BuildSectorRepresentatives(PopulationC,W,RefObj,true);

    Pick = zeros(0,1);
    for s = unique(SectorOff(:))'
        idx = find(SectorOff == s);
        [bestScalar,loc] = min(ScalarOff(idx));
        bestIdx = idx(loc);
        if ~RepC.has(s) || bestScalar < RepC.scalar(s)
            Pick(end+1,1) = bestIdx; %#ok<AGROW>
        end
    end

    if isempty(Pick)
        Migrants = BoundaryOff([]);
    else
        Migrants = BoundaryOff(Pick);
    end
end

%% ========== Environmental selection ==========

function Population = EnvironmentalSelectionC(Population,N)
    Population = KeepUniquePopulation(Population);
    if isempty(Population)
        return;
    end

    Feasible = FilterFeasiblePopulation(Population);
    if numel(Feasible) >= N
        Population = SelectByObjective(Feasible,N);
    else
        Next   = Feasible;
        Need   = N - numel(Next);
        Remain = RemovePopulationByDecision(Population,Next);
        if Need > 0 && ~isempty(Remain)
            Next = [Next,SelectByObjective(Remain,min(Need,numel(Remain)))];
        end
        Population = PadPopulation(Next,N);
    end
end

function Population = EnvironmentalSelectionU(Population,N)
    Population = KeepUniquePopulation(Population);
    Population = SelectByObjective(Population,min(N,numel(Population)));
    Population = PadPopulation(Population,N);
end

function Population = SelectByObjective(Population,N)
    if isempty(Population)
        return;
    end
    if N <= 0
        Population = Population([]);
        return;
    end

    N = min(N,numel(Population));
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next     = FrontNo < MaxFNo;
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    Last     = find(FrontNo == MaxFNo);
    Need     = N - sum(Next);
    if Need > 0
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:Need))) = true;
    end
    Population = Population(Next);
end

function Population = PadPopulation(Population,N)
    if isempty(Population)
        return;
    end
    if numel(Population) < N
        Population = [Population,Population(mod(0:N-numel(Population)-1,numel(Population))+1)];
    else
        Population = Population(1:N);
    end
end

%% ========== Parent-selection indicators ==========

function [Flag,FrontNo,CrowdDis] = ConstraintSideIndicator(Population)
    [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population);
    Flag = double(~all(Population.cons<=0,2)); % feasible:0, infeasible:1
end

function [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population)
    [FrontNo,~] = NDSort(Population.objs,numel(Population));
    CrowdDis    = CrowdingDistance(Population.objs,FrontNo);
end

%% ========== Training archive ==========

function TrainArchive = InitTrainArchive(D)
    TrainArchive = struct( ...
        'Dec',zeros(0,D), ...
        'Label',zeros(0,1), ...
        'Source',zeros(0,1));   % 1: B, 2: Pop_C reps, 3: Pop_U reps
end

function TrainArchive = UpdateTrainArchive(TrainArchive,B,PopulationC,PopulationU,W,MaxTrain)
    RefObj = [PopulationC.objs;PopulationU.objs];
    if ~isempty(B)
        RefObj = [RefObj;B.objs];
    end

    RepC = BuildSectorRepresentatives(PopulationC,W,RefObj,true);
    RepU = BuildSectorRepresentatives(PopulationU,W,RefObj,false);

    TrainArchive = AppendTrainArchive(TrainArchive,B,1);
    if any(RepC.has)
        TrainArchive = AppendTrainArchive(TrainArchive,PopulationC(RepC.idx(RepC.has)),2);
    end
    if any(RepU.has)
        TrainArchive = AppendTrainArchive(TrainArchive,PopulationU(RepU.idx(RepU.has)),3);
    end
    TrainArchive = TrimTrainArchive(TrainArchive,MaxTrain);
end

function TrainArchive = AppendTrainArchive(TrainArchive,Population,Source)
    if isempty(Population)
        return;
    end
    TrainArchive.Dec    = [TrainArchive.Dec;Population.decs];
    TrainArchive.Label  = [TrainArchive.Label;double(all(Population.cons<=0,2))];
    TrainArchive.Source = [TrainArchive.Source;Source*ones(numel(Population),1)];

    Keep = KeepLatestDecisionRowsLocal(TrainArchive.Dec);
    TrainArchive.Dec    = TrainArchive.Dec(Keep,:);
    TrainArchive.Label  = TrainArchive.Label(Keep);
    TrainArchive.Source = TrainArchive.Source(Keep);
end

function TrainArchive = TrimTrainArchive(TrainArchive,MaxTrain)
    Count = size(TrainArchive.Dec,1);
    if Count <= MaxTrain
        return;
    end

    % Prefer boundary samples, but keep labels balanced
    BQuota   = min(round(0.7*MaxTrain),sum(TrainArchive.Source==1));
    KeepB    = SelectLatestBalancedIndices(find(TrainArchive.Source==1),TrainArchive.Label,BQuota);
    RemQuota = MaxTrain - numel(KeepB);
    KeepO    = SelectLatestBalancedIndices(find(TrainArchive.Source~=1),TrainArchive.Label,RemQuota);

    Keep = unique([KeepB;KeepO],'stable');
    if numel(Keep) < MaxTrain
        Rest = setdiff((1:Count)',Keep,'stable');
        Need = min(MaxTrain-numel(Keep),numel(Rest));
        if Need > 0
            Keep = [Keep;Rest(end-Need+1:end)];
        end
    elseif numel(Keep) > MaxTrain
        Keep = Keep(end-MaxTrain+1:end);
    end

    Keep = sort(Keep);
    TrainArchive.Dec    = TrainArchive.Dec(Keep,:);
    TrainArchive.Label  = TrainArchive.Label(Keep);
    TrainArchive.Source = TrainArchive.Source(Keep);
end

function Idx = SelectLatestBalancedIndices(CandidateIdx,Label,Count)
    Idx = zeros(0,1);
    if isempty(CandidateIdx) || Count <= 0
        return;
    end

    CandidateIdx = CandidateIdx(:);
    Pos = CandidateIdx(Label(CandidateIdx)==1);
    Neg = CandidateIdx(Label(CandidateIdx)==0);

    Quota = floor(Count/2);
    if Quota > 0
        KeepPos = Pos(max(1,numel(Pos)-Quota+1):end);
        KeepNeg = Neg(max(1,numel(Neg)-Quota+1):end);
        Idx = unique([KeepPos;KeepNeg],'stable');
    end

    if numel(Idx) < Count
        Rest = setdiff(CandidateIdx,Idx,'stable');
        Need = min(Count-numel(Idx),numel(Rest));
        if Need > 0
            Idx = [Idx;Rest(end-Need+1:end)];
        end
    elseif numel(Idx) > Count
        Idx = Idx(end-Count+1:end);
    end

    Idx = sort(Idx);
end

function Flag = CanTrainBoundaryModel(TrainArchive,minClass)
    Flag = sum(TrainArchive.Label==1) >= minClass && ...
           sum(TrainArchive.Label==0) >= minClass;
end

%% ========== Boundary MLP ==========

function Model = TrainBoundaryMLP(X,Y,Hidden,Epoch,LR,PrevModel)
    if nargin < 6
        PrevModel = [];
    end
    Model = PrevModel;

    if isempty(X) || size(X,1) < 4
        return;
    end

    X = double(X);
    Y = double(Y(:) > 0);
    if numel(unique(Y)) < 2
        return;
    end

    Hidden    = max(2,round(Hidden));
    Epoch     = max(1,round(Epoch));
    LR        = max(double(LR),1e-4);
    [N,D]     = size(X);
    LambdaReg = 1e-4;

    Mu    = mean(X,1);
    Sigma = std(X,0,1);
    Sigma(Sigma<1e-12) = 1;

    if ~isempty(PrevModel) && IsWarmStartCompatible(PrevModel,D,Hidden)
        W1 = PrevModel.W1;  b1 = PrevModel.b1;
        W2 = PrevModel.W2;  b2 = PrevModel.b2;
    else
        W1 = 0.1*randn(D,Hidden);
        b1 = zeros(1,Hidden);
        W2 = 0.1*randn(Hidden,1);
        b2 = 0;
    end

    Xn = (X-Mu)./Sigma;
    [Weight,NormWeight] = BuildClassWeights(Y);

    for e = 1 : Epoch
        H = tanh(Xn*W1 + repmat(b1,N,1));
        Z = H*W2 + b2;
        P = 1./(1+exp(-Z));

        Delta2 = Weight.*(P-Y)./NormWeight;
        dW2 = H'*Delta2 + LambdaReg*W2;
        db2 = sum(Delta2);

        D1  = (Delta2*W2').*(1-H.^2);
        dW1 = Xn'*D1 + LambdaReg*W1;
        db1 = sum(D1,1);

        Step = LR/sqrt(e);
        W1 = W1 - Step*dW1;
        b1 = b1 - Step*db1;
        W2 = W2 - Step*dW2;
        b2 = b2 - Step*db2;
    end

    Model = struct();
    Model.Mu    = Mu;
    Model.Sigma = Sigma;
    Model.W1    = W1;
    Model.b1    = b1;
    Model.W2    = W2;
    Model.b2    = b2;
end

function [Prob,Stats] = PredictBoundaryMLP(Model,X)
    if nargin < 2 || isempty(X)
        Prob  = zeros(0,1);
        Stats = struct('logit',zeros(0,1));
        return;
    end
    if isempty(Model) || ~isfield(Model,'Mu')
        Prob  = 0.5*ones(size(X,1),1);
        Stats = struct('logit',zeros(size(X,1),1));
        return;
    end

    Xn = (double(X)-Model.Mu)./Model.Sigma;
    H  = tanh(Xn*Model.W1 + repmat(Model.b1,size(Xn,1),1));
    Z  = H*Model.W2 + Model.b2;
    Prob = 1./(1+exp(-Z));
    Prob = min(max(Prob,1e-6),1-1e-6);
    Stats = struct('logit',Z(:));
end

function Flag = IsWarmStartCompatible(Model,D,Hidden)
    Flag = ~isempty(Model) && isfield(Model,'W1') && isfield(Model,'W2') && ...
           size(Model.W1,1) == D && size(Model.W1,2) == Hidden && ...
           size(Model.W2,1) == Hidden;
end

function [Weight,NormWeight] = BuildClassWeights(Y)
    N    = numel(Y);
    Pos  = sum(Y==1);
    Neg  = N - Pos;
    WPos = N/(2*max(1,Pos));
    WNeg = N/(2*max(1,Neg));
    Weight     = WNeg + (WPos-WNeg).*Y;
    NormWeight = max(sum(Weight),1);
end

%% ========== Sector tools ==========

function Rep = BuildSectorRepresentatives(Population,W,RefObj,PreferFeasible)
    K   = size(W,1);
    Rep = struct('has',false(K,1), ...
                 'idx',zeros(K,1), ...
                 'scalar',inf(K,1));
    if isempty(Population)
        return;
    end

    Sector = AssociateSectorsLocal(Population.objs,W,RefObj);
    Scalar = ComputeSectorScalar(Population.objs,W,RefObj,Sector);
    Fea    = all(Population.cons<=0,2);

    for s = 1 : K
        idx = find(Sector == s);
        if isempty(idx)
            continue;
        end
        if PreferFeasible
            idxF = idx(Fea(idx));
            if ~isempty(idxF)
                idx = idxF;
            end
        end
        [Rep.scalar(s),loc] = min(Scalar(idx));
        Rep.idx(s) = idx(loc);
        Rep.has(s) = true;
    end
end

function Flag = IsBetween(Value,A,B)
    lower = min(A,B) - 1e-12;
    upper = max(A,B) + 1e-12;
    Flag  = Value >= lower & Value <= upper;
end

function [Sector,Count] = AssociateSectorsLocal(PopObj,W,RefObj)
    if nargin < 3 || isempty(RefObj)
        RefObj = PopObj;
    end
    if isempty(PopObj)
        Sector = zeros(0,1);
        Count  = zeros(size(W,1),1);
        return;
    end
    if isempty(W)
        Sector = ones(size(PopObj,1),1);
        Count  = size(PopObj,1);
        return;
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range  = MaxObj - MinObj;
    Range(Range<1e-12) = 1;

    Obj = (PopObj - MinObj)./Range;
    ObjNorm = sqrt(sum(Obj.^2,2));
    ZeroMask = ObjNorm < 1e-12;
    Obj(ZeroMask,:) = 1;
    ObjNorm(ZeroMask) = sqrt(size(Obj,2));
    Obj = Obj./ObjNorm(:,ones(1,size(Obj,2)));

    WNorm = sqrt(sum(W.^2,2));
    WNorm(WNorm<1e-12) = 1;
    Wn = W./WNorm(:,ones(1,size(W,2)));

    Cosine = Obj*Wn';
    [~,Sector] = max(Cosine,[],2);
    Count = accumarray(Sector,1,[size(W,1),1]);
end

function Value = ComputeSectorScalar(Obj,W,RefObj,Sector)
    if isempty(Obj)
        Value = zeros(0,1);
        return;
    end
    if nargin < 2 || isempty(W)
        W = ones(1,size(Obj,2));
    end
    if nargin < 3 || isempty(RefObj)
        RefObj = Obj;
    end
    if nargin < 4 || isempty(Sector)
        Weight = repmat(W(1,:),size(Obj,1),1);
    else
        Weight = W(Sector,:);
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range  = MaxObj - MinObj;
    Range(Range<1e-12) = 1;
    NormObj = (Obj-MinObj)./Range;

    Value = sum(NormObj.*Weight,2);
end

%% ========== Population utilities ==========

function Population = FilterFeasiblePopulation(Population)
    if isempty(Population)
        return;
    end
    Population = Population(all(Population.cons<=0,2));
end

function Population = KeepUniquePopulation(Population)
    if isempty(Population)
        return;
    end
    Keep = KeepLatestDecisionRowsLocal(Population.decs);
    Population = Population(Keep);
end

function Population = RemovePopulationByDecision(Population,Remove)
    if isempty(Population) || isempty(Remove)
        return;
    end
    Keep = ~ismember(Population.decs,Remove.decs,'rows');
    Population = Population(Keep);
end

function Keep = KeepLatestDecisionRowsLocal(Dec)
    if isempty(Dec)
        Keep = zeros(0,1);
        return;
    end
    [~,Keep] = unique(double(Dec),'rows','last');
    Keep = sort(Keep);
end
```

这版实现的核心优点是：结构干净、机制闭环、每个模块都能直接对应到你的论文叙述。它没有再被旧 PRBCCMO 的“桥接主流程”牵着走，而是把你的确定思想——双群体、显式边界档案、扇区覆盖、MLP 不确定性驱动、边界解有条件回流——完整落到了 PlatEMO 的代码骨架里。fileciteturn0file0



