classdef PRBCCMO < ALGORITHM
% <2026> <multi> <real/integer/label/binary/permutation> <constrained>
% Pareto-relevant boundary CCMO
% type     --- 1    --- Type of operator (1. GA 2. DE)
% bRho     --- 0.2  --- Boundary evaluation ratio relative to N
% afRho    --- 0.5  --- Feasible boundary archive size ratio
% aiRho    --- 0.5  --- Infeasible boundary archive size ratio
% trainRho --- 4    --- Training archive size ratio
% hidden   --- 20   --- Hidden units of the boundary MLP
% epoch    --- 25   --- Training epochs of the boundary MLP
% lr       --- 0.01 --- Learning rate of the boundary MLP
% xRho     --- 0.5  --- Candidate ratio from P_C x P_U
% lsRho    --- 0.25 --- Candidate ratio from local boundary perturbation
% mRho     --- 0.25 --- Candidate ratio from A_I x P_C

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [type,bRho,afRho,aiRho,trainRho,hidden,epoch,lr,xRho,lsRho,mRho] = ...
                Algorithm.ParameterSet(1,0.2,0.5,0.5,4,20,25,0.01,0.5,0.25,0.25);

            BoundaryBudget = max(0,round(bRho*Problem.N));
            ArchiveFMax    = max(0,round(afRho*Problem.N));
            ArchiveIMax    = max(0,round(aiRho*Problem.N));
            TrainMax       = max(0,round(trainRho*Problem.N));
            NumCross       = max(0,round(xRho*Problem.N));
            NumLocal       = max(0,round(lsRho*Problem.N));
            NumMate        = max(0,round(mRho*Problem.N));
            [W,~]          = UniformPoint(max(Problem.N,2),Problem.M);
            NumBoundarySources = 4;

            %% Generate random populations
            PopulationC = Problem.Initialization();
            PopulationU = Problem.Initialization();
            FitnessC    = CalFitness(PopulationC.objs,PopulationC.cons);
            FitnessU    = CalFitness(PopulationU.objs);

            %% Initialize archives
            ArchiveF    = [];
            ArchiveI    = [];
            [TrainDec,TrainLabel] = UpdateTrainingArchive([],[],[PopulationC,PopulationU],[],TrainMax);
            Algorithm.metric = InitializeBoundaryMetrics(Algorithm.metric,NumBoundarySources);
            
            %% Optimization
            while Algorithm.NotTerminated(PopulationC)
                Model = TrainBoundaryMLP(TrainDec,TrainLabel,hidden,epoch,lr);

                [OffspringC,OffspringU] = GenerateRegularOffspring( ...
                    Problem,PopulationC,PopulationU,FitnessC,FitnessU,type);

                CandidatePool = GenerateBoundaryCandidates( ...
                    Problem,PopulationC,PopulationU,ArchiveF,ArchiveI, ...
                    FitnessC,FitnessU,type,W,NumCross,NumLocal,NumMate);
                
                BoundaryBudgetNow = min(BoundaryBudget,max(0,Problem.maxFE-Problem.FE));
                [BoundaryOffspring,BoundaryInfo] = SelectBoundaryCandidates( ...
                    Problem,CandidatePool,PopulationC,Model,W,BoundaryBudgetNow);
                HelperU = SelectHelperPopulation(OffspringU,PopulationC,Model,W,BoundaryBudgetNow);

                if isempty(BoundaryOffspring)
                    CrossBoundary = [];
                else
                    CrossBoundary = BoundaryOffspring(BoundaryInfo.source==1);
                end

                BoundaryBaseCount = numel(PopulationC) + numel(OffspringC) + numel(HelperU);
                [PopulationC,FitnessC,SelectedIdxC] = EnvironmentalSelectionC( ...
                    [PopulationC,OffspringC,HelperU,BoundaryOffspring],Problem.N,Model,W);
                [PopulationU,FitnessU] = EnvironmentalSelectionU( ...
                    [PopulationU,OffspringU,CrossBoundary],Problem.N);

                Algorithm.metric = RecordBoundaryMetrics( ...
                    Algorithm.metric,Problem.FE,BoundaryOffspring,BoundaryInfo, ...
                    SelectedIdxC,BoundaryBaseCount,NumBoundarySources);

                [ArchiveF,ArchiveI] = UpdateBoundaryArchives( ...
                    ArchiveF,ArchiveI,BoundaryOffspring,PopulationC,Model,W, ...
                    ArchiveFMax,ArchiveIMax);

                % Idea.md requires the retraining signal to come from
                % real-evaluated near-boundary samples and their archives.
                [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                    TrainDec,TrainLabel,BoundaryOffspring,[ArchiveF,ArchiveI],TrainMax);
            end
        end
    end
end

function [OffspringC,OffspringU] = GenerateRegularOffspring(Problem,PopulationC,PopulationU,FitnessC,FitnessU,type)
    if type == 1
        MatingPoolC = TournamentSelection(2,Problem.N,FitnessC);
        MatingPoolU = TournamentSelection(2,Problem.N,FitnessU);
        OffspringC  = OperatorGAhalf(Problem,PopulationC(MatingPoolC));
        OffspringU  = OperatorGAhalf(Problem,PopulationU(MatingPoolU));
    else
        MatingPoolC = TournamentSelection(2,2*Problem.N,FitnessC);
        MatingPoolU = TournamentSelection(2,2*Problem.N,FitnessU);
        OffspringC  = OperatorDE(Problem,PopulationC, ...
            PopulationC(MatingPoolC(1:end/2)),PopulationC(MatingPoolC(end/2+1:end)));
        OffspringU  = OperatorDE(Problem,PopulationU, ...
            PopulationU(MatingPoolU(1:end/2)),PopulationU(MatingPoolU(end/2+1:end)));
    end
end

function HelperU = SelectHelperPopulation(OffspringU,PopulationC,Model,W,HelperBudget)
% Reinject a filtered helper subset from P_U into the constrained update.

    HelperU = [];
    if isempty(OffspringU)
        return;
    end
    if nargin < 5 || isempty(HelperBudget)
        HelperBudget = numel(OffspringU);
    end

    FeasibleMask = all(OffspringU.cons<=0,2);
    if any(FeasibleMask)
        HelperU = OffspringU(FeasibleMask);
    end

    Infeasible = OffspringU(~FeasibleMask);
    if isempty(Infeasible)
        return;
    end

    Prob    = PredictBoundaryMLP(Model,Infeasible.decs);
    Score   = 1 - 2*abs(Prob-0.5);
    NearIdx = find(Score>=0.5);
    if isempty(NearIdx)
        if ~isempty(HelperU)
            Keep = KeepLatestDecisionRows(HelperU.decs);
            HelperU = HelperU(Keep);
        end
        return;
    end

    Infeasible = Infeasible(NearIdx);
    Score      = Score(NearIdx);
    FeasibleC  = PopulationC(all(PopulationC.cons<=0,2));
    if isempty(FeasibleC)
        FeasibleObj = zeros(0,size(Infeasible.objs,2));
    else
        FeasibleObj = FeasibleC.objs;
    end
    HelperBudget = min(max(0,HelperBudget),numel(Infeasible));
    Order = RerankBoundaryCandidates(Infeasible.objs,Score,[],FeasibleObj,W,HelperBudget);
    if ~isempty(Order)
        HelperU = [HelperU,Infeasible(Order)];
    end

    if isempty(HelperU)
        return;
    end
    Keep = KeepLatestDecisionRows(HelperU.decs);
    HelperU = HelperU(Keep);
end

function Metric = InitializeBoundaryMetrics(Metric,NumSources)
    Metric.boundary_FE_t       = zeros(0,1);
    Metric.B_selected_t        = zeros(0,1);
    Metric.B_feasible_t        = zeros(0,1);
    Metric.B_enterPc_t         = zeros(0,1);
    Metric.B_selected_source_t = zeros(0,NumSources);
    Metric.B_feasible_source_t = zeros(0,NumSources);
    Metric.B_enterPc_source_t  = zeros(0,NumSources);
end

function Metric = RecordBoundaryMetrics(Metric,CurrentFE,BoundaryOffspring,BoundaryInfo,SelectedIdxC,BoundaryBaseCount,NumSources)
    FeasibleMask = false(0,1);
    EnterIdx     = zeros(0,1);
    if ~isempty(BoundaryOffspring)
        FeasibleMask = all(BoundaryOffspring.cons<=0,2);
        EnterIdx = SelectedIdxC(SelectedIdxC>BoundaryBaseCount) - BoundaryBaseCount;
    end

    Metric.boundary_FE_t(end+1,1) = CurrentFE;
    Metric.B_selected_t(end+1,1)  = numel(BoundaryOffspring);
    Metric.B_feasible_t(end+1,1)  = sum(FeasibleMask);
    Metric.B_enterPc_t(end+1,1)   = numel(EnterIdx);

    Metric.B_selected_source_t(end+1,:) = CountBySource(BoundaryInfo.source,NumSources);
    Metric.B_feasible_source_t(end+1,:) = CountBySource(BoundaryInfo.source(FeasibleMask),NumSources);
    Metric.B_enterPc_source_t(end+1,:)  = CountBySource(BoundaryInfo.source(EnterIdx),NumSources);
end

function Counts = CountBySource(Source,NumSources)
    Counts = zeros(1,NumSources);
    if isempty(Source)
        return;
    end
    Counts = histcounts(Source,0.5:1:(NumSources+0.5));
end
