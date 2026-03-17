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
% xRho     --- 0.5  --- Candidate ratio from bridge source
% lsRho    --- 0.5  --- Candidate ratio from label-aware local source
% mRho     --- 0.5  --- Seed-query ratio within each boundary budget

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
                Algorithm.ParameterSet(1,0.2,0.5,0.5,4,20,25,0.01,0.5,0.5,0.5);

            BoundaryBudget = max(0,floor(bRho*Problem.N));
            ArchiveFMax    = max(0,round(afRho*Problem.N));
            ArchiveIMax    = max(0,round(aiRho*Problem.N));
            TrainMax       = max(0,round(trainRho*Problem.N));
            CalibMax       = max(20,ceil(0.25*TrainMax));
            ProtectedMax   = max(20,ceil(0.20*TrainMax));
            HardNegMax     = max(20,ceil(0.25*TrainMax));
            RatioSum       = max(xRho+lsRho,1e-12);
            PoolSize       = max(0,4*BoundaryBudget);
            NumBridge      = max(0,round(PoolSize*xRho/RatioSum));
            NumLocal       = max(0,PoolSize-NumBridge);
            SeedRatio      = min(max(mRho,0.25),0.75);
            [W,~]          = UniformPoint(max(Problem.N,2),Problem.M);
            UpdateGap      = 5;
            RestartGap     = 25;
            WarmEpoch      = min(epoch,max(5,round(epoch/3)));
            TriggerCount   = max(20,ceil(0.1*TrainMax));

            %% Generate random populations
            PopulationC = Problem.Initialization();
            PopulationU = Problem.Initialization();
            FitnessC    = CalFitness(PopulationC.objs,PopulationC.cons);
            FitnessU    = CalFitness(PopulationU.objs);

            %% Initialize boundary memories
            ArchiveF = [];
            ArchiveI = [];
            HardNegativeArchive.Dec    = zeros(0,Problem.D);
            HardNegativeArchive.Radius = zeros(0,1);
            ProtectedDec   = zeros(0,Problem.D);
            ProtectedLabel = zeros(0,1);
            InitSolutions = [PopulationC,PopulationU];
            [InitTrain,~,InitCalib,InitCalibInfo] = SplitHeldOutBatch( ...
                InitSolutions,NormalizeBoundaryInfo([],Problem.M),CalibMax,Problem.M);
            InitHoldoutDec = SolutionDecs(InitCalib,Problem.D);
            [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                [],[],ProtectedDec,ProtectedLabel,InitTrain,InitHoldoutDec,TrainMax);
            [CalibDec,CalibLabel,CalibNear] = UpdateCalibrationBuffer( ...
                [],[],[],InitCalib,InitCalibInfo,CalibMax);
            Model = TrainBoundaryMLP( ...
                TrainDec,TrainLabel,hidden,epoch,lr,[],CalibDec,CalibLabel,struct());
            LastCalMetric = EvaluateBoundaryCalibration(Model,CalibDec,CalibLabel);
            PendingLabels = 0;
            Generation    = 0;

            %% Optimization
            while Algorithm.NotTerminated(PopulationC)
                Generation = Generation + 1;

                [OffspringC,OffspringU] = GenerateRegularOffspring( ...
                    Problem,PopulationC,PopulationU,FitnessC,FitnessU,type);

                CandidatePool = GenerateBoundaryCandidates( ...
                    Problem,PopulationC,PopulationU,ArchiveF,ArchiveI,type,W,NumBridge,NumLocal);

                BoundaryBudgetNow = min(BoundaryBudget,max(0,Problem.maxFE-Problem.FE));
                SeedBudget = min(BoundaryBudgetNow,max(0,round(SeedRatio*BoundaryBudgetNow)));
                if BoundaryBudgetNow > 0 && SeedBudget == 0
                    SeedBudget = 1;
                end
                [BoundarySeeds,SeedInfo] = SelectBoundaryCandidates( ...
                    Problem,CandidatePool,PopulationC,Model,W,HardNegativeArchive,SeedBudget);

                WorkerBudget = max(0,BoundaryBudgetNow-numel(BoundarySeeds));
                [WorkerOffspring,WorkerInfo,MigrationPool,BracketBatch,HardNegBatch] = ...
                    RefineBoundaryWorkers( ...
                        Problem,BoundarySeeds,SeedInfo,PopulationC,PopulationU, ...
                        ArchiveF,ArchiveI,Model,W,HardNegativeArchive,WorkerBudget);
                [BoundaryOffspring,BoundaryInfo] = MergeBoundaryResults( ...
                    BoundarySeeds,SeedInfo,WorkerOffspring,WorkerInfo,Problem.M);

                HardNegativeArchive = UpdateHardNegativeArchive(HardNegativeArchive,HardNegBatch,HardNegMax);

                ConstrainedBase = KeepUniquePopulation( ...
                    ApplySectorMigration([PopulationC,OffspringC],MigrationPool,W));
                BoundaryFeasible = FilterFeasiblePopulation(BoundaryOffspring);
                ConstrainedPool  = KeepUniquePopulation([ConstrainedBase,BoundaryFeasible]);
                [PopulationC,FitnessC] = EnvironmentalSelectionC( ...
                    ConstrainedPool,Problem.N);
                [PopulationU,FitnessU] = EnvironmentalSelectionU( ...
                    KeepUniquePopulation([PopulationU,OffspringU]),Problem.N);

                [ArchiveF,ArchiveI] = UpdateBoundaryArchives( ...
                    Problem,ArchiveF,ArchiveI,BoundaryOffspring,PopulationC,Model,W, ...
                    HardNegativeArchive,ArchiveFMax,ArchiveIMax);

                [TrainBatch,TrainInfo,CalibBatch,CalibInfo] = SplitHeldOutBatch( ...
                    BoundaryOffspring,BoundaryInfo,CalibMax,Problem.M);
                HoldoutDec = [CalibDec;SolutionDecs(CalibBatch,Problem.D)];
                [ProtectedAddDec,ProtectedAddLabel] = CollectProtectedCases( ...
                    TrainBatch,TrainInfo,BracketBatch,HardNegBatch,Problem.D);
                [ProtectedAddDec,ProtectedAddLabel] = ExcludeLabeledRows( ...
                    ProtectedAddDec,ProtectedAddLabel,HoldoutDec);
                [ProtectedDec,ProtectedLabel] = UpdateProtectedBuffer( ...
                    ProtectedDec,ProtectedLabel,ProtectedAddDec,ProtectedAddLabel,ProtectedMax);
                [ProtectedDec,ProtectedLabel] = ExcludeLabeledRows( ...
                    ProtectedDec,ProtectedLabel,HoldoutDec);
                [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                    TrainDec,TrainLabel,ProtectedDec,ProtectedLabel,TrainBatch,HoldoutDec,TrainMax);
                [CalibDec,CalibLabel,CalibNear] = UpdateCalibrationBuffer( ...
                    CalibDec,CalibLabel,CalibNear,CalibBatch,CalibInfo,CalibMax);

                PendingLabels = PendingLabels + numel(BoundaryOffspring);
                [Model,PendingLabels,LastCalMetric] = UpdateBoundaryModel( ...
                    Model,TrainDec,TrainLabel,CalibDec,CalibLabel,hidden,epoch, ...
                    WarmEpoch,lr,Generation,PendingLabels,TriggerCount, ...
                    UpdateGap,RestartGap,LastCalMetric);
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

function Population = KeepUniquePopulation(Population)
    if isempty(Population)
        return;
    end
    Keep = KeepLatestDecisionRows(Population.decs);
    Population = Population(Keep);
end

function Population = FilterFeasiblePopulation(Population)
    if isempty(Population)
        return;
    end
    Population = Population(all(Population.cons<=0,2));
end

function [TrainSolutions,TrainInfo,CalibSolutions,CalibInfo] = SplitHeldOutBatch(Solutions,Info,CalibMax,M)
    TrainSolutions = Solutions;
    TrainInfo = NormalizeBoundaryInfo(Info,M);
    CalibSolutions = [];
    CalibInfo = NormalizeBoundaryInfo([],M);
    Count = numel(Solutions);
    if Count <= 1 || CalibMax <= 0
        return;
    end

    Info = NormalizeBoundaryInfo(Info,M);
    if numel(Info.source) ~= Count
        Info = DefaultBoundaryInfo(Solutions,M);
    end
    Label = double(all(Solutions.cons<=0,2));
    NearMask = true(Count,1);
    if isfield(Info,'prob') && numel(Info.prob) == Count
        NearMask = abs(Info.prob(:)-0.5) <= 0.1;
    end

    HoldoutQuota = min(max(1,round(0.2*Count)),min(CalibMax,Count-1));
    HoldoutIdx = SelectCalibrationHoldout(Label,NearMask,HoldoutQuota);
    if isempty(HoldoutIdx)
        return;
    end

    TrainMask = true(1,Count);
    TrainMask(HoldoutIdx) = false;
    TrainSolutions = Solutions(TrainMask);
    TrainInfo = SliceBoundaryInfo(Info,find(TrainMask),M);
    CalibSolutions = Solutions(HoldoutIdx);
    CalibInfo = SliceBoundaryInfo(Info,HoldoutIdx,M);
end

function HoldoutIdx = SelectCalibrationHoldout(Label,NearMask,Quota)
    HoldoutIdx = zeros(0,1);
    Total = numel(Label);
    if Total <= 1 || Quota <= 0
        return;
    end

    Quota = min(Quota,Total-1);
    ClassOrder = [1,0];
    BaseQuota = floor(Quota/2);
    HoldoutCell = cell(1,numel(ClassOrder)+1);
    HoldCount = 0;
    for i = 1 : numel(ClassOrder)
        ClassIdx = FindCalibrationCandidates(Label,NearMask,ClassOrder(i));
        Take = min(numel(ClassIdx),BaseQuota);
        if Take > 0
            HoldCount = HoldCount + 1;
            HoldoutCell{HoldCount} = ClassIdx(1:Take);
        end
    end
    HoldoutIdx = vertcat(HoldoutCell{1:HoldCount});

    if numel(HoldoutIdx) < Quota
        Remaining = setdiff((1:Total)',HoldoutIdx,'stable');
        NearFirst = [Remaining(NearMask(Remaining));Remaining(~NearMask(Remaining))];
        Extra = NearFirst(1:min(Quota-numel(HoldoutIdx),numel(NearFirst)));
        HoldoutIdx = [HoldoutIdx;Extra(:)];
    end

    HoldoutIdx = unique(HoldoutIdx,'stable');
end

function Idx = FindCalibrationCandidates(Label,NearMask,ClassValue)
    NearIdx = find(NearMask & Label==ClassValue);
    FarIdx  = find(~NearMask & Label==ClassValue);
    Idx = [NearIdx(:);FarIdx(:)];
end

function Info = DefaultBoundaryInfo(Solutions,M)
    Count = numel(Solutions);
    Info = NormalizeBoundaryInfo([],M);
    Info.source    = zeros(Count,1);
    Info.score     = zeros(Count,1);
    Info.prob      = 0.5*ones(Count,1);
    Info.entropy   = zeros(Count,1);
    Info.hvGain    = zeros(Count,1);
    Info.novelty   = zeros(Count,1);
    Info.penalty   = ones(Count,1);
    Info.utility   = zeros(Count,1);
    Info.sector    = zeros(Count,1);
    Info.proxyObjs = Solutions.objs;
end

function [Dec,Label] = ExcludeLabeledRows(Dec,Label,ExcludeDec)
    if isempty(Dec) || isempty(ExcludeDec)
        return;
    end
    Keep = ~ismember(Dec,ExcludeDec,'rows');
    Dec = Dec(Keep,:);
    Label = Label(Keep);
end

function Dec = SolutionDecs(Solutions,D)
    if isempty(Solutions)
        Dec = zeros(0,D);
        return;
    end
    Dec = Solutions.decs;
end

function [ProtectedDec,ProtectedLabel] = CollectProtectedCases(BoundaryOffspring,BoundaryInfo,BracketBatch,HardNegBatch,D)
    ProtectedDec = zeros(0,D);
    ProtectedLabel = zeros(0,1);
    if ~isempty(BoundaryOffspring)
        Label = double(all(BoundaryOffspring.cons<=0,2));
        NearMask = abs(BoundaryInfo.prob(:)-0.5) <= 0.1;
        MisMask  = (BoundaryInfo.prob(:)>=0.5) ~= logical(Label);
        KeepMask = NearMask | MisMask;
        ProtectedDec = [ProtectedDec;BoundaryOffspring(KeepMask).decs];
        ProtectedLabel = [ProtectedLabel;Label(KeepMask)];
    end
    if ~isempty(BracketBatch.Feasible)
        ProtectedDec = [ProtectedDec;BracketBatch.Feasible.decs;BracketBatch.Infeasible.decs];
        ProtectedLabel = [ProtectedLabel;ones(numel(BracketBatch.Feasible),1);zeros(numel(BracketBatch.Infeasible),1)];
    end
    if ~isempty(HardNegBatch.Dec)
        ProtectedDec = [ProtectedDec;HardNegBatch.Dec];
        ProtectedLabel = [ProtectedLabel;zeros(size(HardNegBatch.Dec,1),1)];
    end
end

function [Model,PendingLabels,LastCalMetric] = UpdateBoundaryModel( ...
    Model,TrainDec,TrainLabel,CalibDec,CalibLabel,Hidden,Epoch,WarmEpoch,LR, ...
    Generation,PendingLabels,TriggerCount,UpdateGap,RestartGap,LastCalMetric)

    CurrentMetric = EvaluateBoundaryCalibration(Model,CalibDec,CalibLabel);
    NeedUpdate = isempty(Model) || ...
        PendingLabels >= TriggerCount || ...
        mod(Generation,UpdateGap) == 0 || ...
        IsCalibrationDrifting(CurrentMetric,LastCalMetric);
    if ~NeedUpdate
        return;
    end

    UseWarmStart = ~isempty(Model) && mod(Generation,RestartGap) ~= 0;
    TrainEpoch   = Epoch;
    PrevModel    = [];
    if UseWarmStart
        PrevModel  = Model;
        TrainEpoch = WarmEpoch;
    end

    Model = TrainBoundaryMLP( ...
        TrainDec,TrainLabel,Hidden,TrainEpoch,LR,PrevModel,CalibDec,CalibLabel, ...
        struct('WarmStart',UseWarmStart));
    PendingLabels = 0;
    LastCalMetric = EvaluateBoundaryCalibration(Model,CalibDec,CalibLabel);
end

function flag = IsCalibrationDrifting(CurrentMetric,LastMetric)
    if isempty(LastMetric) || ~isfinite(LastMetric.brier)
        flag = CurrentMetric.nearCount >= 5 && CurrentMetric.nearGap > 0.15;
        return;
    end

    flag = false;
    if CurrentMetric.nearCount >= 5 && CurrentMetric.nearGap > 0.15
        flag = true;
        return;
    end
    if isfinite(CurrentMetric.brier) && CurrentMetric.brier > LastMetric.brier + 0.02
        flag = true;
        return;
    end
    if isfinite(CurrentMetric.ece) && CurrentMetric.ece > LastMetric.ece + 0.02
        flag = true;
        return;
    end
    if CurrentMetric.nearCount >= 5 && CurrentMetric.nearGap > LastMetric.nearGap + 0.05
        flag = true;
    end
end

function [AllOffspring,AllInfo] = MergeBoundaryResults(Primary,PrimaryInfo,Extra,ExtraInfo,M)
    if isempty(Primary)
        AllOffspring = Extra;
        AllInfo      = NormalizeBoundaryInfo(ExtraInfo,M);
        return;
    end
    if isempty(Extra)
        AllOffspring = Primary;
        AllInfo      = NormalizeBoundaryInfo(PrimaryInfo,M);
        return;
    end

    AllOffspring = [Primary,Extra];
    PrimaryInfo = NormalizeBoundaryInfo(PrimaryInfo,M);
    ExtraInfo   = NormalizeBoundaryInfo(ExtraInfo,M);
    AllInfo.source    = [PrimaryInfo.source;ExtraInfo.source];
    AllInfo.score     = [PrimaryInfo.score;ExtraInfo.score];
    AllInfo.prob      = [PrimaryInfo.prob;ExtraInfo.prob];
    AllInfo.entropy   = [PrimaryInfo.entropy;ExtraInfo.entropy];
    AllInfo.hvGain    = [PrimaryInfo.hvGain;ExtraInfo.hvGain];
    AllInfo.novelty   = [PrimaryInfo.novelty;ExtraInfo.novelty];
    AllInfo.penalty   = [PrimaryInfo.penalty;ExtraInfo.penalty];
    AllInfo.utility   = [PrimaryInfo.utility;ExtraInfo.utility];
    AllInfo.sector    = [PrimaryInfo.sector;ExtraInfo.sector];
    AllInfo.proxyObjs = [PrimaryInfo.proxyObjs;ExtraInfo.proxyObjs];
end

function Info = NormalizeBoundaryInfo(Info,M)
    if nargin < 2
        M = 0;
    end
    if isempty(Info)
        Info.source    = zeros(0,1);
        Info.score     = zeros(0,1);
        Info.prob      = zeros(0,1);
        Info.entropy   = zeros(0,1);
        Info.hvGain    = zeros(0,1);
        Info.novelty   = zeros(0,1);
        Info.penalty   = zeros(0,1);
        Info.utility   = zeros(0,1);
        Info.sector    = zeros(0,1);
        Info.proxyObjs = zeros(0,M);
        return;
    end
    Fields = {'source','score','prob','entropy','hvGain','novelty','penalty','utility','sector'};
    for i = 1 : numel(Fields)
        if ~isfield(Info,Fields{i}) || isempty(Info.(Fields{i}))
            Info.(Fields{i}) = zeros(0,1);
        end
    end
    if ~isfield(Info,'proxyObjs') || isempty(Info.proxyObjs)
        Info.proxyObjs = zeros(0,M);
    end
end

function Info = SliceBoundaryInfo(Info,Idx,M)
    Info = NormalizeBoundaryInfo(Info,M);
    if isempty(Idx)
        Info = NormalizeBoundaryInfo([],M);
        return;
    end
    Info.source    = Info.source(Idx);
    Info.score     = Info.score(Idx);
    Info.prob      = Info.prob(Idx);
    Info.entropy   = Info.entropy(Idx);
    Info.hvGain    = Info.hvGain(Idx);
    Info.novelty   = Info.novelty(Idx);
    Info.penalty   = Info.penalty(Idx);
    Info.utility   = Info.utility(Idx);
    Info.sector    = Info.sector(Idx);
    Info.proxyObjs = Info.proxyObjs(Idx,:);
end
