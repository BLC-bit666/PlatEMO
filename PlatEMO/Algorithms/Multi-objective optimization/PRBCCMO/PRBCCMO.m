classdef PRBCCMO < ALGORITHM
% <2026> <multi> <real/integer/label/binary/permutation> <constrained>
% Pareto-relevant boundary CCMO
% bRho     --- 0.2  --- Boundary evaluation ratio relative to N
% trainRho --- 2    --- Training archive size ratio
% hidden   --- 20   --- Hidden units of the boundary MLP
% epoch    --- 25   --- Training epochs of the boundary MLP
% lr       --- 0.01 --- Learning rate of the boundary MLP
% mRho     --- 0.4  --- Seed-query ratio within each boundary budget
% ensK     --- 3    --- Committee size of shallow boundary MLPs
% calMode  --- 2    --- Calibration mode (1 raw, 2 auto temp/beta, 3 beta)
% dLambda  --- 1    --- Committee disagreement weight in boundary utility
% pairM    --- 0.05 --- Margin for tight bracket pair loss
% lPair    --- 1    --- Weight of bracket pair loss
% lMid     --- 1    --- Weight of midpoint-to-0.5 loss
% selMode  --- 1    --- Section B seed selection mode (1. trusted-query 2. uncertain-only 3. random-bridge)
% localMode--- 1    --- Section B local search mode (1. label-aware 2. isotropic)
% traceOn  --- 0    --- Enable Section B audit trace recording

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
            Params = ResolvePRBCCMOParameters(Algorithm.parameter);
            bRho      = Params.bRho;
            trainRho  = Params.trainRho;
            hidden    = Params.hidden;
            epoch     = Params.epoch;
            lr        = Params.lr;
            mRho      = Params.mRho;
            ensK      = Params.ensK;
            calMode   = Params.calMode;
            dLambda   = Params.dLambda;
            pairM     = Params.pairM;
            lPair     = Params.lPair;
            lMid      = Params.lMid;
            selMode   = Params.selMode;
            localMode = Params.localMode;
            traceOn   = Params.traceOn;
            RuntimeOptions = BuildBoundaryRuntimeOptions(selMode,localMode,traceOn);

            BoundaryBudget = max(0,floor(bRho*Problem.N));
            TrainMax       = max(1,round(trainRho*Problem.N));
            CalibMax       = max(1,Problem.N);
            TestMax        = max(1,Problem.N);
            BracketMax     = max(1,Problem.N);
            HardNegMax     = max(20,ceil(0.25*TrainMax));
            SeedRatio      = min(max(mRho,0),1);
            [W,~]          = UniformPoint(max(Problem.N,2),Problem.M);
            UpdateGap      = 5;
            RestartGap     = 25;
            WarmEpoch      = min(epoch,max(5,round(epoch/3)));
            TriggerCount   = max(1,ceil(0.1*TrainMax));
            TightGap       = 0.03;

            %% Generate random populations
            PopulationC = Problem.Initialization();
            PopulationU = Problem.Initialization();

            %% Initialize boundary memories
            BracketArchive = EmptyBracketArchive(Problem.D);
            HardNegativeArchive.Dec    = zeros(0,Problem.D);
            HardNegativeArchive.Radius = zeros(0,1);
            InitSolutions = [PopulationC,PopulationU];
            [InitTrain,~,InitCalib,InitCalibInfo,InitTest,InitTestInfo] = SplitHeldOutBatch( ...
                InitSolutions,NormalizeBoundaryInfo([],Problem.M),CalibMax,TestMax,Problem.M);
            InitHoldoutDec = [SolutionDecs(InitCalib,Problem.D);SolutionDecs(InitTest,Problem.D)];
            ProtectedDec   = zeros(0,Problem.D);
            ProtectedLabel = zeros(0,1);
            [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                [],[],ProtectedDec,ProtectedLabel,InitTrain,InitHoldoutDec,TrainMax);
            [CalibDec,CalibLabel,CalibNear] = UpdateCalibrationBuffer( ...
                [],[],[],InitCalib,InitCalibInfo,CalibMax);
            [TestDec,TestLabel,TestNear] = UpdateCalibrationBuffer( ...
                [],[],[],InitTest,InitTestInfo,TestMax);
            TrainOptions = BuildBoundaryTrainingOptions( ...
                BracketArchive,ensK,calMode,dLambda,pairM,lPair,lMid,Problem.D,TightGap);
            Model = TrainBoundaryMLP( ...
                TrainDec,TrainLabel,hidden,epoch,lr,[],CalibDec,CalibLabel,TrainOptions);
            Model = RefreshBoundaryTrust(Model,TestDec,TestLabel);
            LastCalMetric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel);
            PendingLabels = 0;
            Generation    = 0;
            [ExternalArchive,~] = UpdateExternalArchive([],FilterFeasiblePopulation(InitSolutions));
            Algorithm.metric.boundaryCalibration = AttachCalibrationContext( ...
                LastCalMetric,Generation,Problem.FE,size(TrainDec,1),size(CalibDec,1), ...
                sum(CalibNear),size(TestDec,1),sum(TestNear));
            Algorithm.metric.sectionB = InitSectionBMetric(Problem.D,RuntimeOptions,ExternalArchive);
            Algorithm.metric.sectionB = AppendSectionBCalibrationTrace( ...
                Algorithm.metric.sectionB,Algorithm.metric.boundaryCalibration);

            %% Optimization
            while Algorithm.NotTerminated(PopulationC)
                Generation = Generation + 1;

                [OffspringC,OffspringU] = GenerateRegularOffspring( ...
                    Problem,PopulationC,PopulationU);

                CandidatePool = GenerateBoundaryCandidates( ...
                    Problem,PopulationC,PopulationU,W,RuntimeOptions);

                BoundaryBudgetNow = min(BoundaryBudget,max(0,Problem.maxFE-Problem.FE));
                SeedBudget = min(BoundaryBudgetNow,max(0,round(SeedRatio*BoundaryBudgetNow)));
                if BoundaryBudgetNow > 0 && SeedBudget == 0
                    SeedBudget = 1;
                end
                [BoundarySeeds,SeedInfo] = SelectBoundaryCandidates( ...
                    Problem,CandidatePool,PopulationC,Model,W,HardNegativeArchive,SeedBudget,RuntimeOptions);

                WorkerBudget = max(0,BoundaryBudgetNow-numel(BoundarySeeds));
                [WorkerOffspring,WorkerInfo,WorkerFeasiblePool,BracketBatch,HardNegBatch,WorkerAudit] = ...
                    RefineBoundaryWorkers( ...
                        Problem,BoundarySeeds,SeedInfo,PopulationC,Model,W, ...
                        HardNegativeArchive,WorkerBudget,RuntimeOptions);
                [BoundaryOffspring,BoundaryInfo] = MergeBoundaryResults( ...
                    BoundarySeeds,SeedInfo,WorkerOffspring,WorkerInfo,Problem.M);
                MigrationPool = ScreenBoundaryMigrants( ...
                    PopulationC,BoundarySeeds,WorkerFeasiblePool,W,RuntimeOptions);

                HardNegativeArchive = UpdateHardNegativeArchive(HardNegativeArchive,HardNegBatch,HardNegMax);
                BracketArchive = UpdateBracketArchive(BracketArchive,BracketBatch,BracketMax,Problem.D,TightGap);

                ConstrainedBase = KeepUniquePopulation([PopulationC,OffspringC]);
                PopulationC = EnvironmentalSelectionC(ConstrainedBase,Problem.N,MigrationPool,W);
                PopulationU = EnvironmentalSelectionU(KeepUniquePopulation([PopulationU,OffspringU]),Problem.N);

                [ExternalArchive,BoundaryGain,BoundaryAdded] = UpdateSectionBExternalArchive( ...
                    ExternalArchive,OffspringC,OffspringU,BoundaryOffspring);
                Algorithm.metric.sectionB.seedAudit = AppendBoundarySeedAuditRows( ...
                    Algorithm.metric.sectionB.seedAudit,BoundarySeeds,SeedInfo,WorkerAudit, ...
                    BoundaryAdded,Generation,Problem.FE,Problem.D);
                Algorithm.metric.sectionB.boundaryGainTrace = AppendBoundaryGainTrace( ...
                    Algorithm.metric.sectionB.boundaryGainTrace,Generation,Problem.FE, ...
                    BoundaryGain,numel(BoundaryAdded),numel(ExternalArchive));
                Algorithm.metric.sectionB.externalArchiveCount = numel(ExternalArchive);
                Algorithm.metric.sectionB.totalBoundaryGain = Algorithm.metric.sectionB.totalBoundaryGain + BoundaryGain;

                [TrainBatch,~,CalibBatch,CalibInfo,TestBatch,TestInfo] = SplitHeldOutBatch( ...
                    BoundaryOffspring,BoundaryInfo,CalibMax,TestMax,Problem.M);
                HoldoutDec = [CalibDec;TestDec; ...
                    SolutionDecs(CalibBatch,Problem.D);SolutionDecs(TestBatch,Problem.D)];
                [ProtectedBracketDec,ProtectedBracketLabel] = BuildBracketProtectedBuffer(BracketArchive,Problem.D);
                [ProtectedBracketDec,ProtectedBracketLabel] = ExcludeLabeledRows( ...
                    ProtectedBracketDec,ProtectedBracketLabel,HoldoutDec);
                ProtectedDec   = ProtectedBracketDec;
                ProtectedLabel = ProtectedBracketLabel;
                [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                    TrainDec,TrainLabel,ProtectedDec,ProtectedLabel,TrainBatch,HoldoutDec,TrainMax);
                [CalibDec,CalibLabel,CalibNear] = UpdateCalibrationBuffer( ...
                    CalibDec,CalibLabel,CalibNear,CalibBatch,CalibInfo,CalibMax);
                [TestDec,TestLabel,TestNear] = UpdateCalibrationBuffer( ...
                    TestDec,TestLabel,TestNear,TestBatch,TestInfo,TestMax);

                PendingLabels = PendingLabels + numel(BoundaryOffspring);
                TrainOptions = BuildBoundaryTrainingOptions( ...
                    BracketArchive,ensK,calMode,dLambda,pairM,lPair,lMid,Problem.D,TightGap);
                [Model,PendingLabels,LastCalMetric] = UpdateBoundaryModel( ...
                    Model,TrainDec,TrainLabel,CalibDec,CalibLabel,TestDec,TestLabel, ...
                    hidden,epoch,WarmEpoch,lr,Generation,PendingLabels,TriggerCount, ...
                    UpdateGap,RestartGap,LastCalMetric,TrainOptions);
                Algorithm.metric.boundaryCalibration = AttachCalibrationContext( ...
                    EvaluateBoundaryCalibration(Model,TestDec,TestLabel), ...
                    Generation,Problem.FE,size(TrainDec,1),size(CalibDec,1), ...
                    sum(CalibNear),size(TestDec,1),sum(TestNear));
                Algorithm.metric.sectionB = AppendSectionBCalibrationTrace( ...
                    Algorithm.metric.sectionB,Algorithm.metric.boundaryCalibration);
            end
        end
    end
end

function [OffspringC,OffspringU] = GenerateRegularOffspring(Problem,PopulationC,PopulationU)
    OffspringC = Problem.Evaluation(OperatorDE_current_rand_1(Problem,PopulationC.decs));
    OffspringU = Problem.Evaluation(OperatorDE_current_rand_1(Problem,PopulationU.decs));
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

function [TrainSolutions,TrainInfo,CalibSolutions,CalibInfo,TestSolutions,TestInfo] = SplitHeldOutBatch(Solutions,Info,CalibMax,TestMax,M)
    TrainSolutions = Solutions;
    TrainInfo = NormalizeBoundaryInfo(Info,M);
    CalibSolutions = [];
    CalibInfo = NormalizeBoundaryInfo([],M);
    TestSolutions = [];
    TestInfo = NormalizeBoundaryInfo([],M);
    Count = numel(Solutions);
    if Count <= 1 || (CalibMax <= 0 && TestMax <= 0)
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

    CalibQuota = min(max(1,round(0.2*Count)),min(CalibMax,Count-1));
    CalibIdx = SelectCalibrationHoldout(Label,NearMask,CalibQuota);
    RemainingMask = true(Count,1);
    RemainingMask(CalibIdx) = false;

    RemainingIdx = find(RemainingMask);
    TestQuota = min(max(1,round(0.2*Count)),TestMax);
    TestQuota = min(TestQuota,max(0,numel(RemainingIdx)-1));
    if TestQuota > 0
        TestLocalIdx = SelectCalibrationHoldout(Label(RemainingIdx),NearMask(RemainingIdx),TestQuota);
        TestIdx = RemainingIdx(TestLocalIdx);
    else
        TestIdx = zeros(0,1);
    end

    TrainMask = true(Count,1);
    TrainMask(CalibIdx) = false;
    TrainMask(TestIdx) = false;
    TrainIdx = find(TrainMask);
    TrainSolutions = Solutions(TrainIdx);
    TrainInfo = SliceBoundaryInfo(Info,TrainIdx,M);
    CalibSolutions = Solutions(CalibIdx);
    CalibInfo = SliceBoundaryInfo(Info,CalibIdx,M);
    TestSolutions = Solutions(TestIdx);
    TestInfo = SliceBoundaryInfo(Info,TestIdx,M);
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
    D = 0;
    if Count > 0
        D = size(Solutions.decs,2);
    end
    Info = NormalizeBoundaryInfo([],M,D);
    Info.source        = zeros(Count,1);
    Info.score         = zeros(Count,1);
    Info.prob          = 0.5*ones(Count,1);
    Info.queryScore    = zeros(Count,1);
    Info.disagreement  = zeros(Count,1);
    Info.paretoValue   = zeros(Count,1);
    Info.reliability   = zeros(Count,1);
    Info.boundaryTrust = zeros(Count,1);
    Info.utility       = zeros(Count,1);
    Info.sector        = zeros(Count,1);
    Info.eligible      = true(Count,1);
    Info.proxyObjs     = Solutions.objs;
    Info.anchorDec     = zeros(Count,D);
    Info.anchorObj     = zeros(Count,M);
    Info.helperDec     = zeros(Count,D);
    Info.helperObj     = zeros(Count,M);
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

function [ProtectedDec,ProtectedLabel] = BuildBracketProtectedBuffer(BracketArchive,D)
    ProtectedDec = zeros(0,D);
    ProtectedLabel = zeros(0,1);
    if isempty(BracketArchive) || isempty(BracketArchive.FeasibleDec)
        return;
    end
    ProtectedDec = [BracketArchive.FeasibleDec;BracketArchive.InfeasibleDec];
    ProtectedLabel = [ones(size(BracketArchive.FeasibleDec,1),1);zeros(size(BracketArchive.InfeasibleDec,1),1)];
end

function Archive = EmptyBracketArchive(D)
    Archive.FeasibleDec   = zeros(0,D);
    Archive.InfeasibleDec = zeros(0,D);
    Archive.Gap           = zeros(0,1);
end

function Archive = UpdateBracketArchive(Archive,NewPairs,MaxPairs,D,TightGap)
    if nargin < 1 || isempty(Archive)
        Archive = EmptyBracketArchive(D);
    end
    if nargin < 3 || MaxPairs <= 0
        Archive = EmptyBracketArchive(D);
        return;
    end
    if nargin < 5 || isempty(TightGap)
        TightGap = 0.03;
    end

    NewF = zeros(0,D);
    NewI = zeros(0,D);
    NewG = zeros(0,1);
    if ~isempty(NewPairs) && isfield(NewPairs,'Feasible') && ~isempty(NewPairs.Feasible)
        NewF = SolutionDecs(NewPairs.Feasible,D);
        NewI = SolutionDecs(NewPairs.Infeasible,D);
        NewG = NewPairs.Gap(:);
    end

    AllF = [Archive.FeasibleDec;NewF];
    AllI = [Archive.InfeasibleDec;NewI];
    AllG = [Archive.Gap(:);NewG];
    KeepTight = isfinite(AllG) & AllG <= TightGap;
    AllF = AllF(KeepTight,:);
    AllI = AllI(KeepTight,:);
    AllG = AllG(KeepTight);
    if isempty(AllF)
        Archive = EmptyBracketArchive(D);
        return;
    end

    PairKey = [AllF,AllI];
    Keep = KeepLatestDecisionRows(PairKey);
    AllF = AllF(Keep,:);
    AllI = AllI(Keep,:);
    AllG = AllG(Keep);
    [~,Order] = sort(AllG,'ascend');
    Order = Order(1:min(MaxPairs,numel(Order)));
    Archive.FeasibleDec   = AllF(Order,:);
    Archive.InfeasibleDec = AllI(Order,:);
    Archive.Gap           = AllG(Order);
end

function Options = BuildBoundaryTrainingOptions( ...
    BracketArchive,EnsembleSize,CalMode,DisagreementWeight,PairMargin,LambdaPair,LambdaMid,D,TightGap)

    Options = struct();
    Options.EnsembleSize       = EnsembleSize;
    Options.CalibratorCandidates = DecodeCalibrationMode(CalMode);
    Options.DisagreementWeight = max(DisagreementWeight,0);
    Options.PairMargin         = max(PairMargin,0);
    Options.LambdaBrier        = 0.5;
    Options.LambdaPair         = max(LambdaPair,0);
    Options.LambdaMid          = max(LambdaMid,0);
    Options.PairFeasibleDec    = zeros(0,D);
    Options.PairInfeasibleDec  = zeros(0,D);
    Options.MidDec             = zeros(0,D);
    if nargin < 9 || isempty(TightGap)
        TightGap = 0.03;
    end

    if ~isempty(BracketArchive) && ~isempty(BracketArchive.FeasibleDec)
        Options.PairFeasibleDec   = BracketArchive.FeasibleDec;
        Options.PairInfeasibleDec = BracketArchive.InfeasibleDec;
        TightMask = true(size(BracketArchive.FeasibleDec,1),1);
        if isfield(BracketArchive,'Gap') && numel(BracketArchive.Gap) == size(BracketArchive.FeasibleDec,1)
            TightMask = isfinite(BracketArchive.Gap(:)) & BracketArchive.Gap(:) <= TightGap;
        end
        if any(TightMask)
            Options.MidDec = 0.5*(BracketArchive.FeasibleDec(TightMask,:) + ...
                BracketArchive.InfeasibleDec(TightMask,:));
        end
    end
end

function Type = DecodeCalibrationMode(CalMode)
    switch round(CalMode)
        case 1
            Type = {'raw'};
        case 2
            Type = {'temperature','beta'};
        case 3
            Type = {'beta'};
        otherwise
            Type = {'temperature','beta'};
    end
end

function [Model,PendingLabels,LastCalMetric] = UpdateBoundaryModel( ...
    Model,TrainDec,TrainLabel,CalibDec,CalibLabel,TestDec,TestLabel, ...
    Hidden,Epoch,WarmEpoch,LR,Generation,PendingLabels,TriggerCount, ...
    UpdateGap,RestartGap,LastCalMetric,TrainOptions)

    CurrentMetric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel);
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

    if nargin < 18 || ~isstruct(TrainOptions)
        TrainOptions = struct();
    end
    TrainOptions.WarmStart = UseWarmStart;
    Model = TrainBoundaryMLP( ...
        TrainDec,TrainLabel,Hidden,TrainEpoch,LR,PrevModel,CalibDec,CalibLabel, ...
        TrainOptions);
    Model = RefreshBoundaryTrust(Model,TestDec,TestLabel);
    PendingLabels = 0;
    LastCalMetric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel);
end

function Model = RefreshBoundaryTrust(Model,TestDec,TestLabel)
    if isempty(Model)
        return;
    end

    Metric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel);
    TrustGate = isfinite(Metric.ece) && Metric.ece <= 0.05 ...
        && isfinite(Metric.nearGap) && Metric.nearGap <= 0.05 ...
        && Metric.nearCount >= 20;
    Model.TrustGate = TrustGate;
    Model.TrustMetric = Metric;
    if isfield(Metric,'binEdges') && isfield(Metric,'bin') && isfield(Metric.bin,'count')
        Model.ReliabilityBinEdges = Metric.binEdges;
        Reliability = zeros(numel(Metric.bin.count),1);
        Valid = Metric.bin.count > 0 & isfinite(Metric.bin.feasibleRate);
        Reliability(Valid) = max(0,1 - 2*abs(Metric.bin.feasibleRate(Valid)-0.5));
        if any(Valid)
            Reliability = FillReliabilityGaps(Reliability,Valid);
        end
        Model.ReliabilityScore = Reliability;
    else
        Model.ReliabilityBinEdges = linspace(0,1,11);
        Model.ReliabilityScore = zeros(10,1);
    end
end

function Reliability = FillReliabilityGaps(Reliability,Valid)
    ValidIdx = find(Valid);
    for i = 1 : numel(Reliability)
        if Valid(i)
            continue;
        end
        [~,Best] = min(abs(ValidIdx-i));
        Reliability(i) = Reliability(ValidIdx(Best));
    end
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

function Metric = AttachCalibrationContext(Metric,Generation,FE,TrainCount,CalibCount,CalNearCount,TestCount,TestNearCount)
    Metric.generation        = Generation;
    Metric.FE                = FE;
    Metric.trainingCount     = TrainCount;
    Metric.calibrationCount  = CalibCount;
    Metric.calibrationNearCount = CalNearCount;
    Metric.testCount         = TestCount;
    Metric.testNearCount     = TestNearCount;
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
    D = max(ResolveBoundaryInfoDecisionWidth(PrimaryInfo),ResolveBoundaryInfoDecisionWidth(ExtraInfo));
    PrimaryInfo = NormalizeBoundaryInfo(PrimaryInfo,M,D);
    ExtraInfo   = NormalizeBoundaryInfo(ExtraInfo,M,D);
    AllInfo     = NormalizeBoundaryInfo([],M,D);
    VectorFields = BoundaryInfoVectorFields();
    for i = 1 : numel(VectorFields)
        Field = VectorFields{i};
        AllInfo.(Field) = [PrimaryInfo.(Field);ExtraInfo.(Field)];
    end
    MatrixFields = BoundaryInfoMatrixFields();
    for i = 1 : numel(MatrixFields)
        Field = MatrixFields{i};
        AllInfo.(Field) = [PrimaryInfo.(Field);ExtraInfo.(Field)];
    end
end

function Info = NormalizeBoundaryInfo(Info,M,D)
    if nargin < 2
        M = 0;
    end
    if nargin < 3 || isempty(D)
        D = ResolveBoundaryInfoDecisionWidth(Info);
    end
    Count = ResolveBoundaryInfoCount(Info);
    if isempty(Info)
        Info = struct();
    end

    Info.source        = ResolveBoundaryVector(Info,'source',Count,0);
    Info.score         = ResolveBoundaryVector(Info,'score',Count,0);
    Info.prob          = ResolveBoundaryVector(Info,'prob',Count,0);
    Info.queryScore    = ResolveBoundaryVector(Info,'queryScore',Count,0);
    Info.disagreement  = ResolveBoundaryVector(Info,'disagreement',Count,0);
    Info.paretoValue   = ResolveBoundaryVector(Info,'paretoValue',Count,0);
    Info.reliability   = ResolveBoundaryVector(Info,'reliability',Count,0);
    Info.boundaryTrust = ResolveBoundaryVector(Info,'boundaryTrust',Count,[]);
    if isempty(Info.boundaryTrust)
        Info.boundaryTrust = Info.reliability .* Info.queryScore;
    end
    Info.utility       = ResolveBoundaryVector(Info,'utility',Count,[]);
    if isempty(Info.utility)
        Info.utility = Info.score;
    end
    Info.sector        = ResolveBoundaryVector(Info,'sector',Count,0);
    Info.eligible      = ResolveBoundaryEligible(Info,Count);
    Info.proxyObjs     = ResolveBoundaryMatrix(Info,'proxyObjs',Count,M);
    Info.anchorDec     = ResolveBoundaryMatrix(Info,'anchorDec',Count,D);
    Info.anchorObj     = ResolveBoundaryMatrix(Info,'anchorObj',Count,M);
    Info.helperDec     = ResolveBoundaryMatrix(Info,'helperDec',Count,D);
    Info.helperObj     = ResolveBoundaryMatrix(Info,'helperObj',Count,M);
end

function Info = SliceBoundaryInfo(Info,Idx,M)
    D = ResolveBoundaryInfoDecisionWidth(Info);
    Info = NormalizeBoundaryInfo(Info,M,D);
    if isempty(Idx)
        Info = NormalizeBoundaryInfo([],M,D);
        return;
    end
    VectorFields = BoundaryInfoVectorFields();
    for i = 1 : numel(VectorFields)
        Field = VectorFields{i};
        Info.(Field) = Info.(Field)(Idx);
    end
    MatrixFields = BoundaryInfoMatrixFields();
    for i = 1 : numel(MatrixFields)
        Field = MatrixFields{i};
        Info.(Field) = Info.(Field)(Idx,:);
    end
end

function Fields = BoundaryInfoVectorFields()
    Fields = {'source','score','prob','queryScore','disagreement','paretoValue', ...
        'reliability','boundaryTrust','utility','sector','eligible'};
end

function Fields = BoundaryInfoMatrixFields()
    Fields = {'proxyObjs','anchorDec','anchorObj','helperDec','helperObj'};
end

function Count = ResolveBoundaryInfoCount(Info)
    Count = 0;
    if ~isstruct(Info)
        return;
    end
    VectorFields = BoundaryInfoVectorFields();
    for i = 1 : numel(VectorFields)
        Field = VectorFields{i};
        if isfield(Info,Field) && ~isempty(Info.(Field))
            Count = numel(Info.(Field));
            return;
        end
    end
    MatrixFields = BoundaryInfoMatrixFields();
    for i = 1 : numel(MatrixFields)
        Field = MatrixFields{i};
        if isfield(Info,Field) && ~isempty(Info.(Field))
            Count = size(Info.(Field),1);
            return;
        end
    end
end

function D = ResolveBoundaryInfoDecisionWidth(Info)
    D = 0;
    if ~isstruct(Info)
        return;
    end
    if isfield(Info,'anchorDec') && ~isempty(Info.anchorDec)
        D = size(Info.anchorDec,2);
        return;
    end
    if isfield(Info,'helperDec') && ~isempty(Info.helperDec)
        D = size(Info.helperDec,2);
    end
end

function Value = ResolveBoundaryVector(Info,Field,Count,Default)
    if nargin < 4
        Default = 0;
    end
    if isstruct(Info) && isfield(Info,Field) && ~isempty(Info.(Field))
        Value = Info.(Field)(:);
        return;
    end
    if isempty(Default)
        Value = [];
    else
        Value = repmat(Default,Count,1);
    end
end

function Value = ResolveBoundaryEligible(Info,Count)
    if isstruct(Info) && isfield(Info,'eligible') && ~isempty(Info.eligible)
        Value = logical(Info.eligible(:));
        return;
    end
    Value = true(Count,1);
end

function Value = ResolveBoundaryMatrix(Info,Field,Count,Width)
    if nargin < 4
        Width = 0;
    end
    if isstruct(Info) && isfield(Info,Field) && ~isempty(Info.(Field))
        Value = Info.(Field);
        return;
    end
    Value = zeros(Count,Width);
end

function Params = ResolvePRBCCMOParameters(ParameterCell)
    if nargin >= 1 && ~isempty(ParameterCell)
        if ~iscell(ParameterCell)
            ParameterCell = {ParameterCell};
        end
    else
        ParameterCell = {};
    end

    LegacyDefaults = {1,0.2,0.5,0.5,2,20,25,0.01,0.5,0.5,0.4, ...
        3,2,1,0.05,1,1,1,1,1,0};
    LegacyMap = struct( ...
        'bRho',2,'trainRho',5,'hidden',6,'epoch',7,'lr',8,'mRho',11, ...
        'ensK',12,'calMode',13,'dLambda',14,'pairM',15,'lPair',16, ...
        'lMid',17,'selMode',19,'localMode',20,'traceOn',21);
    ActiveNames = {'bRho','trainRho','hidden','epoch','lr','mRho','ensK', ...
        'calMode','dLambda','pairM','lPair','lMid','selMode','localMode','traceOn'};
    ActiveDefaults = {0.2,2,20,25,0.01,0.4,3,2,1,0.05,1,1,1,1,0};

    Params = cell2struct(ActiveDefaults,ActiveNames,2);
    if numel(ParameterCell) >= numel(LegacyDefaults)
        for i = 1 : numel(ActiveNames)
            Index = LegacyMap.(ActiveNames{i});
            if numel(ParameterCell) >= Index && ~isempty(ParameterCell{Index})
                Params.(ActiveNames{i}) = ParameterCell{Index};
            elseif ~isempty(LegacyDefaults{Index})
                Params.(ActiveNames{i}) = LegacyDefaults{Index};
            end
        end
        return;
    end

    Limit = min(numel(ParameterCell),numel(ActiveNames));
    for i = 1 : Limit
        if ~isempty(ParameterCell{i})
            Params.(ActiveNames{i}) = ParameterCell{i};
        end
    end
end

function Metric = InitSectionBMetric(D,RuntimeOptions,ExternalArchive)
    if nargin < 1 || isempty(D)
        D = 0;
    end
    if nargin < 2 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    if nargin < 3 || isempty(ExternalArchive)
        ExternalArchive = [];
    end

    Metric = struct();
    Metric.selectionMode = SafeRuntimeOption(RuntimeOptions,'SelectionMode',1);
    Metric.selectionName = SafeRuntimeOption(RuntimeOptions,'SelectionName','trusted_query');
    Metric.localMode     = SafeRuntimeOption(RuntimeOptions,'LocalMode',1);
    Metric.localName     = SafeRuntimeOption(RuntimeOptions,'LocalName','label_aware');
    Metric.traceFlag     = logical(SafeRuntimeOption(RuntimeOptions,'TraceFlag',false));
    Metric.seedAudit = repmat(InitBoundarySeedAuditRow(D),0,1);
    Metric.boundaryGainTrace = repmat(InitBoundaryGainTraceRow(),0,1);
    Metric.calibrationTrace = repmat(InitSectionBCalibrationTraceRow(),0,1);
    Metric.totalBoundaryGain = 0;
    Metric.externalArchiveCount = numel(ExternalArchive);
end

function Value = SafeRuntimeOption(RuntimeOptions,Field,Default)
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,Field) && ~isempty(RuntimeOptions.(Field))
        Value = RuntimeOptions.(Field);
    else
        Value = Default;
    end
end

function Row = InitBoundarySeedAuditRow(D)
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'source',NaN, ...
        'seedFeasible',false, ...
        'prob',NaN, ...
        'queryScore',NaN, ...
        'disagreement',NaN, ...
        'paretoValue',NaN, ...
        'reliability',NaN, ...
        'boundaryTrust',NaN, ...
        'eligible',false, ...
        'utility',NaN, ...
        'localEvalCount',0, ...
        'frrSuccess',false, ...
        'ubySuccess',false, ...
        'bracketGap',NaN, ...
        'hardNegativeConfirmed',false, ...
        'seedDec',zeros(1,D), ...
        'lineageFeasibleDec',zeros(0,D));
end

function Rows = AppendBoundarySeedAuditRows(Rows,BoundarySeeds,SeedInfo,WorkerAudit,BoundaryAdded,Generation,FE,D)
    if nargin < 8 || isempty(D)
        if isempty(BoundarySeeds)
            D = 0;
        else
            D = numel(BoundarySeeds(1).dec);
        end
    end
    if isempty(BoundarySeeds)
        if isempty(Rows)
            Rows = repmat(InitBoundarySeedAuditRow(D),0,1);
        end
        return;
    end

    UBYSuccess = ResolveBoundarySeedYield(WorkerAudit,BoundaryAdded);
    AddRows = repmat(InitBoundarySeedAuditRow(D),numel(BoundarySeeds),1);
    for i = 1 : numel(BoundarySeeds)
        AddRows(i).generation = Generation;
        AddRows(i).FE         = FE;
        AddRows(i).source     = SafeInfoValue(SeedInfo,'source',i,NaN);
        AddRows(i).seedFeasible = all(BoundarySeeds(i).cons<=0,2);
        AddRows(i).prob       = SafeInfoValue(SeedInfo,'prob',i,NaN);
        AddRows(i).queryScore = SafeInfoValue(SeedInfo,'queryScore',i,NaN);
        AddRows(i).disagreement = SafeInfoValue(SeedInfo,'disagreement',i,NaN);
        AddRows(i).paretoValue = SafeInfoValue(SeedInfo,'paretoValue',i,NaN);
        AddRows(i).reliability = SafeInfoValue(SeedInfo,'reliability',i,NaN);
        AddRows(i).boundaryTrust = SafeInfoValue(SeedInfo,'boundaryTrust',i,NaN);
        AddRows(i).eligible   = logical(SafeInfoValue(SeedInfo,'eligible',i,false));
        AddRows(i).utility    = SafeInfoValue(SeedInfo,'utility',i,NaN);
        AddRows(i).seedDec    = BoundarySeeds(i).dec;
        if nargin >= 4 && numel(WorkerAudit) >= i
            AddRows(i).localEvalCount = WorkerAudit(i).localEvalCount;
            AddRows(i).frrSuccess = WorkerAudit(i).frrSuccess;
            AddRows(i).bracketGap = WorkerAudit(i).bracketGap;
            AddRows(i).hardNegativeConfirmed = WorkerAudit(i).hardNegativeConfirmed;
            AddRows(i).lineageFeasibleDec = WorkerAudit(i).lineageFeasibleDec;
        end
        if numel(UBYSuccess) >= i
            AddRows(i).ubySuccess = UBYSuccess(i);
        end
    end

    if isempty(Rows)
        Rows = AddRows;
    else
        Rows = [Rows;AddRows];
    end
end

function Value = SafeInfoValue(Info,Field,Index,Default)
    Value = Default;
    if ~isstruct(Info) || ~isfield(Info,Field) || isempty(Info.(Field))
        return;
    end
    Data = Info.(Field);
    if numel(Data) < Index
        return;
    end
    Value = Data(Index);
end

function Success = ResolveBoundarySeedYield(WorkerAudit,BoundaryAdded)
    Success = false(numel(WorkerAudit),1);
    if isempty(WorkerAudit) || isempty(BoundaryAdded)
        return;
    end
    AddedDec = BoundaryAdded.decs;
    if isempty(AddedDec)
        return;
    end
    for i = 1 : numel(WorkerAudit)
        LineageDec = WorkerAudit(i).lineageFeasibleDec;
        if isempty(LineageDec)
            continue;
        end
        Success(i) = any(ismember(LineageDec,AddedDec,'rows'));
    end
end

function Row = InitBoundaryGainTraceRow()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'boundaryGain',0, ...
        'cumulativeBoundaryGain',0, ...
        'boundaryAddedCount',0, ...
        'externalArchiveCount',0);
end

function Trace = AppendBoundaryGainTrace(Trace,Generation,FE,BoundaryGain,BoundaryAddedCount,ExternalArchiveCount)
    if isempty(Trace)
        Trace = repmat(InitBoundaryGainTraceRow(),0,1);
    end
    Row = InitBoundaryGainTraceRow();
    Row.generation = Generation;
    Row.FE = FE;
    Row.boundaryGain = BoundaryGain;
    if isempty(Trace)
        Row.cumulativeBoundaryGain = BoundaryGain;
    else
        Row.cumulativeBoundaryGain = Trace(end).cumulativeBoundaryGain + BoundaryGain;
    end
    Row.boundaryAddedCount = BoundaryAddedCount;
    Row.externalArchiveCount = ExternalArchiveCount;
    Trace(end+1,1) = Row;
end

function [ExternalArchive,BoundaryGain,BoundaryAdded] = UpdateSectionBExternalArchive(ExternalArchive,OffspringC,OffspringU,BoundaryOffspring)
    if nargin < 1 || isempty(ExternalArchive)
        ExternalArchive = [];
    end
    if nargin < 2 || isempty(OffspringC)
        OffspringC = [];
    end
    if nargin < 3 || isempty(OffspringU)
        OffspringU = [];
    end
    if nargin < 4 || isempty(BoundaryOffspring)
        BoundaryOffspring = [];
    end

    [ExternalArchive,~] = UpdateExternalArchive(ExternalArchive,[OffspringC,OffspringU]);
    BaseArchive = ExternalArchive;
    [ExternalArchive,BoundaryAdded] = UpdateExternalArchive(ExternalArchive,BoundaryOffspring);
    BoundaryGain = EstimateArchiveHVGain(BaseArchive,BoundaryAdded);
end

function Gain = EstimateArchiveHVGain(Archive,Added)
    Gain = 0;
    if isempty(Added)
        return;
    end

    OldObj = zeros(0,size(Added.objs,2));
    if ~isempty(Archive)
        OldObj = Archive.objs;
    end
    NewObj = [OldObj;Added.objs];
    if isempty(NewObj)
        return;
    end

    Ref = max(NewObj,[],1) + 1;
    try
        if isempty(OldObj)
            OldHV = 0;
        else
            OldHV = HV(OldObj,Ref);
        end
        Gain = max(HV(NewObj,Ref) - OldHV,0);
    catch
        Gain = 0;
    end
end

function Metric = AppendSectionBCalibrationTrace(Metric,CalMetric)
    if ~isstruct(Metric) || ~isfield(Metric,'traceFlag') || ~Metric.traceFlag
        return;
    end
    if ~isfield(Metric,'calibrationTrace') || isempty(Metric.calibrationTrace)
        Trace = repmat(InitSectionBCalibrationTraceRow(),0,1);
    else
        Trace = Metric.calibrationTrace;
    end
    FE = FieldOrDefaultMetric(CalMetric,'FE',NaN);
    if ~isempty(Trace) && isequaln(Trace(end).FE,FE)
        Metric.calibrationTrace = Trace;
        return;
    end

    Row = InitSectionBCalibrationTraceRow();
    Row.generation = FieldOrDefaultMetric(CalMetric,'generation',NaN);
    Row.FE = FE;
    Row.count = FieldOrDefaultMetric(CalMetric,'count',0);
    Row.feasible_rate = FieldOrDefaultMetric(CalMetric,'feasibleRate',NaN);
    Row.mean_prob = FieldOrDefaultMetric(CalMetric,'meanProb',NaN);
    Row.brier = FieldOrDefaultMetric(CalMetric,'brier',NaN);
    Row.ece = FieldOrDefaultMetric(CalMetric,'ece',NaN);
    Row.log_loss = FieldOrDefaultMetric(CalMetric,'logLoss',NaN);
    Row.near_count = FieldOrDefaultMetric(CalMetric,'nearCount',0);
    Row.near_mean_prob = FieldOrDefaultMetric(CalMetric,'nearMeanProb',NaN);
    Row.near_feasible_rate = FieldOrDefaultMetric(CalMetric,'nearFeasibleRate',NaN);
    Row.near_gap = FieldOrDefaultMetric(CalMetric,'nearGap',NaN);
    Row.training_count = FieldOrDefaultMetric(CalMetric,'trainingCount',NaN);
    Row.calibration_count = FieldOrDefaultMetric(CalMetric,'calibrationCount',NaN);
    Row.calibration_near_count = FieldOrDefaultMetric(CalMetric,'calibrationNearCount',NaN);
    Row.test_count = FieldOrDefaultMetric(CalMetric,'testCount',NaN);
    Row.test_near_count = FieldOrDefaultMetric(CalMetric,'testNearCount',NaN);
    Trace(end+1,1) = Row;
    Metric.calibrationTrace = Trace;
end

function Row = InitSectionBCalibrationTraceRow()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'count',0, ...
        'feasible_rate',NaN, ...
        'mean_prob',NaN, ...
        'brier',NaN, ...
        'ece',NaN, ...
        'log_loss',NaN, ...
        'near_count',0, ...
        'near_mean_prob',NaN, ...
        'near_feasible_rate',NaN, ...
        'near_gap',NaN, ...
        'training_count',NaN, ...
        'calibration_count',NaN, ...
        'calibration_near_count',NaN, ...
        'test_count',NaN, ...
        'test_near_count',NaN);
end

function Value = FieldOrDefaultMetric(S,Field,Default)
    if isstruct(S) && isfield(S,Field) && ~isempty(S.(Field))
        Value = S.(Field);
    else
        Value = Default;
    end
end
