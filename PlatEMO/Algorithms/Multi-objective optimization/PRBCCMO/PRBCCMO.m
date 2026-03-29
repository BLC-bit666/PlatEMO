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
% dLambda  --- 1    --- Committee disagreement weight in boundary utility
% pairM    --- 0.05 --- Margin for tight bracket pair loss
% lPair    --- 1    --- Weight of bracket pair loss
% lMid     --- 1    --- Weight of midpoint-to-0.5 loss
% calibrator --- 'beta' --- Boundary calibrator candidates ('beta'/'raw')
% PRBCCMO1 removes the legacy bookkeeping and keeps only the
% core boundary-search, trust, and update logic for normal runs.

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
            dLambda   = Params.dLambda;
            pairM     = Params.pairM;
            lPair     = Params.lPair;
            lMid      = Params.lMid;
            CalibratorCandidates = Params.calibratorCandidates;
            RuntimeOptions = BuildBoundaryRuntimeOptions(Params.runtimeOptions);

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
            TightGap       = SafeRuntimeOption(RuntimeOptions,'BracketTightGap',0.03);

            %% Generate random populations
            PopulationC = Problem.Initialization();
            PopulationU = Problem.Initialization();

            %% Initialize boundary memories
            BracketArchive = EmptyBracketArchive(Problem.D);
            HardNegativeArchive.Dec    = zeros(0,Problem.D);
            HardNegativeArchive.Radius = zeros(0,1);
            TrainDec       = zeros(0,Problem.D);
            TrainLabel     = zeros(0,1);
            CalibDec       = zeros(0,Problem.D);
            CalibLabel     = zeros(0,1);
            CalibNear      = false(0,1);
            TestDec        = zeros(0,Problem.D);
            TestLabel      = zeros(0,1);
            TestNear       = false(0,1);
            Model          = [];
            LastCalMetric  = EvaluateBoundaryCalibration(Model,TestDec,TestLabel,TestNear);
            PendingLabels  = 0;
            Generation     = 0;
            InitSolutions  = [PopulationC,PopulationU];
            [ExternalArchive,~] = UpdateExternalArchive([],FilterFeasiblePopulation(InitSolutions));

            %% Optimization
            while Algorithm.NotTerminated(PopulationC)
                Generation = Generation + 1;

                [OffspringC,OffspringU] = GenerateRegularOffspring( ...
                    Problem,PopulationC,PopulationU);
                RegularFeasible = FilterFeasiblePopulation([OffspringC,OffspringU]);
                FeasibleAnchorPool = BuildFeasibleAnchorPool( ...
                    PopulationC,RegularFeasible,ExternalArchive);
                FeasibleAnchorObj = SolutionObjs(FeasibleAnchorPool,Problem.M);

                CandidatePool = GenerateBoundaryCandidates( ...
                    Problem,FeasibleAnchorPool,PopulationU,W,RuntimeOptions);

                BoundaryBudgetNow = min(BoundaryBudget,max(0,Problem.maxFE-Problem.FE));
                SeedBudget = min(BoundaryBudgetNow,max(0,round(SeedRatio*BoundaryBudgetNow)));
                if BoundaryBudgetNow >= 2
                    % Reserve one post-label local evaluation slot per selected seed whenever possible.
                    SeedBudget = min(SeedBudget,BoundaryBudgetNow-SeedBudget);
                end
                if BoundaryBudgetNow > 0 && SeedBudget == 0
                    SeedBudget = 1;
                end
                [BoundarySeeds,SeedInfo] = SelectBoundaryCandidates( ...
                    Problem,CandidatePool,FeasibleAnchorObj,Model,W,HardNegativeArchive, ...
                    SeedBudget,RuntimeOptions);

                WorkerBudget = max(0,BoundaryBudgetNow-numel(BoundarySeeds));
                [WorkerOffspring,WorkerInfo,WorkerFeasiblePool,BracketBatch,HardNegBatch] = ...
                    RefineBoundaryWorkers( ...
                        Problem,BoundarySeeds,SeedInfo,FeasibleAnchorObj,Model,W, ...
                        HardNegativeArchive,WorkerBudget,RuntimeOptions);
                [BoundaryOffspring,BoundaryInfo] = MergeBoundaryResults( ...
                    BoundarySeeds,SeedInfo,WorkerOffspring,WorkerInfo,Problem.M);
                MigrationPool = ScreenBoundaryMigrants(PopulationC,WorkerFeasiblePool,W);

                HardNegativeArchive = UpdateHardNegativeArchive( ...
                    HardNegativeArchive,HardNegBatch,HardNegMax);
                BracketArchive = UpdateBracketArchive( ...
                    BracketArchive,BracketBatch,BracketMax,Problem.D,TightGap);

                ConstrainedBase = KeepUniquePopulation([PopulationC,OffspringC]);
                PopulationC = EnvironmentalSelectionC(ConstrainedBase,Problem.N,MigrationPool,W);
                PopulationU = EnvironmentalSelectionU( ...
                    KeepUniquePopulation([PopulationU,OffspringU]),Problem.N);
                ExternalArchive = UpdateSectionBExternalArchive( ...
                    ExternalArchive,OffspringC,OffspringU,MigrationPool);

                [HoldoutSolutions,HoldoutInfo] = PrepareHoldoutFeed( ...
                    BoundaryOffspring,BoundaryInfo,Problem.M);
                [TrainBatch,~,CalibBatch,CalibInfo,TestBatch,TestInfo] = SplitHeldOutBatch( ...
                    HoldoutSolutions,HoldoutInfo,CalibMax,TestMax,Problem.M);
                [ProtectedBracketDec,ProtectedBracketLabel] = BuildBracketProtectedBuffer( ...
                    BracketArchive,HardNegativeArchive,Problem.D);
                [CalibFallbackDec,CalibFallbackLabel,TestFallbackDec,TestFallbackLabel] = ...
                    BuildFallbackCalibrationPools(ProtectedBracketDec,ProtectedBracketLabel);
                [CalibDec,CalibLabel,CalibNear] = ExcludeBufferRows(CalibDec,CalibLabel,CalibNear,TestDec);
                CalibBatch = ExcludeSolutionsByDec(CalibBatch,TestDec);
                [CalibFallbackDec,CalibFallbackLabel] = ExcludeLabeledRows( ...
                    CalibFallbackDec,CalibFallbackLabel,TestDec);
                [CalibDec,CalibLabel,CalibNear,~] = UpdateCalibrationBuffer( ...
                    CalibDec,CalibLabel,CalibNear,CalibBatch,CalibInfo,CalibMax, ...
                    CalibFallbackDec,CalibFallbackLabel);
                [TestDec,TestLabel,TestNear] = ExcludeBufferRows(TestDec,TestLabel,TestNear,CalibDec);
                TestBatch = ExcludeSolutionsByDec(TestBatch,CalibDec);
                [TestFallbackDec,TestFallbackLabel] = ExcludeLabeledRows( ...
                    TestFallbackDec,TestFallbackLabel,CalibDec);
                [TestDec,TestLabel,TestNear,~] = UpdateCalibrationBuffer( ...
                    TestDec,TestLabel,TestNear,TestBatch,TestInfo,TestMax, ...
                    TestFallbackDec,TestFallbackLabel);
                HoldoutDec = [CalibDec;TestDec];
                [ProtectedBracketDec,ProtectedBracketLabel] = ExcludeLabeledRows( ...
                    ProtectedBracketDec,ProtectedBracketLabel, ...
                    [CalibFallbackDec;TestFallbackDec;HoldoutDec]);
                [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                    TrainDec,TrainLabel,ProtectedBracketDec,ProtectedBracketLabel, ...
                    TrainBatch,HoldoutDec,TrainMax);
                AssertBoundaryBufferSeparation(TrainDec,CalibDec,TestDec);

                PendingLabels = PendingLabels + numel(HoldoutSolutions);
                TrainOptions = BuildBoundaryTrainingOptions( ...
                    Problem,BracketArchive,ensK,dLambda,pairM,lPair,lMid,TightGap, ...
                    CalibratorCandidates);
                [Model,PendingLabels,LastCalMetric] = UpdateBoundaryModel( ...
                    Model,TrainDec,TrainLabel,CalibDec,CalibLabel,TestDec,TestLabel,TestNear, ...
                    hidden,epoch,WarmEpoch,lr,Generation,PendingLabels,TriggerCount, ...
                    UpdateGap,RestartGap,LastCalMetric,TrainOptions,RuntimeOptions);
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

function Population = BuildFeasibleAnchorPool(PopulationC,RegularFeasible,ExternalArchive)
    Population = KeepUniquePopulation([ ...
        FilterFeasiblePopulation(PopulationC), ...
        FilterFeasiblePopulation(RegularFeasible), ...
        FilterFeasiblePopulation(ExternalArchive)]);
end

function Obj = SolutionObjs(Solutions,M)
    if nargin < 2
        M = 0;
    end
    if isempty(Solutions)
        Obj = zeros(0,M);
        return;
    end
    Obj = Solutions.objs;
end

function [HoldoutSolutions,HoldoutInfo] = PrepareHoldoutFeed(BoundaryOffspring,BoundaryInfo,M)
    HoldoutInfo = NormalizeBoundaryInfo(BoundaryInfo,M);
    HoldoutSolutions = BoundaryOffspring;
    if isempty(BoundaryOffspring)
        return;
    end

    Keep = true(numel(BoundaryOffspring),1);
    if isfield(HoldoutInfo,'trainKeep') && numel(HoldoutInfo.trainKeep) == numel(BoundaryOffspring)
        % Keep selected bridge probes and tight bracket refinements for model updates.
        Keep = logical(HoldoutInfo.trainKeep(:));
    end
    HoldoutSolutions = BoundaryOffspring(Keep);
    HoldoutInfo = SliceBoundaryInfo(HoldoutInfo,find(Keep),M);
end

function [TrainSolutions,TrainInfo,CalibSolutions,CalibInfo,TestSolutions,TestInfo] = ...
    SplitHeldOutBatch(Solutions,Info,CalibMax,TestMax,M)
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
    NearMask = ResolveBoundaryNearMask(Info,Count,0.10);

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

function NearMask = ResolveBoundaryNearMask(Info,Count,Delta)
    if nargin < 3 || isempty(Delta)
        Delta = 0.10;
    end
    NearMask = false(Count,1);
    if isstruct(Info) && isfield(Info,'prob') && numel(Info.prob) == Count
        NearMask = NearMask | abs(Info.prob(:)-0.5) <= Delta;
    end
    if isstruct(Info) && isfield(Info,'boundaryLocal') && numel(Info.boundaryLocal) == Count
        NearMask = NearMask | logical(Info.boundaryLocal(:));
    end
    if ~any(NearMask) && Count > 0
        NearMask = true(Count,1);
    end
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
    Info.boundaryLocal = false(Count,1);
    Info.trainKeep     = false(Count,1);
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

function [Dec,Label,Near] = ExcludeBufferRows(Dec,Label,Near,ExcludeDec)
    if isempty(Dec) || isempty(ExcludeDec)
        return;
    end
    Keep = ~ismember(Dec,ExcludeDec,'rows');
    Dec = Dec(Keep,:);
    Label = Label(Keep);
    Near = Near(Keep);
end

function Solutions = ExcludeSolutionsByDec(Solutions,ExcludeDec)
    if isempty(Solutions) || isempty(ExcludeDec)
        return;
    end
    Keep = ~ismember(Solutions.decs,ExcludeDec,'rows');
    Solutions = Solutions(Keep);
end

function Dec = SolutionDecs(Solutions,D)
    if isempty(Solutions)
        Dec = zeros(0,D);
        return;
    end
    if isstruct(Solutions) && isfield(Solutions,'dec')
        Dec = vertcat(Solutions.dec);
    elseif isstruct(Solutions) && isfield(Solutions,'decs')
        Dec = Solutions.decs;
    else
        Dec = Solutions.decs;
    end
end

function [ProtectedDec,ProtectedLabel] = BuildBracketProtectedBuffer(BracketArchive,HardNegativeArchive,D)
    ProtectedDec = zeros(0,D);
    ProtectedLabel = zeros(0,1);
    if nargin < 2
        HardNegativeArchive = [];
    end
    if ~isempty(BracketArchive) && isfield(BracketArchive,'FeasibleDec') ...
            && ~isempty(BracketArchive.FeasibleDec)
        ProtectedDec = [ProtectedDec;BracketArchive.FeasibleDec;BracketArchive.InfeasibleDec];
        ProtectedLabel = [ProtectedLabel; ...
            ones(size(BracketArchive.FeasibleDec,1),1); ...
            zeros(size(BracketArchive.InfeasibleDec,1),1)];
    end
    if nargin >= 2 && ~isempty(HardNegativeArchive) && isfield(HardNegativeArchive,'Dec') ...
            && ~isempty(HardNegativeArchive.Dec)
        ProtectedDec = [ProtectedDec;HardNegativeArchive.Dec];
        ProtectedLabel = [ProtectedLabel;zeros(size(HardNegativeArchive.Dec,1),1)];
    end
end

function [CalibDec,CalibLabel,TestDec,TestLabel] = BuildFallbackCalibrationPools(ProtectedDec,ProtectedLabel)
    if nargin < 2
        ProtectedLabel = zeros(0,1);
    end
    D = 0;
    if ~isempty(ProtectedDec)
        D = size(ProtectedDec,2);
    end
    CalibDec = zeros(0,D);
    CalibLabel = zeros(0,1);
    TestDec = zeros(0,D);
    TestLabel = zeros(0,1);
    if isempty(ProtectedDec)
        return;
    end

    PerClassPerPool = 2;
    for ClassValue = [1,0]
        ClassIdx = find(ProtectedLabel == ClassValue);
        if isempty(ClassIdx)
            continue;
        end
        ClassIdx = flipud(ClassIdx(:));
        CalTake = ClassIdx(1:2:min(numel(ClassIdx),2*PerClassPerPool));
        TestTake = ClassIdx(2:2:min(numel(ClassIdx),2*PerClassPerPool));
        CalTake = sort(CalTake,'ascend');
        TestTake = sort(TestTake,'ascend');
        if ~isempty(CalTake)
            CalibDec = [CalibDec;ProtectedDec(CalTake,:)]; %#ok<AGROW>
            CalibLabel = [CalibLabel;ProtectedLabel(CalTake)]; %#ok<AGROW>
        end
        if ~isempty(TestTake)
            TestDec = [TestDec;ProtectedDec(TestTake,:)]; %#ok<AGROW>
            TestLabel = [TestLabel;ProtectedLabel(TestTake)]; %#ok<AGROW>
        end
    end
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
    if ~isempty(NewPairs) && isfield(NewPairs,'FeasibleDec') && ~isempty(NewPairs.FeasibleDec)
        NewF = NewPairs.FeasibleDec;
        NewI = NewPairs.InfeasibleDec;
        NewG = NewPairs.Gap(:);
    elseif ~isempty(NewPairs) && isfield(NewPairs,'Feasible') && ~isempty(NewPairs.Feasible)
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
    Problem,BracketArchive,EnsembleSize,DisagreementWeight,PairMargin,LambdaPair,LambdaMid, ...
    TightGap,CalibratorCandidates)

    Options = struct();
    Options.EnsembleSize         = EnsembleSize;
    Options.CalibratorCandidates = {'beta'};
    if nargin >= 9 && ~isempty(CalibratorCandidates)
        Options.CalibratorCandidates = CalibratorCandidates;
    end
    Options.DisagreementWeight   = max(DisagreementWeight,0);
    Options.PairMargin           = max(PairMargin,0);
    Options.LambdaPair           = max(LambdaPair,0);
    Options.LambdaMid            = max(LambdaMid,0);
    Options.BracketOversampleFactor = 3;
    D = Problem.D;
    Options.PairFeasibleDec      = zeros(0,D);
    Options.PairInfeasibleDec    = zeros(0,D);
    Options.MidDec               = zeros(0,D);
    if nargin < 8 || isempty(TightGap)
        TightGap = 0.03;
    end

    if ~isempty(BracketArchive) && ~isempty(BracketArchive.FeasibleDec)
        TightMask = true(size(BracketArchive.FeasibleDec,1),1);
        if isfield(BracketArchive,'Gap') && numel(BracketArchive.Gap) == size(BracketArchive.FeasibleDec,1)
            TightMask = isfinite(BracketArchive.Gap(:)) & BracketArchive.Gap(:) <= TightGap;
        end
        if any(TightMask)
            Options.PairFeasibleDec = BracketArchive.FeasibleDec(TightMask,:);
            Options.PairInfeasibleDec = BracketArchive.InfeasibleDec(TightMask,:);
            Options.MidDec = BuildTrainingMidpoints( ...
                Problem,Options.PairFeasibleDec,Options.PairInfeasibleDec);
        end
    end
end

function MidDec = BuildTrainingMidpoints(Problem,FeasibleDec,InfeasibleDec)
    Count = min(size(FeasibleDec,1),size(InfeasibleDec,1));
    MidDec = zeros(Count,Problem.D);
    for i = 1 : Count
        MidDec(i,:) = InterpolateBoundaryPoint( ...
            Problem,FeasibleDec(i,:),InfeasibleDec(i,:),0.5);
    end
end

function [Model,PendingLabels,LastCalMetric] = UpdateBoundaryModel( ...
    Model,TrainDec,TrainLabel,CalibDec,CalibLabel,TestDec,TestLabel,TestNear, ...
    Hidden,Epoch,WarmEpoch,LR,Generation,PendingLabels,TriggerCount, ...
    UpdateGap,RestartGap,LastCalMetric,TrainOptions,RuntimeOptions)

    CurrentMetric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel,TestNear);
    NeedUpdate = isempty(Model) || PendingLabels >= TriggerCount ...
        || mod(Generation,UpdateGap) == 0 ...
        || IsCalibrationDrifting(CurrentMetric,LastCalMetric);
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

    if nargin < 19 || ~isstruct(TrainOptions)
        TrainOptions = struct();
    end
    TrainOptions.WarmStart = UseWarmStart;
    UpdatedModel = TrainBoundaryMLP( ...
        TrainDec,TrainLabel,Hidden,TrainEpoch,LR,PrevModel,CalibDec,CalibLabel, ...
        TrainOptions);
    if isempty(UpdatedModel)
        LastCalMetric = CurrentMetric;
        return;
    end

    UpdatedModel.BoundaryLocalDelta = SafeRuntimeOption(RuntimeOptions,'BoundaryLocalDelta',0.10);
    Model = RefreshBoundaryTrust(UpdatedModel,TestDec,TestLabel,TestNear,RuntimeOptions);
    PendingLabels = 0;
    LastCalMetric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel,TestNear);
end

function Model = RefreshBoundaryTrust(Model,TestDec,TestLabel,TestNear,RuntimeOptions)
    if isempty(Model)
        return;
    end

    Metric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel,TestNear);
    if nargin < 5 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    TauE = SafeRuntimeOption(RuntimeOptions,'TrustTauE',0.10);
    TauN = SafeRuntimeOption(RuntimeOptions,'TrustTauN',0.10);
    MinCoreCount = SafeRuntimeOption(RuntimeOptions,'TrustMinCoreCount',20);
    BoundaryEce = FieldOrDefaultMetric(Metric,'boundaryEce',FieldOrDefaultMetric(Metric,'ece',inf));
    BoundaryGap = FieldOrDefaultMetric( ...
        Metric,'boundaryGap',FieldOrDefaultMetric(Metric,'coreNearGap',FieldOrDefaultMetric(Metric,'nearGap',inf)));
    RawTrustWeight = 0;
    if logical(FieldOrDefaultMetric(Metric,'valid',false)) ...
            && isfinite(BoundaryEce) && isfinite(BoundaryGap)
        RawTrustWeight = max(0,1-BoundaryEce/TauE) * max(0,1-BoundaryGap/TauN);
        RawTrustWeight = min(max(RawTrustWeight,0),1);
    end
    AuditPass = logical(FieldOrDefaultMetric(Metric,'valid',false)) ...
        && isfinite(BoundaryEce) && BoundaryEce <= TauE ...
        && isfinite(BoundaryGap) && BoundaryGap <= TauN ...
        && FieldOrDefaultMetric(Metric,'boundaryCount',FieldOrDefaultMetric(Metric,'coreNearCount',0)) >= MinCoreCount;
    Model.TrustWeightRaw = RawTrustWeight;
    Model.TrustWeight = RawTrustWeight;
    Model.TrustAuditPass = AuditPass;
    Model.TrustMinCoreCount = MinCoreCount;
    Model.TrustGate = RawTrustWeight > 0;
    Model.TrustMetric = PruneBoundaryTrustMetric(Metric);
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

function Compact = PruneBoundaryTrustMetric(Metric)
    Compact = struct();
    Compact.valid = logical(FieldOrDefaultMetric(Metric,'valid',false));
    Compact.ece = FieldOrDefaultMetric(Metric,'ece',inf);
    Compact.nearGap = FieldOrDefaultMetric(Metric,'nearGap',inf);
    Compact.coreNearGap = FieldOrDefaultMetric( ...
        Metric,'coreNearGap',FieldOrDefaultMetric(Metric,'nearGap',inf));
    Compact.boundaryEce = FieldOrDefaultMetric( ...
        Metric,'boundaryEce',FieldOrDefaultMetric(Metric,'ece',inf));
    Compact.boundaryGap = FieldOrDefaultMetric( ...
        Metric,'boundaryGap',FieldOrDefaultMetric(Metric,'coreNearGap',FieldOrDefaultMetric(Metric,'nearGap',inf)));
    Compact.boundaryCount = FieldOrDefaultMetric( ...
        Metric,'boundaryCount',FieldOrDefaultMetric(Metric,'coreNearCount',0));
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

function Flag = IsCalibrationDrifting(CurrentMetric,LastMetric)
    if isempty(LastMetric) || ~isfinite(LastMetric.brier)
        Flag = CurrentMetric.nearCount >= 5 && CurrentMetric.nearGap > 0.15;
        return;
    end

    Flag = false;
    if CurrentMetric.nearCount >= 5 && CurrentMetric.nearGap > 0.15
        Flag = true;
        return;
    end
    if isfinite(CurrentMetric.brier) && CurrentMetric.brier > LastMetric.brier + 0.02
        Flag = true;
        return;
    end
    if isfinite(CurrentMetric.ece) && CurrentMetric.ece > LastMetric.ece + 0.02
        Flag = true;
        return;
    end
    if CurrentMetric.nearCount >= 5 && CurrentMetric.nearGap > LastMetric.nearGap + 0.05
        Flag = true;
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
    Info.boundaryLocal = ResolveBoundaryEligible( ...
        struct('eligible',ResolveBoundaryVector(Info,'boundaryLocal',Count,false)),Count);
    Info.trainKeep     = ResolveBoundaryEligible( ...
        struct('eligible',ResolveBoundaryVector(Info,'trainKeep',Count,false)),Count);
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
        'reliability','boundaryTrust','utility','sector','eligible','boundaryLocal','trainKeep'};
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
    ActiveNames = {'bRho','trainRho','hidden','epoch','lr','mRho','ensK', ...
        'dLambda','pairM','lPair','lMid'};
    ActiveDefaults = {0.2,2,20,25,0.01,0.4,3,1,0.05,1,1};
    LegacyMap = struct( ...
        'bRho',2,'trainRho',5,'hidden',6,'epoch',7,'lr',8,'mRho',11, ...
        'ensK',12,'dLambda',14,'pairM',15,'lPair',16,'lMid',17);
    LegacyThreshold = max(struct2array(LegacyMap));

    Params = cell2struct(ActiveDefaults,ActiveNames,2);
    Params.calibratorCandidates = {'beta'};
    Params.runtimeOptions = struct();

    if nargin < 1 || isempty(ParameterCell)
        return;
    end
    if isstruct(ParameterCell)
        Params = ApplyPRBCCMOParameterStruct(Params,ParameterCell,ActiveNames);
        return;
    end
    if ~iscell(ParameterCell)
        ParameterCell = {ParameterCell};
    end

    StructMask = cellfun(@isstruct,ParameterCell);
    if any(StructMask)
        StructEntries = ParameterCell(StructMask);
        for i = 1 : numel(StructEntries)
            Params = ApplyPRBCCMOParameterStruct(Params,StructEntries{i},ActiveNames);
        end
    end

    if numel(ParameterCell) >= LegacyThreshold
        for i = 1 : numel(ActiveNames)
            Name = ActiveNames{i};
            Index = LegacyMap.(Name);
            if numel(ParameterCell) >= Index && ~StructMask(Index) && ~isempty(ParameterCell{Index})
                Params.(Name) = ParameterCell{Index};
            end
        end
        return;
    end

    Limit = min(numel(ParameterCell),numel(ActiveNames));
    for i = 1 : Limit
        if StructMask(i) || isempty(ParameterCell{i})
            continue;
        end
        Params.(ActiveNames{i}) = ParameterCell{i};
    end
end

function Params = ApplyPRBCCMOParameterStruct(Params,Overrides,ActiveNames)
    if nargin < 3
        ActiveNames = fieldnames(Params);
    end
    Fields = fieldnames(Overrides);
    for i = 1 : numel(Fields)
        Field = Fields{i};
        Value = Overrides.(Field);
        if isempty(Value)
            continue;
        end
        if strcmpi(Field,'runtimeOptions')
            Params.runtimeOptions = BuildBoundaryRuntimeOptions( ...
                Params.runtimeOptions,Value);
        elseif strcmpi(Field,'calibratorCandidates')
            Params.calibratorCandidates = NormalizeCalibratorCandidates(Value);
        elseif strcmpi(Field,'calibrator')
            Params.calibratorCandidates = NormalizeCalibratorCandidates(Value);
        elseif any(strcmp(Field,ActiveNames))
            Params.(Field) = Value;
        end
    end
end

function Candidates = NormalizeCalibratorCandidates(Value)
    if isempty(Value)
        Candidates = {'beta'};
        return;
    end
    if ischar(Value) || (isstring(Value) && isscalar(Value))
        Value = {char(Value)};
    end
    if ~iscell(Value)
        error('PRBCCMO:InvalidCalibratorCandidates', ...
            'Calibrator candidates must be a string or a cell array of strings.');
    end
    Candidates = cell(1,0);
    for i = 1 : numel(Value)
        Name = lower(strtrim(char(Value{i})));
        switch Name
            case {'beta','raw'}
                if ~any(strcmp(Candidates,Name))
                    Candidates{end+1} = Name; %#ok<AGROW>
                end
            otherwise
                error('PRBCCMO:UnsupportedCalibrator', ...
                    'Unsupported calibrator candidate ''%s''.',char(Value{i}));
        end
    end
    if isempty(Candidates)
        Candidates = {'beta'};
    end
end

function AssertBoundaryBufferSeparation(TrainDec,CalibDec,TestDec)
    D = 0;
    Inputs = {TrainDec,CalibDec,TestDec};
    for i = 1 : numel(Inputs)
        Dec = Inputs{i};
        if isempty(Dec)
            continue;
        end
        if D == 0
            D = size(Dec,2);
        elseif size(Dec,2) ~= D
            error('PRBCCMO:BoundaryBufferDimension', ...
                'Boundary buffers must share the same decision width.');
        end
    end
    if isempty(TrainDec)
        TrainDec = zeros(0,D);
    end
    if isempty(CalibDec)
        CalibDec = zeros(0,D);
    end
    if isempty(TestDec)
        TestDec = zeros(0,D);
    end

    TrainCalOverlap = CountDecisionOverlap(TrainDec,CalibDec);
    TrainTestOverlap = CountDecisionOverlap(TrainDec,TestDec);
    CalTestOverlap = CountDecisionOverlap(CalibDec,TestDec);
    if TrainCalOverlap > 0 || TrainTestOverlap > 0 || CalTestOverlap > 0
        error('PRBCCMO:BoundaryBufferLeakage', ...
            'Boundary buffers overlap: train-cal=%d, train-test=%d, cal-test=%d.', ...
            TrainCalOverlap,TrainTestOverlap,CalTestOverlap);
    end
end

function Count = CountDecisionOverlap(A,B)
    Count = 0;
    if isempty(A) || isempty(B)
        return;
    end
    if size(A,2) ~= size(B,2)
        error('PRBCCMO:BoundaryBufferDimension', ...
            'Boundary buffers must share the same decision width.');
    end
    A = unique(A,'rows','stable');
    B = unique(B,'rows','stable');
    Count = sum(ismember(A,B,'rows'));
end

function ExternalArchive = UpdateSectionBExternalArchive(ExternalArchive,OffspringC,OffspringU,BoundaryMigrants)
    if nargin < 1 || isempty(ExternalArchive)
        ExternalArchive = [];
    end
    if nargin < 2 || isempty(OffspringC)
        OffspringC = [];
    end
    if nargin < 3 || isempty(OffspringU)
        OffspringU = [];
    end
    if nargin < 4 || isempty(BoundaryMigrants)
        BoundaryMigrants = [];
    end

    [ExternalArchive,~] = UpdateExternalArchive(ExternalArchive,[OffspringC,OffspringU]);
    [ExternalArchive,~] = UpdateExternalArchive(ExternalArchive,BoundaryMigrants);
end

function Value = SafeRuntimeOption(RuntimeOptions,Field,Default)
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,Field) && ~isempty(RuntimeOptions.(Field))
        Value = RuntimeOptions.(Field);
    else
        Value = Default;
    end
end

function Value = FieldOrDefaultMetric(S,Field,Default)
    if isstruct(S) && isfield(S,Field) && ~isempty(S.(Field))
        Value = S.(Field);
    else
        Value = Default;
    end
end
