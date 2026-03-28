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
% The PRBCCMO-BoundaryCore mainline follows TopKPair bridge generation,
% bridge-conditioned p≈0.5 localization, Pareto-then-boundary selection,
% one-step feasible exploit / one-step infeasible bracket shrink, and calibrated trust.

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
            CalibStatus    = InitDefaultBufferStatus();
            TestDec        = zeros(0,Problem.D);
            TestLabel      = zeros(0,1);
            TestNear       = false(0,1);
            TestStatus     = InitDefaultBufferStatus();
            Model = [];
            LastCalMetric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel,TestNear);
            PendingLabels = 0;
            Generation    = 0;
            BoundaryAuditStarted = false;
            InitSolutions = [PopulationC,PopulationU];
            [ExternalArchive,~] = UpdateExternalArchive([],FilterFeasiblePopulation(InitSolutions));
            AuditState = BuildBoundaryAuditState(CalibStatus,TestStatus,0,BoundaryAuditStarted);
            Algorithm.metric.boundaryCalibration = AttachCalibrationContext( ...
                LastCalMetric,Generation,Problem.FE,size(TrainDec,1),size(CalibDec,1), ...
                sum(CalibNear),size(TestDec,1),sum(TestNear),CalibStatus,TestStatus,AuditState);
            Algorithm.metric.sectionB = InitSectionBMetric(Problem.D,RuntimeOptions,ExternalArchive);
            InitBufferAudit = BuildBoundaryBufferAudit(TrainDec,CalibDec,TestDec);
            Algorithm.metric.sectionB = AppendBoundaryUpdateAudit( ...
                Algorithm.metric.sectionB,Algorithm.metric.boundaryCalibration, ...
                InitBufferAudit,'initial_train',true);
            Algorithm.metric.sectionB = AppendSectionBCalibrationTrace( ...
                Algorithm.metric.sectionB,Algorithm.metric.boundaryCalibration);
            InitFeasibleAnchorPool = BuildFeasibleAnchorPool(PopulationC,[],ExternalArchive);
            Algorithm.metric.sectionB.activationTrace = AppendBoundaryActivationTrace( ...
                Algorithm.metric.sectionB.activationTrace,Generation,Problem.FE, ...
                0,numel(ExternalArchive),numel(InitFeasibleAnchorPool),0,0,0);

            %% Optimization
            while Algorithm.NotTerminated(PopulationC)
                Generation = Generation + 1;

                [OffspringC,OffspringU] = GenerateRegularOffspring( ...
                    Problem,PopulationC,PopulationU);
                RegularFeasible = FilterFeasiblePopulation([OffspringC,OffspringU]);
                FeasibleAnchorPool = BuildFeasibleAnchorPool( ...
                    PopulationC,RegularFeasible,ExternalArchive);
                FeasibleAnchorObj = SolutionObjs(FeasibleAnchorPool,Problem.M);

                [CandidatePool,BridgeDiag] = GenerateBoundaryCandidates( ...
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
                [BoundarySeeds,SeedInfo,SelectionDiag,CandidateAudit] = SelectBoundaryCandidates( ...
                    Problem,CandidatePool,FeasibleAnchorObj,Model,W,HardNegativeArchive,SeedBudget,RuntimeOptions);

                WorkerBudget = max(0,BoundaryBudgetNow-numel(BoundarySeeds));
                [WorkerOffspring,WorkerInfo,WorkerFeasiblePool,BracketBatch,HardNegBatch,WorkerAudit] = ...
                    RefineBoundaryWorkers( ...
                        Problem,BoundarySeeds,SeedInfo,FeasibleAnchorObj,Model,W, ...
                        HardNegativeArchive,WorkerBudget,RuntimeOptions);
                [BoundaryOffspring,BoundaryInfo] = MergeBoundaryResults( ...
                    BoundarySeeds,SeedInfo,WorkerOffspring,WorkerInfo,Problem.M);
                BoundaryBatchCount = max(numel(BoundarySeeds),numel(BoundaryOffspring));
                BoundaryAuditStarted = BoundaryAuditStarted || BoundaryBatchCount > 0;
                MigrationPool = ScreenBoundaryMigrants(PopulationC,WorkerFeasiblePool,W);

                HardNegativeArchive = UpdateHardNegativeArchive(HardNegativeArchive,HardNegBatch,HardNegMax);
                BracketArchive = UpdateBracketArchive(BracketArchive,BracketBatch,BracketMax,Problem.D,TightGap);

                ConstrainedBase = KeepUniquePopulation([PopulationC,OffspringC]);
                PopulationC = EnvironmentalSelectionC(ConstrainedBase,Problem.N,MigrationPool,W);
                PopulationU = EnvironmentalSelectionU(KeepUniquePopulation([PopulationU,OffspringU]),Problem.N);

                [ExternalArchive,BoundaryGain,BoundaryAdded,ArchiveEvents] = UpdateSectionBExternalArchive( ...
                    ExternalArchive,OffspringC,OffspringU,MigrationPool,Generation,Problem.FE);
                Algorithm.metric.sectionB.candidateAudit = AppendBoundaryCandidateAuditRows( ...
                    Algorithm.metric.sectionB.candidateAudit,CandidateAudit,Generation,Problem.FE);
                Algorithm.metric.sectionB.seedAudit = AppendBoundarySeedAuditRows( ...
                    Algorithm.metric.sectionB.seedAudit,BoundarySeeds,SeedInfo,WorkerAudit, ...
                    BoundaryAdded,Generation,Problem.FE,Problem.D);
                Algorithm.metric.sectionB.boundaryLineage = AppendBoundaryLineageRows( ...
                    Algorithm.metric.sectionB.boundaryLineage,BoundarySeeds,SeedInfo,WorkerAudit, ...
                    Generation,Problem.FE,Problem.D);
                Algorithm.metric.sectionB.selectionTrace = AppendBoundarySelectionTrace( ...
                    Algorithm.metric.sectionB.selectionTrace,Generation,Problem.FE,SelectionDiag);
                Algorithm.metric.sectionB.bridgeTrace = AppendBoundaryBridgeTrace( ...
                    Algorithm.metric.sectionB.bridgeTrace,Generation,Problem.FE,BridgeDiag);
                Algorithm.metric.sectionB.boundaryGainTrace = AppendBoundaryGainTrace( ...
                    Algorithm.metric.sectionB.boundaryGainTrace,Generation,Problem.FE, ...
                    BoundaryGain,numel(BoundaryAdded),numel(ExternalArchive));
                Algorithm.metric.sectionB.archiveEvent = AppendBoundaryArchiveEventRows( ...
                    Algorithm.metric.sectionB.archiveEvent,ArchiveEvents);
                Algorithm.metric.sectionB.activationTrace = AppendBoundaryActivationTrace( ...
                    Algorithm.metric.sectionB.activationTrace,Generation,Problem.FE, ...
                    numel(RegularFeasible),numel(ExternalArchive),numel(FeasibleAnchorPool), ...
                    numel(CandidatePool.sector),numel(BoundarySeeds),numel(BoundaryOffspring));
                Algorithm.metric.sectionB.externalArchiveCount = numel(ExternalArchive);
                Algorithm.metric.sectionB.totalBoundaryGain = Algorithm.metric.sectionB.totalBoundaryGain + BoundaryGain;

                [HoldoutSolutions,HoldoutInfo] = PrepareHoldoutFeed( ...
                    BoundaryOffspring,BoundaryInfo,Problem.M);
                [TrainBatch,~,CalibBatch,CalibInfo,TestBatch,TestInfo] = SplitHeldOutBatch( ...
                    HoldoutSolutions,HoldoutInfo,CalibMax,TestMax,Problem.M);
                [ProtectedBracketDec,ProtectedBracketLabel] = BuildBracketProtectedBuffer( ...
                    BracketArchive,HardNegativeArchive,Problem.D);
                [CalibFallbackDec,CalibFallbackLabel,TestFallbackDec,TestFallbackLabel] = ...
                    BuildAuditFallbackPools(ProtectedBracketDec,ProtectedBracketLabel);
                [CalibDec,CalibLabel,CalibNear] = ExcludeBufferRows(CalibDec,CalibLabel,CalibNear,TestDec);
                CalibBatch = ExcludeSolutionsByDec(CalibBatch,TestDec);
                [CalibFallbackDec,CalibFallbackLabel] = ExcludeLabeledRows( ...
                    CalibFallbackDec,CalibFallbackLabel,TestDec);
                [CalibDec,CalibLabel,CalibNear,CalibStatus] = UpdateCalibrationBuffer( ...
                    CalibDec,CalibLabel,CalibNear,CalibBatch,CalibInfo,CalibMax, ...
                    CalibFallbackDec,CalibFallbackLabel);
                [TestDec,TestLabel,TestNear] = ExcludeBufferRows(TestDec,TestLabel,TestNear,CalibDec);
                TestBatch = ExcludeSolutionsByDec(TestBatch,CalibDec);
                [TestFallbackDec,TestFallbackLabel] = ExcludeLabeledRows( ...
                    TestFallbackDec,TestFallbackLabel,CalibDec);
                [TestDec,TestLabel,TestNear,TestStatus] = UpdateCalibrationBuffer( ...
                    TestDec,TestLabel,TestNear,TestBatch,TestInfo,TestMax, ...
                    TestFallbackDec,TestFallbackLabel);
                HoldoutDec = [CalibDec;TestDec];
                [ProtectedBracketDec,ProtectedBracketLabel] = ExcludeLabeledRows( ...
                    ProtectedBracketDec,ProtectedBracketLabel, ...
                    [CalibFallbackDec;TestFallbackDec;HoldoutDec]);
                ProtectedDec   = ProtectedBracketDec;
                ProtectedLabel = ProtectedBracketLabel;
                [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                    TrainDec,TrainLabel,ProtectedDec,ProtectedLabel,TrainBatch,HoldoutDec,TrainMax);
                BufferAudit = BuildBoundaryBufferAudit(TrainDec,CalibDec,TestDec);

                PendingLabels = PendingLabels + numel(HoldoutSolutions);
                TrainOptions = BuildBoundaryTrainingOptions( ...
                    Problem,BracketArchive,ensK,dLambda,pairM,lPair,lMid,TightGap, ...
                    CalibratorCandidates);
                [Model,PendingLabels,LastCalMetric,ModelUpdated,UpdateReason] = UpdateBoundaryModel( ...
                    Model,TrainDec,TrainLabel,CalibDec,CalibLabel,TestDec,TestLabel,TestNear, ...
                    hidden,epoch,WarmEpoch,lr,Generation,PendingLabels,TriggerCount, ...
                    UpdateGap,RestartGap,LastCalMetric,TrainOptions,RuntimeOptions);
                AuditState = BuildBoundaryAuditState( ...
                    CalibStatus,TestStatus,BoundaryBatchCount,BoundaryAuditStarted);
                Algorithm.metric.boundaryCalibration = AttachCalibrationContext( ...
                    EvaluateBoundaryCalibration(Model,TestDec,TestLabel,TestNear), ...
                    Generation,Problem.FE,size(TrainDec,1),size(CalibDec,1), ...
                    sum(CalibNear),size(TestDec,1),sum(TestNear),CalibStatus,TestStatus,AuditState);
                if ModelUpdated
                    Algorithm.metric.sectionB = AppendBoundaryUpdateAudit( ...
                        Algorithm.metric.sectionB,Algorithm.metric.boundaryCalibration, ...
                        BufferAudit,UpdateReason,true);
                end
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
        % The model update keeps selected bridge probes and tight bracket refinements only.
        Keep = logical(HoldoutInfo.trainKeep(:));
    end
    HoldoutSolutions = BoundaryOffspring(Keep);
    HoldoutInfo = SliceBoundaryInfo(HoldoutInfo,find(Keep),M);
end

function Flag = SafeBufferValid(Status)
    Flag = isstruct(Status) && isfield(Status,'valid') && logical(Status.valid);
end

function State = BuildBoundaryAuditState(CalibStatus,TestStatus,BoundaryBatchCount,BoundaryAuditStarted)

    State = struct();
    State.boundaryBatchCount = max(0,BoundaryBatchCount);
    State.boundaryStarted = logical(BoundaryAuditStarted);
    State.auditReady = SafeBufferValid(CalibStatus) && SafeBufferValid(TestStatus) ...
        && State.boundaryStarted;
    if State.auditReady
        State.auditPhase = 'boundary_auditable';
    else
        State.auditPhase = 'not_yet_auditable';
    end
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

function [CalibDec,CalibLabel,TestDec,TestLabel] = BuildAuditFallbackPools(ProtectedDec,ProtectedLabel)
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
    Options.EnsembleSize       = EnsembleSize;
    Options.CalibratorCandidates = {'beta'};
    if nargin >= 9 && ~isempty(CalibratorCandidates)
        Options.CalibratorCandidates = CalibratorCandidates;
    end
    Options.DisagreementWeight = max(DisagreementWeight,0);
    Options.PairMargin         = max(PairMargin,0);
    Options.LambdaPair         = max(LambdaPair,0);
    Options.LambdaMid          = max(LambdaMid,0);
    Options.BracketOversampleFactor = 3;
    D = Problem.D;
    Options.PairFeasibleDec    = zeros(0,D);
    Options.PairInfeasibleDec  = zeros(0,D);
    Options.MidDec             = zeros(0,D);
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

function [Model,PendingLabels,LastCalMetric,ModelUpdated,UpdateReason] = UpdateBoundaryModel( ...
    Model,TrainDec,TrainLabel,CalibDec,CalibLabel,TestDec,TestLabel,TestNear, ...
    Hidden,Epoch,WarmEpoch,LR,Generation,PendingLabels,TriggerCount, ...
    UpdateGap,RestartGap,LastCalMetric,TrainOptions,RuntimeOptions)

    ModelUpdated = false;
    CurrentMetric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel,TestNear);
    ReasonFlags = cell(1,0);
    if isempty(Model)
        ReasonFlags{end+1} = 'missing_model';
    end
    if PendingLabels >= TriggerCount
        ReasonFlags{end+1} = 'pending_labels';
    end
    if mod(Generation,UpdateGap) == 0
        ReasonFlags{end+1} = 'periodic';
    end
    if IsCalibrationDrifting(CurrentMetric,LastCalMetric)
        ReasonFlags{end+1} = 'calibration_drift';
    end
    NeedUpdate = ~isempty(ReasonFlags);
    if NeedUpdate
        UpdateReason = strjoin(ReasonFlags,'|');
    else
        UpdateReason = 'no_update';
    end
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
    ModelUpdated = true;
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
    TrustWeight = RawTrustWeight;
    TrustGate = TrustWeight > 0;
    Model.TrustWeightRaw = RawTrustWeight;
    Model.TrustWeight = TrustWeight;
    Model.TrustAuditPass = AuditPass;
    Model.TrustMinCoreCount = MinCoreCount;
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

function Metric = AttachCalibrationContext( ...
    Metric,Generation,FE,TrainCount,CalibCount,CalNearCount,TestCount,TestNearCount, ...
    CalibStatus,TestStatus,AuditState)

    if nargin < 9 || ~isstruct(CalibStatus)
        CalibStatus = InitDefaultBufferStatus();
    end
    if nargin < 10 || ~isstruct(TestStatus)
        TestStatus = InitDefaultBufferStatus();
    end
    if nargin < 11 || ~isstruct(AuditState)
        AuditState = BuildBoundaryAuditState(CalibStatus,TestStatus,0,false);
    end

    Metric.generation        = Generation;
    Metric.FE                = FE;
    Metric.trainingCount     = TrainCount;
    Metric.calibrationCount  = CalibCount;
    Metric.calibrationNearCount = CalNearCount;
    Metric.testCount         = TestCount;
    Metric.testNearCount     = TestNearCount;
    Metric.calibrationBufferValid = logical(FieldOrDefaultMetric(CalibStatus,'valid',false));
    Metric.calibrationBufferSingleClass = logical(FieldOrDefaultMetric(CalibStatus,'singleClass',false));
    Metric.calibrationBufferClassCount = FieldOrDefaultMetric(CalibStatus,'classCount',0);
    Metric.calibrationBufferStatus = FieldOrDefaultMetric(CalibStatus,'status','invalid_empty');
    Metric.testBufferValid = logical(FieldOrDefaultMetric(TestStatus,'valid',false));
    Metric.testBufferSingleClass = logical(FieldOrDefaultMetric(TestStatus,'singleClass',false));
    Metric.testBufferClassCount = FieldOrDefaultMetric(TestStatus,'classCount',0);
    Metric.testBufferStatus = FieldOrDefaultMetric(TestStatus,'status','invalid_empty');
    Metric.auditReady = logical(FieldOrDefaultMetric(AuditState,'auditReady',false));
    Metric.auditPhase = FieldOrDefaultMetric(AuditState,'auditPhase','not_yet_auditable');
    Metric.coldStartActive = logical(FieldOrDefaultMetric(AuditState,'coldStartActive',false));
    Metric.coldStartBatchCount = FieldOrDefaultMetric(AuditState,'coldStartBatchCount',0);
    Metric.boundaryBatchCount = FieldOrDefaultMetric(AuditState,'boundaryBatchCount',0);
    Metric.boundaryStarted = logical(FieldOrDefaultMetric(AuditState,'boundaryStarted',false));

    if ~Metric.calibrationBufferValid
        Metric.valid = false;
        if Metric.calibrationBufferSingleClass
            Metric.singleClass = true;
            Metric.invalidReason = 'invalid_single_class';
        elseif isempty(FieldOrDefaultMetric(Metric,'invalidReason',''))
            Metric.invalidReason = 'invalid_calibration_buffer';
        end
    end
    if ~Metric.testBufferValid
        Metric.valid = false;
        if Metric.testBufferSingleClass
            Metric.singleClass = true;
            Metric.invalidReason = 'invalid_single_class';
        elseif isempty(FieldOrDefaultMetric(Metric,'invalidReason',''))
            Metric.invalidReason = 'invalid_test_buffer';
        end
    end
    Metric.auditReady = Metric.auditReady && logical(FieldOrDefaultMetric(Metric,'valid',false));
    if ~Metric.auditReady && strcmp(FieldOrDefaultMetric(Metric,'auditPhase',''), 'boundary_auditable')
        Metric.auditPhase = 'boundary_invalid';
    end
end

function Status = InitDefaultBufferStatus()
    Status = struct( ...
        'valid',false, ...
        'singleClass',false, ...
        'classCount',0, ...
        'status','invalid_empty');
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
    Metric.selectionName = 'boundary_core';
    Metric.selectionMode = 4;
    Metric.localMode     = 1;
    Metric.localName     = 'boundary_core';
    Metric.bridgeName    = 'topk_pair';
    Metric.bridgeTopK    = SafeRuntimeOption(RuntimeOptions,'BridgeTopK',5);
    Metric.gateMode      = 1;
    Metric.gateName      = 'two_stage';
    Metric.traceFlag     = logical(SafeRuntimeOption(RuntimeOptions,'TraceFlag',false));
    Metric.traceProbLabel = logical(SafeRuntimeOption(RuntimeOptions,'TraceProbLabel',false));
    Metric.candidateAudit = repmat(InitBoundaryCandidateAuditRow(D),0,1);
    Metric.seedAudit = repmat(InitBoundarySeedAuditRow(D),0,1);
    Metric.boundaryLineage = repmat(InitBoundaryLineageRow(D),0,1);
    Metric.selectionTrace = repmat(InitBoundarySelectionTraceRow(),0,1);
    Metric.bridgeTrace = repmat(InitBoundaryBridgeTraceRow(),0,1);
    Metric.boundaryGainTrace = repmat(InitBoundaryGainTraceRow(),0,1);
    Metric.archiveEvent = repmat(InitBoundaryArchiveEventRow(D,0),0,1);
    Metric.activationTrace = repmat(InitBoundaryActivationTraceRow(),0,1);
    Metric.calibrationTrace = repmat(InitSectionBCalibrationTraceRow(),0,1);
    Metric.updateAudit = repmat(InitBoundaryUpdateAuditRow(),0,1);
    Metric.totalBoundaryGain = 0;
    Metric.externalArchiveCount = numel(ExternalArchive);
end

function Audit = BuildBoundaryBufferAudit(TrainDec,CalibDec,TestDec)
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
            error('PRBCCMO:BoundaryBufferAuditDimension', ...
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

    Audit = struct();
    Audit.train_count = size(TrainDec,1);
    Audit.cal_count = size(CalibDec,1);
    Audit.test_count = size(TestDec,1);
    Audit.train_cal_overlap = CountDecisionOverlap(TrainDec,CalibDec);
    Audit.train_test_overlap = CountDecisionOverlap(TrainDec,TestDec);
    Audit.cal_test_overlap = CountDecisionOverlap(CalibDec,TestDec);
    Audit.strict_separation = Audit.train_cal_overlap == 0 && ...
        Audit.train_test_overlap == 0 && Audit.cal_test_overlap == 0;
    if ~Audit.strict_separation
        error('PRBCCMO:BoundaryBufferLeakage', ...
            'Boundary buffers overlap: train-cal=%d, train-test=%d, cal-test=%d.', ...
            Audit.train_cal_overlap,Audit.train_test_overlap,Audit.cal_test_overlap);
    end
end

function Count = CountDecisionOverlap(A,B)
    Count = 0;
    if isempty(A) || isempty(B)
        return;
    end
    if size(A,2) ~= size(B,2)
        error('PRBCCMO:BoundaryBufferAuditDimension', ...
            'Boundary buffers must share the same decision width.');
    end
    A = unique(A,'rows','stable');
    B = unique(B,'rows','stable');
    Count = sum(ismember(A,B,'rows'));
end

function Metric = AppendBoundaryUpdateAudit(Metric,CalMetric,BufferAudit,Reason,ForceProbLabel)
    if nargin < 5
        ForceProbLabel = false;
    end
    if ~isstruct(Metric)
        Metric = struct();
    end
    if ~isfield(Metric,'updateAudit') || isempty(Metric.updateAudit)
        Metric.updateAudit = repmat(InitBoundaryUpdateAuditRow(),0,1);
    end
    if nargin < 2 || ~isstruct(CalMetric)
        CalMetric = struct();
    end
    if nargin < 3 || ~isstruct(BufferAudit)
        BufferAudit = struct();
    end
    if nargin < 4 || isempty(Reason)
        Reason = '';
    end

    Row = InitBoundaryUpdateAuditRow();
    Row.generation = FieldOrDefaultMetric(CalMetric,'generation',NaN);
    Row.FE = FieldOrDefaultMetric(CalMetric,'FE',NaN);
    Row.reason = Reason;
    Row.calibrator = FieldOrDefaultMetric(CalMetric,'calibrator','raw');
    Row.valid = logical(FieldOrDefaultMetric(CalMetric,'valid',false));
    Row.audit_ready = logical(FieldOrDefaultMetric(CalMetric,'auditReady',false));
    Row.audit_phase = FieldOrDefaultMetric(CalMetric,'auditPhase','not_yet_auditable');
    Row.brier = FieldOrDefaultMetric(CalMetric,'brier',NaN);
    Row.ece = FieldOrDefaultMetric(CalMetric,'ece',NaN);
    Row.near_gap = FieldOrDefaultMetric(CalMetric,'nearGap',NaN);
    Row.core_near_gap = FieldOrDefaultMetric(CalMetric,'coreNearGap',NaN);
    Row.train_count = FieldOrDefaultMetric(BufferAudit,'train_count',0);
    Row.cal_count = FieldOrDefaultMetric(BufferAudit,'cal_count',0);
    Row.test_count = FieldOrDefaultMetric(BufferAudit,'test_count',0);
    Row.strict_separation = logical(FieldOrDefaultMetric(BufferAudit,'strict_separation',false));
    Row.train_cal_overlap = FieldOrDefaultMetric(BufferAudit,'train_cal_overlap',inf);
    Row.train_test_overlap = FieldOrDefaultMetric(BufferAudit,'train_test_overlap',inf);
    Row.cal_test_overlap = FieldOrDefaultMetric(BufferAudit,'cal_test_overlap',inf);
    if ForceProbLabel || isfield(CalMetric,'prob') || isfield(CalMetric,'label')
        Row.prob = FieldOrDefaultMetric(CalMetric,'prob',zeros(0,1));
        Row.label = FieldOrDefaultMetric(CalMetric,'label',zeros(0,1));
    end
    Row.count = FieldOrDefaultMetric(CalMetric,'count',numel(Row.label));
    Metric.updateAudit(end+1,1) = Row;
end

function Row = InitBoundaryUpdateAuditRow()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'reason','', ...
        'calibrator','raw', ...
        'valid',false, ...
        'audit_ready',false, ...
        'audit_phase','not_yet_auditable', ...
        'count',0, ...
        'prob',zeros(0,1), ...
        'label',zeros(0,1), ...
        'brier',NaN, ...
        'ece',NaN, ...
        'near_gap',NaN, ...
        'core_near_gap',NaN, ...
        'train_count',0, ...
        'cal_count',0, ...
        'test_count',0, ...
        'strict_separation',false, ...
        'train_cal_overlap',0, ...
        'train_test_overlap',0, ...
        'cal_test_overlap',0);
end

function Row = InitBoundaryCandidateAuditRow(D)
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'source',NaN, ...
        'selectionMode',4, ...
        'sector',NaN, ...
        'eligible',false, ...
        'selected',false, ...
        'prob',NaN, ...
        'queryScore',NaN, ...
        'disagreement',NaN, ...
        'reliability',NaN, ...
        'paretoValue',NaN, ...
        'boundaryTrust',NaN, ...
        'trustWeight',NaN, ...
        'utility',NaN, ...
        'fullV2Utility',NaN, ...
        'fullV2Shortlisted',false, ...
        'candidateDec',zeros(1,D), ...
        'anchorDec',zeros(1,D), ...
        'helperDec',zeros(1,D));
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
        'feasibleForwardSuccess',false, ...
        'frrSuccess',false, ...
        'ubySuccess',false, ...
        'initialBracketGap',NaN, ...
        'bracketGap',NaN, ...
        'shrinkSuccess',false, ...
        'tightSuccess',false, ...
        'hardNegativeConfirmed',false, ...
        'seedDec',zeros(1,D), ...
        'lineageFeasibleDec',zeros(0,D));
end

function Row = InitBoundarySelectionTraceRow()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'budget',0, ...
        'selectionMode',4, ...
        'hasModel',false, ...
        'trustGate',false, ...
        'trustWeight',0, ...
        'refineQuota',0, ...
        'refineUseCount',0, ...
        'refineGain',0, ...
        'candidateCount',0, ...
        'eligibleCount',0, ...
        'ineligibleCount',0, ...
        'finiteScoreCount',0, ...
        'validCount',0, ...
        'selectedCount',0, ...
        'positiveParetoCount',0, ...
        'maxRankScore',NaN, ...
        'maxParetoValue',NaN, ...
        'maxQueryScore',NaN, ...
        'maxBoundaryTrust',NaN);
end

function Row = InitBoundaryBridgeTraceRow()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'feasibleAnchorCount',0, ...
        'infeasibleHelperCount',0, ...
        'feasibleSectorCount',0, ...
        'infeasibleSectorCount',0, ...
        'sharedSectorCount',0, ...
        'activeSectorCount',0, ...
        'strictActiveSectorCount',0, ...
        'weakActiveSectorCount',0, ...
        'positiveMarginPairCount',0, ...
        'strictPositivePairCount',0, ...
        'gatePassedPairCount',0, ...
        'usedWeakGate',false, ...
        'deltaG',0, ...
        'minRawMargin',NaN, ...
        'medianRawMargin',NaN, ...
        'maxRawMargin',NaN, ...
        'minActivationMargin',NaN, ...
        'medianActivationMargin',NaN, ...
        'maxActivationMargin',NaN);
end

function Trace = AppendBoundarySelectionTrace(Trace,Generation,FE,Diag)
    if isempty(Trace)
        Trace = repmat(InitBoundarySelectionTraceRow(),0,1);
    end
    Row = InitBoundarySelectionTraceRow();
    Row.generation = Generation;
    Row.FE = FE;
    if nargin >= 4 && isstruct(Diag)
        Row.budget = BoundaryDiagValue(Diag,'budget',0);
        Row.selectionMode = BoundaryDiagValue(Diag,'selectionMode',1);
        Row.hasModel = logical(BoundaryDiagValue(Diag,'hasModel',false));
        Row.trustGate = logical(BoundaryDiagValue(Diag,'trustGate',false));
        Row.trustWeight = BoundaryDiagValue(Diag,'trustWeight',0);
        Row.refineQuota = BoundaryDiagValue(Diag,'refineQuota',0);
        Row.refineUseCount = BoundaryDiagValue(Diag,'refineUseCount',0);
        Row.refineGain = BoundaryDiagValue(Diag,'refineGain',0);
        Row.candidateCount = BoundaryDiagValue(Diag,'candidateCount',0);
        Row.eligibleCount = BoundaryDiagValue(Diag,'eligibleCount',0);
        Row.ineligibleCount = BoundaryDiagValue(Diag,'ineligibleCount',0);
        Row.finiteScoreCount = BoundaryDiagValue(Diag,'finiteScoreCount',0);
        Row.validCount = BoundaryDiagValue(Diag,'validCount',0);
        Row.selectedCount = BoundaryDiagValue(Diag,'selectedCount',0);
        Row.positiveParetoCount = BoundaryDiagValue(Diag,'positiveParetoCount',0);
        Row.maxRankScore = BoundaryDiagValue(Diag,'maxRankScore',NaN);
        Row.maxParetoValue = BoundaryDiagValue(Diag,'maxParetoValue',NaN);
        Row.maxQueryScore = BoundaryDiagValue(Diag,'maxQueryScore',NaN);
        Row.maxBoundaryTrust = BoundaryDiagValue(Diag,'maxBoundaryTrust',NaN);
    end
    if ~isempty(Trace) && isequaln(Trace(end).FE,FE)
        Trace(end) = Row;
    else
        Trace(end+1,1) = Row;
    end
end

function Trace = AppendBoundaryBridgeTrace(Trace,Generation,FE,Diag)
    if isempty(Trace)
        Trace = repmat(InitBoundaryBridgeTraceRow(),0,1);
    end
    Row = InitBoundaryBridgeTraceRow();
    Row.generation = Generation;
    Row.FE = FE;
    if nargin >= 4 && isstruct(Diag)
        Row.feasibleAnchorCount = BoundaryDiagValue(Diag,'feasibleAnchorCount',0);
        Row.infeasibleHelperCount = BoundaryDiagValue(Diag,'infeasibleHelperCount',0);
        Row.feasibleSectorCount = BoundaryDiagValue(Diag,'feasibleSectorCount',0);
        Row.infeasibleSectorCount = BoundaryDiagValue(Diag,'infeasibleSectorCount',0);
        Row.sharedSectorCount = BoundaryDiagValue(Diag,'sharedSectorCount',0);
        Row.activeSectorCount = BoundaryDiagValue(Diag,'activeSectorCount',0);
        Row.strictActiveSectorCount = BoundaryDiagValue(Diag,'strictActiveSectorCount',0);
        Row.weakActiveSectorCount = BoundaryDiagValue(Diag,'weakActiveSectorCount',0);
        Row.positiveMarginPairCount = BoundaryDiagValue(Diag,'positiveMarginPairCount',0);
        Row.strictPositivePairCount = BoundaryDiagValue(Diag,'strictPositivePairCount',0);
        Row.gatePassedPairCount = BoundaryDiagValue(Diag,'gatePassedPairCount',0);
        Row.usedWeakGate = logical(BoundaryDiagValue(Diag,'usedWeakGate',false));
        Row.deltaG = BoundaryDiagValue(Diag,'deltaG',0);
        Row.minRawMargin = BoundaryDiagValue(Diag,'minRawMargin',NaN);
        Row.medianRawMargin = BoundaryDiagValue(Diag,'medianRawMargin',NaN);
        Row.maxRawMargin = BoundaryDiagValue(Diag,'maxRawMargin',NaN);
        Row.minActivationMargin = BoundaryDiagValue(Diag,'minActivationMargin',NaN);
        Row.medianActivationMargin = BoundaryDiagValue(Diag,'medianActivationMargin',NaN);
        Row.maxActivationMargin = BoundaryDiagValue(Diag,'maxActivationMargin',NaN);
    end
    if ~isempty(Trace) && isequaln(Trace(end).FE,FE)
        Trace(end) = Row;
    else
        Trace(end+1,1) = Row;
    end
end

function Value = BoundaryDiagValue(Diag,Field,Default)
    if isstruct(Diag) && isfield(Diag,Field) && ~isempty(Diag.(Field))
        Value = Diag.(Field);
    else
        Value = Default;
    end
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
            AddRows(i).feasibleForwardSuccess = WorkerAudit(i).feasibleForwardSuccess;
            AddRows(i).frrSuccess = WorkerAudit(i).frrSuccess;
            AddRows(i).initialBracketGap = WorkerAudit(i).initialBracketGap;
            AddRows(i).bracketGap = WorkerAudit(i).bracketGap;
            AddRows(i).shrinkSuccess = WorkerAudit(i).shrinkSuccess;
            AddRows(i).tightSuccess = WorkerAudit(i).tightSuccess;
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

function Rows = AppendBoundaryCandidateAuditRows(Rows,CandidateAudit,Generation,FE)
    if isempty(CandidateAudit)
        return;
    end
    AddRows = CandidateAudit;
    for i = 1 : numel(AddRows)
        AddRows(i).generation = Generation;
        AddRows(i).FE = FE;
    end
    if isempty(Rows)
        Rows = AddRows;
    else
        Rows = [Rows;AddRows];
    end
end

function Row = InitBoundaryLineageRow(D)
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'seedIndex',0, ...
        'source',NaN, ...
        'workerMode','boundary_core', ...
        'seedFeasible',false, ...
        'seedProb',NaN, ...
        'seedUtility',NaN, ...
        'seedDec',zeros(1,D), ...
        'descendantDec',zeros(0,D), ...
        'descendantLabel',zeros(0,1), ...
        'feasibleDescendantDec',zeros(0,D), ...
        'localEvalCount',0, ...
        'feasibleForwardSuccess',false, ...
        'frrSuccess',false, ...
        'initialBracketGap',NaN, ...
        'bracketGap',NaN, ...
        'shrinkSuccess',false, ...
        'tightSuccess',false, ...
        'hardNegativeConfirmed',false);
end

function Rows = AppendBoundaryLineageRows(Rows,BoundarySeeds,SeedInfo,WorkerAudit,Generation,FE,D)
    if nargin < 7 || isempty(D)
        D = 0;
    end
    if isempty(BoundarySeeds)
        return;
    end
    AddRows = repmat(InitBoundaryLineageRow(D),numel(BoundarySeeds),1);
    for i = 1 : numel(BoundarySeeds)
        AddRows(i).generation = Generation;
        AddRows(i).FE = FE;
        AddRows(i).seedIndex = i;
        AddRows(i).source = SafeInfoValue(SeedInfo,'source',i,NaN);
        AddRows(i).workerMode = 'boundary_core';
        AddRows(i).seedFeasible = all(BoundarySeeds(i).cons<=0,2);
        AddRows(i).seedProb = SafeInfoValue(SeedInfo,'prob',i,NaN);
        AddRows(i).seedUtility = SafeInfoValue(SeedInfo,'utility',i,NaN);
        AddRows(i).seedDec = BoundarySeeds(i).dec;
        if numel(WorkerAudit) >= i
            AddRows(i).descendantDec = FieldOrDefaultMetric(WorkerAudit(i),'lineageDescDec',zeros(0,D));
            AddRows(i).descendantLabel = FieldOrDefaultMetric(WorkerAudit(i),'lineageDescLabel',zeros(0,1));
            AddRows(i).feasibleDescendantDec = FieldOrDefaultMetric(WorkerAudit(i),'lineageFeasibleDec',zeros(0,D));
            AddRows(i).localEvalCount = FieldOrDefaultMetric(WorkerAudit(i),'localEvalCount',0);
            AddRows(i).feasibleForwardSuccess = logical(FieldOrDefaultMetric(WorkerAudit(i),'feasibleForwardSuccess',false));
            AddRows(i).frrSuccess = logical(FieldOrDefaultMetric(WorkerAudit(i),'frrSuccess',false));
            AddRows(i).initialBracketGap = FieldOrDefaultMetric(WorkerAudit(i),'initialBracketGap',NaN);
            AddRows(i).bracketGap = FieldOrDefaultMetric(WorkerAudit(i),'bracketGap',NaN);
            AddRows(i).shrinkSuccess = logical(FieldOrDefaultMetric(WorkerAudit(i),'shrinkSuccess',false));
            AddRows(i).tightSuccess = logical(FieldOrDefaultMetric(WorkerAudit(i),'tightSuccess',false));
            AddRows(i).hardNegativeConfirmed = logical(FieldOrDefaultMetric(WorkerAudit(i),'hardNegativeConfirmed',false));
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

function Row = InitBoundaryActivationTraceRow()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'regularFeasibleCount',0, ...
        'externalArchiveCount',0, ...
        'feasibleAnchorCount',0, ...
        'candidatePoolSize',0, ...
        'boundarySeedCount',0, ...
        'boundaryOffspringCount',0);
end

function Trace = AppendBoundaryActivationTrace( ...
    Trace,Generation,FE,RegularFeasibleCount,ExternalArchiveCount, ...
    FeasibleAnchorCount,CandidatePoolSize,BoundarySeedCount,BoundaryOffspringCount)

    if isempty(Trace)
        Trace = repmat(InitBoundaryActivationTraceRow(),0,1);
    end
    if ~isempty(Trace) && isequaln(Trace(end).FE,FE)
        Trace(end).generation = Generation;
        Trace(end).regularFeasibleCount = RegularFeasibleCount;
        Trace(end).externalArchiveCount = ExternalArchiveCount;
        Trace(end).feasibleAnchorCount = FeasibleAnchorCount;
        Trace(end).candidatePoolSize = CandidatePoolSize;
        Trace(end).boundarySeedCount = BoundarySeedCount;
        Trace(end).boundaryOffspringCount = BoundaryOffspringCount;
        return;
    end

    Row = InitBoundaryActivationTraceRow();
    Row.generation = Generation;
    Row.FE = FE;
    Row.regularFeasibleCount = RegularFeasibleCount;
    Row.externalArchiveCount = ExternalArchiveCount;
    Row.feasibleAnchorCount = FeasibleAnchorCount;
    Row.candidatePoolSize = CandidatePoolSize;
    Row.boundarySeedCount = BoundarySeedCount;
    Row.boundaryOffspringCount = BoundaryOffspringCount;
    Trace(end+1,1) = Row;
end

function Row = InitBoundaryArchiveEventRow(D,M)
    if nargin < 1
        D = 0;
    end
    if nargin < 2
        M = 0;
    end
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'hvGain',0, ...
        'solutionDec',zeros(1,D), ...
        'solutionObj',zeros(1,M));
end

function Rows = AppendBoundaryArchiveEventRows(Rows,AddRows)
    if isempty(AddRows)
        return;
    end
    if isempty(Rows)
        Rows = AddRows;
    else
        Rows = [Rows;AddRows];
    end
end

function [ExternalArchive,BoundaryGain,BoundaryAdded,ArchiveEvents] = UpdateSectionBExternalArchive( ...
    ExternalArchive,OffspringC,OffspringU,BoundaryMigrants,Generation,FE)
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
    if nargin < 5
        Generation = NaN;
    end
    if nargin < 6
        FE = NaN;
    end

    [ExternalArchive,~] = UpdateExternalArchive(ExternalArchive,[OffspringC,OffspringU]);
    BaseArchive = ExternalArchive;
    [ExternalArchive,BoundaryAdded] = UpdateExternalArchive(ExternalArchive,BoundaryMigrants);
    BoundaryGain = EstimateArchiveHVGain(BaseArchive,BoundaryAdded);
    ArchiveEvents = BuildBoundaryArchiveEvents(BaseArchive,BoundaryAdded,Generation,FE);
end

function Rows = BuildBoundaryArchiveEvents(BaseArchive,BoundaryAdded,Generation,FE)
    D = 0;
    M = 0;
    if ~isempty(BoundaryAdded)
        D = size(BoundaryAdded.decs,2);
        M = size(BoundaryAdded.objs,2);
    end
    Rows = repmat(InitBoundaryArchiveEventRow(D,M),0,1);
    if isempty(BoundaryAdded)
        return;
    end

    RunningArchive = BaseArchive;
    Rows = repmat(InitBoundaryArchiveEventRow(D,M),numel(BoundaryAdded),1);
    for i = 1 : numel(BoundaryAdded)
        Rows(i).generation = Generation;
        Rows(i).FE = FE;
        Rows(i).solutionDec = BoundaryAdded(i).dec;
        Rows(i).solutionObj = BoundaryAdded(i).obj;
        Rows(i).hvGain = EstimateArchiveHVGain(RunningArchive,BoundaryAdded(i));
        [RunningArchive,~] = UpdateExternalArchive(RunningArchive,BoundaryAdded(i));
    end
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
    SaveProbLabel = isfield(Metric,'traceProbLabel') && Metric.traceProbLabel;
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
    Row.core_near_count = FieldOrDefaultMetric(CalMetric,'coreNearCount',0);
    Row.core_near_mean_prob = FieldOrDefaultMetric(CalMetric,'coreNearMeanProb',NaN);
    Row.core_near_feasible_rate = FieldOrDefaultMetric(CalMetric,'coreNearFeasibleRate',NaN);
    Row.core_near_gap = FieldOrDefaultMetric(CalMetric,'coreNearGap',NaN);
    Row.relaxed_near_count = FieldOrDefaultMetric(CalMetric,'relaxedNearCount',0);
    Row.relaxed_near_mean_prob = FieldOrDefaultMetric(CalMetric,'relaxedNearMeanProb',NaN);
    Row.relaxed_near_feasible_rate = FieldOrDefaultMetric(CalMetric,'relaxedNearFeasibleRate',NaN);
    Row.relaxed_near_gap = FieldOrDefaultMetric(CalMetric,'relaxedNearGap',NaN);
    Row.valid = logical(FieldOrDefaultMetric(CalMetric,'valid',false));
    Row.invalid_reason = FieldOrDefaultMetric(CalMetric,'invalidReason','');
    Row.single_class = logical(FieldOrDefaultMetric(CalMetric,'singleClass',false));
    Row.class_count = FieldOrDefaultMetric(CalMetric,'classCount',0);
    Row.trust_gate = logical(FieldOrDefaultMetric(CalMetric,'trustGate',false));
    Row.trust_audit_pass = logical(FieldOrDefaultMetric(CalMetric,'trustAuditPass',false));
    Row.trust_weight = FieldOrDefaultMetric(CalMetric,'trustWeight',NaN);
    Row.trust_weight_raw = FieldOrDefaultMetric(CalMetric,'trustWeightRaw',NaN);
    Row.calibrator = FieldOrDefaultMetric(CalMetric,'calibrator','raw');
    Row.training_count = FieldOrDefaultMetric(CalMetric,'trainingCount',NaN);
    Row.calibration_count = FieldOrDefaultMetric(CalMetric,'calibrationCount',NaN);
    Row.calibration_near_count = FieldOrDefaultMetric(CalMetric,'calibrationNearCount',NaN);
    Row.calibration_buffer_valid = logical(FieldOrDefaultMetric(CalMetric,'calibrationBufferValid',false));
    Row.calibration_buffer_single_class = logical(FieldOrDefaultMetric(CalMetric,'calibrationBufferSingleClass',false));
    Row.calibration_buffer_class_count = FieldOrDefaultMetric(CalMetric,'calibrationBufferClassCount',0);
    Row.calibration_buffer_status = FieldOrDefaultMetric(CalMetric,'calibrationBufferStatus','invalid_empty');
    Row.test_count = FieldOrDefaultMetric(CalMetric,'testCount',NaN);
    Row.test_near_count = FieldOrDefaultMetric(CalMetric,'testNearCount',NaN);
    Row.test_buffer_valid = logical(FieldOrDefaultMetric(CalMetric,'testBufferValid',false));
    Row.test_buffer_single_class = logical(FieldOrDefaultMetric(CalMetric,'testBufferSingleClass',false));
    Row.test_buffer_class_count = FieldOrDefaultMetric(CalMetric,'testBufferClassCount',0);
    Row.test_buffer_status = FieldOrDefaultMetric(CalMetric,'testBufferStatus','invalid_empty');
    Row.audit_ready = logical(FieldOrDefaultMetric(CalMetric,'auditReady',false));
    Row.audit_phase = FieldOrDefaultMetric(CalMetric,'auditPhase','not_yet_auditable');
    Row.coldstart_active = logical(FieldOrDefaultMetric(CalMetric,'coldStartActive',false));
    Row.coldstart_batch_count = FieldOrDefaultMetric(CalMetric,'coldStartBatchCount',0);
    Row.boundary_batch_count = FieldOrDefaultMetric(CalMetric,'boundaryBatchCount',0);
    Row.boundary_started = logical(FieldOrDefaultMetric(CalMetric,'boundaryStarted',false));
    Bin = FieldOrDefaultMetric(CalMetric,'bin',struct());
    Row.ece_bin_count = ResolveMetricBinField(Bin,'count');
    Row.ece_bin_prob_sum = ResolveMetricBinField(Bin,'probSum');
    Row.ece_bin_label_sum = ResolveMetricBinField(Bin,'labelSum');
    if SaveProbLabel
        Row.prob = FieldOrDefaultMetric(CalMetric,'prob',zeros(0,1));
        Row.label = FieldOrDefaultMetric(CalMetric,'label',zeros(0,1));
    end
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
        'core_near_count',0, ...
        'core_near_mean_prob',NaN, ...
        'core_near_feasible_rate',NaN, ...
        'core_near_gap',NaN, ...
        'relaxed_near_count',0, ...
        'relaxed_near_mean_prob',NaN, ...
        'relaxed_near_feasible_rate',NaN, ...
        'relaxed_near_gap',NaN, ...
        'valid',false, ...
        'invalid_reason','', ...
        'single_class',false, ...
        'class_count',0, ...
        'trust_gate',false, ...
        'trust_audit_pass',false, ...
        'trust_weight',NaN, ...
        'trust_weight_raw',NaN, ...
        'calibrator','raw', ...
        'training_count',NaN, ...
        'calibration_count',NaN, ...
        'calibration_near_count',NaN, ...
        'calibration_buffer_valid',false, ...
        'calibration_buffer_single_class',false, ...
        'calibration_buffer_class_count',0, ...
        'calibration_buffer_status','invalid_empty', ...
        'test_count',NaN, ...
        'test_near_count',NaN, ...
        'test_buffer_valid',false, ...
        'test_buffer_single_class',false, ...
        'test_buffer_class_count',0, ...
        'test_buffer_status','invalid_empty', ...
        'audit_ready',false, ...
        'audit_phase','not_yet_auditable', ...
        'coldstart_active',false, ...
        'coldstart_batch_count',0, ...
        'boundary_batch_count',0, ...
        'boundary_started',false, ...
        'ece_bin_count',zeros(1,10), ...
        'ece_bin_prob_sum',zeros(1,10), ...
        'ece_bin_label_sum',zeros(1,10), ...
        'prob',zeros(0,1), ...
        'label',zeros(0,1));
end

function Value = ResolveMetricBinField(Bin,Field)
    if isstruct(Bin) && isfield(Bin,Field) && ~isempty(Bin.(Field))
        Value = double(Bin.(Field)(:))';
    else
        Value = zeros(1,10);
    end
end

function Value = FieldOrDefaultMetric(S,Field,Default)
    if isstruct(S) && isfield(S,Field) && ~isempty(S.(Field))
        Value = S.(Field);
    else
        Value = Default;
    end
end
