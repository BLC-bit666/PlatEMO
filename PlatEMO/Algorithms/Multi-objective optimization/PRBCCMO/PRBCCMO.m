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
% ensK     --- 3    --- Committee size of shallow boundary MLPs
% calMode  --- 2    --- Calibration mode (1 raw, 2 temperature, 3 sigmoid)
% dLambda  --- 1    --- Committee disagreement weight in boundary utility
% pairM    --- 0.05 --- Margin for tight bracket pair loss
% lPair    --- 1    --- Weight of bracket pair loss
% lMid     --- 1    --- Weight of midpoint-to-0.5 loss
% lHard    --- 1    --- Weight of hard-negative loss
% selMode  --- 1    --- Section B seed selection mode (1. full 2. uncertain-only 3. random-boundary)
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
            type      = Params.type;
            bRho      = Params.bRho;
            afRho     = Params.afRho;
            aiRho     = Params.aiRho;
            trainRho  = Params.trainRho;
            hidden    = Params.hidden;
            epoch     = Params.epoch;
            lr        = Params.lr;
            xRho      = Params.xRho;
            lsRho     = Params.lsRho;
            mRho      = Params.mRho;
            ensK      = Params.ensK;
            calMode   = Params.calMode;
            dLambda   = Params.dLambda;
            pairM     = Params.pairM;
            lPair     = Params.lPair;
            lMid      = Params.lMid;
            lHard     = Params.lHard;
            selMode   = Params.selMode;
            localMode = Params.localMode;
            traceOn   = Params.traceOn;
            RuntimeOptions = BuildBoundaryRuntimeOptions(selMode,localMode,traceOn);

            BoundaryBudget = max(0,floor(bRho*Problem.N));
            ArchiveFMax    = max(0,round(afRho*Problem.N));
            ArchiveIMax    = max(0,round(aiRho*Problem.N));
            TrainMax       = max(0,round(trainRho*Problem.N));
            CalibMax       = max(20,ceil(0.25*TrainMax));
            TestMax        = max(20,ceil(0.25*TrainMax));
            ProtectedMax   = max(20,ceil(0.25*TrainMax));
            ProtectedBracketMax = max(10,floor(0.5*ProtectedMax));
            ProtectedOtherMax   = max(10,ProtectedMax-ProtectedBracketMax);
            BracketMax     = max(1,floor(0.5*ProtectedBracketMax));
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
            BracketArchive = EmptyBracketArchive(Problem.D);
            HardNegativeArchive.Dec    = zeros(0,Problem.D);
            HardNegativeArchive.Radius = zeros(0,1);
            ProtectedOtherDec   = zeros(0,Problem.D);
            ProtectedOtherLabel = zeros(0,1);
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
                BracketArchive,HardNegativeArchive,ensK,calMode,dLambda,pairM,lPair,lMid,lHard,Problem.D);
            Model = TrainBoundaryMLP( ...
                TrainDec,TrainLabel,hidden,epoch,lr,[],CalibDec,CalibLabel,TrainOptions);
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
                    Problem,PopulationC,PopulationU,FitnessC,FitnessU,type);

                CandidatePool = GenerateBoundaryCandidates( ...
                    Problem,PopulationC,PopulationU,ArchiveF,ArchiveI,type,W,NumBridge,NumLocal,RuntimeOptions);

                BoundaryBudgetNow = min(BoundaryBudget,max(0,Problem.maxFE-Problem.FE));
                SeedBudget = min(BoundaryBudgetNow,max(0,round(SeedRatio*BoundaryBudgetNow)));
                if BoundaryBudgetNow > 0 && SeedBudget == 0
                    SeedBudget = 1;
                end
                [BoundarySeeds,SeedInfo] = SelectBoundaryCandidates( ...
                    Problem,CandidatePool,PopulationC,Model,W,HardNegativeArchive,SeedBudget,RuntimeOptions);

                WorkerBudget = max(0,BoundaryBudgetNow-numel(BoundarySeeds));
                [WorkerOffspring,WorkerInfo,MigrationPool,BracketBatch,HardNegBatch,WorkerAudit] = ...
                    RefineBoundaryWorkers( ...
                        Problem,BoundarySeeds,SeedInfo,PopulationC,PopulationU, ...
                        ArchiveF,ArchiveI,Model,W,HardNegativeArchive,WorkerBudget,RuntimeOptions);
                [BoundaryOffspring,BoundaryInfo] = MergeBoundaryResults( ...
                    BoundarySeeds,SeedInfo,WorkerOffspring,WorkerInfo,Problem.M);

                HardNegativeArchive = UpdateHardNegativeArchive(HardNegativeArchive,HardNegBatch,HardNegMax);
                BracketArchive = UpdateBracketArchive(BracketArchive,BracketBatch,BracketMax,Problem.D);

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
                    HardNegativeArchive,ArchiveFMax,ArchiveIMax,RuntimeOptions);

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

                [TrainBatch,TrainInfo,CalibBatch,CalibInfo,TestBatch,TestInfo] = SplitHeldOutBatch( ...
                    BoundaryOffspring,BoundaryInfo,CalibMax,TestMax,Problem.M);
                HoldoutDec = [CalibDec;TestDec; ...
                    SolutionDecs(CalibBatch,Problem.D);SolutionDecs(TestBatch,Problem.D)];
                [ProtectedBracketDec,ProtectedBracketLabel] = BuildBracketProtectedBuffer(BracketArchive,Problem.D);
                [ProtectedBracketDec,ProtectedBracketLabel] = ExcludeLabeledRows( ...
                    ProtectedBracketDec,ProtectedBracketLabel,HoldoutDec);
                [ProtectedOtherAddDec,ProtectedOtherAddLabel] = CollectOtherProtectedCases( ...
                    TrainBatch,TrainInfo,HardNegBatch,Problem.D);
                [ProtectedOtherAddDec,ProtectedOtherAddLabel] = ExcludeLabeledRows( ...
                    ProtectedOtherAddDec,ProtectedOtherAddLabel,HoldoutDec);
                [ProtectedOtherDec,ProtectedOtherLabel] = UpdateProtectedBuffer( ...
                    ProtectedOtherDec,ProtectedOtherLabel,ProtectedOtherAddDec,ProtectedOtherAddLabel,ProtectedOtherMax);
                [ProtectedOtherDec,ProtectedOtherLabel] = ExcludeLabeledRows( ...
                    ProtectedOtherDec,ProtectedOtherLabel,HoldoutDec);
                ProtectedDec   = [ProtectedBracketDec;ProtectedOtherDec];
                ProtectedLabel = [ProtectedBracketLabel;ProtectedOtherLabel];
                [TrainDec,TrainLabel] = UpdateTrainingArchive( ...
                    TrainDec,TrainLabel,ProtectedDec,ProtectedLabel,TrainBatch,HoldoutDec,TrainMax);
                [CalibDec,CalibLabel,CalibNear] = UpdateCalibrationBuffer( ...
                    CalibDec,CalibLabel,CalibNear,CalibBatch,CalibInfo,CalibMax);
                [TestDec,TestLabel,TestNear] = UpdateCalibrationBuffer( ...
                    TestDec,TestLabel,TestNear,TestBatch,TestInfo,TestMax);

                PendingLabels = PendingLabels + numel(BoundaryOffspring);
                TrainOptions = BuildBoundaryTrainingOptions( ...
                    BracketArchive,HardNegativeArchive,ensK,calMode,dLambda,pairM,lPair,lMid,lHard,Problem.D);
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

function [ProtectedDec,ProtectedLabel] = CollectOtherProtectedCases(BoundaryOffspring,BoundaryInfo,HardNegBatch,D)
    ProtectedDec = zeros(0,D);
    ProtectedLabel = zeros(0,1);
    if ~isempty(BoundaryOffspring)
        Label = double(all(BoundaryOffspring.cons<=0,2));
        NearMask = abs(BoundaryInfo.prob(:)-0.5) <= 0.1;
        MisMask  = (BoundaryInfo.prob(:)>=0.5) ~= logical(Label);
        KeepMask = NearMask & MisMask;
        ProtectedDec = [ProtectedDec;BoundaryOffspring(KeepMask).decs];
        ProtectedLabel = [ProtectedLabel;Label(KeepMask)];
    end
    if ~isempty(HardNegBatch.Dec)
        ProtectedDec = [ProtectedDec;HardNegBatch.Dec];
        ProtectedLabel = [ProtectedLabel;zeros(size(HardNegBatch.Dec,1),1)];
    end
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

function Archive = UpdateBracketArchive(Archive,NewPairs,MaxPairs,D)
    if nargin < 1 || isempty(Archive)
        Archive = EmptyBracketArchive(D);
    end
    if nargin < 3 || MaxPairs <= 0
        Archive = EmptyBracketArchive(D);
        return;
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
    BracketArchive,HardNegativeArchive,EnsembleSize,CalMode,DisagreementWeight, ...
    PairMargin,LambdaPair,LambdaMid,LambdaHard,D)

    Options = struct();
    Options.EnsembleSize       = EnsembleSize;
    Options.Calibrator         = DecodeCalibrationMode(CalMode);
    Options.DisagreementWeight = max(DisagreementWeight,0);
    Options.PairMargin         = max(PairMargin,0);
    Options.LambdaPair         = max(LambdaPair,0);
    Options.LambdaMid          = max(LambdaMid,0);
    Options.LambdaHardNeg      = max(LambdaHard,0);
    Options.PairFeasibleDec    = zeros(0,D);
    Options.PairInfeasibleDec  = zeros(0,D);
    Options.MidDec             = zeros(0,D);
    Options.HardNegDec         = zeros(0,D);

    if ~isempty(BracketArchive) && ~isempty(BracketArchive.FeasibleDec)
        Options.PairFeasibleDec   = BracketArchive.FeasibleDec;
        Options.PairInfeasibleDec = BracketArchive.InfeasibleDec;
        Options.MidDec = 0.5*(BracketArchive.FeasibleDec + BracketArchive.InfeasibleDec);
    end
    if ~isempty(HardNegativeArchive) && isfield(HardNegativeArchive,'Dec') && ~isempty(HardNegativeArchive.Dec)
        Options.HardNegDec = HardNegativeArchive.Dec;
    end
end

function Type = DecodeCalibrationMode(CalMode)
    switch round(CalMode)
        case 1
            Type = 'raw';
        case 2
            Type = 'temperature';
        case 3
            Type = 'sigmoid';
        otherwise
            Type = 'temperature';
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
    PendingLabels = 0;
    LastCalMetric = EvaluateBoundaryCalibration(Model,TestDec,TestLabel);
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

function Params = ResolvePRBCCMOParameters(ParameterCell)
    Defaults = {1,0.2,0.5,0.5,4,20,25,0.01,0.5,0.5,0.5, ...
        3,2,1,0.05,1,1,1,1,1,0};
    Names = {'type','bRho','afRho','aiRho','trainRho','hidden','epoch','lr', ...
        'xRho','lsRho','mRho','ensK','calMode','dLambda','pairM','lPair', ...
        'lMid','lHard','selMode','localMode','traceOn'};
    Values = Defaults;
    if nargin >= 1 && ~isempty(ParameterCell)
        if ~iscell(ParameterCell)
            ParameterCell = {ParameterCell};
        end
        Limit = min(numel(ParameterCell),numel(Defaults));
        for i = 1 : Limit
            if ~isempty(ParameterCell{i})
                Values{i} = ParameterCell{i};
            end
        end
    end

    Params = struct();
    for i = 1 : numel(Names)
        Params.(Names{i}) = Values{i};
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
    Metric.selectionName = SafeRuntimeOption(RuntimeOptions,'SelectionName','full');
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
        'entropy',NaN, ...
        'hvGain',NaN, ...
        'novelty',NaN, ...
        'penalty',NaN, ...
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
        AddRows(i).entropy    = SafeInfoValue(SeedInfo,'entropy',i,NaN);
        AddRows(i).hvGain     = SafeInfoValue(SeedInfo,'hvGain',i,NaN);
        AddRows(i).novelty    = SafeInfoValue(SeedInfo,'novelty',i,NaN);
        AddRows(i).penalty    = SafeInfoValue(SeedInfo,'penalty',i,NaN);
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
    Trace(end+1,1) = Row; %#ok<AGROW>
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
    Trace(end+1,1) = Row; %#ok<AGROW>
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
