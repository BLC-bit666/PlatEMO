classdef PRBCCMO < ALGORITHM
% <2026> <multi> <real> <constrained>
% PRBCCMO1
% Pareto-relevant bridge-driven CCMO for binary feasible/infeasible feedback
% bRho     --- 0.25 --- Boundary evaluation ratio relative to N
% trainRho --- 3    --- Training archive size ratio
% hidden   --- 20   --- Hidden units of the boundary MLP
% epoch    --- 25   --- Training epochs of the boundary MLP
% lr       --- 0.01 --- Learning rate of the boundary MLP
%
% PRBCCMO1 keeps only:
%   1) dual populations with independent DE,
%   2) one binary MLP for feasible/infeasible classification,
%   3) coverage-first bridge scheduling with one bridge per sector,
%   4) topK=5 local bridge pairing inside each sector,
%   5) three coarse probes {0.25,0.5,0.75} for boundary localization,
%   6) bracket-only local refine and midpoint shrink,
%   7) one migration rule: move the best locally evaluated feasible point
%      into P_C only if it replaces the current sector champion.
% The paper-facing scope is real-coded CMOP/BC with binary feasible /
% infeasible feedback. Runtime overrides keep only the shrink ablation.

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
            Params         = ResolvePRBCCMOParameters(Algorithm.parameter);
            RuntimeOptions = BuildBoundaryRuntimeOptions(Params.runtimeOptions);
            [W,~]          = UniformPoint(max(Problem.N,2),Problem.M);

            BoundaryBudget = max(0,floor(double(Params.bRho)*Problem.N));
            TrainMax       = max(8,round(double(Params.trainRho)*Problem.N));
            MaxArchive     = max(2*TrainMax,2*Problem.N);
            UpdateGap      = 5;
            TriggerCount   = max(1,ceil(0.1*TrainMax));

            PopulationC = Problem.Initialization();
            PopulationU = Problem.Initialization();

            ExternalArchive = UpdateExternalArchiveLocal([], ...
                FilterFeasiblePopulation([PopulationC,PopulationU]));
            LabelArchive = InitLabelArchive(Problem.D);
            LabelArchive = AppendLabelArchive(LabelArchive,[PopulationC,PopulationU], ...
                false(numel([PopulationC,PopulationU]),1),MaxArchive);

            Model = RefitBoundaryModel( ...
                [],LabelArchive,Params.hidden,Params.epoch,Params.lr,TrainMax);

            PendingBoundaryLabels = 0;
            Generation           = 0;

            while Algorithm.NotTerminated(PopulationC)
                Generation = Generation + 1;

                [OffspringC,OffspringU] = GenerateRegularOffspring( ...
                    Problem,PopulationC,PopulationU,RuntimeOptions);
                LabelArchive = AppendLabelArchive( ...
                    LabelArchive,[OffspringC,OffspringU], ...
                    false(numel([OffspringC,OffspringU]),1),MaxArchive);

                RemainingFE = max(0,Problem.maxFE-Problem.FE);
                BoundaryBudgetNow = min(BoundaryBudget,RemainingFE);
                FeasibleAnchors = BuildFeasibleAnchorPool( ...
                    PopulationC,OffspringC,OffspringU,ExternalArchive);
                CoverageAnchors = BuildCoverageAnchorPool( ...
                    PopulationC,ExternalArchive);
                HelperPool = BuildHelperPool(PopulationU,OffspringU);

                [MigrationPool,BoundaryBatch] = ExecuteBoundaryCore( ...
                    Problem,FeasibleAnchors,HelperPool,CoverageAnchors,Model,W, ...
                    BoundaryBudgetNow,RuntimeOptions);

                LabelArchive = AppendLabelArchive( ...
                    LabelArchive,BoundaryBatch.population,BoundaryBatch.Boundary,MaxArchive);
                PendingBoundaryLabels = PendingBoundaryLabels + BoundaryBatch.count;

                PopulationC = EnvironmentalSelectionCored( ...
                    [PopulationC,OffspringC],Problem.N,MigrationPool,W);
                PopulationU = EnvironmentalSelectionHelper( ...
                    [PopulationU,OffspringU],Problem.N);

                ExternalArchive = UpdateExternalArchiveLocal( ...
                    ExternalArchive,FilterFeasiblePopulation([OffspringC,OffspringU]));
                ExternalArchive = UpdateExternalArchiveLocal( ...
                    ExternalArchive,MigrationPool);

                NeedUpdate = isempty(Model) || PendingBoundaryLabels >= TriggerCount ...
                    || mod(Generation,UpdateGap) == 0;
                if NeedUpdate
                    Model = RefitBoundaryModel( ...
                        Model,LabelArchive,Params.hidden,Params.epoch,Params.lr,TrainMax);
                    PendingBoundaryLabels = 0;
                end
            end
        end
    end
end

function Params = ResolvePRBCCMOParameters(ParameterCell)
    ActiveNames = {'bRho','trainRho','hidden','epoch','lr'};
    ActiveDefaults = {0.25,3,20,25,0.01};

    Params = cell2struct(ActiveDefaults,ActiveNames,2);
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

    NonStructMask = ~StructMask & ~cellfun(@isempty,ParameterCell);
    NonStructMask = NonStructMask(:);
    UnsupportedIndex = find(NonStructMask & (1:numel(ParameterCell))' > numel(ActiveNames),1);
    if ~isempty(UnsupportedIndex)
        error('PRBCCMO:UnsupportedParameter', ...
            'Unsupported positional parameter at index %d. PRBCCMO1 only accepts the core 5 parameters.', ...
            UnsupportedIndex);
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
    Fields = fieldnames(Overrides);
    for i = 1 : numel(Fields)
        Field = Fields{i};
        Value = Overrides.(Field);
        if isempty(Value)
            continue;
        end
        if strcmpi(Field,'runtimeOptions')
            Params.runtimeOptions = BuildBoundaryRuntimeOptions(Params.runtimeOptions,Value);
        else
            FieldIndex = find(strcmpi(Field,ActiveNames),1);
            if ~isempty(FieldIndex)
                Params.(ActiveNames{FieldIndex}) = Value;
            else
                error('PRBCCMO:UnsupportedParameterField', ...
                    'Unsupported PRBCCMO1 parameter ''%s''.',Field);
            end
        end
    end
end

function Options = BuildBoundaryRuntimeOptions(varargin)
    Defaults = struct( ...
        'DisableInfeasibleShrink',false);
    Options = Defaults;
    if nargin == 0
        Options = NormalizeRuntimeOptions(Options,Defaults);
        return;
    end

    Override = struct();
    if all(cellfun(@isstruct,varargin))
        for i = 1 : nargin
            Fields = fieldnames(varargin{i});
            for j = 1 : numel(Fields)
                Field = Fields{j};
                Value = varargin{i}.(Field);
                if isempty(Value)
                    continue;
                end
                Override.(Field) = Value;
            end
        end
    else
        if mod(nargin,2) ~= 0
            error('PRBCCMO:RuntimeOptionsInput', ...
                'Runtime overrides must be a struct or name-value pairs.');
        end
        for i = 1 : 2 : nargin
            Name = varargin{i};
            if ~(ischar(Name) || (isstring(Name) && isscalar(Name)))
                error('PRBCCMO:RuntimeOptionsInput', ...
                    'Runtime override names must be character vectors or scalars.');
            end
            Override.(char(Name)) = varargin{i+1};
        end
    end

    Fields = fieldnames(Override);
    for i = 1 : numel(Fields)
        if ~isfield(Defaults,Fields{i})
            error('PRBCCMO:UnsupportedRuntimeOption', ...
                'Unsupported PRBCCMO1 runtime option ''%s''.',Fields{i});
        end
        Options.(Fields{i}) = Override.(Fields{i});
    end
    Options = NormalizeRuntimeOptions(Options,Defaults);
end

function Options = NormalizeRuntimeOptions(Options,~)
    Options.DisableInfeasibleShrink = logical(Options.DisableInfeasibleShrink);
end

function Lambdas = NormalizeProbeLambdas(Lambdas,DefaultLambdas)
    if nargin < 2 || isempty(DefaultLambdas)
        DefaultLambdas = [0.25,0.50,0.75];
    end
    if isempty(Lambdas)
        Lambdas = DefaultLambdas;
    end
    Lambdas = double(Lambdas(:)');
    Lambdas = Lambdas(isfinite(Lambdas));
    Lambdas = Lambdas(Lambdas > 0 & Lambdas < 1);
    if isempty(Lambdas)
        Lambdas = DefaultLambdas;
    end
    Lambdas = unique(Lambdas,'stable');
end

function [OffspringC,OffspringU] = GenerateRegularOffspring( ...
    Problem,PopulationC,PopulationU,~)
    OffspringC = Problem.Evaluation(OperatorDECurrentRand1(Problem,PopulationC.decs,PopulationC.decs));
    OffspringU = Problem.Evaluation(OperatorDECurrentRand1(Problem,PopulationU.decs,PopulationU.decs));
end

function Offspring = OperatorDECurrentRand1(Problem,BaseDec,PoolDec)
    [N,D] = size(BaseDec);
    if nargin < 3 || isempty(PoolDec)
        PoolDec = BaseDec;
    end
    PoolSize = size(PoolDec,1);
    Fm       = [0.6,0.8,1.0];
    CRm      = [0.1,0.2,1.0];
    F        = Fm(randi(numel(Fm),N,1));
    CR       = CRm(randi(numel(CRm),N,1));
    F        = F(:);
    CR       = CR(:);
    F        = F(:,ones(1,D));

    P1 = PoolDec(randi(PoolSize,N,1),:);
    P2 = PoolDec(randi(PoolSize,N,1),:);
    P3 = PoolDec(randi(PoolSize,N,1),:);
    Site = rand(N,D) < CR(:,ones(1,D));
    Offspring = BaseDec;
    Offspring(Site) = BaseDec(Site) + F(Site).*(P1(Site)-BaseDec(Site)) ...
        + F(Site).*(P2(Site)-P3(Site));

    proM  = 1;
    disM  = 20;
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp) + (Upper(temp)-Lower(temp)).*((2.*mu(temp) + ...
        (1-2.*mu(temp)).*(1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)) - 1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp) + (Upper(temp)-Lower(temp)).*(1 - (2.*(1-mu(temp)) + ...
        2.*(mu(temp)-0.5).*(1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.CalDec(Offspring);
end

function Pool = BuildFeasibleAnchorPool(PopulationC,OffspringC,OffspringU,ExternalArchive)
    Pool = KeepUniquePopulation([ ...
        FilterFeasiblePopulation(PopulationC), ...
        FilterFeasiblePopulation(OffspringC), ...
        FilterFeasiblePopulation(OffspringU), ...
        FilterFeasiblePopulation(ExternalArchive)]);
end

function Pool = BuildCoverageAnchorPool(PopulationC,ExternalArchive)
    Pool = KeepUniquePopulation([ ...
        FilterFeasiblePopulation(PopulationC), ...
        FilterFeasiblePopulation(ExternalArchive)]);
end

function Pool = BuildHelperPool(PopulationU,OffspringU)
    Pool = KeepUniquePopulation([PopulationU,OffspringU]);
    if isempty(Pool)
        return;
    end
    Pool = Pool(any(Pool.cons>0,2));
end

function [MigrationPool,BoundaryBatch] = ExecuteBoundaryCore( ...
    Problem,FeasibleAnchors,HelperPool,CoveragePool,Model,W,Budget,RuntimeOptions)

    BoundaryBatch  = InitBoundaryBatch();
    MigrationPool  = [];

    [BridgePool,Order] = BuildSingleBridgePool( ...
        FeasibleAnchors,HelperPool,CoveragePool,W);
    if isempty(BridgePool.sector) || Budget <= 0
        return;
    end

    RemainingBudget = Budget;
    Lambdas         = ResolveProbeLambda(RuntimeOptions);
    RefineStep      = ResolveRefineStep();
    DisableShrink   = SafeRuntimeOption(RuntimeOptions,'DisableInfeasibleShrink',false);
    BestMigrants    = InitBestMigrants();
    CoarseSectorCount = min(numel(Order),floor(RemainingBudget/numel(Lambdas)));
    if CoarseSectorCount <= 0
        return;
    end

    CoarseOrder = Order(1:CoarseSectorCount);
    SectorStates = cell(CoarseSectorCount,1);

    for i = 1 : CoarseSectorCount
        Index = CoarseOrder(i);
        AnchorDec = BridgePool.anchorDec(Index,:);
        HelperDec = BridgePool.helperDec(Index,:);
        [ProbeDec,ProbeLambda] = BuildProbeDecisions(Problem,AnchorDec,HelperDec,Lambdas);
        ProbeSolutions = Problem.Evaluation(ProbeDec);
        RemainingBudget = RemainingBudget - numel(Lambdas);

        [~,ProbeScore] = EvaluateProbeScores(Model,ProbeDec,ProbeLambda);
        [CoarseBoundaryMask,LocalBracket] = ResolveCoarseBoundaryBracket( ...
            ProbeSolutions,ProbeLambda,ProbeScore);
        BoundaryBatch = AppendBoundaryBatch( ...
            BoundaryBatch,ProbeSolutions,CoarseBoundaryMask);
        [BestLocal,~] = ResolveRefineSeedIndex( ...
            ProbeScore,ProbeLambda,LocalBracket);
        SectorStates{i} = struct( ...
            'sector',BridgePool.sector(Index), ...
            'anchorDec',AnchorDec, ...
            'helperDec',HelperDec, ...
            'anchorScalar',BridgePool.anchorScalar(Index), ...
            'population',ProbeSolutions, ...
            'lambda',ProbeLambda(:), ...
            'bracket',LocalBracket, ...
            'seedLambda',ProbeLambda(BestLocal));
    end

    for i = 1 : CoarseSectorCount
        State = SectorStates{i};
        if State.bracket.active && RemainingBudget > 0
            RefineLambda = BuildLocalRefineLambdas( ...
                State.seedLambda,State.lambda,State.bracket,RefineStep);
            RefineLambda = RefineLambda(1:min(numel(RefineLambda),RemainingBudget));
            if ~isempty(RefineLambda)
                [RefineDec,RefineLambda] = BuildProbeDecisions( ...
                    Problem,State.anchorDec,State.helperDec,RefineLambda);
                RefineSolutions = Problem.Evaluation(RefineDec);
                RemainingBudget = RemainingBudget - numel(RefineLambda);
                BoundaryBatch = AppendBoundaryBatch( ...
                    BoundaryBatch,RefineSolutions,true(numel(RefineSolutions),1));
                State = AppendBoundarySectorEvaluations( ...
                    State,RefineSolutions,RefineLambda);
            end
        end
        SectorStates{i} = State;
    end

    if ~DisableShrink
        for i = 1 : CoarseSectorCount
            State = SectorStates{i};
            if ~State.bracket.active || RemainingBudget <= 0
                continue;
            end
            ShrinkLambda = ResolveLocalShrinkLambda( ...
                State.population,State.lambda,State.bracket);
            ShrinkLambda = RemoveKnownLambdas(ShrinkLambda,State.lambda);
            if ~isempty(ShrinkLambda)
                [ShrinkDec,ShrinkLambda] = BuildProbeDecisions( ...
                    Problem,State.anchorDec,State.helperDec,ShrinkLambda(1));
                ShrinkSol = Problem.Evaluation(ShrinkDec);
                RemainingBudget = RemainingBudget - 1;
                BoundaryBatch = AppendBoundaryBatch(BoundaryBatch,ShrinkSol,true);
                State = AppendBoundarySectorEvaluations(State,ShrinkSol,ShrinkLambda);
            end
            SectorStates{i} = State;
        end
    end

    for i = 1 : CoarseSectorCount
        State = SectorStates{i};
        [BestLocalSolution,BestLocalScalar,HasLocalFeasible] = ...
            SelectLocalBestFeasibleSolution(State.population,State.sector,W,BridgePool.refObj);
        if HasLocalFeasible && BestLocalScalar < State.anchorScalar
            BestMigrants = UpdateBestMigrant( ...
                BestMigrants,State.sector,BestLocalSolution,BestLocalScalar,State.anchorScalar);
        end
    end

    MigrationPool = BestMigrants.population;
end

function [BridgePool,Order] = BuildSingleBridgePool(FeasibleAnchors,HelperPool,CoveragePool,W)
    BridgePool = InitBridgePool();
    Order = zeros(0,1);
    if isempty(FeasibleAnchors) || isempty(HelperPool)
        return;
    end

    RefObj  = [FeasibleAnchors.objs;HelperPool.objs];
    if nargin >= 3 && ~isempty(CoveragePool)
        RefObj = [RefObj;CoveragePool.objs];
    end
    SectorF = AssociateSectorsLocal(FeasibleAnchors.objs,W,RefObj);
    SectorU = AssociateSectorsLocal(HelperPool.objs,W,RefObj);
    ScalarF = ComputeSectorScalar(FeasibleAnchors.objs,W,RefObj,SectorF);
    ScalarU = ComputeSectorScalar(HelperPool.objs,W,RefObj,SectorU);
    SharedSector = intersect(unique(SectorF(:),'stable'),unique(SectorU(:),'stable'),'stable');
    if isempty(SharedSector)
        return;
    end

    Count = numel(SharedSector);
    BridgePool.sector      = SharedSector(:);
    BridgePool.anchorDec   = zeros(Count,size(FeasibleAnchors.decs,2));
    BridgePool.helperDec   = zeros(Count,size(HelperPool.decs,2));
    BridgePool.anchorScalar = zeros(Count,1);
    BridgePool.coverageCount = zeros(Count,1);
    BridgePool.refObj       = RefObj;
    BridgeTopK              = ResolveBridgeTopK();

    if nargin >= 3 && ~isempty(CoveragePool)
        CoverageSector = AssociateSectorsLocal(CoveragePool.objs,W,RefObj);
        CoverageCount = accumarray(CoverageSector,1,[size(W,1),1]);
    else
        CoverageCount = zeros(size(W,1),1);
    end

    for i = 1 : Count
        SectorID = SharedSector(i);
        FIdx = find(SectorF == SectorID);
        UIdx = find(SectorU == SectorID);
        [AnchorIdx,HelperIdx,BestAnchorScalar] = SelectTopKBridgePair( ...
            FIdx,UIdx,ScalarF,ScalarU,BridgeTopK);
        BridgePool.anchorDec(i,:) = FeasibleAnchors(AnchorIdx).dec;
        BridgePool.helperDec(i,:) = HelperPool(HelperIdx).dec;
        BridgePool.anchorScalar(i) = BestAnchorScalar;
        BridgePool.coverageCount(i) = CoverageCount(SectorID);
    end

    [~,Order] = sortrows([ ...
        BridgePool.coverageCount, ...
        BridgePool.anchorScalar, ...
        BridgePool.sector],[1 2 3]);
end

function Pool = InitBridgePool()
    Pool = struct( ...
        'sector',zeros(0,1), ...
        'anchorDec',zeros(0,0), ...
        'helperDec',zeros(0,0), ...
        'anchorScalar',zeros(0,1), ...
        'coverageCount',zeros(0,1), ...
        'refObj',zeros(0,0));
end

function [ProbeDec,ProbeLambda] = BuildProbeDecisions(Problem,AnchorDec,HelperDec,Lambdas)
    Count = numel(Lambdas);
    ProbeDec = zeros(Count,Problem.D);
    for i = 1 : Count
        ProbeDec(i,:) = InterpolateBoundaryPointLocal(Problem,AnchorDec,HelperDec,Lambdas(i));
    end
    ProbeLambda = Lambdas(:);
end

function Lambdas = ResolveProbeLambda(~)
    DefaultLambdas = [0.25,0.50,0.75];
    Lambdas = NormalizeProbeLambdas(DefaultLambdas,DefaultLambdas);
end

function Step = ResolveRefineStep()
    Step = 0.125;
end

function K = ResolveBridgeTopK()
    K = 5;
end

function [AnchorIdx,HelperIdx,AnchorScalar] = SelectTopKBridgePair( ...
    FIdx,UIdx,ScalarF,ScalarU,TopK)
    [~,FOrder] = sort(ScalarF(FIdx),'ascend');
    [~,UOrder] = sort(ScalarU(UIdx),'ascend');
    FIdx = FIdx(FOrder(1:min(numel(FOrder),TopK)));
    UIdx = UIdx(UOrder(1:min(numel(UOrder),TopK)));
    FScalar = ScalarF(FIdx(:));
    UScalar = ScalarU(UIdx(:));
    FMat = repmat(FScalar(:),1,numel(UScalar));
    UMat = repmat(UScalar(:)',numel(FScalar),1);
    Gap  = abs(FMat-UMat);
    Preferred = UMat <= FMat;
    if any(Preferred(:))
        Candidate = find(Preferred);
    else
        Candidate = (1:numel(Gap))';
    end
    Key = [Gap(Candidate),FMat(Candidate),UMat(Candidate)];
    [LocalIdx,~] = minrows(Key);
    [AnchorLocal,HelperLocal] = ind2sub(size(Gap),Candidate(LocalIdx));
    AnchorIdx = FIdx(AnchorLocal);
    HelperIdx = UIdx(HelperLocal);
    AnchorScalar = ScalarF(AnchorIdx);
end

function [Prob,Score] = EvaluateProbeScores(Model,ProbeDec,ProbeLambda)
    if isempty(ProbeDec)
        Prob  = zeros(0,1);
        Score = zeros(0,1);
        return;
    end
    if isempty(Model)
        Prob  = 0.5*ones(size(ProbeDec,1),1);
        Score = abs(ProbeLambda(:) - 0.5);
        return;
    end
    [Prob,Stats] = PredictBoundaryMLP(Model,ProbeDec);
    if isfield(Stats,'logit') && ~isempty(Stats.logit)
        Score = abs(Stats.logit(:));
    else
        Score = abs(Prob(:)-0.5);
    end
end

function [BoundaryMask,Bracket] = ResolveCoarseBoundaryBracket( ...
    ProbeSolutions,ProbeLambda,ProbeScore)
    Count = numel(ProbeLambda);
    BoundaryMask = false(Count,1);
    Bracket = InitLocalBracket();
    if Count < 2 || isempty(ProbeSolutions)
        return;
    end

    FeasibleMask = all(ProbeSolutions.cons<=0,2);
    FlipIdx = find(FeasibleMask(1:end-1) ~= FeasibleMask(2:end));
    if isempty(FlipIdx)
        return;
    end

    for i = 1 : numel(FlipIdx)
        BoundaryMask([FlipIdx(i);FlipIdx(i)+1]) = true;
    end

    [BestLocal,~] = SelectBestProbe(ProbeScore,ProbeLambda);
    PairPos = SelectPreferredBracketPair(FlipIdx,ProbeLambda,BestLocal);
    LowerIdx = FlipIdx(PairPos);
    Bracket.active = true;
    Bracket.lower = ProbeLambda(LowerIdx);
    Bracket.upper = ProbeLambda(LowerIdx+1);
    Bracket.indices = [LowerIdx;LowerIdx+1];
end

function [BestLocal,BestScore] = ResolveRefineSeedIndex( ...
    ProbeScore,ProbeLambda,Bracket)
    CandidateIdx = [];
    if nargin >= 3 && isstruct(Bracket) && isfield(Bracket,'active') ...
            && Bracket.active && isfield(Bracket,'indices') ...
            && ~isempty(Bracket.indices)
        CandidateIdx = unique(Bracket.indices(:),'stable');
    end
    [BestLocal,BestScore] = SelectBestProbe( ...
        ProbeScore,ProbeLambda,CandidateIdx);
end

function PairPos = SelectPreferredBracketPair(FlipIdx,ProbeLambda,BestLocal)
    Candidate = find(FlipIdx == BestLocal | FlipIdx+1 == BestLocal);
    if isempty(Candidate)
        Candidate = (1:numel(FlipIdx))';
    else
        Candidate = Candidate(:);
    end
    MidLambda = 0.5*(ProbeLambda(FlipIdx(Candidate)) + ProbeLambda(FlipIdx(Candidate)+1));
    Key = [ ...
        abs(MidLambda - ProbeLambda(BestLocal)), ...
        abs(MidLambda - 0.5), ...
        FlipIdx(Candidate)];
    [LocalIdx,~] = minrows(Key);
    PairPos = Candidate(LocalIdx);
end

function Bracket = InitLocalBracket()
    Bracket = struct( ...
        'active',false, ...
        'lower',0, ...
        'upper',1, ...
        'indices',zeros(0,1));
end

function RefineLambda = BuildLocalRefineLambdas( ...
    SeedLambda,ExistingLambda,Bracket,Step)
    Lower = 0;
    Upper = 1;
    if nargin >= 3 && isstruct(Bracket) && isfield(Bracket,'active') && Bracket.active
        Lower = Bracket.lower;
        Upper = Bracket.upper;
    end
    RefineLambda = [ ...
        min(max(SeedLambda-Step,Lower),Upper), ...
        min(max(SeedLambda+Step,Lower),Upper)];
    RefineLambda = unique(RefineLambda,'stable');
    RefineLambda = RemoveKnownLambdas(RefineLambda,ExistingLambda);
end

function Lambda = RemoveKnownLambdas(Lambda,KnownLambda)
    if isempty(Lambda) || isempty(KnownLambda)
        return;
    end
    Tol = 1e-12;
    Keep = true(size(Lambda));
    for i = 1 : numel(Lambda)
        Keep(i) = ~any(abs(KnownLambda(:)-Lambda(i)) <= Tol);
    end
    Lambda = Lambda(Keep);
end

function State = AppendBoundarySectorEvaluations(State,Population,Lambda)
    if isempty(Population)
        return;
    end
    if isempty(State.population)
        State.population = Population;
    else
        State.population = [State.population,Population];
    end
    State.lambda = [State.lambda;Lambda(:)];
end

function Lambda = ResolveLocalShrinkLambda(Population,KnownLambda,Bracket)
    Lambda = zeros(0,1);
    if isempty(Population) || isempty(KnownLambda) || nargin < 3 ...
            || ~isstruct(Bracket) || ~isfield(Bracket,'active') || ~Bracket.active
        return;
    end
    Tol = 1e-12;
    InBracket = KnownLambda >= Bracket.lower-Tol & KnownLambda <= Bracket.upper+Tol;
    if sum(InBracket) < 2
        return;
    end
    LocalPop = Population(InBracket);
    LocalLambda = KnownLambda(InBracket);
    FeasibleMask = all(LocalPop.cons<=0,2);
    FIdx = find(FeasibleMask);
    IIdx = find(~FeasibleMask);
    if isempty(FIdx) || isempty(IIdx)
        return;
    end
    FLambda = LocalLambda(FIdx);
    ILambda = LocalLambda(IIdx);
    FMat = FLambda(:,ones(1,numel(ILambda)));
    IMat = ILambda(ones(numel(FLambda),1),:);
    Gap = abs(FMat-IMat);
    Mid = 0.5*(FMat+IMat);
    Key = [Gap(:),abs(Mid(:)-0.5),Mid(:)];
    [LocalIdx,~] = minrows(Key);
    Lambda = Mid(LocalIdx);
end

function [BestSolution,BestScalar,Found] = SelectLocalBestFeasibleSolution( ...
    Population,SectorID,W,RefObj)
    BestSolution = [];
    BestScalar   = inf;
    Found        = false;
    FeasiblePop = FilterFeasiblePopulation(Population);
    if isempty(FeasiblePop)
        return;
    end
    Sector = repmat(SectorID,numel(FeasiblePop),1);
    Scalar = ComputeSectorScalar(FeasiblePop.objs,W,RefObj,Sector);
    [BestScalar,BestIdx] = min(Scalar);
    BestSolution = FeasiblePop(BestIdx);
    Found = true;
end

function [BestIndex,BestScore] = SelectBestProbe(Score,Lambda,CandidateIdx)
    if nargin < 3 || isempty(CandidateIdx)
        CandidateIdx = (1:numel(Score))';
    else
        CandidateIdx = CandidateIdx(:);
    end
    if isempty(Score) || isempty(CandidateIdx)
        BestIndex = 1;
        BestScore = NaN;
        return;
    end
    OrderKey = [ ...
        Score(CandidateIdx), ...
        abs(Lambda(CandidateIdx)-0.5), ...
        Lambda(CandidateIdx)];
    [LocalIdx,~] = minrows(OrderKey);
    BestIndex = CandidateIdx(LocalIdx);
    BestScore = Score(BestIndex);
end

function [Index,Value] = minrows(Key)
    [~,Order] = sortrows(Key,[1 2 3]);
    Index = Order(1);
    Value = Key(Index,1);
end

function Population = EnvironmentalSelectionCored(Population,N,MigrationPool,W)
    Population    = KeepUniquePopulation(Population);
    MigrationPool = KeepUniquePopulation(MigrationPool);
    Reserved      = SelectReservedMigrants(Population,MigrationPool,W);
    if numel(Reserved) > N
        Reserved = Reserved(1:N);
    end

    Pool = KeepUniquePopulation([Population,MigrationPool]);
    Pool = RemovePopulationByDecision(Pool,Reserved);
    Next = Reserved;
    Need = max(0,N-numel(Next));
    if Need > 0
        FeasiblePool = FilterFeasiblePopulation(Pool);
        Next = [Next,SelectByNSGA2(FeasiblePool,min(Need,numel(FeasiblePool)),true)];
    end
    Need = max(0,N-numel(Next));
    if Need > 0
        Remain = RemovePopulationByDecision(Pool,Next);
        Next = [Next,SelectByNSGA2(Remain,min(Need,numel(Remain)),false)];
    end
    if isempty(Next)
        Population = Population([]);
        return;
    end
    if numel(Next) < N
        Next = [Next,Next(mod(0:N-numel(Next)-1,numel(Next))+1)];
    end
    Population = Next(1:N);
end

function Population = EnvironmentalSelectionHelper(Population,N)
    Population = KeepUniquePopulation(Population);
    Population = SelectByNSGA2(Population,min(N,numel(Population)),false);
    if isempty(Population)
        return;
    end
    if numel(Population) < N
        Population = [Population,Population(mod(0:N-numel(Population)-1,numel(Population))+1)];
    end
    Population = Population(1:N);
end

function Reserved = SelectReservedMigrants(BasePool,MigrationPool,W)
    Reserved = [];
    if isempty(MigrationPool)
        return;
    end
    FeasibleBase = FilterFeasiblePopulation(BasePool);
    FeasibleMig  = FilterFeasiblePopulation(MigrationPool);
    if isempty(FeasibleMig)
        return;
    end
    if isempty(W)
        W = ones(1,size(FeasibleMig.objs,2));
    end

    RefObj = FeasibleMig.objs;
    if ~isempty(FeasibleBase)
        RefObj = [FeasibleBase.objs;FeasibleMig.objs];
    end
    SectorMig = AssociateSectorsLocal(FeasibleMig.objs,W,RefObj);
    MigValue  = ComputeSectorScalar(FeasibleMig.objs,W,RefObj,SectorMig);
    if isempty(FeasibleBase)
        SectorBase = zeros(0,1);
        BaseValue  = zeros(0,1);
    else
        SectorBase = AssociateSectorsLocal(FeasibleBase.objs,W,RefObj);
        BaseValue  = ComputeSectorScalar(FeasibleBase.objs,W,RefObj,SectorBase);
    end

    Keep = false(1,numel(FeasibleMig));
    Improve = -inf(1,numel(FeasibleMig));
    for s = unique(SectorMig(:))'
        MIdx = find(SectorMig == s);
        [BestMig,Local] = min(MigValue(MIdx));
        BestIdx = MIdx(Local);
        BIdx = find(SectorBase == s);
        if isempty(BIdx)
            Keep(BestIdx) = true;
            Improve(BestIdx) = inf;
        else
            Champion = min(BaseValue(BIdx));
            if BestMig < Champion
                Keep(BestIdx) = true;
                Improve(BestIdx) = Champion - BestMig;
            end
        end
    end
    Reserved = FeasibleMig(Keep);
    if isempty(Reserved)
        return;
    end
    Improve = Improve(Keep);
    [~,Order] = sort(Improve,'descend');
    Reserved = Reserved(Order);
end

function Population = SelectByNSGA2(Population,N,UseConstraint)
    if isempty(Population) || N <= 0
        Population = Population([]);
        return;
    end
    N = min(N,numel(Population));
    if UseConstraint
        [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    else
        [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    end
    Next     = FrontNo < MaxFNo;
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    Last     = find(FrontNo == MaxFNo);
    Need     = N - sum(Next);
    if Need > 0
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:Need))) = true;
    end
    Population = Population(Next);
    FrontNo  = FrontNo(Next);
    CrowdDis = CrowdDis(Next);
    [~,Order] = sortrows([FrontNo(:),-CrowdDis(:)],[1 2]);
    Population = Population(Order);
end

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

function [Archive,Added] = UpdateExternalArchiveLocal(Archive,NewSolutions)
    Added = [];
    if nargin < 1 || isempty(Archive)
        Archive = [];
    end
    if nargin < 2 || isempty(NewSolutions)
        return;
    end
    NewSolutions = FilterFeasiblePopulation(NewSolutions);
    if isempty(NewSolutions)
        return;
    end
    Previous = Archive;
    Pool = [Archive,NewSolutions];
    Pool = KeepUniquePopulation(Pool);
    if isempty(Pool)
        Archive = [];
        return;
    end
    FrontNo = NDSort(Pool.objs,1);
    Archive = Pool(FrontNo == 1);
    if isempty(Previous)
        Added = Archive;
        return;
    end
    Keep = ~ismember(Archive.decs,Previous.decs,'rows');
    Added = Archive(Keep);
end

function LabelArchive = InitLabelArchive(D)
    LabelArchive = struct( ...
        'Dec',zeros(0,D), ...
        'Label',zeros(0,1), ...
        'Boundary',false(0,1));
end

function LabelArchive = AppendLabelArchive(LabelArchive,Population,IsBoundary,MaxCount)
    if isempty(Population)
        return;
    end
    Dec = Population.decs;
    Label = double(all(Population.cons<=0,2));
    if nargin < 3 || isempty(IsBoundary)
        IsBoundary = false(size(Label));
    end
    IsBoundary = logical(IsBoundary(:));
    if numel(IsBoundary) ~= numel(Label)
        error('PRBCCMO:LabelArchiveSizeMismatch', ...
            'Boundary flags must align with the appended label batch.');
    end
    LabelArchive.Dec = [LabelArchive.Dec;Dec];
    LabelArchive.Label = [LabelArchive.Label;Label(:)];
    LabelArchive.Boundary = [LabelArchive.Boundary;IsBoundary];
    Keep = KeepLatestDecisionRowsLocal(LabelArchive.Dec);
    LabelArchive.Dec = LabelArchive.Dec(Keep,:);
    LabelArchive.Label = LabelArchive.Label(Keep);
    LabelArchive.Boundary = LabelArchive.Boundary(Keep);
    LabelArchive = TrimLabelArchive(LabelArchive,MaxCount);
end

function LabelArchive = TrimLabelArchive(LabelArchive,MaxCount)
    Count = size(LabelArchive.Dec,1);
    if Count <= MaxCount
        return;
    end
    BoundaryQuota = min(Count,floor(0.6*MaxCount));
    BoundaryIdx = find(LabelArchive.Boundary);
    Keep = SelectLatestBalancedIndices( ...
        BoundaryIdx,LabelArchive.Label,min(BoundaryQuota,numel(BoundaryIdx)));
    Remaining = setdiff((1:Count)',Keep,'stable');
    NonBoundaryIdx = Remaining(~LabelArchive.Boundary(Remaining));
    ResidualQuota = MaxCount - numel(Keep);
    ResidualKeep = SelectLatestBalancedIndices( ...
        NonBoundaryIdx,LabelArchive.Label,min(ResidualQuota,numel(NonBoundaryIdx)));
    Keep = unique([Keep;ResidualKeep],'stable');
    if numel(Keep) < MaxCount
        Rest = setdiff((1:Count)',Keep,'stable');
        Need = MaxCount - numel(Keep);
        Keep = [Keep;Rest(max(1,end-Need+1):end)];
    elseif numel(Keep) > MaxCount
        Keep = Keep(end-MaxCount+1:end);
    end
    Keep = sort(Keep);
    LabelArchive.Dec = LabelArchive.Dec(Keep,:);
    LabelArchive.Label = LabelArchive.Label(Keep);
    LabelArchive.Boundary = LabelArchive.Boundary(Keep);
end

function [TrainSplit,EvalSplit,TestSplit] = SplitLabelArchive(LabelArchive,TrainMax,TestMax)
    TrainSplit = struct('Dec',zeros(0,size(LabelArchive.Dec,2)),'Label',zeros(0,1),'Boundary',false(0,1));
    EvalSplit  = TrainSplit;
    TestSplit  = TrainSplit;
    if isempty(LabelArchive.Dec)
        return;
    end

    BoundaryIdx = find(LabelArchive.Boundary);
    TestIdx = SelectLatestBalancedIndices(BoundaryIdx,LabelArchive.Label,min(TestMax,numel(BoundaryIdx)));
    if numel(TestIdx) < min(TestMax,size(LabelArchive.Dec,1))
        Remain = setdiff((1:size(LabelArchive.Dec,1))',TestIdx,'stable');
        Fill = SelectLatestBalancedIndices(Remain,LabelArchive.Label, ...
            min(TestMax-numel(TestIdx),numel(Remain)));
        TestIdx = unique([TestIdx;Fill],'stable');
    end
    RemainIdx = setdiff((1:size(LabelArchive.Dec,1))',TestIdx,'stable');
    TrainIdx = SelectLatestBalancedIndices(RemainIdx,LabelArchive.Label,min(TrainMax,numel(RemainIdx)));
    EvalIdx  = TestIdx;

    TrainSplit = BuildLabelSplit(LabelArchive,TrainIdx);
    EvalSplit  = BuildLabelSplit(LabelArchive,EvalIdx);
    TestSplit  = EvalSplit;
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
    KeepPos = Pos(max(1,end-Quota+1):end);
    KeepNeg = Neg(max(1,end-Quota+1):end);
    Idx = unique([KeepPos;KeepNeg],'stable');
    if numel(Idx) < Count
        Rest = setdiff(CandidateIdx,Idx,'stable');
        Need = Count - numel(Idx);
        Idx = [Idx;Rest(max(1,end-Need+1):end)];
    elseif numel(Idx) > Count
        Idx = Idx(end-Count+1:end);
    end
    Idx = sort(Idx);
end

function Split = BuildLabelSplit(LabelArchive,Idx)
    D = size(LabelArchive.Dec,2);
    Split = struct('Dec',zeros(0,D),'Label',zeros(0,1),'Boundary',false(0,1));
    if isempty(Idx)
        return;
    end
    Split.Dec = LabelArchive.Dec(Idx,:);
    Split.Label = LabelArchive.Label(Idx);
    Split.Boundary = LabelArchive.Boundary(Idx);
end

function BoundaryBatch = InitBoundaryBatch()
    BoundaryBatch = struct( ...
        'population',[], ...
        'count',0, ...
        'Boundary',false(0,1));
end

function BoundaryBatch = AppendBoundaryBatch(BoundaryBatch,Population,IsBoundary)
    if isempty(Population)
        return;
    end
    if isempty(BoundaryBatch.population)
        BoundaryBatch.population = Population;
    else
        BoundaryBatch.population = [BoundaryBatch.population,Population];
    end
    BoundaryBatch.count = BoundaryBatch.count + numel(Population);
    BoundaryBatch.Boundary = [BoundaryBatch.Boundary;logical(IsBoundary(:))];
end

function Model = RefitBoundaryModel( ...
    PrevModel,LabelArchive,Hidden,Epoch,LR,TrainMax)

    [TrainSplit,~,~] = SplitLabelArchive(LabelArchive,TrainMax,0);
    Model = TrainBoundaryMLP(TrainSplit.Dec,TrainSplit.Label,Hidden,Epoch,LR,PrevModel);
end

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

    Hidden = max(2,round(Hidden));
    Epoch  = max(1,round(Epoch));
    LR     = max(double(LR),1e-4);
    [N,D]  = size(X);
    LambdaReg = 1e-4;

    % Always recompute normalization from the current training set;
    % only warm-start the weights (W1,b1,W2,b2) from the previous model.
    Mu    = mean(X,1);
    Sigma = std(X,0,1);
    Sigma(Sigma<1e-12) = 1;
    if ~isempty(PrevModel) && IsWarmStartCompatible(PrevModel,D,Hidden)
        W1    = PrevModel.W1;
        b1    = PrevModel.b1;
        W2    = PrevModel.W2;
        b2    = PrevModel.b2;
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
    Model.Mu = Mu;
    Model.Sigma = Sigma;
    Model.W1 = W1;
    Model.b1 = b1;
    Model.W2 = W2;
    Model.b2 = b2;
end

function Flag = IsWarmStartCompatible(Model,D,Hidden)
    Flag = ~isempty(Model) && isfield(Model,'W1') && isfield(Model,'W2') ...
        && size(Model.W1,1) == D && size(Model.W1,2) == Hidden ...
        && size(Model.W2,1) == Hidden;
end

function [Weight,NormWeight] = BuildClassWeights(Y)
    N = numel(Y);
    Pos = sum(Y==1);
    Neg = N - Pos;
    WPos = N/(2*max(1,Pos));
    WNeg = N/(2*max(1,Neg));
    Weight = WNeg + (WPos-WNeg).*Y;
    NormWeight = max(sum(Weight),1);
end

function [Prob,Stats] = PredictBoundaryMLP(Model,X)
    if nargin < 2 || isempty(X)
        Prob = zeros(0,1);
        Stats = struct('logit',zeros(0,1));
        return;
    end
    if isempty(Model) || ~isfield(Model,'Mu')
        Prob = 0.5*ones(size(X,1),1);
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

function Value = SafeRuntimeOption(RuntimeOptions,Field,Default)
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,Field) && ~isempty(RuntimeOptions.(Field))
        Value = RuntimeOptions.(Field);
    else
        Value = Default;
    end
end

function [Sector,Count] = AssociateSectorsLocal(PopObj,W,RefObj)
    if nargin < 3 || isempty(RefObj)
        RefObj = PopObj;
    end
    if isempty(PopObj)
        Sector = zeros(0,1);
        Count = zeros(size(W,1),1);
        return;
    end
    if isempty(W)
        Sector = ones(size(PopObj,1),1);
        Count  = size(PopObj,1);
        return;
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range = MaxObj - MinObj;
    Range(Range<1e-12) = 1;

    Obj = (PopObj-MinObj)./Range;
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
    Range = MaxObj - MinObj;
    Range(Range<1e-12) = 1;
    NormObj = (Obj-MinObj)./Range;
    Value = sum(NormObj.*Weight,2);
end

function Dec = InterpolateBoundaryPointLocal(Problem,FeasibleDec,InfeasibleDec,Lambda)
    Dec = FeasibleDec;
    RealIdx = find(Problem.encoding<=2);
    if ~isempty(RealIdx)
        Dec(RealIdx) = FeasibleDec(RealIdx) + Lambda*(InfeasibleDec(RealIdx)-FeasibleDec(RealIdx));
    end
    OtherIdx = setdiff(1:Problem.D,RealIdx);
    if ~isempty(OtherIdx) && Lambda > 0.5
        Dec(OtherIdx) = InfeasibleDec(OtherIdx);
    end
    Dec = Problem.CalDec(Dec);
end

function State = InitBestMigrants()
    State = struct('sector',zeros(0,1),'scalar',zeros(0,1),'population',[]);
end

function State = UpdateBestMigrant(State,SectorID,Solution,Scalar,AnchorScalar)
    if Scalar >= AnchorScalar
        return;
    end
    Idx = find(State.sector == SectorID,1,'first');
    if isempty(Idx)
        State.sector(end+1,1) = SectorID;
        State.scalar(end+1,1) = Scalar;
        if isempty(State.population)
            State.population = Solution;
        else
            State.population(end+1) = Solution;
        end
        return;
    end
    if Scalar < State.scalar(Idx)
        State.scalar(Idx) = Scalar;
        State.population(Idx) = Solution;
    end
end
