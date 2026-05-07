classdef PRBCCMO < ALGORITHM
% <2026> <multi> <real> <constrained>
% PRBCCMO
% Boundary-band MLP driven CCMO for binary unknown constraints
%
% hidden --- 20   --- Hidden neurons of the boundary MLP
% epoch  --- 20   --- Training epochs per update
% lr     --- 0.01 --- Learning rate
% kappa  --- 10   --- Maximum boundary samples kept per sector
% rho    --- 0.12 --- Boundary evaluation budget ratio

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
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            [hidden,epoch,lr,~,rho] = Algorithm.ParameterSet(20,20,0.01,10,0.12);

            kappa = 10;
            rho   = max(0,min(0.5,double(rho)));

            PopulationC = Problem.Initialization();
            PopulationU = Problem.Initialization();
            B           = PopulationC([]);
            Model       = [];
            ModelState  = InitBoundaryModelState(size(W,1));
            RecentBoundaryOff = PopulationC([]);

            while Algorithm.NotTerminated(PopulationC)
                OffspringC = GenerateDEOffspring(Problem,PopulationC,true);
                OffspringU = GenerateDEOffspring(Problem,PopulationU,false);

                [BoundaryOff,BoundaryEvidence] = GenerateBoundaryOffspring( ...
                    Problem,B,PopulationC,PopulationU,W,Model,rho);

                QC = KeepUniquePopulation([PopulationC,OffspringC,OffspringU,BoundaryOff]);
                QU = KeepUniquePopulation([PopulationU,OffspringC,OffspringU,BoundaryOff]);
                PopulationC = EnvironmentalSelectionC( ...
                    QC,Problem.N,W,Model,B,RecentBoundaryOff,PopulationU);
                PopulationU = EnvironmentalSelectionU(QU,Problem.N,W);

                RecentBoundaryOff = BoundaryOff;
                B = UpdateBoundaryArchive(B,RecentBoundaryOff,PopulationC,PopulationU,W,Model,kappa);
                [Model,ModelState] = UpdateBoundaryModelIfNeeded( ...
                    Model,ModelState,B,BoundaryEvidence,PopulationC,PopulationU,W,hidden,epoch,lr,Problem,Problem.FE);
            end
        end
    end
end

%% ========== Main-population offspring ==========

function Offspring = GenerateDEOffspring(Problem,Population,useConstraintIndicator)
    if isempty(Population)
        Offspring = Population;
        return;
    end

    N = numel(Population);
    if useConstraintIndicator
        [Flag,FrontNo,CrowdDis] = ConstraintSideIndicator(Population);
        MatingPool = TournamentSelection(2,2*N,Flag,FrontNo,-CrowdDis);
    else
        [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population);
        MatingPool = TournamentSelection(2,2*N,FrontNo,-CrowdDis);
    end

    Base = Population(randi(N,N,1));
    Offspring = OperatorDE(Problem,Base, ...
        Population(MatingPool(1:N)),Population(MatingPool(N+1:end)));
end

%% ========== Model state ==========

function ModelState = InitBoundaryModelState(K)
    ModelState = struct( ...
        'trainBuffer',InitTrainingBuffer(), ...
        'lastTrainSize',0, ...
        'lastMixedSectorMask',false(K,1), ...
        'lastBoundaryCount',0, ...
        'lastTrainFE',0, ...
        'lastValidationBrier',inf, ...
        'lastCandidateDec',zeros(0,0));
    ModelState.trainBuffer = InitTrainingBuffer();
end

%% ========== Boundary archive ==========

function B = UpdateBoundaryArchive(B,BoundarySamples,PopulationC,PopulationU,W,Model,kappa)
    Support = KeepUniquePopulation([BoundarySamples,PopulationC,PopulationU]);
    SupportBase = BuildOppositeSupportBase(Support,W);
    PairPool = KeepUniquePopulation([BoundarySamples,PopulationC,PopulationU]);
    Pairs = BuildBoundaryPairs(PairPool,W,max(4*size(W,1),2*numel(PairPool)));
    TightEndpoints = ExtractTightPairEndpoints(PairPool,Pairs,kappa);
    RevalidatedB = RevalidateBoundaryMemory(B,SupportBase,W,Model,PopulationC,PopulationU);
    [Candidates,CandidateSource] = BuildArchiveCandidateSet(BoundarySamples,TightEndpoints,RevalidatedB);
    if isempty(Candidates)
        B = Candidates;
        return;
    end

    Meta = BuildBoundaryMeta(Candidates,W,Model,PopulationC,PopulationU,SupportBase);
    B = SelectTopKPerSector(Candidates,Meta,CandidateSource,kappa,~isempty(Model));
end

function Endpoints = ExtractTightPairEndpoints(Pool,Pairs,kappa)
    Endpoints = Pool([]);
    if isempty(Pool) || isempty(Pairs.FeasibleIndex)
        return;
    end

    Limit = max(2,2*max(1,kappa)*max(1,numel(unique(Pairs.Sector))));
    Gate = ResolvePairGapThreshold(Pairs,'archive');
    idx = find(Pairs.BoundaryGap <= Gate);
    if isempty(idx)
        return;
    end
    Key = [Pairs.BoundaryGap(idx),Pairs.GapDec(idx),Pairs.GapObj(idx),idx(:)];
    [~,ord] = sortrows(Key,[1 2 3 4]);
    idx = idx(ord(1:min(numel(ord),Limit)));
    Pick = unique([Pairs.FeasibleIndex(idx);Pairs.InfeasibleIndex(idx)],'stable');
    Endpoints = Pool(Pick);
end

function Revalidated = RevalidateBoundaryMemory(B,SupportBase,W,Model,PopulationC,PopulationU)
    Revalidated = B([]);
    if isempty(B) || isempty(SupportBase.Dec)
        return;
    end

    Meta = BuildBoundaryMeta(B,W,Model,PopulationC,PopulationU,SupportBase);
    Keep = BoundaryEligible(Meta,~isempty(Model),'evidence');
    if any(Keep)
        Revalidated = B(Keep);
    end
end

function [Candidates,Source] = BuildArchiveCandidateSet(BoundarySamples,TightEndpoints,RevalidatedB)
    Raw = [RevalidatedB,TightEndpoints,BoundarySamples];
    Source = [ ...
        3*ones(numel(RevalidatedB),1); ...
        2*ones(numel(TightEndpoints),1); ...
        ones(numel(BoundarySamples),1)];
    Candidates = Raw;
    if isempty(Raw)
        return;
    end

    Keep = KeepLatestDecisionRowsLocal(Raw.decs);
    Candidates = Raw(Keep);
    Source = Source(Keep);
end

function Pairs = BuildBoundaryPairs(Pool,W,MaxPairs)
    D = ResolveDecisionDimension(Pool);
    M = 0;
    if ~isempty(Pool)
        M = size(Pool.objs,2);
    end
    Pairs = InitBoundaryPairs(D,M);
    if isempty(Pool) || MaxPairs <= 0
        return;
    end

    Pairs = BuildBoundaryPairsFromArrays( ...
        double(Pool.decs),double(Pool.objs),double(all(Pool.cons<=0,2)),W,MaxPairs);
end

function Pairs = BuildBoundaryPairsFromArrays(Dec,Obj,Label,W,MaxPairs)
    D = size(Dec,2);
    M = size(Obj,2);
    Pairs = InitBoundaryPairs(D,M);
    if isempty(Dec) || MaxPairs <= 0
        return;
    end

    Label = ColumnVector(double(Label));
    Fea = find(Label == 1);
    Inf = find(Label == 0);
    if isempty(Fea) || isempty(Inf)
        return;
    end

    DecScale = ResolveDistanceScale(Dec);
    ObjScale = ResolveDistanceScale(Obj);
    Dist = zeros(numel(Fea),numel(Inf));
    GapDec = zeros(numel(Fea),numel(Inf));
    GapObj = zeros(numel(Fea),numel(Inf));
    for i = 1 : numel(Fea)
        DecDelta = Dec(Inf,:) - repmat(Dec(Fea(i),:),numel(Inf),1);
        ObjDelta = Obj(Inf,:) - repmat(Obj(Fea(i),:),numel(Inf),1);
        GapDec(i,:) = sqrt(sum(DecDelta.^2,2))'./max(DecScale,1e-6);
        GapObj(i,:) = sqrt(sum(ObjDelta.^2,2))'./max(ObjScale,1e-6);
        Dist(i,:) = GapDec(i,:) + 0.25*GapObj(i,:);
    end

    [~,NearInfForFea] = min(Dist,[],2);
    [~,NearFeaForInf] = min(Dist,[],1);
    PairMask = false(size(Dist));
    for i = 1 : numel(Fea)
        j = NearInfForFea(i);
        if NearFeaForInf(j) == i
            PairMask(i,j) = true;
        end
    end
    if ~any(PairMask(:))
        [~,ord] = sort(Dist(:),'ascend');
        PairMask(ord(1:min(numel(ord),MaxPairs))) = true;
    end
    [I,J] = find(PairMask);
    PairGap = Dist(sub2ind(size(Dist),I,J));
    [~,ord] = sort(PairGap,'ascend');
    ord = ord(1:min(numel(ord),MaxPairs));
    I = I(ord);
    J = J(ord);
    PairGap = PairGap(ord);
    Count = numel(I);
    if Count <= 0
        return;
    end

    Pairs.FeasibleIndex = Fea(I);
    Pairs.InfeasibleIndex = Inf(J);
    Pairs.FeasibleDec = Dec(Pairs.FeasibleIndex,:);
    Pairs.InfeasibleDec = Dec(Pairs.InfeasibleIndex,:);
    Pairs.FeasibleObj = Obj(Pairs.FeasibleIndex,:);
    Pairs.InfeasibleObj = Obj(Pairs.InfeasibleIndex,:);
    MidObj = 0.5*(Pairs.FeasibleObj + Pairs.InfeasibleObj);
    Pairs.Sector = AssociateSectorsLocal(MidObj,W,min(Obj,[],1));
    Pairs.GapDec = GapDec(sub2ind(size(GapDec),I,J));
    Pairs.GapObj = GapObj(sub2ind(size(GapObj),I,J));
    Pairs.BoundaryGap = PairGap;
end

function Pairs = InitBoundaryPairs(D,M)
    Pairs = struct( ...
        'FeasibleIndex',zeros(0,1), ...
        'InfeasibleIndex',zeros(0,1), ...
        'FeasibleDec',zeros(0,D), ...
        'InfeasibleDec',zeros(0,D), ...
        'FeasibleObj',zeros(0,M), ...
        'InfeasibleObj',zeros(0,M), ...
        'Sector',zeros(0,1), ...
        'GapDec',zeros(0,1), ...
        'GapObj',zeros(0,1), ...
        'BoundaryGap',zeros(0,1));
end

function Gate = ResolvePairGapThreshold(Pairs,Mode)
    if isempty(Pairs.BoundaryGap)
        Gate = inf;
        return;
    end
    Gate = BoundaryGapTau(Mode);
end

function B = SelectTopKPerSector(Candidates,Meta,Source,kappa,ModelReady)
    if isempty(Candidates)
        B = Candidates;
        return;
    end
    if nargin < 3 || isempty(Source)
        Source = 5*ones(numel(Candidates),1);
    end
    Source = MatchLength(Source,numel(Candidates),5);
    if nargin < 5
        ModelReady = true;
    end

    kappa = min(10,max(1,round(kappa)));
    SideQuota = max(1,ceil(kappa/2));
    Eligible = BoundaryEligible(Meta,ModelReady,'archive');
    if isempty(find(Meta.sector > 0,1))
        ord = SortByArchiveSourcePriority(Meta,find(Eligible),Source,kappa);
        if isempty(ord)
            B = Candidates([]);
        else
            B = Candidates(ord(1:min(numel(ord),kappa)));
        end
        return;
    end

    Sectors = unique(Meta.sector(Meta.sector > 0))';
    Pick = zeros(min(numel(Candidates),numel(Sectors)*kappa),1);
    PickCount = 0;
    for s = Sectors
        idx = find(Meta.sector == s & Eligible);
        if isempty(idx)
            continue;
        end

        Pos = SortByArchiveSourcePriority(Meta,idx(Meta.feasible(idx) == 1),Source,SideQuota);
        Neg = SortByArchiveSourcePriority(Meta,idx(Meta.feasible(idx) == 0),Source,SideQuota);
        Sel = zeros(min(kappa,numel(Pos)+numel(Neg)),1);
        SelCount = 0;
        for j = 1 : max(numel(Pos),numel(Neg))
            if j <= numel(Pos)
                SelCount = SelCount + 1;
                Sel(SelCount) = Pos(j);
            end
            if SelCount >= numel(Sel)
                break;
            end
            if j <= numel(Neg)
                SelCount = SelCount + 1;
                Sel(SelCount) = Neg(j);
            end
            if SelCount >= numel(Sel)
                break;
            end
        end
        Sel = Sel(1:SelCount);
        Sel = Sel(1:min(kappa,numel(Sel)));
        AddCount = min(numel(Sel),numel(Pick)-PickCount);
        if AddCount > 0
            Pick(PickCount+1:PickCount+AddCount) = Sel(1:AddCount);
            PickCount = PickCount + AddCount;
        end
    end
    Pick = Pick(1:PickCount);

    if isempty(Pick)
        B = Candidates([]);
    else
        B = Candidates(unique(Pick,'stable'));
    end
end

function Order = SortByArchiveSourcePriority(Meta,idx,Source,MaxCount)
    Order = zeros(min(numel(idx),MaxCount),1);
    idx = ColumnVector(idx);
    if isempty(idx) || MaxCount <= 0
        Order = zeros(0,1);
        return;
    end
    Source = MatchLength(Source,numel(Meta.sector),5);
    Count = 0;
    for priority = 1 : 3
        Tier = idx(Source(idx) == priority);
        Tier = SortByBoundaryKey(Meta,Tier,Source);
        Need = min(MaxCount, numel(Order)) - Count;
        if Need <= 0
            break;
        end
        Add = min(Need,numel(Tier));
        if Add > 0
            Order(Count+1:Count+Add) = Tier(1:Add);
            Count = Count + Add;
        end
    end
    if Count < min(MaxCount,numel(Order))
        Rest = setdiff(idx,Order(1:Count),'stable');
        Rest = SortByBoundaryKey(Meta,Rest,Source);
        Need = min(MaxCount,numel(Order)) - Count;
        Add = min(Need,numel(Rest));
        if Add > 0
            Order(Count+1:Count+Add) = Rest(1:Add);
            Count = Count + Add;
        end
    end
    Order = Order(1:Count);
end

%% ========== Boundary meta ==========

function Meta = BuildBoundaryMeta(Candidates,W,Model,PopulationC,PopulationU,Support)
    N = numel(Candidates);
    Meta = InitBoundaryMeta(N);
    if isempty(Candidates)
        return;
    end

    RefObj = ComputeIdealPoint(Candidates,PopulationC,PopulationU);
    Sector = AssociateSectorsLocal(Candidates.objs,W,RefObj);
    Prob   = PredictBoundaryMLP(Model,Candidates.decs);
    ObjScore = ComputeBoundaryObjectiveScore(Candidates,PopulationC,W,RefObj,Sector);
    SupportCache = ResolveOppositeSupportCache(Support,W,RefObj);
    OppSupport = ComputeOppositeSupport(Candidates,SupportCache,Sector);
    [LocalPairExists,ShellDistDec,BetweenScoreObj] = ComputeBoundaryShellMetrics( ...
        Candidates,SupportCache,Sector);
    Feasible = double(all(Candidates.cons<=0,2));
    Meta = ComposeBoundaryMeta( ...
        Sector,Feasible,Prob,OppSupport,ObjScore, ...
        LocalPairExists,ShellDistDec,BetweenScoreObj);
end

function OppSupport = ComputeOppositeSupport(Candidates,SupportCache,Sector)
    if isempty(Candidates) || isempty(SupportCache.Dec)
        OppSupport = zeros(numel(Candidates),1);
        return;
    end

    CandidateLabel = ColumnVector(double(all(Candidates.cons<=0,2)));
    OppSupport = ComputeOppositeSupportFromCache( ...
        Candidates.decs,CandidateLabel,Sector,SupportCache);
end

function [LocalPairExists,ShellDistDec,BetweenScoreObj] = ComputeBoundaryShellMetrics( ...
    Candidates,SupportCache,Sector)
    %#ok<INUSD>
    N = numel(Candidates);
    LocalPairExists = false(N,1);
    ShellDistDec = inf(N,1);
    BetweenScoreObj = inf(N,1);
    if isempty(Candidates) || isempty(SupportCache.Dec)
        return;
    end

    CandidateLabel = ColumnVector(double(all(Candidates.cons<=0,2)));
    CandidateDec = double(Candidates.decs);
    CandidateObj = double(Candidates.objs);
    PairDec = [CandidateDec;SupportCache.Dec];
    PairObj = [CandidateObj;SupportCache.Obj];
    PairLabel = [CandidateLabel;SupportCache.Label];
    Pairs = BuildBoundaryPairsFromArrays(PairDec,PairObj,PairLabel,[],max(2*N,2*size(PairDec,1)));
    for p = 1 : numel(Pairs.BoundaryGap)
        FMatch = find(CandidateLabel == 1 & ismember(CandidateDec,Pairs.FeasibleDec(p,:),'rows'));
        IMatch = find(CandidateLabel == 0 & ismember(CandidateDec,Pairs.InfeasibleDec(p,:),'rows'));
        CandidateIdx = [FMatch;IMatch];
        for k = 1 : numel(CandidateIdx)
            i = CandidateIdx(k);
            if Pairs.BoundaryGap(p) < ShellDistDec(i) + 0.25*BetweenScoreObj(i)
                LocalPairExists(i) = true;
                ShellDistDec(i) = Pairs.GapDec(p);
                BetweenScoreObj(i) = Pairs.GapObj(p);
            end
        end
    end
end

function Eligible = BoundaryEligible(Meta,ModelReady,Mode)
    if nargin < 2
        ModelReady = true;
    end
    if nargin < 3 || isempty(Mode)
        Mode = 'archive';
    end
    Eligible = false(numel(Meta.sector),1);
    if isempty(Meta.sector)
        return;
    end

    LocalPair = Meta.localPairExists > 0;
    if ~any(LocalPair)
        return;
    end
    GapGate = ResolveBoundaryGapGate(Meta,LocalPair,Mode);
    Eligible = LocalPair & Meta.boundaryGap <= GapGate;
end

function Gate = ResolveBoundaryGapGate(Meta,Mask,Mode)
    %#ok<INUSD>
    Gate = BoundaryGapTau(Mode);
end

function Tau = BoundaryGapTau(Mode)
    if nargin >= 1 && strcmpi(Mode,'evidence')
        Tau = 1.25;
    else
        Tau = 0.55;
    end
end

function SupportBase = BuildOppositeSupportBase(Support,W)
    SupportBase = struct( ...
        'isSupportBase',true, ...
        'Dec',zeros(0,0), ...
        'Obj',zeros(0,0), ...
        'Label',zeros(0,1), ...
        'Tau',1, ...
        'Neighbors',{BuildSectorNeighbors(W,min(3,max(size(W,1)-1,0)))});
    if isempty(Support)
        return;
    end

    SupportBase.Dec = double(Support.decs);
    SupportBase.Obj = Support.objs;
    SupportBase.Label = ColumnVector(double(all(Support.cons<=0,2)));
    SupportBase.Tau = ResolveDistanceScale(SupportBase.Dec);
end

function SupportCache = ResolveOppositeSupportCache(SupportInfo,W,RefObj)
    if IsOppositeSupportBase(SupportInfo)
        SupportBase = SupportInfo;
    else
        SupportBase = BuildOppositeSupportBase(SupportInfo,W);
    end

    SupportCache = struct( ...
        'Dec',zeros(0,0), ...
        'Obj',zeros(0,0), ...
        'Label',zeros(0,1), ...
        'Sector',zeros(0,1), ...
        'Tau',1, ...
        'Neighbors',{SupportBase.Neighbors});
    if isempty(SupportBase.Dec)
        return;
    end

    SupportCache.Dec = SupportBase.Dec;
    SupportCache.Obj = SupportBase.Obj;
    SupportCache.Label = SupportBase.Label;
    SupportCache.Sector = ColumnVector(AssociateSectorsLocal(SupportBase.Obj,W,RefObj));
    SupportCache.Tau = SupportBase.Tau;
end

function Flag = IsOppositeSupportBase(Value)
    Flag = isstruct(Value) && isfield(Value,'isSupportBase') && isequal(Value.isSupportBase,true);
end

function ObjScore = ComputeBoundaryObjectiveScore(Candidates,PopulationC,W,RefObj,Sector)
    N = numel(Candidates);
    ObjScore = zeros(N,1);
    if isempty(Candidates)
        return;
    end

    [FrontNo,~] = NDSort(Candidates.objs,numel(Candidates));
    FrontScore  = NormalizeRange(FrontNo(:)-1);
    if isempty(W)
        CrowdScore = 1 - NormalizeRange(CrowdingDistance(Candidates.objs,FrontNo)');
        ObjScore = max(0,min(1,0.7*FrontScore + 0.3*CrowdScore(:)));
        return;
    end

    Scalar = ComputeSectorScalar(Candidates.objs,W,RefObj,Sector);
    ScalarScore = NormalizeRange(Scalar);
    ImprovePenalty = zeros(N,1);
    if ~isempty(PopulationC)
        RefSector = AssociateSectorsLocal(PopulationC.objs,W,RefObj);
        RefScalar = ComputeSectorScalar(PopulationC.objs,W,RefObj,RefSector);
        Best = inf(size(W,1),1);
        for s = 1 : size(W,1)
            idx = find(RefSector == s);
            if ~isempty(idx)
                Best(s) = min(RefScalar(idx));
            end
        end
        for i = 1 : N
            if Sector(i) > 0 && Sector(i) <= numel(Best) && isfinite(Best(Sector(i)))
                ImprovePenalty(i) = double(Scalar(i) > Best(Sector(i)));
            end
        end
    end
    ObjScore = 0.55*FrontScore + 0.35*ScalarScore + 0.10*ImprovePenalty;
    ObjScore = max(0,min(1,ObjScore));
end

%% ========== Boundary dataset + MLP ==========

function [Model,ModelState] = UpdateBoundaryModelIfNeeded( ...
    Model,ModelState,B,BoundaryEvidence,PopulationC,PopulationU,W,hidden,epoch,lr,Problem,CurrentFE)
    MaxTrain = max(6*Problem.N,4*size(W,1));
    Candidates = BuildTrainingCandidateSet(B,BoundaryEvidence,PopulationC,PopulationU);
    [ModelState.trainBuffer,BufferDiag,ModelState.lastCandidateDec] = UpdateTrainingBuffer( ...
        ModelState.trainBuffer,Candidates,W,MaxTrain,CurrentFE,ModelState.lastCandidateDec);
    Dataset = BuildBoundaryDatasetFromBuffer(ModelState.trainBuffer,W,Problem.N);
    [NeedTrain,Summary] = ShouldRetrainBoundaryModel( ...
        Problem,Model,ModelState,Dataset,BufferDiag,CurrentFE);
    if ~NeedTrain
        return;
    end

    if isempty(Dataset.Dec) || numel(unique(Dataset.Label)) < 2
        return;
    end

    Mode = ResolveBoundaryTrainingMode(Model,Dataset,hidden,Summary);
    Model = TrainBoundaryMLP(Model,Dataset,hidden,epoch,lr,Mode);
    ModelState.lastTrainSize      = size(Dataset.Dec,1);
    ModelState.lastMixedSectorMask = Summary.mixed_sector_mask;
    ModelState.lastBoundaryCount  = Summary.boundary_count;
    ModelState.lastTrainFE        = CurrentFE;
    Stats = EvaluateBoundaryValidation(Model,Dataset);
    ModelState.lastValidationBrier = Stats.brier;
end

function [NeedTrain,Summary] = ShouldRetrainBoundaryModel( ...
    Problem,Model,ModelState,Dataset,BufferDiag,CurrentFE)
    Summary = SummarizeBoundaryData(Problem,Model,ModelState,Dataset,BufferDiag,CurrentFE);
    NeedTrain = Summary.ready && (isempty(Model) || ...
        Summary.new_sample_ratio >= 0.20 || ...
        Summary.mixed_sector_gain > 0 || ...
        Summary.validation_brier > ModelState.lastValidationBrier + 0.02);
end

function Summary = SummarizeBoundaryData(Problem,Model,ModelState,Dataset,BufferDiag,~)
    Stats = EvaluateBoundaryValidation(Model,Dataset);
    MixedMask = BufferDiag.mixed_sector_mask;
    PrevMask = MatchLogicalLength(ModelState.lastMixedSectorMask,numel(MixedMask));
    Summary = struct( ...
        'ready',false, ...
        'new_sample_ratio',BufferDiag.new_sample_ratio, ...
        'mixed_sector_gain',sum(MixedMask & ~PrevMask), ...
        'mixed_sector_mask',MixedMask, ...
        'validation_brier',Stats.brier, ...
        'boundary_count',BufferDiag.buffer_size);
    Summary.ready = size(Dataset.Dec,1) >= max(6,2*Problem.M) && ...
        numel(unique(Dataset.Label)) >= 2;
end

function Mode = ResolveBoundaryTrainingMode(Model,Dataset,Hidden,Summary)
    D = size(Dataset.Dec,2);
    if isempty(Model) || ~CanWarmStartBoundaryMLP(Model,D,Hidden) || ...
            Summary.validation_brier > 0.40
        Mode = 'cold';
    else
        Mode = 'finetune';
    end
end

function Buffer = InitTrainingBuffer()
    Buffer = struct( ...
        'Dec',zeros(0,0), ...
        'Obj',zeros(0,0), ...
        'Label',zeros(0,1), ...
        'Sector',zeros(0,1), ...
        'Source',zeros(0,1), ...
        'OppDist',zeros(0,1), ...
        'PairId',zeros(0,1), ...
        'GapDec',zeros(0,1), ...
        'GapObj',zeros(0,1), ...
        'BoundaryGap',zeros(0,1), ...
        'Time',zeros(0,1));
end

function [Buffer,Diag,LastCandidateDec] = UpdateTrainingBuffer( ...
    Buffer,Candidates,W,MaxTrain,Time,PreviousCandidateDec)
    Diag = InitTrainingBufferDiag(W,Buffer);
    LastCandidateDec = ExtractCandidateDecisionRows(Candidates);
    if isempty(Candidates.Population)
        return;
    end
    if ~HaveDecisionRowsChanged(LastCandidateDec,PreviousCandidateDec)
        Diag.mixed_sector_mask = ResolveMixedSectorMaskFromBuffer(Buffer,size(W,1));
        return;
    end

    Eligible = ResolveTrainingCandidateEligibility( ...
        Candidates.Population.decs,Buffer.Dec,Candidates.Source);
    if ~any(Eligible)
        Diag.mixed_sector_mask = ResolveMixedSectorMaskFromBuffer(Buffer,size(W,1));
        return;
    end

    RefObj = ComputeIdealPoint(Candidates.Population);
    Sector = AssociateSectorsLocal(Candidates.Population.objs,W,RefObj);
    Label  = double(all(Candidates.Population.cons<=0,2));
    Band = CollectSectorBoundaryBand( ...
        Candidates.Population,Label,Sector,Candidates.Source,W,Eligible);
    if isempty(Band.Dec)
        Diag.mixed_sector_mask = ResolveMixedSectorMaskFromBuffer(Buffer,size(W,1));
        return;
    end

    OldDec = Buffer.Dec;
    Buffer = AppendTrainingBuffer(Buffer,Band,Time);
    Buffer = TrimTrainingBuffer(Buffer,max(1,round(MaxTrain)));
    NewSize = size(Buffer.Dec,1);
    if isempty(OldDec)
        NewRows = true(NewSize,1);
    else
        NewRows = ~ismember(Buffer.Dec,OldDec,'rows');
    end
    Diag.buffer_size = NewSize;
    Diag.new_sample_ratio = sum(NewRows)/max(NewSize,1);
    Diag.mixed_sector_mask = ResolveMixedSectorMaskFromBuffer(Buffer,size(W,1));
end

function Dec = ExtractCandidateDecisionRows(Candidates)
    if isempty(Candidates.Population)
        Dec = zeros(0,0);
    else
        Dec = double(Candidates.Population.decs);
    end
end

function Flag = HaveDecisionRowsChanged(CurrentDec,PreviousDec)
    if isempty(CurrentDec) && isempty(PreviousDec)
        Flag = false;
        return;
    end
    if ~isequal(size(CurrentDec),size(PreviousDec))
        Flag = true;
        return;
    end
    Flag = any(CurrentDec(:) ~= PreviousDec(:));
end

function Eligible = ResolveTrainingCandidateEligibility(Dec,BufferDec,Source)
    Eligible = ColumnVector(Source <= 2);
    if isempty(Dec)
        return;
    end
    if isempty(BufferDec)
        Eligible = true(size(Dec,1),1);
    else
        Eligible = Eligible | ~ismember(Dec,BufferDec,'rows');
    end
end

function Diag = InitTrainingBufferDiag(W,Buffer)
    K = size(W,1);
    Diag = struct( ...
        'buffer_size',size(Buffer.Dec,1), ...
        'new_sample_ratio',0, ...
        'mixed_sector_mask',false(K,1));
end

function Candidates = BuildTrainingCandidateSet(B,BoundaryEvidence,PopulationC,PopulationU)
    [Population,Source] = MergeTrainingSourcePopulations(B,BoundaryEvidence,PopulationC,PopulationU);
    if isempty(Population)
        Candidates = struct('Population',Population,'Source',Source);
        return;
    end

    Keep = KeepLatestDecisionRowsLocal(Population.decs);
    Population = Population(Keep);
    Source = Source(Keep);
    Candidates = struct('Population',Population,'Source',Source);
end

function [Population,Source] = MergeTrainingSourcePopulations(B,BoundaryEvidence,PopulationC,PopulationU)
    Population = [B,BoundaryEvidence,PopulationC,PopulationU];
    Source = [ ...
        ones(numel(B),1); ...
        2*ones(numel(BoundaryEvidence),1); ...
        3*ones(numel(PopulationC),1); ...
        4*ones(numel(PopulationU),1)];
end

function Band = CollectSectorBoundaryBand(Population,Label,Sector,Source,W,Eligible)
    %#ok<INUSD>
    Band = InitBoundaryBand(size(Population.decs,2));
    if isempty(Population)
        return;
    end
    if nargin < 6 || isempty(Eligible)
        Eligible = true(numel(Population),1);
    else
        Eligible = ColumnVector(logical(Eligible));
    end
    if ~any(Eligible)
        return;
    end

    EligibleIdx = find(Eligible);
    PairPool = Population(EligibleIdx);
    Pairs = BuildBoundaryPairs(PairPool,W,max(2*size(W,1),numel(PairPool)));
    if isempty(Pairs.FeasibleIndex)
        return;
    end
    MaxPairsPerSector = 2;
    PairPick = false(numel(Pairs.BoundaryGap),1);
    Sectors = unique(Pairs.Sector(Pairs.Sector > 0))';
    if isempty(Sectors)
        Sectors = 1;
    end
    for s = Sectors
        if isempty(W)
            idx = (1:numel(Pairs.BoundaryGap))';
        else
            idx = find(Pairs.Sector == s);
        end
        if isempty(idx)
            continue;
        end
        Key = [Pairs.BoundaryGap(idx),Pairs.GapDec(idx),Pairs.GapObj(idx),idx(:)];
        [~,ord] = sortrows(Key,[1 2 3 4]);
        idx = idx(ord(1:min(MaxPairsPerSector,numel(ord))));
        PairPick(idx) = true;
    end
    SelectedPairIdx = find(PairPick);
    if isempty(SelectedPairIdx)
        return;
    end

    FeaIdx = EligibleIdx(Pairs.FeasibleIndex(SelectedPairIdx));
    InfIdx = EligibleIdx(Pairs.InfeasibleIndex(SelectedPairIdx));
    PairId = (1:numel(SelectedPairIdx))';
    Band.Dec     = [Population(FeaIdx).decs;Population(InfIdx).decs];
    Band.Obj     = [Population(FeaIdx).objs;Population(InfIdx).objs];
    Band.Label   = [ones(numel(SelectedPairIdx),1);zeros(numel(SelectedPairIdx),1)];
    Band.Sector  = [Pairs.Sector(SelectedPairIdx);Pairs.Sector(SelectedPairIdx)];
    Band.Source  = [Source(FeaIdx);Source(InfIdx)];
    Band.OppDist = [Pairs.GapDec(SelectedPairIdx);Pairs.GapDec(SelectedPairIdx)];
    Band.PairId  = [PairId;PairId];
    Band.GapDec  = [Pairs.GapDec(SelectedPairIdx);Pairs.GapDec(SelectedPairIdx)];
    Band.GapObj  = [Pairs.GapObj(SelectedPairIdx);Pairs.GapObj(SelectedPairIdx)];
    Band.BoundaryGap = [Pairs.BoundaryGap(SelectedPairIdx);Pairs.BoundaryGap(SelectedPairIdx)];
end

function Band = InitBoundaryBand(D)
    Band = struct( ...
        'Dec',zeros(0,D), ...
        'Obj',zeros(0,0), ...
        'Label',zeros(0,1), ...
        'Sector',zeros(0,1), ...
        'Source',zeros(0,1), ...
        'OppDist',zeros(0,1), ...
        'PairId',zeros(0,1), ...
        'GapDec',zeros(0,1), ...
        'GapObj',zeros(0,1), ...
        'BoundaryGap',zeros(0,1));
end

function Priority = SourcePriority(Source)
    Priority = 5*ones(numel(Source),1);
    Priority(Source == 1) = 1;
    Priority(Source == 2) = 2;
    Priority(Source == 3) = 3;
    Priority(Source == 4) = 4;
end

function Buffer = AppendTrainingBuffer(Buffer,Band,Time)
    if isempty(Band.Dec)
        return;
    end
    Band = OffsetTrainingPairIds(Band,Buffer.PairId);
    if isempty(Buffer.Dec)
        Buffer.Dec = Band.Dec;
        Buffer.Obj = Band.Obj;
        Buffer.Label = Band.Label;
        Buffer.Sector = Band.Sector;
        Buffer.Source = Band.Source;
        Buffer.OppDist = Band.OppDist;
        Buffer.PairId = Band.PairId;
        Buffer.GapDec = Band.GapDec;
        Buffer.GapObj = Band.GapObj;
        Buffer.BoundaryGap = Band.BoundaryGap;
        Buffer.Time = Time*ones(size(Band.Label));
    else
        Buffer.Dec = [Buffer.Dec;Band.Dec];
        Buffer.Obj = [Buffer.Obj;Band.Obj];
        Buffer.Label = [Buffer.Label;Band.Label];
        Buffer.Sector = [Buffer.Sector;Band.Sector];
        Buffer.Source = [Buffer.Source;Band.Source];
        Buffer.OppDist = [Buffer.OppDist;Band.OppDist];
        Buffer.PairId = [Buffer.PairId;Band.PairId];
        Buffer.GapDec = [Buffer.GapDec;Band.GapDec];
        Buffer.GapObj = [Buffer.GapObj;Band.GapObj];
        Buffer.BoundaryGap = [Buffer.BoundaryGap;Band.BoundaryGap];
        Buffer.Time = [Buffer.Time;Time*ones(size(Band.Label))];
    end
    Keep = KeepLatestDecisionRowsLocal(Buffer.Dec);
    Buffer = SliceTrainingBuffer(Buffer,Keep);
end

function Band = OffsetTrainingPairIds(Band,ExistingPairId)
    if isempty(Band.PairId)
        return;
    end
    Offset = max([0;ColumnVector(ExistingPairId)]);
    Mask = Band.PairId > 0;
    Band.PairId(Mask) = Band.PairId(Mask) + Offset;
end

function Buffer = TrimTrainingBuffer(Buffer,MaxTrain)
    if size(Buffer.Dec,1) <= MaxTrain
        return;
    end
    Key = [ColumnVector(Buffer.BoundaryGap),ColumnVector(Buffer.GapDec), ...
           ColumnVector(Buffer.GapObj),ColumnVector(-Buffer.Time), ...
           ColumnVector(SourcePriority(Buffer.Source)),ColumnVector((1:size(Buffer.Dec,1))')];
    [~,ord] = sortrows(Key,[1 2 3 4 5 6]);
    Keep = sort(ord(1:MaxTrain));
    Buffer = SliceTrainingBuffer(Buffer,Keep);
end

function Buffer = SliceTrainingBuffer(Buffer,Keep)
    Buffer.Dec = Buffer.Dec(Keep,:);
    Buffer.Obj = Buffer.Obj(Keep,:);
    Buffer.Label = Buffer.Label(Keep);
    Buffer.Sector = Buffer.Sector(Keep);
    Buffer.Source = Buffer.Source(Keep);
    Buffer.OppDist = Buffer.OppDist(Keep);
    Buffer.PairId = Buffer.PairId(Keep);
    Buffer.GapDec = Buffer.GapDec(Keep);
    Buffer.GapObj = Buffer.GapObj(Keep);
    Buffer.BoundaryGap = Buffer.BoundaryGap(Keep);
    Buffer.Time = Buffer.Time(Keep);
end

function Mask = ResolveMixedSectorMaskFromBuffer(Buffer,K)
    Mask = false(K,1);
    if K <= 0 || isempty(Buffer.Dec)
        return;
    end
    for s = 1 : K
        idx = Buffer.Sector == s;
        Mask(s) = any(Buffer.Label(idx) == 1) && any(Buffer.Label(idx) == 0);
    end
end

function Meta = BuildBoundaryMetaFromBuffer(Buffer,W)
    Count = size(Buffer.Dec,1);
    Meta = InitBoundaryMeta(Count);
    if Count <= 0
        return;
    end

    LocalPairExists = isfinite(Buffer.BoundaryGap);
    ShellDistDec = Buffer.GapDec;
    BetweenScoreObj = Buffer.GapObj;
    Prob = 0.5*ones(Count,1);
    OppSupport = exp(-Buffer.BoundaryGap);
    ObjScore = NormalizeFiniteScore(BetweenScoreObj);
    Meta = ComposeBoundaryMeta( ...
        Buffer.Sector,Buffer.Label,Prob,OppSupport,ObjScore, ...
        LocalPairExists,ShellDistDec,BetweenScoreObj);
end

function Dataset = BuildBoundaryDatasetFromBuffer(Buffer,W,N)
    Dataset = struct('Dec',zeros(0,0),'Label',zeros(0,1));
    if isempty(Buffer.Dec) || numel(unique(Buffer.Label)) < 2
        return;
    end

    Meta = BuildBoundaryMetaFromBuffer(Buffer,W);
    Eligible = BoundaryEligible(Meta,false,'evidence');
    if ~any(Eligible)
        return;
    end

    MaxPairsPerSector = 2;
    Sectors = unique(Buffer.Sector(Buffer.Sector > 0))';
    Ranked = zeros(min(size(Buffer.Dec,1),2*MaxPairsPerSector*numel(Sectors)),1);
    RankedCount = 0;
    for s = Sectors
        Add = SelectReplayPairsInSector(Buffer,Meta,Eligible,s,MaxPairsPerSector);
        AddCount = min(numel(Add),numel(Ranked)-RankedCount);
        if AddCount > 0
            Ranked(RankedCount+1:RankedCount+AddCount) = Add(1:AddCount);
            RankedCount = RankedCount + AddCount;
        end
    end
    Ranked = Ranked(1:RankedCount);
    if isempty(Ranked)
        return;
    end
    Ranked = unique(Ranked,'stable');
    Dataset.Dec = Buffer.Dec(Ranked,:);
    Dataset.Label = Buffer.Label(Ranked);
end

function ReplayRows = SelectReplayPairsInSector(Buffer,Meta,Eligible,Sector,MaxPairs)
    ReplayRows = zeros(0,1);
    PairIds = unique(Buffer.PairId(Buffer.Sector == Sector & Eligible & Buffer.PairId > 0),'stable');
    if isempty(PairIds)
        return;
    end
    PairKey = inf(numel(PairIds),4);
    PairMembers = cell(numel(PairIds),1);
    for i = 1 : numel(PairIds)
        idx = find(Buffer.PairId == PairIds(i) & Buffer.Sector == Sector & Eligible);
        Pos = idx(Buffer.Label(idx) == 1);
        Neg = idx(Buffer.Label(idx) == 0);
        if isempty(Pos) || isempty(Neg)
            continue;
        end
        Pos = SortByBoundaryKey(Meta,Pos,Buffer.Source);
        Neg = SortByBoundaryKey(Meta,Neg,Buffer.Source);
        PairMembers{i} = [Pos(1);Neg(1)];
        PairKey(i,:) = [min(Meta.boundaryGap(idx)),min(Meta.gapDec(idx)), ...
            min(Meta.gapObj(idx)),PairIds(i)];
    end
    Valid = find(isfinite(PairKey(:,1)));
    if isempty(Valid)
        return;
    end
    [~,ord] = sortrows(PairKey(Valid,:),[1 2 3 4]);
    Valid = Valid(ord(1:min(MaxPairs,numel(ord))));
    ReplayRows = vertcat(PairMembers{Valid});
end

function Model = TrainBoundaryMLP(Model,Dataset,Hidden,Epoch,LR,Mode)
    X = double(Dataset.Dec);
    Y = double(Dataset.Label(:) > 0);
    Hidden = max(2,round(Hidden));
    Epoch  = max(1,round(Epoch));
    LR     = max(double(LR),1e-4);
    if isempty(X) || size(X,1) < 2 || numel(unique(Y)) < 2
        return;
    end

    Mu    = mean(X,1);
    Sigma = std(X,0,1);
    Sigma(Sigma < 1e-12) = 1;
    Xn = (X-Mu)./Sigma;

    [~,D] = size(Xn);
    if strcmpi(Mode,'finetune') && CanWarmStartBoundaryMLP(Model,D,Hidden)
        [W1,b1,W2,b2] = RebaseBoundaryMLPNormalization(Model,Mu,Sigma);
    else
        W1 = 0.1*randn(D,Hidden);
        b1 = zeros(1,Hidden);
        W2 = 0.1*randn(Hidden,1);
        b2 = 0;
    end

    for e = 1 : Epoch
        H = tanh(Xn*W1 + repmat(b1,size(Xn,1),1));
        Z = H*W2 + b2;
        P = 1./(1+exp(-Z));
        Delta2 = (P-Y)./max(size(Xn,1),1);
        D1 = (Delta2*W2').*(1-H.^2);

        Step = LR/sqrt(e);
        W2 = W2 - Step*(H'*Delta2);
        b2 = b2 - Step*sum(Delta2);
        W1 = W1 - Step*(Xn'*D1);
        b1 = b1 - Step*sum(D1,1);
    end

    Z = tanh(Xn*W1 + repmat(b1,size(Xn,1),1))*W2 + b2;
    Temperature = FitBoundaryTemperature(Z,Y);
    Model = struct('Mu',Mu,'Sigma',Sigma,'W1',W1,'b1',b1,'W2',W2,'b2',b2,'Temperature',Temperature);
end

function [W1,b1,W2,b2] = RebaseBoundaryMLPNormalization(Model,Mu,Sigma)
    OldSigma = Model.Sigma;
    OldSigma(OldSigma < 1e-12) = 1;
    Scale = Sigma./OldSigma;
    Shift = (Mu-Model.Mu)./OldSigma;
    W1 = Model.W1.*repmat(Scale(:),1,size(Model.W1,2));
    b1 = Model.b1 + Shift*Model.W1;
    W2 = Model.W2;
    b2 = Model.b2;
end

function Flag = CanWarmStartBoundaryMLP(Model,D,Hidden)
    Flag = ~isempty(Model) && isfield(Model,'W1') && isfield(Model,'W2') && ...
        size(Model.W1,1) == D && size(Model.W1,2) == Hidden && ...
        size(Model.W2,1) == Hidden;
end

function Stats = EvaluateBoundaryValidation(Model,Dataset)
    Stats = struct('brier',inf,'balanced_accuracy',0);
    if isempty(Model) || isempty(Dataset.Dec) || numel(unique(Dataset.Label)) < 2
        return;
    end
    Prob = PredictBoundaryMLP(Model,Dataset.Dec);
    Label = double(Dataset.Label(:) > 0);
    Stats.brier = mean((Prob(:)-Label).^2);
    Pred = double(Prob(:) >= 0.5);
    TPR = mean(Pred(Label == 1) == 1);
    TNR = mean(Pred(Label == 0) == 0);
    Stats.balanced_accuracy = mean([TPR,TNR]);
end

function Prob = PredictBoundaryMLP(Model,X)
    if nargin < 2 || isempty(X)
        Prob = zeros(0,1);
        return;
    end
    if isempty(Model) || ~isfield(Model,'Mu')
        Prob = 0.5*ones(size(X,1),1);
        return;
    end

    Xn = (double(X)-Model.Mu)./Model.Sigma;
    H  = tanh(Xn*Model.W1 + repmat(Model.b1,size(Xn,1),1));
    Z  = H*Model.W2 + Model.b2;
    if isfield(Model,'Temperature')
        Z = Z./max(Model.Temperature,1e-6);
    end
    Prob = min(max(1./(1+exp(-Z)),1e-6),1-1e-6);
end

function Temperature = FitBoundaryTemperature(Logit,Label)
    Label = double(Label(:) > 0);
    Logit = double(Logit(:));
    Grid = linspace(0.5,5,19);
    Loss = inf(numel(Grid),1);
    for i = 1 : numel(Grid)
        P = min(max(1./(1+exp(-Logit./Grid(i))),1e-6),1-1e-6);
        Loss(i) = -mean(Label.*log(P) + (1-Label).*log(1-P));
    end
    [~,best] = min(Loss);
    Temperature = Grid(best);
end

%% ========== Boundary offspring ==========

function [Offspring,Evidence] = GenerateBoundaryOffspring(Problem,B,PopulationC,PopulationU,W,Model,rho)
    Evidence = B([]);
    Budget = min(max(0,floor(rho*Problem.N)),max(0,Problem.maxFE-Problem.FE));
    if Budget <= 0
        Offspring = B([]);
        return;
    end

    AnchorPool = KeepUniquePopulation([B,PopulationC,PopulationU]);
    if isempty(AnchorPool)
        Offspring = B([]);
        return;
    end

    Support = AnchorPool;
    Pair = BuildBoundaryPairs(AnchorPool,W,max(Problem.N,2*Budget));
    if isempty(Pair.FeasibleDec)
        Offspring = B([]);
        return;
    end

    PairCount = min(size(Pair.FeasibleDec,1),max(1,ceil(Budget/2)));
    FirstDec = 0.5*(Pair.FeasibleDec(1:PairCount,:) + Pair.InfeasibleDec(1:PairCount,:));
    FirstDec = ClipDecisionRows(FirstDec,Problem);
    FirstEval = Problem.Evaluation(FirstDec);

    Remaining = max(0,min([PairCount,Budget-PairCount,Problem.maxFE-Problem.FE]));
    if Remaining > 0
        SecondDec = zeros(Remaining,size(FirstDec,2));
        for i = 1 : Remaining
            if all(FirstEval(i).cons <= 0)
                SecondDec(i,:) = 0.5*(double(FirstEval(i).decs) + Pair.InfeasibleDec(i,:));
            else
                SecondDec(i,:) = 0.5*(Pair.FeasibleDec(i,:) + double(FirstEval(i).decs));
            end
        end
        CandidateDec = ClipDecisionRows(SecondDec,Problem);
        EvaluatedCandidate = Problem.Evaluation(CandidateDec);
    else
        CandidateDec = FirstDec;
        EvaluatedCandidate = FirstEval;
    end

    Evidence = KeepUniquePopulation([FirstEval,EvaluatedCandidate]);
    if isempty(CandidateDec)
        Offspring = B([]);
        return;
    end

    CandidateSupportBase = BuildOppositeSupportBase( ...
        KeepUniquePopulation([Support,FirstEval,EvaluatedCandidate]),W);
    CandidateMeta = BuildBoundaryMeta(EvaluatedCandidate,W,Model,PopulationC,PopulationU,CandidateSupportBase);
    Relevant = BoundaryEligible(CandidateMeta,~isempty(Model),'evidence');
    if ~any(Relevant)
        Offspring = B([]);
        return;
    end
    if isempty(W)
        ord = SortByBoundaryKey(CandidateMeta,find(Relevant));
        Pick = ord(1:min(Budget,numel(ord)));
    else
        Ranked = cell(size(W,1),1);
        for s = 1 : size(W,1)
            idx = find(CandidateMeta.sector == s & Relevant);
            if isempty(idx)
                continue;
            end
            Ranked{s} = SortByBoundaryKey(CandidateMeta,idx);
        end
        Order = SectorRoundRobinPick(Ranked,Budget);
        Pick = Order(:,2);
        if numel(Pick) < min(Budget,size(CandidateDec,1))
            Rest = setdiff(find(Relevant),Pick,'stable');
            ord = SortByBoundaryKey(CandidateMeta,Rest);
            Need = min(Budget-numel(Pick),numel(ord));
            PickCount = numel(Pick);
            ExpandedPick = zeros(PickCount+Need,1);
            ExpandedPick(1:PickCount) = Pick;
            ExpandedPick(PickCount+1:end) = ord(1:Need);
            Pick = ExpandedPick;
        end
    end

    if isempty(Pick)
        Offspring = B([]);
    else
        Offspring = EvaluatedCandidate(Pick);
    end
end

function Dec = ClipDecisionRows(Dec,Problem)
    if isempty(Dec)
        return;
    end
    Lower = repmat(Problem.lower,size(Dec,1),1);
    Upper = repmat(Problem.upper,size(Dec,1),1);
    Dec = min(max(Dec,Lower),Upper);
end

%% ========== Environmental selection ==========

function Population = EnvironmentalSelectionC(Population,N,W,Model,B,RecentBoundaryOff,PopulationU)
    Population = KeepUniquePopulation(Population);
    if isempty(Population)
        return;
    end

    Feasible = FilterFeasiblePopulation(Population);
    if numel(Feasible) >= N
        Population = ObjectiveSelectionWithLastSectorTruncation(Feasible,N,W);
        return;
    end

    Next = Feasible;
    Need = N - numel(Next);
    Infeasible = Population(any(Population.cons>0,2));
    if Need > 0 && ~isempty(Infeasible)
        Next = [Next,SelectInfeasibleByBoundaryMeta( ...
            Infeasible,Need,W,Model,Feasible,PopulationU,B,RecentBoundaryOff)];
    end

    if numel(Next) < N
        Rest = RemovePopulationByDecision(Population,Next);
        Rest = ObjectiveSelectionWithLastSectorTruncation(Rest,min(N-numel(Next),numel(Rest)),W);
        Next = [Next,Rest];
    end
    Population = PadPopulation(Next,N);
end

function Pick = SelectInfeasibleByBoundaryMeta( ...
    Population,N,W,Model,PopulationC,PopulationU,B,RecentBoundaryOff)
    Pick = Population([]);
    if isempty(Population) || N <= 0
        return;
    end

    Population = Population(any(Population.cons>0,2));
    if isempty(Population)
        return;
    end
    N = min(N,numel(Population));

    Support = KeepUniquePopulation([B,RecentBoundaryOff,PopulationC,PopulationU]);
    SupportBase = BuildOppositeSupportBase(Support,W);
    Meta = BuildBoundaryMeta(Population,W,Model,PopulationC,PopulationU,SupportBase);
    Trusted = FindSupportedBoundaryCandidates(Meta,W,~isempty(Model)) & Meta.feasible == 0;
    if ~any(Trusted)
        return;
    end
    if isempty(W)
        ord = SortBySelectionKey(Meta,find(Trusted));
        N = min(N,numel(ord));
        Pick = Population(ord(1:N));
        return;
    end

    Ranked = cell(size(W,1),1);
    for s = 1 : size(W,1)
        idx = find(Meta.sector == s & Trusted);
        if isempty(idx)
            continue;
        end
        Ranked{s} = SortBySelectionKey(Meta,idx);
    end
    Order = SectorRoundRobinPick(Ranked,N);
    if isempty(Order)
        ord = SortBySelectionKey(Meta,find(Trusted));
        N = min(N,numel(ord));
        Pick = Population(ord(1:N));
    else
        Pick = Population(Order(:,2));
    end
end

function Trusted = FindSupportedBoundaryCandidates(Meta,W,ModelReady)
    Trusted = false(numel(Meta.sector),1);
    if isempty(Meta.sector)
        return;
    end
    if nargin < 3
        ModelReady = true;
    end
    Trusted = BoundaryEligible(Meta,ModelReady,'evidence');
end

function Population = EnvironmentalSelectionU(Population,N,W)
    Population = KeepUniquePopulation(Population);
    Population = ObjectiveSelectionWithLastSectorTruncation(Population,min(N,numel(Population)),W);
    Population = PadPopulation(Population,N);
end

function Population = ObjectiveSelectionWithLastSectorTruncation(Population,N,W)
    if isempty(Population)
        return;
    end
    if N <= 0
        Population = Population([]);
        return;
    end

    N = min(N,numel(Population));
    if numel(Population) <= N
        return;
    end

    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo < MaxFNo;
    Last = find(FrontNo == MaxFNo);
    Need = N - sum(Next);
    if Need > 0 && ~isempty(Last)
        PickLast = SelectLastFrontBySector(Population(Last),Need,W,Population.objs);
        Next(Last(PickLast)) = true;
    end
    Population = Population(Next);
end

function Pick = SelectLastFrontBySector(LastPopulation,N,W,RefObj)
    Pick = zeros(0,1);
    if isempty(LastPopulation) || N <= 0
        return;
    end

    N = min(N,numel(LastPopulation));
    if isempty(W)
        CrowdDis = CrowdingDistance(LastPopulation.objs,ones(1,numel(LastPopulation)));
        [~,ord]  = sort(CrowdDis,'descend');
        Pick     = ord(1:N);
        return;
    end

    Sector   = AssociateSectorsLocal(LastPopulation.objs,W,RefObj);
    Scalar   = ComputeSectorScalar(LastPopulation.objs,W,RefObj,Sector);
    CrowdDis = CrowdingDistance(LastPopulation.objs,ones(1,numel(LastPopulation)));
    Ranked   = cell(size(W,1),1);
    for s = 1 : size(W,1)
        idx = find(Sector == s);
        if isempty(idx)
            continue;
        end
        Key = [ColumnVector(-CrowdDis(idx)),ColumnVector(Scalar(idx)),ColumnVector(idx)];
        [~,ord] = sortrows(Key,[1 2 3]);
        Ranked{s} = idx(ord);
    end
    Order = SectorRoundRobinPick(Ranked,N);
    Pick  = Order(:,2);
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
    Flag = sum(max(0,Population.cons),2);
end

function [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population)
    [FrontNo,~] = NDSort(Population.objs,numel(Population));
    CrowdDis    = CrowdingDistance(Population.objs,FrontNo);
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

function D = ResolveDecisionDimension(varargin)
    D = 0;
    for i = 1 : nargin
        Population = varargin{i};
        if ~isempty(Population)
            D = size(Population.decs,2);
            return;
        end
    end
end

function Keep = KeepLatestDecisionRowsLocal(Dec)
    if isempty(Dec)
        Keep = zeros(0,1);
        return;
    end
    [~,Keep] = unique(double(Dec),'rows','last');
    Keep = sort(Keep);
end

function Value = ColumnVector(Value)
    Value = Value(:);
end

function Meta = InitBoundaryMeta(Count)
    Meta = struct( ...
        'sector',zeros(Count,1), ...
        'feasible',zeros(Count,1), ...
        'prob',0.5*ones(Count,1), ...
        'margin',ones(Count,1), ...
        'oppSupport',zeros(Count,1), ...
        'localPairExists',false(Count,1), ...
        'betweenScoreObj',inf(Count,1), ...
        'shellDistDec',inf(Count,1), ...
        'gapDec',inf(Count,1), ...
        'gapObj',inf(Count,1), ...
        'boundaryGap',inf(Count,1), ...
        'objScore',zeros(Count,1), ...
        'score',ones(Count,1));
end

function Meta = ComposeBoundaryMeta( ...
    Sector,Feasible,Prob,OppSupport,ObjScore,LocalPairExists,ShellDistDec,BetweenScoreObj)
    Count = max([numel(Sector),numel(Feasible),numel(Prob),numel(OppSupport), ...
        numel(ObjScore),numel(LocalPairExists),numel(ShellDistDec),numel(BetweenScoreObj),0]);
    Meta = InitBoundaryMeta(Count);
    if Count == 0
        return;
    end

    Sector = MatchLength(ColumnVector(Sector),Count,0);
    Feasible = MatchLength(ColumnVector(Feasible),Count,0);
    Prob = MatchLength(ColumnVector(Prob),Count,0.5);
    OppSupport = MatchLength(ColumnVector(OppSupport),Count,0);
    ObjScore = MatchLength(ColumnVector(ObjScore),Count,0);
    LocalPairExists = MatchLength(logical(ColumnVector(LocalPairExists)),Count,false);
    ShellDistDec = MatchLength(ColumnVector(ShellDistDec),Count,inf);
    BetweenScoreObj = MatchLength(ColumnVector(BetweenScoreObj),Count,inf);
    Margin = 2*abs(Prob - 0.5);

    Meta.sector     = Sector;
    Meta.feasible   = Feasible;
    Meta.prob       = Prob;
    Meta.margin     = Margin;
    Meta.oppSupport = OppSupport;
    Meta.localPairExists = LocalPairExists;
    Meta.betweenScoreObj = BetweenScoreObj;
    Meta.shellDistDec = ShellDistDec;
    Meta.gapDec     = ShellDistDec;
    Meta.gapObj     = BetweenScoreObj;
    Meta.boundaryGap = ShellDistDec + 0.25*BetweenScoreObj;
    Meta.objScore   = ObjScore;
    Meta.score      = 0.55*NormalizeFiniteScore(Meta.boundaryGap) + ...
        0.25*Margin + 0.10*NormalizeFiniteScore(ShellDistDec) + 0.10*ObjScore;
end

function Score = NormalizeFiniteScore(Value)
    Score = ones(size(Value));
    Mask = isfinite(Value);
    if any(Mask)
        Score(Mask) = NormalizeRange(Value(Mask));
    end
end

function Order = SortByBoundaryKey(Meta,idx,Source)
    idx = ColumnVector(idx);
    if isempty(idx)
        Order = idx;
        return;
    end
    if nargin < 3 || isempty(Source)
        Source = 5*ones(numel(Meta.sector),1);
    end
    Source = MatchLength(Source,numel(Meta.sector),5);
    Key = [ColumnVector(Meta.boundaryGap(idx)),ColumnVector(Meta.gapDec(idx)), ...
        ColumnVector(Meta.gapObj(idx)),ColumnVector(SourcePriority(Source(idx))), ...
        ColumnVector(Meta.margin(idx)),ColumnVector(Meta.objScore(idx)),ColumnVector(idx)];
    [~,ord] = sortrows(Key,[1 2 3 4 5 6 7]);
    Order = idx(ord);
end

function Order = SortBySelectionKey(Meta,idx)
    idx = ColumnVector(idx);
    if isempty(idx)
        Order = idx;
        return;
    end
    Key = [ColumnVector(Meta.boundaryGap(idx)),ColumnVector(Meta.gapDec(idx)), ...
        ColumnVector(Meta.gapObj(idx)),ColumnVector(Meta.margin(idx)), ...
        ColumnVector(Meta.objScore(idx)),ColumnVector(idx)];
    [~,ord] = sortrows(Key,[1 2 3 4 5 6]);
    Order = idx(ord);
end

%% ========== Sector and distance utilities ==========

function Z = ComputeIdealPoint(varargin)
    Z = zeros(0,0);
    for i = 1 : nargin
        Population = varargin{i};
        if isempty(Population)
            continue;
        end
        if isempty(Z)
            Z = min(Population.objs,[],1);
        else
            Z = min([Z;min(Population.objs,[],1)],[],1);
        end
    end
end

function [Sector,Scalar] = AssociateSectorsLocal(Obj,W,RefObj)
    if isempty(Obj)
        Sector = zeros(0,1);
        if nargout > 1
            Scalar = zeros(0,1);
        end
        return;
    end
    if isempty(W)
        Sector = ones(size(Obj,1),1);
        if nargout > 1
            Scalar = zeros(size(Obj,1),1);
        end
        return;
    end

    if nargin < 3 || isempty(RefObj)
        Z = min(Obj,[],1);
    else
        Z = min(RefObj,[],1);
    end
    Shift = max(Obj - Z,0);
    ShiftNorm = sqrt(sum(Shift.^2,2));
    ShiftNorm(ShiftNorm < 1e-12) = 1;
    WNorm = sqrt(sum(W.^2,2));
    WNorm(WNorm < 1e-12) = 1;
    Cosine = (Shift./ShiftNorm)*(W./WNorm)';
    [~,Sector] = max(Cosine,[],2);
    if nargout > 1
        Scalar = ComputeSectorScalar(Obj,W,RefObj,Sector);
    end
end

function Scalar = ComputeSectorScalar(Obj,W,RefObj,Sector)
    if isempty(Obj)
        Scalar = zeros(0,1);
        return;
    end
    if nargin < 4 || isempty(Sector)
        if isempty(W)
            Sector = ones(size(Obj,1),1);
        else
            Sector = AssociateSectorsLocal(Obj,W,RefObj);
        end
    end
    if nargin < 3 || isempty(RefObj)
        Z = min(Obj,[],1);
    else
        Z = min(RefObj,[],1);
    end
    Shift = max(Obj - Z,0);
    if isempty(W)
        Scalar = sum(Shift,2);
        return;
    end

    Scalar = zeros(size(Obj,1),1);
    MeanW = max(mean(W,1),1e-6);
    for i = 1 : size(Obj,1)
        if Sector(i) > 0 && Sector(i) <= size(W,1)
            Weight = max(W(Sector(i),:),1e-6);
        else
            Weight = MeanW;
        end
        Scalar(i) = max(Shift(i,:).*Weight,[],2);
    end
end

function OppSupport = ComputeOppositeSupportFromCache( ...
    CandidateDec,CandidateLabel,Sector,SupportCache)
    CandidateDec = double(CandidateDec);
    CandidateLabel = ColumnVector(CandidateLabel);
    Sector = ColumnVector(Sector);
    OppSupport = zeros(size(CandidateDec,1),1);
    if isempty(CandidateDec) || isempty(SupportCache.Dec)
        return;
    end

    for i = 1 : size(CandidateDec,1)
        CandidateRow = reshape(CandidateDec(i,:),1,[]);
        Local = ColumnVector(ResolveLocalSectorSet(Sector(i),SupportCache.Neighbors));
        Mask = ColumnVector((SupportCache.Label ~= CandidateLabel(i)) & ismember(SupportCache.Sector,Local));
        if any(Mask)
            LocalSupportDec = SupportCache.Dec(Mask,:);
            Delta = LocalSupportDec - repmat(CandidateRow,size(LocalSupportDec,1),1);
            Dist = sqrt(sum(Delta.^2,2));
            OppSupport(i) = exp(-min(Dist)/max(SupportCache.Tau,1e-6));
        end
    end
end

function Value = MatchLength(Value,Count,Fill)
    Value = ColumnVector(Value);
    if isempty(Value)
        Value = Fill*ones(Count,1);
    elseif numel(Value) > Count
        Value = Value(1:Count);
    elseif numel(Value) < Count
        Value(end+1:Count,1) = Value(end);
    end
end

function Value = MatchLogicalLength(Value,Count)
    Value = logical(ColumnVector(Value));
    if numel(Value) > Count
        Value = Value(1:Count);
    elseif numel(Value) < Count
        Value(end+1:Count,1) = false;
    end
end

function Value = NormalizeRange(Value)
    if isempty(Value)
        return;
    end
    Lower = min(Value);
    Upper = max(Value);
    if Upper - Lower < 1e-12
        Value = zeros(size(Value));
    else
        Value = (Value - Lower)./(Upper - Lower);
    end
end

function Pick = SectorRoundRobinPick(Ranked,MaxPick)
    Pick = zeros(MaxPick,2);
    if isempty(Ranked) || MaxPick <= 0
        Pick = zeros(0,2);
        return;
    end

    Ptr = ones(numel(Ranked),1);
    PickCount = 0;
    while PickCount < MaxPick
        Changed = false;
        for s = 1 : numel(Ranked)
            if Ptr(s) <= numel(Ranked{s})
                PickCount = PickCount + 1;
                Pick(PickCount,:) = [s,Ranked{s}(Ptr(s))];
                Ptr(s) = Ptr(s) + 1;
                Changed = true;
                if PickCount >= MaxPick
                    break;
                end
            end
        end
        if ~Changed
            break;
        end
    end
    Pick = Pick(1:PickCount,:);
end

function Neighbors = BuildSectorNeighbors(W,NeighborCount)
    K = size(W,1);
    Neighbors = cell(K,1);
    if K <= 1 || NeighborCount <= 0
        return;
    end

    WNorm = sqrt(sum(W.^2,2));
    WNorm(WNorm < 1e-12) = 1;
    Wn = W./WNorm;
    Cosine = Wn*Wn';
    Cosine(1:K+1:end) = -inf;
    for i = 1 : K
        [~,ord] = sort(Cosine(i,:),'descend');
        Neighbors{i} = ord(1:min(NeighborCount,K-1));
    end
end

function Local = ResolveLocalSectorSet(Sector,Neighbors)
    if isempty(Neighbors)
        Local = 1;
        return;
    end
    K = numel(Neighbors);
    if Sector <= 0 || Sector > K
        Local = (1:K)';
        return;
    end
    Local = unique([Sector;Neighbors{Sector}(:)],'stable');
end

function Scale = ResolveDistanceScale(Dec)
    if isempty(Dec) || size(Dec,1) <= 1
        Scale = 1;
        return;
    end

    SampleCount = min(32,size(Dec,1));
    Index = randperm(size(Dec,1),SampleCount);
    Sample = double(Dec(Index,:));
    Dist = zeros(SampleCount*(SampleCount-1)/2,1);
    DistCount = 0;
    for i = 1 : SampleCount-1
        Delta = Sample(i+1:end,:) - Sample(i,:);
        Count = size(Delta,1);
        Dist(DistCount+1:DistCount+Count) = sqrt(sum(Delta.^2,2));
        DistCount = DistCount + Count;
    end
    Dist = Dist(1:DistCount);
    Dist = Dist(isfinite(Dist) & Dist > 0);
    if isempty(Dist)
        Scale = 1;
    else
        Scale = median(Dist);
    end
end
