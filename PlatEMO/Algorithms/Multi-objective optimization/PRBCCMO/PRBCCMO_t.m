classdef PRBCCMO_t < ALGORITHM
% <2026> <multi> <real> <constrained>
% PRBCCMO_t
% Traced PRBCCMO with compact CSV diagnostics
%
% hidden --- 20   --- Hidden neurons of the boundary MLP
% epoch  --- 20   --- Training epochs per update
% lr     --- 0.01 --- Learning rate
% kappa  --- 10   --- Maximum boundary samples kept per sector
% rho    --- 0.12 --- Boundary evaluation budget ratio
%
% PRBCCMO_t mirrors the boundary-band PRBCCMO and writes CSV-only
% diagnostics under Data/PRBCCMO_t/<run-folder>. No MAT files are saved.

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
            Generation  = 0;

            Observer = InitObserver(Algorithm,Problem,[hidden,epoch,lr,kappa,rho]);
            Algorithm.metric.analysis_folder       = Observer.folder;
            Algorithm.metric.analysis_meta_csv     = Observer.meta_file;
            Algorithm.metric.analysis_summary_csv  = Observer.summary_file;
            Algorithm.metric.analysis_boundary_csv = Observer.boundary_file;
            Algorithm.metric.analysis_archive_csv  = Observer.archive_file;
            Algorithm.metric.analysis_objective_csv = Observer.objective_file;
            Algorithm.metric.analysis_mlp_csv      = Observer.mlp_file;

            while Algorithm.NotTerminated(PopulationC)
                Generation = Generation + 1;

                OffspringC = GenerateDEOffspring(Problem,PopulationC,true);
                OffspringU = GenerateDEOffspring(Problem,PopulationU,false);

                SeedB = B;
                SeedSupport = KeepUniquePopulation([SeedB,PopulationC,PopulationU]);
                SeedSupportBase = BuildOppositeSupportBase(SeedSupport,W);
                SeedMeta = BuildBoundaryMeta(SeedB,W,Model,PopulationC,PopulationU,SeedSupportBase);

                [BoundaryOff,BoundaryEvidence,BoundaryDiag] = GenerateBoundaryOffspring( ...
                    Problem,B,PopulationC,PopulationU,W,Model,rho);

                QC = KeepUniquePopulation([PopulationC,OffspringC,OffspringU,BoundaryOff]);
                QU = KeepUniquePopulation([PopulationU,OffspringC,OffspringU,BoundaryOff]);
                [PopulationC,SelectionDiag] = EnvironmentalSelectionC( ...
                    QC,Problem.N,W,Model,B,RecentBoundaryOff,PopulationU);
                PopulationU = EnvironmentalSelectionU(QU,Problem.N,W);

                BoundarySurvival = ComputeBoundarySurvival(BoundaryOff,PopulationC,PopulationU);
                RecentBoundaryOff = BoundaryOff;
                B = UpdateBoundaryArchive(B,RecentBoundaryOff,PopulationC,PopulationU,W,Model,kappa);
                [Model,ModelState,MLPDiag] = UpdateBoundaryModelIfNeeded( ...
                    Model,ModelState,B,BoundaryEvidence,PopulationC,PopulationU,W, ...
                    hidden,epoch,lr,Generation,Problem,Problem.FE);
                Observer = LogMLPEvent(Observer,MLPDiag);

                Support = KeepUniquePopulation([B,RecentBoundaryOff,PopulationC,PopulationU]);
                SupportBase = BuildOppositeSupportBase(Support,W);
                ArchiveMeta = BuildBoundaryMeta(B,W,Model,PopulationC,PopulationU,SupportBase);
                Observer = LogGenerationDiagnostics( ...
                    Observer,Generation,Problem,W,PopulationC,PopulationU, ...
                    OffspringC,OffspringU,BoundaryOff,BoundaryEvidence,BoundaryDiag,BoundarySurvival, ...
                    SeedB,SeedMeta,B,ArchiveMeta,MLPDiag,SelectionDiag,Model);
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
        Need = min(MaxCount,numel(Order)) - Count;
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

function [Model,ModelState,Diag] = UpdateBoundaryModelIfNeeded( ...
    Model,ModelState,B,BoundaryEvidence,PopulationC,PopulationU,W,hidden,epoch,lr,Generation,Problem,CurrentFE)

    MaxTrain = max(6*Problem.N,4*size(W,1));
    Candidates = BuildTrainingCandidateSet(B,BoundaryEvidence,PopulationC,PopulationU);
    [ModelState.trainBuffer,BufferDiag,ModelState.lastCandidateDec] = UpdateTrainingBuffer( ...
        ModelState.trainBuffer,Candidates,W,MaxTrain,CurrentFE,ModelState.lastCandidateDec);
    [Dataset,DataDiag] = BuildBoundaryDatasetFromBuffer(ModelState.trainBuffer,W,Problem.N);
    [NeedTrain,Summary] = ShouldRetrainBoundaryModel( ...
        Problem,Model,ModelState,Dataset,BufferDiag,CurrentFE);
    Diag = InitMLPDiag(Generation,CurrentFE,ModelState,Summary,~isempty(Model));
    Diag.need_train = double(NeedTrain);
    Diag.train_size       = DataDiag.train_size;
    Diag.pos_count        = DataDiag.pos_count;
    Diag.neg_count        = DataDiag.neg_count;
    Diag.sector_coverage  = DataDiag.sector_coverage;
    Diag.train_dual_sectors = DataDiag.mixed_sector_count;
    Diag.src_b              = DataDiag.src_b;
    Diag.src_recent_boundary = DataDiag.src_recent_boundary;
    Diag.src_boundary_evidence = DataDiag.src_boundary_evidence;
    Diag.src_pop_c          = DataDiag.src_pop_c;
    Diag.src_pop_u          = DataDiag.src_pop_u;
    Diag.feasible_endpoint_count = DataDiag.pos_count;
    Diag.pair_count         = DataDiag.neg_count;
    Diag.mean_pair_dist     = DataDiag.mean_opp_dist;
    if ~NeedTrain
        return;
    end

    Diag.can_train          = double(Summary.ready);
    if ~isempty(Dataset.Dec)
        Diag.stats_before = EvaluateBinaryPredictions(Model,Dataset.Dec,Dataset.Label);
        Diag.stats_after  = Diag.stats_before;
    end
    if isempty(Dataset.Dec) || numel(unique(Dataset.Label)) < 2
        return;
    end

    Mode = ResolveBoundaryTrainingMode(Model,Dataset,hidden,Summary);
    Model = TrainBoundaryMLP(Model,Dataset,hidden,epoch,lr,Mode);
    Diag.trained           = 1;
    Diag.model_ready_after = double(~isempty(Model));
    Diag.stats_after       = EvaluateBinaryPredictions(Model,Dataset.Dec,Dataset.Label);
    ModelState.lastTrainSize      = size(Dataset.Dec,1);
    ModelState.lastMixedSectorMask = Summary.mixed_sector_mask;
    ModelState.lastBoundaryCount  = Summary.boundary_count;
    ModelState.lastTrainFE        = CurrentFE;
    ModelState.lastValidationBrier = Diag.stats_after.brier;
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
    Stats = EvaluateBinaryPredictions(Model,Dataset.Dec,Dataset.Label);
    MixedMask = BufferDiag.mixed_sector_mask;
    PrevMask = MatchLogicalLength(ModelState.lastMixedSectorMask,numel(MixedMask));
    Summary = struct( ...
        'ready',false, ...
        'new_sample_ratio',BufferDiag.new_sample_ratio, ...
        'mixed_sector_gain',sum(MixedMask & ~PrevMask), ...
        'mixed_sector_mask',MixedMask, ...
        'validation_brier',Stats.brier, ...
        'boundary_count',BufferDiag.buffer_size, ...
        'feasible_endpoint_count',BufferDiag.pos_count, ...
        'pair_count',BufferDiag.neg_count, ...
        'mean_pair_dist',BufferDiag.mean_opp_dist);
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
    Diag.pos_count = sum(Buffer.Label == 1);
    Diag.neg_count = sum(Buffer.Label == 0);
    Diag.mean_opp_dist = MeanOrNaN(Buffer.OppDist);
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
        'mixed_sector_mask',false(K,1), ...
        'pos_count',sum(Buffer.Label == 1), ...
        'neg_count',sum(Buffer.Label == 0), ...
        'mean_opp_dist',MeanOrNaN(Buffer.OppDist));
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

function [Dataset,Diag] = BuildBoundaryDatasetFromBuffer(Buffer,W,N)
    Dataset = struct('Dec',zeros(0,0),'Label',zeros(0,1));
    Diag = struct( ...
        'train_size',0, ...
        'pos_count',0, ...
        'neg_count',0, ...
        'sector_coverage',0, ...
        'mixed_sector_count',0, ...
        'mean_opp_dist',NaN, ...
        'src_b',0, ...
        'src_recent_boundary',0, ...
        'src_boundary_evidence',0, ...
        'src_pop_c',0, ...
        'src_pop_u',0);
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
    Dataset.Source = Buffer.Source(Ranked);
    Dataset.Sector = Buffer.Sector(Ranked);
    Dataset.OppDist = Buffer.OppDist(Ranked);
    Diag.train_size = size(Dataset.Dec,1);
    Diag.pos_count = sum(Dataset.Label == 1);
    Diag.neg_count = sum(Dataset.Label == 0);
    Diag.sector_coverage = numel(unique(Dataset.Sector(Dataset.Sector > 0)));
    Diag.mixed_sector_count = countDualSectors(Dataset.Sector,Dataset.Label);
    Diag.mean_opp_dist = MeanOrNaN(Dataset.OppDist);
    Diag.src_b = sum(Dataset.Source == 1);
    Diag.src_recent_boundary = sum(Dataset.Source == 2);
    Diag.src_boundary_evidence = Diag.src_recent_boundary;
    Diag.src_pop_c = sum(Dataset.Source == 3);
    Diag.src_pop_u = sum(Dataset.Source == 4);
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

function [Offspring,Evidence,Diag] = GenerateBoundaryOffspring(Problem,B,PopulationC,PopulationU,W,Model,rho)
    Diag = InitBoundaryDiag(0,Problem.FE,0);
    Evidence = B([]);
    Budget = min(max(0,floor(rho*Problem.N)),max(0,Problem.maxFE-Problem.FE));
    Diag.budget = Budget;
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
    Diag.attempts = PairCount;

    CandidateSector = Pair.Sector(1:PairCount);
    CandidateFeasibleEndpointProb = PredictBoundaryMLP(Model,Pair.FeasibleDec(1:PairCount,:));
    CandidateInfeasibleEndpointProb = PredictBoundaryMLP(Model,Pair.InfeasibleDec(1:PairCount,:));
    CandidateFeasibleEndpointScore = Pair.BoundaryGap(1:PairCount);
    CandidateInfeasibleEndpointScore = Pair.BoundaryGap(1:PairCount);
    CandidatePairDist = Pair.BoundaryGap(1:PairCount);
    CandidateFeasibleEndpointSide = ones(PairCount,1);
    CandidateInfeasibleEndpointSide = zeros(PairCount,1);
    CandidatePairPredCross = double((CandidateFeasibleEndpointProb-0.5).*(CandidateInfeasibleEndpointProb-0.5) <= 0);

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
        CandidateSector = CandidateSector(1:Remaining);
        CandidateFeasibleEndpointProb = CandidateFeasibleEndpointProb(1:Remaining);
        CandidateFeasibleEndpointScore = CandidateFeasibleEndpointScore(1:Remaining);
        CandidateInfeasibleEndpointProb = CandidateInfeasibleEndpointProb(1:Remaining);
        CandidateInfeasibleEndpointScore = CandidateInfeasibleEndpointScore(1:Remaining);
        CandidatePairDist = CandidatePairDist(1:Remaining);
        CandidateFeasibleEndpointSide = CandidateFeasibleEndpointSide(1:Remaining);
        CandidateInfeasibleEndpointSide = CandidateInfeasibleEndpointSide(1:Remaining);
        CandidatePairPredCross = CandidatePairPredCross(1:Remaining);
        CandidateSegCrossDist = 0.25*CandidatePairDist;
    else
        CandidateDec = FirstDec;
        EvaluatedCandidate = FirstEval;
        CandidateSegCrossDist = 0.5*CandidatePairDist;
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

    Diag.selected = numel(Pick);
    if isempty(Pick)
        Offspring = B([]);
        return;
    end

    Offspring = EvaluatedCandidate(Pick);
    Diag.events = buildBoundaryEvents( ...
        Diag.events,Problem.FE,Pick,CandidateSector,CandidateFeasibleEndpointSide,CandidateInfeasibleEndpointSide, ...
        CandidateFeasibleEndpointProb,CandidateFeasibleEndpointScore, ...
        CandidateInfeasibleEndpointProb,CandidateInfeasibleEndpointScore, ...
        CandidatePairPredCross,CandidatePairDist,CandidateSegCrossDist,CandidateMeta,Offspring);
end

function Dec = ClipDecisionRows(Dec,Problem)
    if isempty(Dec)
        return;
    end
    Lower = repmat(Problem.lower,size(Dec,1),1);
    Upper = repmat(Problem.upper,size(Dec,1),1);
    Dec = min(max(Dec,Lower),Upper);
end

function Rows = buildBoundaryEvents(Rows,FE,Pick,Sector,FeasibleEndpointSide,InfeasibleEndpointSide, ...
    FeasibleEndpointProb,FeasibleEndpointScore,InfeasibleEndpointProb,InfeasibleEndpointScore, ...
    PairPredCross,PairDist,SegCrossDist,CandidateMeta,Offspring)
    Rows = cell(numel(Pick),24);
    for i = 1 : numel(Pick)
        idx = Pick(i);
        PredLabel = double(CandidateMeta.prob(idx) >= 0.5);
        Feasible = double(all(Offspring(i).cons<=0,2));
        OppDist = SupportDistance(CandidateMeta.oppSupport(idx));
        Rows(i,:) = { ...
            NaN,FE,i,Sector(idx), ...
            FeasibleEndpointSide(idx),InfeasibleEndpointSide(idx),FeasibleEndpointProb(idx),FeasibleEndpointScore(idx), ...
            InfeasibleEndpointProb(idx),InfeasibleEndpointScore(idx),PairPredCross(idx),PairDist(idx),SegCrossDist(idx), ...
            CandidateMeta.prob(idx),PredLabel,CandidateMeta.margin(idx),CandidateMeta.oppSupport(idx), ...
            OppDist,CandidateMeta.objScore(idx),CandidateMeta.score(idx),Feasible,double(PredLabel == Feasible),0,0};
    end
end

%% ========== Environmental selection ==========

function [Population,Diag] = EnvironmentalSelectionC(Population,N,W,Model,B,RecentBoundaryOff,PopulationU)
    Population = KeepUniquePopulation(Population);
    Diag = InitInfeasibleSelectionDiag();
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
        [Selected,Diag] = SelectInfeasibleByBoundaryMeta( ...
            Infeasible,Need,W,Model,Feasible,PopulationU,B,RecentBoundaryOff);
        Next = [Next,Selected];
    elseif ~isempty(Infeasible)
        Diag.pool_size = numel(Infeasible);
    end

    if numel(Next) < N
        Rest = RemovePopulationByDecision(Population,Next);
        Rest = ObjectiveSelectionWithLastSectorTruncation(Rest,min(N-numel(Next),numel(Rest)),W);
        Next = [Next,Rest];
    end
    Population = PadPopulation(Next,N);
end

function [Pick,Diag] = SelectInfeasibleByBoundaryMeta( ...
    Population,N,W,Model,PopulationC,PopulationU,B,RecentBoundaryOff)

    Pick = Population([]);
    Diag = InitInfeasibleSelectionDiag();
    if isempty(Population) || N <= 0
        return;
    end

    Population = Population(any(Population.cons>0,2));
    Diag.pool_size = numel(Population);
    if isempty(Population)
        return;
    end
    N = min(N,numel(Population));

    Support = KeepUniquePopulation([B,RecentBoundaryOff,PopulationC,PopulationU]);
    SupportBase = BuildOppositeSupportBase(Support,W);
    Meta = BuildBoundaryMeta(Population,W,Model,PopulationC,PopulationU,SupportBase);
    Diag.pool_mean_score       = MeanOrNaN(Meta.score);
    Diag.pool_mean_prob        = MeanOrNaN(Meta.prob);
    Diag.pool_mean_margin      = MeanOrNaN(Meta.margin);
    Diag.pool_mean_opp_support = MeanOrNaN(Meta.oppSupport);
    Diag.pool_mean_obj_score   = MeanOrNaN(Meta.objScore);
    Diag.pool_lowmargin_ratio  = LowMarginRatio(Meta.margin);

    Trusted = FindSupportedBoundaryCandidates(Meta,W,~isempty(Model)) & Meta.feasible == 0;
    if ~any(Trusted)
        return;
    end
    if isempty(W)
        PickIndex = SortBySelectionKey(Meta,find(Trusted));
        N = min(N,numel(PickIndex));
        PickIndex = PickIndex(1:N);
    else
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
            PickIndex = SortBySelectionKey(Meta,find(Trusted));
            N = min(N,numel(PickIndex));
            PickIndex = PickIndex(1:N);
        else
            PickIndex = Order(:,2);
        end
    end

    Pick = Population(PickIndex);
    Diag.selected_count = numel(PickIndex);
    Diag.selected_ratio = SafeDivide(Diag.selected_count,max(Diag.pool_size,1));
    Diag.selected_mean_score       = MeanOrNaN(Meta.score(PickIndex));
    Diag.selected_mean_prob        = MeanOrNaN(Meta.prob(PickIndex));
    Diag.selected_mean_margin      = MeanOrNaN(Meta.margin(PickIndex));
    Diag.selected_mean_opp_support = MeanOrNaN(Meta.oppSupport(PickIndex));
    Diag.selected_mean_obj_score   = MeanOrNaN(Meta.objScore(PickIndex));
    Diag.selected_lowmargin_ratio  = LowMarginRatio(Meta.margin(PickIndex));
    Diag.score_gain       = Diag.pool_mean_score - Diag.selected_mean_score;
    Diag.prob_gain        = Diag.selected_mean_prob - Diag.pool_mean_prob;
    Diag.margin_gain      = Diag.pool_mean_margin - Diag.selected_mean_margin;
    Diag.opp_support_gain = Diag.selected_mean_opp_support - Diag.pool_mean_opp_support;
    Diag.obj_score_gain   = Diag.pool_mean_obj_score - Diag.selected_mean_obj_score;
    Diag.lowmargin_ratio_gain = Diag.selected_lowmargin_ratio - Diag.pool_lowmargin_ratio;
    Diag.selected_sector_coverage = numel(unique(Meta.sector(PickIndex)));
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

%% ========== Observer ==========

function Observer = InitObserver(Algorithm,Problem,Params)
    RootDir    = fileparts(which('platemo'));
    BaseFolder = fullfile(RootDir,'Data','PRBCCMO_t');
    [~,~]      = mkdir(BaseFolder);
    [~,Token]  = fileparts(tempname(BaseFolder));
    RunFolder  = fullfile(BaseFolder,sprintf('%s_%s_run%d_%s', ...
        class(Algorithm),class(Problem),resolveRunId(Algorithm),Token));
    [~,~]      = mkdir(RunFolder);

    Observer = struct( ...
        'folder',RunFolder, ...
        'meta_file',fullfile(RunFolder,'run_meta.csv'), ...
        'summary_file',fullfile(RunFolder,'generation_summary.csv'), ...
        'boundary_file',fullfile(RunFolder,'boundary_event.csv'), ...
        'archive_file',fullfile(RunFolder,'archive_members.csv'), ...
        'objective_file',fullfile(RunFolder,'objective_snapshot.csv'), ...
        'mlp_file',fullfile(RunFolder,'mlp_events.csv'), ...
        'objective_marks',[0.05 0.10 0.25 0.30 0.50 0.70 0.75 0.90 1.00], ...
        'objective_logged',false(1,9));

    WriteCsvHeader(Observer.meta_file,{ ...
        'algorithm','problem','family','run','M','D','N','maxFE', ...
        'hidden','epoch','lr','kappa','rho','output_folder'});
    AppendCsvRows(Observer.meta_file,{ ...
        class(Algorithm),class(Problem),familyOfProblem(class(Problem)), ...
        resolveRunId(Algorithm),Problem.M,Problem.D,Problem.N,Problem.maxFE, ...
        Params(1),Params(2),Params(3),Params(4),Params(5),RunFolder});

    WriteCsvHeader(Observer.summary_file,{ ...
        'generation','fe','fe_ratio', ...
        'popc_feasible_ratio','popu_feasible_ratio', ...
        'offspringc_feasible_ratio','offspringu_feasible_ratio', ...
        'model_ready','model_trained', ...
        'boundary_budget','boundary_candidates','boundary_selected','boundary_off_count', ...
        'boundary_evidence_count','boundary_evidence_feasible_ratio','boundary_feasible_ratio', ...
        'boundary_sector_coverage','boundary_mean_pair_dist','boundary_lowmargin_pair_hit','boundary_seg_cross_dist', ...
        'boundary_margin_oppdist_corr','boundary_lowmargin_mean_opp_dist','boundary_highmargin_mean_opp_dist', ...
        'boundary_lowmargin_oppdist_gain','boundary_lowmargin_mean_opp_support','boundary_highmargin_mean_opp_support', ...
        'pair_skip_duplicate', ...
        'seed_b_mean_pair_dist','seed_b_lowmargin_pair_hit', ...
        'b_size','b_lowmargin_ratio','b_mean_dist_to_true_boundary','b_p90_dist_to_true_boundary', ...
        'boundary_off_hit_rate_eps','archive_hit_rate_eps', ...
        'b_sector_coverage','b_mean_pair_dist','b_lowmargin_pair_hit', ...
        'b_margin_true_boundary_corr','b_margin_oppdist_corr','b_lowmargin_mean_opp_dist','b_highmargin_mean_opp_dist', ...
        'b_lowmargin_oppdist_gain','b_lowmargin_mean_opp_support','b_highmargin_mean_opp_support', ...
        'train_size','train_pos','train_neg','train_sector_coverage', ...
        'train_feasible_endpoint_count','train_pair_count','train_boundary_evidence_count','train_mean_pair_dist','train_can_train', ...
        'train_acc_before','train_bal_acc_before','train_brier_before','train_logloss_before', ...
        'train_acc_after','train_bal_acc_after','train_brier_after','train_logloss_after', ...
        'boundary_pred_acc','boundary_pred_bal_acc','boundary_pred_brier', ...
        'boundary_survive_c','boundary_survive_u','boundary_survive_c_rate','boundary_survive_u_rate', ...
        'inf_pool_size','inf_selected','inf_selected_ratio', ...
        'inf_pool_mean_score','inf_selected_mean_score','inf_score_gain', ...
        'inf_pool_mean_prob','inf_selected_mean_prob','inf_prob_gain', ...
        'inf_pool_mean_margin','inf_selected_mean_margin','inf_margin_gain', ...
        'inf_pool_mean_opp_support','inf_selected_mean_opp_support','inf_opp_support_gain', ...
        'inf_pool_mean_obj_score','inf_selected_mean_obj_score','inf_obj_score_gain', ...
        'inf_pool_lowmargin_ratio','inf_selected_lowmargin_ratio','inf_lowmargin_ratio_gain', ...
        'inf_selected_sector_coverage'});
    WriteCsvHeader(Observer.boundary_file,{ ...
        'generation','fe','event_index','sector', ...
        'pair_feasible_endpoint_side','pair_infeasible_endpoint_side', ...
        'pair_feasible_endpoint_prob','pair_feasible_endpoint_score', ...
        'pair_infeasible_endpoint_prob','pair_infeasible_endpoint_score', ...
        'pair_pred_cross','pair_dist','seg_cross_dist', ...
        'prob','pred_label','margin','opp_support','opp_dist','obj_score','score', ...
        'feasible','correct','survive_c','survive_u'});
    WriteCsvHeader(Observer.archive_file,{ ...
        'generation','fe','member_index','sector', ...
        'feasible','pred_label','correct','prob','margin','opp_support','opp_dist','obj_score','score'});
    WriteCsvHeader(Observer.objective_file,ComposeObjectiveSnapshotHeader(Problem.M));
    WriteCsvHeader(Observer.mlp_file,{ ...
        'generation','fe','model_ready_before','need_train','trained','model_ready_after', ...
        'can_train','boundary_count','last_train_size', ...
        'train_size','pos_count','neg_count','sector_coverage','feasible_endpoint_count','pair_count','mean_pair_dist', ...
        'src_b','src_recent_boundary','src_boundary_evidence','src_pop_c','src_pop_u', ...
        'acc_before','bal_acc_before','brier_before','logloss_before', ...
        'acc_after','bal_acc_after','brier_after','logloss_after'});
end

function Observer = LogMLPEvent(Observer,Diag)
    AppendCsvRows(Observer.mlp_file,{ ...
        Diag.generation,Diag.fe,Diag.model_ready_before,Diag.need_train,Diag.trained,Diag.model_ready_after, ...
        Diag.can_train,Diag.boundary_count,Diag.last_train_size, ...
        Diag.train_size,Diag.pos_count,Diag.neg_count,Diag.sector_coverage,Diag.feasible_endpoint_count,Diag.pair_count,Diag.mean_pair_dist, ...
        Diag.src_b,Diag.src_recent_boundary,Diag.src_boundary_evidence,Diag.src_pop_c,Diag.src_pop_u, ...
        Diag.stats_before.acc,Diag.stats_before.bal_acc,Diag.stats_before.brier,Diag.stats_before.logloss, ...
        Diag.stats_after.acc,Diag.stats_after.bal_acc,Diag.stats_after.brier,Diag.stats_after.logloss});
end

function Observer = LogGenerationDiagnostics( ...
    Observer,Generation,Problem,W,PopulationC,PopulationU,OffspringC,OffspringU, ...
    BoundaryOff,BoundaryEvidence,BoundaryDiag,BoundarySurvival,SeedB,SeedMeta,B,ArchiveMeta,MLPDiag,SelectionDiag,Model)

    BoundarySupport = KeepUniquePopulation([B,PopulationC,PopulationU]);
    BoundarySupportBase = BuildOppositeSupportBase(BoundarySupport,W);
    BoundaryMeta = BuildBoundaryMeta(BoundaryOff,W,Model,PopulationC,PopulationU,BoundarySupportBase);
    BoundaryEvidenceMeta = BuildBoundaryMeta(BoundaryEvidence,W,Model,PopulationC,PopulationU,BoundarySupportBase);
    RootEvidence = BuildSegmentRootEvidence( ...
        Problem,KeepUniquePopulation([B,BoundaryEvidence,PopulationC,PopulationU]), ...
        PopulationC,PopulationU,W,Model,BoundarySupportBase,max(12,min(36,Problem.N)));
    BoundaryStats = SummarizeArchive(BoundaryOff,BoundaryMeta,RootEvidence);
    ArchiveStats  = SummarizeArchive(B,ArchiveMeta,RootEvidence);
    SeedStats = SummarizeArchive(SeedB,SeedMeta,RootEvidence);
    BoundaryPred  = EvaluatePopulationPredictions(Model,BoundaryOff);

    PopCFea = feasibleRatio(PopulationC);
    PopUFea = feasibleRatio(PopulationU);
    OffCFea = feasibleRatio(OffspringC);
    OffUFea = feasibleRatio(OffspringU);
    BoundarySegCrossDist = MeanOrNaN(ExtractBoundaryEventNumericColumn(BoundaryDiag.events,13));
    BoundarySurviveCRate = SafeDivide(BoundarySurvival.selected_c,BoundaryDiag.selected);
    BoundarySurviveURate = SafeDivide(BoundarySurvival.selected_u,BoundaryDiag.selected);

    AppendCsvRows(Observer.summary_file,{ ...
        Generation,Problem.FE,min(Problem.FE/max(Problem.maxFE,1),1), ...
        PopCFea,PopUFea,OffCFea,OffUFea, ...
        MLPDiag.model_ready_after,MLPDiag.trained, ...
        BoundaryDiag.budget,BoundaryDiag.attempts,BoundaryDiag.selected,BoundaryDiag.selected, ...
        numel(BoundaryEvidence),feasibleRatio(BoundaryEvidence),BoundaryStats.feasible_ratio, ...
        BoundaryStats.sector_coverage,BoundaryStats.mean_pair_dist,BoundaryStats.lowmargin_pair_hit,BoundarySegCrossDist, ...
        BoundaryStats.margin_oppdist_corr,BoundaryStats.lowmargin_mean_opp_dist,BoundaryStats.highmargin_mean_opp_dist, ...
        BoundaryStats.lowmargin_oppdist_gain,BoundaryStats.lowmargin_mean_opp_support,BoundaryStats.highmargin_mean_opp_support, ...
        BoundaryDiag.skipped_duplicate, ...
        SeedStats.mean_pair_dist,SeedStats.lowmargin_pair_hit, ...
        ArchiveStats.size,ArchiveStats.lowmargin_ratio,ArchiveStats.mean_dist_to_true_boundary,ArchiveStats.p90_dist_to_true_boundary, ...
        BoundaryStats.hit_rate_eps,ArchiveStats.hit_rate_eps, ...
        ArchiveStats.sector_coverage,ArchiveStats.mean_pair_dist,ArchiveStats.lowmargin_pair_hit, ...
        ArchiveStats.margin_true_boundary_corr,ArchiveStats.margin_oppdist_corr,ArchiveStats.lowmargin_mean_opp_dist,ArchiveStats.highmargin_mean_opp_dist, ...
        ArchiveStats.lowmargin_oppdist_gain,ArchiveStats.lowmargin_mean_opp_support,ArchiveStats.highmargin_mean_opp_support, ...
        MLPDiag.train_size,MLPDiag.pos_count,MLPDiag.neg_count,MLPDiag.sector_coverage, ...
        MLPDiag.feasible_endpoint_count,MLPDiag.pair_count,MLPDiag.src_boundary_evidence,MLPDiag.mean_pair_dist,MLPDiag.can_train, ...
        MLPDiag.stats_before.acc,MLPDiag.stats_before.bal_acc,MLPDiag.stats_before.brier,MLPDiag.stats_before.logloss, ...
        MLPDiag.stats_after.acc,MLPDiag.stats_after.bal_acc,MLPDiag.stats_after.brier,MLPDiag.stats_after.logloss, ...
        BoundaryPred.acc,BoundaryPred.bal_acc,BoundaryPred.brier, ...
        BoundarySurvival.selected_c,BoundarySurvival.selected_u,BoundarySurviveCRate,BoundarySurviveURate, ...
        SelectionDiag.pool_size,SelectionDiag.selected_count,SelectionDiag.selected_ratio, ...
        SelectionDiag.pool_mean_score,SelectionDiag.selected_mean_score,SelectionDiag.score_gain, ...
        SelectionDiag.pool_mean_prob,SelectionDiag.selected_mean_prob,SelectionDiag.prob_gain, ...
        SelectionDiag.pool_mean_margin,SelectionDiag.selected_mean_margin,SelectionDiag.margin_gain, ...
        SelectionDiag.pool_mean_opp_support,SelectionDiag.selected_mean_opp_support,SelectionDiag.opp_support_gain, ...
        SelectionDiag.pool_mean_obj_score,SelectionDiag.selected_mean_obj_score,SelectionDiag.obj_score_gain, ...
        SelectionDiag.pool_lowmargin_ratio,SelectionDiag.selected_lowmargin_ratio,SelectionDiag.lowmargin_ratio_gain, ...
        SelectionDiag.selected_sector_coverage});

    Observer = LogBoundaryEvents(Observer,Generation,Problem.FE,BoundaryOff,PopulationC,PopulationU,BoundaryDiag.events);
    Observer = LogArchiveMembers(Observer,Generation,Problem.FE,B,ArchiveMeta);
    Observer = LogObjectiveSnapshot( ...
        Observer,Generation,Problem,W,PopulationC,PopulationU, ...
        BoundaryOff,BoundaryMeta,BoundaryEvidence,BoundaryEvidenceMeta,B,ArchiveMeta,Model,BoundarySupportBase,RootEvidence);
end

function Observer = LogBoundaryEvents(Observer,Generation,FE,BoundaryOff,PopulationC,PopulationU,Rows)
    if isempty(BoundaryOff) || isempty(Rows)
        return;
    end

    InC = false(numel(BoundaryOff),1);
    InU = false(numel(BoundaryOff),1);
    if ~isempty(PopulationC)
        InC = ismember(BoundaryOff.decs,PopulationC.decs,'rows');
    end
    if ~isempty(PopulationU)
        InU = ismember(BoundaryOff.decs,PopulationU.decs,'rows');
    end

    for i = 1 : numel(BoundaryOff)
        Rows{i,1} = Generation;
        Rows{i,2} = FE;
        Rows{i,23} = double(InC(i));
        Rows{i,24} = double(InU(i));
    end
    AppendCsvRows(Observer.boundary_file,Rows);
end

function Observer = LogArchiveMembers(Observer,Generation,FE,B,Meta)
    if isempty(B)
        return;
    end
    Rows = cell(numel(B),13);
    for i = 1 : numel(B)
        predLabel = double(Meta.prob(i) >= 0.5);
        OppDist = SupportDistance(Meta.oppSupport(i));
        Rows(i,:) = { ...
            Generation,FE,i,Meta.sector(i), ...
            Meta.feasible(i),predLabel,double(predLabel == Meta.feasible(i)),Meta.prob(i), ...
            Meta.margin(i),Meta.oppSupport(i),OppDist,Meta.objScore(i),Meta.score(i)};
    end
    AppendCsvRows(Observer.archive_file,Rows);
end

function Observer = LogObjectiveSnapshot( ...
    Observer,Generation,Problem,W,PopulationC,PopulationU, ...
    BoundaryOff,BoundaryMeta,BoundaryEvidence,BoundaryEvidenceMeta,B,ArchiveMeta,Model,SupportBase,RootEvidence)

    [Observer,WriteContext] = UpdateObjectiveSnapshotCheckpoints( ...
        Observer,Problem.FE,Problem.maxFE);
    WriteBoundaryOff = ~isempty(BoundaryOff);
    WriteBoundaryEvidence = ~isempty(BoundaryEvidence);
    WriteContext = WriteContext || WriteBoundaryOff || WriteBoundaryEvidence;
    if ~WriteContext && ~WriteBoundaryOff && ~WriteBoundaryEvidence
        return;
    end

    ColCount = 16 + 2*Problem.M;
    Rows = cell(0,ColCount);
    if WriteContext
        PopCMeta = BuildBoundaryMeta(PopulationC,W,Model,PopulationC,PopulationU,SupportBase);
        PopUMeta = BuildBoundaryMeta(PopulationU,W,Model,PopulationC,PopulationU,SupportBase);
        InC = true(numel(PopulationC),1);
        InU = false(numel(PopulationC),1);
        Rows = [Rows;BuildObjectiveSnapshotRows( ...
            Generation,Problem.FE,Problem.maxFE,'pop_c',PopulationC,PopCMeta,InC,InU,Problem.M,RootEvidence)]; %#ok<AGROW>
        InC = false(numel(PopulationU),1);
        InU = true(numel(PopulationU),1);
        Rows = [Rows;BuildObjectiveSnapshotRows( ...
            Generation,Problem.FE,Problem.maxFE,'pop_u',PopulationU,PopUMeta,InC,InU,Problem.M,RootEvidence)]; %#ok<AGROW>
        [InC,InU] = ComputePopulationMembership(B,PopulationC,PopulationU);
        Rows = [Rows;BuildObjectiveSnapshotRows( ...
            Generation,Problem.FE,Problem.maxFE,'archive_b',B,ArchiveMeta,InC,InU,Problem.M,RootEvidence)]; %#ok<AGROW>
        Rows = [Rows;BuildSegmentRootEvidenceRows( ...
            Generation,Problem.FE,Problem.maxFE,RootEvidence,Problem.M)]; %#ok<AGROW>
    end

    if WriteBoundaryOff
        [InC,InU] = ComputePopulationMembership(BoundaryOff,PopulationC,PopulationU);
        Rows = [Rows;BuildObjectiveSnapshotRows( ...
            Generation,Problem.FE,Problem.maxFE,'boundary_off',BoundaryOff,BoundaryMeta,InC,InU,Problem.M,RootEvidence)]; %#ok<AGROW>
    end
    if WriteBoundaryEvidence
        [InC,InU] = ComputePopulationMembership(BoundaryEvidence,PopulationC,PopulationU);
        Rows = [Rows;BuildObjectiveSnapshotRows( ...
            Generation,Problem.FE,Problem.maxFE,'boundary_evidence',BoundaryEvidence,BoundaryEvidenceMeta,InC,InU,Problem.M,RootEvidence)]; %#ok<AGROW>
    end

    AppendCsvRows(Observer.objective_file,Rows);
end

function [Observer,WriteContext] = UpdateObjectiveSnapshotCheckpoints(Observer,FE,MaxFE)
    Ratio = min(double(FE)/max(double(MaxFE),1),1);
    Hit = ~Observer.objective_logged & Observer.objective_marks <= Ratio + 1e-12;
    WriteContext = any(Hit);
    Observer.objective_logged(Hit) = true;
end

function [InC,InU] = ComputePopulationMembership(Population,PopulationC,PopulationU)
    InC = false(numel(Population),1);
    InU = false(numel(Population),1);
    if isempty(Population)
        return;
    end
    if ~isempty(PopulationC)
        InC = ismember(Population.decs,PopulationC.decs,'rows');
    end
    if ~isempty(PopulationU)
        InU = ismember(Population.decs,PopulationU.decs,'rows');
    end
end

function Rows = BuildObjectiveSnapshotRows(Generation,FE,MaxFE,Role,Population,Meta,InC,InU,M,RootEvidence)
    Count = numel(Population);
    Rows = cell(Count,16 + 2*M);
    if Count <= 0
        return;
    end

    Ratio = min(double(FE)/max(double(MaxFE),1),1);
    InC = MatchLength(logical(ColumnVector(InC)),Count,false);
    InU = MatchLength(logical(ColumnVector(InU)),Count,false);
    Obj = Population.objs;
    [RootDist,RootObj] = NearestSegmentRoot(Population,RootEvidence,M);
    for i = 1 : Count
        ObjRow = NaN(1,M);
        ObjCount = min(M,size(Obj,2));
        if ObjCount > 0
            ObjRow(1:ObjCount) = Obj(i,1:ObjCount);
        end
        Rows(i,:) = [{ ...
            Generation,FE,Ratio,Role,i,Meta.sector(i),Meta.feasible(i), ...
            Meta.prob(i),Meta.margin(i),Meta.oppSupport(i),SupportDistance(Meta.oppSupport(i)), ...
            RootDist(i),Meta.objScore(i),Meta.score(i),double(InC(i)),double(InU(i))}, ...
            num2cell(RootObj(i,:)),num2cell(ObjRow)];
    end
end

function Rows = BuildSegmentRootEvidenceRows(Generation,FE,MaxFE,RootEvidence,M)
    Count = numel(RootEvidence.Sector);
    Rows = cell(3*Count,16 + 2*M);
    if Count <= 0
        return;
    end

    Ratio = min(double(FE)/max(double(MaxFE),1),1);
    Row = 0;
    Roles = {'pair_feasible_endpoint','pair_infeasible_endpoint','segment_root'};
    ObjSets = {RootEvidence.FeasibleObj,RootEvidence.InfeasibleObj,RootEvidence.RootObj};
    DistValues = [0.5*RootEvidence.PairDist,0.5*RootEvidence.PairDist,zeros(Count,1)];
    FeasibleValues = [ones(Count,1),zeros(Count,1),NaN(Count,1)];
    for r = 1 : numel(Roles)
        Obj = ObjSets{r};
        for i = 1 : Count
            Row = Row + 1;
            ObjRow = NaN(1,M);
            RootObj = NaN(1,M);
            ObjCount = min(M,size(Obj,2));
            if ObjCount > 0
                ObjRow(1:ObjCount) = Obj(i,1:ObjCount);
            end
            RootCount = min(M,size(RootEvidence.RootObj,2));
            if RootCount > 0
                RootObj(1:RootCount) = RootEvidence.RootObj(i,1:RootCount);
            end
            Rows(Row,:) = [{ ...
                Generation,FE,Ratio,Roles{r},i,RootEvidence.Sector(i),FeasibleValues(i,r), ...
                NaN,NaN,NaN,NaN,DistValues(i,r),NaN,NaN,0,0}, ...
                num2cell(RootObj),num2cell(ObjRow)];
        end
    end
end

function Header = ComposeObjectiveSnapshotHeader(M)
    Header = { ...
        'generation','fe','fe_ratio','role','index','sector','feasible', ...
        'prob','margin','opp_support','opp_dist','true_boundary_dist','obj_score','score', ...
        'survive_c','survive_u'};
    RootHeader = arrayfun(@(i)sprintf('root_obj%d',i),1:M,'UniformOutput',false);
    ObjHeader = arrayfun(@(i)sprintf('obj%d',i),1:M,'UniformOutput',false);
    Header = [Header,RootHeader,ObjHeader];
end

%% ========== Diagnostics helpers ==========

function Diag = InitMLPDiag(Generation,FE,ModelState,Summary,ModelReady)
    EmptyStats = InitPredictionStats();
    Diag = struct( ...
        'generation',Generation, ...
        'fe',FE, ...
        'model_ready_before',double(ModelReady), ...
        'need_train',0, ...
        'trained',0, ...
        'model_ready_after',double(ModelReady), ...
        'can_train',double(Summary.ready), ...
        'boundary_count',Summary.boundary_count, ...
        'last_train_size',ModelState.lastTrainSize, ...
        'train_size',0, ...
        'pos_count',0, ...
        'neg_count',0, ...
        'sector_coverage',0, ...
        'train_dual_sectors',0, ...
        'feasible_endpoint_count',Summary.feasible_endpoint_count, ...
        'pair_count',Summary.pair_count, ...
        'mean_pair_dist',Summary.mean_pair_dist, ...
        'src_b',0, ...
        'src_recent_boundary',0, ...
        'src_boundary_evidence',0, ...
        'src_pop_c',0, ...
        'src_pop_u',0, ...
        'stats_before',EmptyStats, ...
        'stats_after',EmptyStats);
end

function Diag = InitInfeasibleSelectionDiag()
    Diag = struct( ...
        'pool_size',0, ...
        'selected_count',0, ...
        'selected_ratio',0, ...
        'pool_mean_score',NaN, ...
        'selected_mean_score',NaN, ...
        'score_gain',NaN, ...
        'pool_mean_prob',NaN, ...
        'selected_mean_prob',NaN, ...
        'prob_gain',NaN, ...
        'pool_mean_margin',NaN, ...
        'selected_mean_margin',NaN, ...
        'margin_gain',NaN, ...
        'pool_mean_opp_support',NaN, ...
        'selected_mean_opp_support',NaN, ...
        'opp_support_gain',NaN, ...
        'pool_mean_obj_score',NaN, ...
        'selected_mean_obj_score',NaN, ...
        'obj_score_gain',NaN, ...
        'pool_lowmargin_ratio',NaN, ...
        'selected_lowmargin_ratio',NaN, ...
        'lowmargin_ratio_gain',NaN, ...
        'selected_sector_coverage',0);
end

function Stats = InitPredictionStats()
    Stats = struct( ...
        'acc',NaN, ...
        'bal_acc',NaN, ...
        'brier',NaN, ...
        'logloss',NaN, ...
        'pos_recall',NaN, ...
        'neg_recall',NaN);
end

function Stats = EvaluatePopulationPredictions(Model,Population)
    if isempty(Population)
        Stats = InitPredictionStats();
        return;
    end
    Stats = EvaluateBinaryPredictions(Model,Population.decs,double(all(Population.cons<=0,2)));
end

function Stats = EvaluateBinaryPredictions(Model,X,Y)
    Stats = InitPredictionStats();
    if nargin < 3 || isempty(X)
        return;
    end

    Y = double(Y(:) > 0);
    Prob = PredictBoundaryMLP(Model,X);
    Pred = double(Prob >= 0.5);
    Stats.acc = mean(Pred == Y);
    PosMask = Y == 1;
    NegMask = Y == 0;
    if any(PosMask)
        Stats.pos_recall = mean(Pred(PosMask) == 1);
    end
    if any(NegMask)
        Stats.neg_recall = mean(Pred(NegMask) == 0);
    end
    if any(PosMask) && any(NegMask)
        Stats.bal_acc = 0.5*(Stats.pos_recall + Stats.neg_recall);
    else
        Stats.bal_acc = NaN;
    end
    Stats.brier = mean((Prob - Y).^2);
    Stats.logloss = -mean(Y.*log(max(Prob,1e-6)) + (1-Y).*log(max(1-Prob,1e-6)));
end

function Stats = SummarizeArchive(B,Meta,RootEvidence)
    if nargin < 3
        RootEvidence = InitSegmentRootEvidence(ResolveDecisionDimension(B),0);
    end
    Stats = struct( ...
        'size',numel(B), ...
        'feasible_ratio',NaN, ...
        'sector_coverage',0, ...
        'mixed_sectors',0, ...
        'mean_margin',NaN, ...
        'lowmargin_ratio',NaN, ...
        'mean_opp_support',NaN, ...
        'mean_pair_dist',NaN, ...
        'mean_dist_to_true_boundary',NaN, ...
        'p90_dist_to_true_boundary',NaN, ...
        'hit_rate_eps',NaN, ...
        'margin_true_boundary_corr',NaN, ...
        'lowmargin_pair_hit',NaN, ...
        'margin_oppdist_corr',NaN, ...
        'lowmargin_mean_opp_dist',NaN, ...
        'highmargin_mean_opp_dist',NaN, ...
        'lowmargin_oppdist_gain',NaN, ...
        'lowmargin_mean_opp_support',NaN, ...
        'highmargin_mean_opp_support',NaN, ...
        'mean_obj_score',NaN, ...
        'mean_score',NaN);
    if isempty(B)
        return;
    end

    Stats.feasible_ratio   = SafeDivide(sum(Meta.feasible == 1),numel(B));
    Stats.sector_coverage  = numel(unique(Meta.sector(Meta.sector > 0)));
    Stats.mean_margin      = MeanOrNaN(Meta.margin);
    Stats.lowmargin_ratio  = LowMarginRatio(Meta.margin);
    Stats.mean_opp_support = MeanOrNaN(Meta.oppSupport);
    OppDist = SupportDistance(Meta.oppSupport);
    Stats.mean_pair_dist   = MeanOrNaN(OppDist);
    TrueBoundaryDist = DistanceToSegmentRoots(B,RootEvidence);
    Stats.mean_dist_to_true_boundary = MeanOrNaN(TrueBoundaryDist);
    Stats.p90_dist_to_true_boundary = PercentileOrNaN(TrueBoundaryDist,90);
    Stats.hit_rate_eps = SafeDivide(sum(TrueBoundaryDist <= 0.10),sum(isfinite(TrueBoundaryDist)));
    Stats.margin_true_boundary_corr = SafeCorr(Meta.margin,TrueBoundaryDist);
    Stats.mean_obj_score   = MeanOrNaN(Meta.objScore);
    Stats.mean_score       = MeanOrNaN(Meta.score);
    LowMask = Meta.margin <= 0.10;
    HighMask = Meta.margin > 0.10;
    if any(LowMask)
        Stats.lowmargin_pair_hit = SafeDivide(sum(Meta.oppSupport(LowMask) > 0),sum(LowMask));
    end
    Stats.margin_oppdist_corr = SafeCorr(Meta.margin,OppDist);
    Stats.lowmargin_mean_opp_dist = MeanOrNaN(OppDist(LowMask));
    Stats.highmargin_mean_opp_dist = MeanOrNaN(OppDist(HighMask));
    Stats.lowmargin_oppdist_gain = Stats.highmargin_mean_opp_dist - Stats.lowmargin_mean_opp_dist;
    Stats.lowmargin_mean_opp_support = MeanOrNaN(Meta.oppSupport(LowMask));
    Stats.highmargin_mean_opp_support = MeanOrNaN(Meta.oppSupport(HighMask));

    Sectors = unique(Meta.sector(Meta.sector > 0))';
    Mixed = 0;
    for s = Sectors
        idx = find(Meta.sector == s);
        if any(Meta.feasible(idx) == 1) && any(Meta.feasible(idx) == 0)
            Mixed = Mixed + 1;
        end
    end
    Stats.mixed_sectors = Mixed;
end

function RootEvidence = BuildSegmentRootEvidence(Problem,AnchorPool,PopulationC,PopulationU,W,Model,SupportBase,MaxPairs)
    %#ok<INUSD>
    Pair = BuildBoundaryPairs(AnchorPool,W,MaxPairs);
    RootEvidence = InitSegmentRootEvidence(ResolveDecisionDimension(AnchorPool,PopulationC,PopulationU),Problem.M);
    PairCount = size(Pair.FeasibleDec,1);
    if PairCount <= 0
        return;
    end

    FeasibleDec = Problem.CalDec(Pair.FeasibleDec);
    InfeasibleDec = Problem.CalDec(Pair.InfeasibleDec);
    RootDec = BisectSegmentRoots(Problem,FeasibleDec,InfeasibleDec,6);
    RootEvidence.Dec = RootDec;
    RootEvidence.Obj = Problem.CalObj(RootDec);
    RootEvidence.RootObj = RootEvidence.Obj;
    RootEvidence.FeasibleDec = FeasibleDec;
    RootEvidence.InfeasibleDec = InfeasibleDec;
    RootEvidence.FeasibleObj = Problem.CalObj(FeasibleDec);
    RootEvidence.InfeasibleObj = Problem.CalObj(InfeasibleDec);
    RootEvidence.Sector = Pair.Sector;
    RootEvidence.PairDist = sqrt(sum((FeasibleDec - InfeasibleDec).^2,2));
end

function RootEvidence = InitSegmentRootEvidence(D,M)
    RootEvidence = struct( ...
        'Dec',zeros(0,D), ...
        'Obj',zeros(0,M), ...
        'RootObj',zeros(0,M), ...
        'FeasibleDec',zeros(0,D), ...
        'InfeasibleDec',zeros(0,D), ...
        'FeasibleObj',zeros(0,M), ...
        'InfeasibleObj',zeros(0,M), ...
        'Sector',zeros(0,1), ...
        'PairDist',zeros(0,1));
end

function RootDec = BisectSegmentRoots(Problem,FeasibleDec,InfeasibleDec,Steps)
    Lo = FeasibleDec;
    Hi = InfeasibleDec;
    for step = 1 : max(1,round(Steps))
        Mid = Problem.CalDec(0.5*(Lo + Hi));
        MidFeasible = all(Problem.CalCon(Mid) <= 0,2);
        Lo(MidFeasible,:) = Mid(MidFeasible,:);
        Hi(~MidFeasible,:) = Mid(~MidFeasible,:);
    end
    RootDec = Problem.CalDec(0.5*(Lo + Hi));
end

function Dist = DistanceToSegmentRoots(Population,RootEvidence)
    Dist = NaN(numel(Population),1);
    if isempty(Population) || isempty(RootEvidence.Dec)
        return;
    end
    RootDec = double(RootEvidence.Dec);
    for i = 1 : numel(Population)
        Delta = RootDec - repmat(double(Population(i).decs),size(RootDec,1),1);
        Dist(i) = min(sqrt(sum(Delta.^2,2)));
    end
end

function [Dist,RootObj] = NearestSegmentRoot(Population,RootEvidence,M)
    Dist = NaN(numel(Population),1);
    RootObj = NaN(numel(Population),M);
    if isempty(Population) || isempty(RootEvidence.Dec)
        return;
    end
    RootDec = double(RootEvidence.Dec);
    for i = 1 : numel(Population)
        Delta = RootDec - repmat(double(Population(i).decs),size(RootDec,1),1);
        D = sqrt(sum(Delta.^2,2));
        [Dist(i),idx] = min(D);
        ObjCount = min(M,size(RootEvidence.RootObj,2));
        if ObjCount > 0
            RootObj(i,1:ObjCount) = RootEvidence.RootObj(idx,1:ObjCount);
        end
    end
end

function Diag = InitBoundaryDiag(Generation,FE,Budget)
    Diag = struct( ...
        'generation',Generation, ...
        'fe',FE, ...
        'budget',Budget, ...
        'attempts',0, ...
        'selected',0, ...
        'skipped_duplicate',0, ...
        'events',{cell(0,24)});
end

function Value = SupportDistance(OppSupport)
    OppSupport = max(double(OppSupport),1e-6);
    Value = -log(OppSupport);
end

function Ratio = LowMarginRatio(Margin)
    Margin = ColumnVector(Margin);
    Mask = isfinite(Margin);
    Ratio = SafeDivide(sum(Margin(Mask) <= 0.10),sum(Mask));
end

function Value = SafeCorr(X,Y)
    X = ColumnVector(X);
    Y = ColumnVector(Y);
    Mask = isfinite(X) & isfinite(Y);
    X = X(Mask);
    Y = Y(Mask);
    if numel(X) < 3 || std(X) < 1e-12 || std(Y) < 1e-12
        Value = NaN;
        return;
    end
    C = corrcoef(X,Y);
    Value = C(1,2);
end

function Values = ExtractBoundaryEventNumericColumn(Rows,Column)
    Values = zeros(0,1);
    if isempty(Rows)
        return;
    end
    Values = NaN(size(Rows,1),1);
    for i = 1 : size(Rows,1)
        Values(i) = double(Rows{i,Column});
    end
end

function Stats = ComputeBoundarySurvival(BoundaryOff,PopulationC,PopulationU)
    Stats = struct('selected_c',0,'selected_u',0);
    if isempty(BoundaryOff)
        return;
    end
    if ~isempty(PopulationC)
        Stats.selected_c = sum(ismember(BoundaryOff.decs,PopulationC.decs,'rows'));
    end
    if ~isempty(PopulationU)
        Stats.selected_u = sum(ismember(BoundaryOff.decs,PopulationU.decs,'rows'));
    end
end

function Ratio = feasibleRatio(Population)
    if isempty(Population)
        Ratio = NaN;
        return;
    end
    Ratio = SafeDivide(sum(all(Population.cons<=0,2)),numel(Population));
end

function Count = countDualSectors(Sector,Label)
    Count = 0;
    if isempty(Sector)
        return;
    end
    for s = unique(Sector(:))'
        idx = find(Sector == s);
        if any(Label(idx) == 0) && any(Label(idx) == 1)
            Count = Count + 1;
        end
    end
end

function Value = MeanOrNaN(Value)
    if isempty(Value)
        Value = NaN;
        return;
    end
    Value = Value(isfinite(Value));
    if isempty(Value)
        Value = NaN;
    else
        Value = mean(Value);
    end
end

function Value = PercentileOrNaN(Value,Percentile)
    Value = ColumnVector(Value);
    Value = sort(Value(isfinite(Value)));
    if isempty(Value)
        Value = NaN;
        return;
    end
    Rank = max(1,min(numel(Value),ceil(double(Percentile)/100*numel(Value))));
    Value = Value(Rank);
end

function Value = SafeDivide(Numerator,Denominator)
    if Denominator <= 0
        Value = NaN;
    else
        Value = Numerator/Denominator;
    end
end

function RunId = resolveRunId(Algorithm)
    if isempty(Algorithm.run)
        RunId = 0;
    else
        RunId = double(Algorithm.run);
    end
end

function Family = familyOfProblem(problem)
    problem = char(string(problem));
    if startsWith(problem,'DASCMOP')
        Family = 'DASCMOP_BC';
    elseif startsWith(problem,'LIRCMOP')
        Family = 'LIRCMOP_BC';
    else
        Family = 'OTHER';
    end
end

function WriteCsvHeader(FilePath,Header)
    fid = fopen(FilePath,'w');
    Cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s\n',strjoin(Header,','));
end

function AppendCsvRows(FilePath,Rows)
    if isempty(Rows)
        return;
    end
    if ~iscell(Rows)
        Rows = num2cell(Rows);
    end
    if isvector(Rows)
        Rows = reshape(Rows,1,[]);
    end
    fid = fopen(FilePath,'a');
    Cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    for i = 1 : size(Rows,1)
        Line = cell(1,size(Rows,2));
        for j = 1 : size(Rows,2)
            Line{j} = ToCsvValue(Rows{i,j});
        end
        fprintf(fid,'%s\n',strjoin(Line,','));
    end
end

function S = ToCsvValue(Value)
    if isstring(Value)
        Value = char(Value);
    end
    if ischar(Value)
        Value = strrep(Value,'"','""');
        S = ['"' Value '"'];
    elseif isempty(Value)
        S = '';
    elseif isnumeric(Value) || islogical(Value)
        if isscalar(Value)
            if isnan(Value)
                S = 'NaN';
            elseif isinf(Value)
                if Value > 0
                    S = 'Inf';
                else
                    S = '-Inf';
                end
            else
                S = sprintf('%.15g',double(Value));
            end
        else
            Flat = Value(:)';
            Parts = arrayfun(@(x)sprintf('%.15g',double(x)),Flat,'UniformOutput',false);
            S = ['"' strjoin(Parts,';') '"'];
        end
    else
        S = ['"' char(string(Value)) '"'];
    end
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
