classdef PRBCCMO < ALGORITHM
% <2026> <multi> <real> <constrained>
% PRBCCMO
% Boundary-band MLP driven CCMO for binary unknown constraints
%
% hidden --- 20   --- Hidden neurons of the boundary MLP
% epoch  --- 20   --- Training epochs per update
% lr     --- 0.01 --- Learning rate
% kappa  --- 3    --- Maximum boundary samples kept per sector
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
            [hidden,epoch,lr,kappa,rho] = Algorithm.ParameterSet(20,20,0.01,3,0.12);

            kappa = max(1,round(kappa));
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

                BoundaryOff = GenerateBoundaryOffspring(Problem,B,PopulationC,PopulationU,W,Model,rho);

                QC = KeepUniquePopulation([PopulationC,OffspringC,OffspringU,BoundaryOff]);
                QU = KeepUniquePopulation([PopulationU,OffspringC,OffspringU,BoundaryOff]);
                PopulationC = EnvironmentalSelectionC( ...
                    QC,Problem.N,W,Model,B,RecentBoundaryOff,PopulationU);
                PopulationU = EnvironmentalSelectionU(QU,Problem.N,W);

                RecentBoundaryOff = BoundaryOff;
                B = UpdateBoundaryArchive(B,RecentBoundaryOff,PopulationC,PopulationU,W,Model,kappa);
                [Model,ModelState] = UpdateBoundaryModelIfNeeded( ...
                    Model,ModelState,B,RecentBoundaryOff,PopulationC,PopulationU,W,hidden,epoch,lr,Problem,Problem.FE);
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
    Support = KeepUniquePopulation([B,BoundarySamples,PopulationC,PopulationU]);
    SupportBase = BuildOppositeSupportBase(Support,W);
    Representatives = BuildBoundaryRepresentatives(B,PopulationC,PopulationU,W,Model,SupportBase);
    Candidates = KeepUniquePopulation([B,BoundarySamples,Representatives]);
    if isempty(Candidates)
        B = Candidates;
        return;
    end

    Meta = BuildBoundaryMeta(Candidates,W,Model,PopulationC,PopulationU,SupportBase);
    B = SelectTopKPerSector(Candidates,Meta,kappa);
end

function Representatives = BuildBoundaryRepresentatives(B,PopulationC,PopulationU,W,Model,SupportBase)
    Representatives = B([]);
    if isempty(W)
        return;
    end

    BMeta = BuildBoundaryMeta(B,W,Model,PopulationC,PopulationU,SupportBase);
    RepC = BuildRepresentativeIndexBySector(PopulationC,W);
    RepU = BuildRepresentativeIndexBySector(PopulationU,W);
    PickC = zeros(size(W,1),1);
    PickU = zeros(size(W,1),1);
    CountC = 0;
    CountU = 0;
    for s = 1 : size(W,1)
        if NeedsRepresentativeRefill(BMeta,s)
            if RepC.has(s)
                CountC = CountC + 1;
                PickC(CountC) = RepC.index(s);
            end
            if RepU.has(s)
                CountU = CountU + 1;
                PickU(CountU) = RepU.index(s);
            end
        end
    end
    RepCOut = PopulationC([]);
    RepUOut = PopulationU([]);
    if CountC > 0
        RepCOut = PopulationC(PickC(1:CountC)');
    end
    if CountU > 0
        RepUOut = PopulationU(PickU(1:CountU)');
    end
    Representatives = KeepUniquePopulation([RepCOut,RepUOut]);
end

function Flag = NeedsRepresentativeRefill(BMeta,s)
    if isempty(BMeta.sector)
        Flag = true;
        return;
    end

    idx = find(BMeta.sector == s);
    if isempty(idx)
        Flag = true;
        return;
    end
    Label = BMeta.feasible(idx);
    Flag = ~(any(Label == 1) && any(Label == 0));
end

function Rep = BuildRepresentativeIndexBySector(Population,W)
    K = size(W,1);
    Rep = struct('has',false(K,1),'index',zeros(K,1));
    if isempty(Population) || K <= 0
        return;
    end

    Z = ComputeIdealPoint(Population);
    Sector = AssociateSectorsLocal(Population.objs,W,Z);
    [FrontNo,~] = NDSort(Population.objs,numel(Population));
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    for s = 1 : K
        idx = find(Sector == s);
        if isempty(idx)
            continue;
        end
        Key = [ColumnVector(FrontNo(idx)),ColumnVector(-CrowdDis(idx)),ColumnVector(idx)];
        [~,ord] = sortrows(Key,[1 2 3]);
        Rep.index(s) = idx(ord(1));
        Rep.has(s) = true;
    end
end

function B = SelectTopKPerSector(Candidates,Meta,kappa)
    if isempty(Candidates)
        B = Candidates;
        return;
    end

    kappa = max(1,round(kappa));
    if isempty(find(Meta.sector > 0,1))
        ord = SortByBoundaryKey(Meta,(1:numel(Candidates))');
        B = Candidates(ord(1:min(numel(ord),kappa)));
        return;
    end

    Sectors = unique(Meta.sector(Meta.sector > 0))';
    Pick = zeros(min(numel(Candidates),numel(Sectors)*kappa),1);
    PickCount = 0;
    for s = Sectors
        idx = find(Meta.sector == s);
        if isempty(idx)
            continue;
        end

        ord = SortByBoundaryKey(Meta,idx);
        Sel = ord(1:min(kappa,numel(ord)));
        Pos = SortByBoundaryKey(Meta,idx(Meta.feasible(idx) == 1));
        Neg = SortByBoundaryKey(Meta,idx(Meta.feasible(idx) == 0));
        if kappa > 1 && ~isempty(Pos) && ~isempty(Neg)
            SelLabel = Meta.feasible(Sel);
            if ~(any(SelLabel == 1) && any(SelLabel == 0))
                Sel = unique([Pos(1);Neg(1);Sel(:)],'stable');
                Sel = Sel(1:min(kappa,numel(Sel)));
            end
        end
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
    Feasible = double(all(Candidates.cons<=0,2));
    Meta = ComposeBoundaryMeta(Sector,Feasible,Prob,OppSupport,ObjScore);
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

function Cache = BuildOppositeSupportCache(Support,W,RefObj)
    SupportBase = BuildOppositeSupportBase(Support,W);
    Cache = ResolveOppositeSupportCache(SupportBase,W,RefObj);
end

function SupportCache = ResolveOppositeSupportCache(SupportInfo,W,RefObj)
    if IsOppositeSupportBase(SupportInfo)
        SupportBase = SupportInfo;
    else
        SupportBase = BuildOppositeSupportBase(SupportInfo,W);
    end

    SupportCache = struct( ...
        'Dec',zeros(0,0), ...
        'Label',zeros(0,1), ...
        'Sector',zeros(0,1), ...
        'Tau',1, ...
        'Neighbors',{SupportBase.Neighbors});
    if isempty(SupportBase.Dec)
        return;
    end

    SupportCache.Dec = SupportBase.Dec;
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
    Model,ModelState,B,RecentBoundaryOff,PopulationC,PopulationU,W,hidden,epoch,lr,Problem,CurrentFE)
    MaxTrain = max(6*Problem.N,4*size(W,1));
    Candidates = BuildTrainingCandidateSet(B,RecentBoundaryOff,PopulationC,PopulationU);
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
        'Label',zeros(0,1), ...
        'Sector',zeros(0,1), ...
        'Source',zeros(0,1), ...
        'OppDist',zeros(0,1), ...
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

function Candidates = BuildTrainingCandidateSet(B,RecentBoundaryOff,PopulationC,PopulationU)
    [Population,Source] = MergeTrainingSourcePopulations(B,RecentBoundaryOff,PopulationC,PopulationU);
    if isempty(Population)
        Candidates = struct('Population',Population,'Source',Source);
        return;
    end

    Keep = KeepLatestDecisionRowsLocal(Population.decs);
    Population = Population(Keep);
    Source = Source(Keep);
    Candidates = struct('Population',Population,'Source',Source);
end

function [Population,Source] = MergeTrainingSourcePopulations(B,RecentBoundaryOff,PopulationC,PopulationU)
    Population = [B,RecentBoundaryOff,PopulationC,PopulationU];
    Source = [ ...
        ones(numel(B),1); ...
        2*ones(numel(RecentBoundaryOff),1); ...
        3*ones(numel(PopulationC),1); ...
        4*ones(numel(PopulationU),1)];
end

function Band = CollectSectorBoundaryBand(Population,Label,Sector,Source,W,Eligible)
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

    OppDist = ComputeOppositeDistanceForCandidates(Population.decs,Label,Sector,W,Eligible);
    K = max(size(W,1),max([Sector(:);1]));
    Pick = false(numel(Population),1);
    Neighbors = BuildSectorNeighbors(W,min(3,max(size(W,1)-1,0)));
    for s = 1 : K
        Local = ResolveLocalSectorSet(s,Neighbors);
        idx = find(Eligible & ismember(Sector,Local) & isfinite(OppDist));
        if isempty(idx)
            continue;
        end
        Pos = idx(Label(idx) == 1);
        Neg = idx(Label(idx) == 0);
        if isempty(Pos) || isempty(Neg)
            continue;
        end
        Quota = ResolveSectorBandQuota(numel(Pos),numel(Neg),numel(idx));
        Pos = SortByDistanceThenSource(OppDist,Source,Pos);
        Neg = SortByDistanceThenSource(OppDist,Source,Neg);
        Pick(Pos(1:min(Quota,numel(Pos)))) = true;
        Pick(Neg(1:min(Quota,numel(Neg)))) = true;
    end

    idx = find(Pick);
    if isempty(idx)
        return;
    end
    Band.Dec     = Population(idx).decs;
    Band.Label   = Label(idx);
    Band.Sector  = Sector(idx);
    Band.Source  = Source(idx);
    Band.OppDist = OppDist(idx);
end

function Band = InitBoundaryBand(D)
    Band = struct( ...
        'Dec',zeros(0,D), ...
        'Label',zeros(0,1), ...
        'Sector',zeros(0,1), ...
        'Source',zeros(0,1), ...
        'OppDist',zeros(0,1));
end

function Quota = ResolveSectorBandQuota(PosCount,NegCount,LocalCount)
    Evidence = min(PosCount,NegCount);
    Quota = max(1,ceil(sqrt(max(Evidence,1)) + LocalCount/25));
    Quota = min(5,Quota);
end

function Order = SortByDistanceThenSource(OppDist,Source,idx)
    idx = ColumnVector(idx);
    Key = [ColumnVector(OppDist(idx)),ColumnVector(SourcePriority(Source(idx))),ColumnVector(idx)];
    [~,ord] = sortrows(Key,[1 2 3]);
    Order = idx(ord);
end

function Priority = SourcePriority(Source)
    Priority = 5*ones(numel(Source),1);
    Priority(Source == 1) = 1;
    Priority(Source == 2) = 2;
    Priority(Source == 3) = 3;
    Priority(Source == 4) = 4;
end

function OppDist = ComputeOppositeDistanceForCandidates(Dec,Label,Sector,W,Eligible)
    Label = ColumnVector(Label);
    Sector = ColumnVector(Sector);
    OppDist = inf(size(Dec,1),1);
    if isempty(Dec)
        return;
    end
    if nargin < 5 || isempty(Eligible)
        Eligible = true(size(Dec,1),1);
    else
        Eligible = ColumnVector(logical(Eligible));
    end

    Neighbors = BuildSectorNeighbors(W,min(3,max(size(W,1)-1,0)));
    Target = find(Eligible);
    for k = 1 : numel(Target)
        i = Target(k);
        Local = ResolveLocalSectorSet(Sector(i),Neighbors);
        Mask = (Label ~= Label(i)) & ismember(Sector,Local);
        if ~any(Mask)
            Mask = Label ~= Label(i);
        end
        if any(Mask)
            Delta = double(Dec(Mask,:)) - repmat(double(Dec(i,:)),sum(Mask),1);
            Dist = sqrt(sum(Delta.^2,2));
            OppDist(i) = min(Dist);
        end
    end
end

function Buffer = AppendTrainingBuffer(Buffer,Band,Time)
    if isempty(Band.Dec)
        return;
    end
    if isempty(Buffer.Dec)
        Buffer.Dec = Band.Dec;
        Buffer.Label = Band.Label;
        Buffer.Sector = Band.Sector;
        Buffer.Source = Band.Source;
        Buffer.OppDist = Band.OppDist;
        Buffer.Time = Time*ones(size(Band.Label));
    else
        Buffer.Dec = [Buffer.Dec;Band.Dec];
        Buffer.Label = [Buffer.Label;Band.Label];
        Buffer.Sector = [Buffer.Sector;Band.Sector];
        Buffer.Source = [Buffer.Source;Band.Source];
        Buffer.OppDist = [Buffer.OppDist;Band.OppDist];
        Buffer.Time = [Buffer.Time;Time*ones(size(Band.Label))];
    end
    Keep = KeepLatestDecisionRowsLocal(Buffer.Dec);
    Buffer = SliceTrainingBuffer(Buffer,Keep);
end

function Buffer = TrimTrainingBuffer(Buffer,MaxTrain)
    if size(Buffer.Dec,1) <= MaxTrain
        return;
    end
    Key = [ColumnVector(Buffer.OppDist),ColumnVector(-Buffer.Time), ...
           ColumnVector(SourcePriority(Buffer.Source)),ColumnVector((1:size(Buffer.Dec,1))')];
    [~,ord] = sortrows(Key,[1 2 3 4]);
    Keep = sort(ord(1:MaxTrain));
    Buffer = SliceTrainingBuffer(Buffer,Keep);
end

function Buffer = SliceTrainingBuffer(Buffer,Keep)
    Buffer.Dec = Buffer.Dec(Keep,:);
    Buffer.Label = Buffer.Label(Keep);
    Buffer.Sector = Buffer.Sector(Keep);
    Buffer.Source = Buffer.Source(Keep);
    Buffer.OppDist = Buffer.OppDist(Keep);
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

function Dataset = BuildBoundaryDatasetFromBuffer(Buffer,W,N)
    Dataset = struct('Dec',zeros(0,0),'Label',zeros(0,1));
    if isempty(Buffer.Dec) || numel(unique(Buffer.Label)) < 2
        return;
    end

    MaxPerSide = max(2,min(5,ceil(N/max(size(W,1),1))));
    Sectors = unique(Buffer.Sector(Buffer.Sector > 0))';
    Ranked = zeros(min(size(Buffer.Dec,1),2*MaxPerSide*numel(Sectors)),1);
    RankedCount = 0;
    for s = Sectors
        idx = find(Buffer.Sector == s);
        Pos = SortByDistanceThenSource(Buffer.OppDist,Buffer.Source,idx(Buffer.Label(idx) == 1));
        Neg = SortByDistanceThenSource(Buffer.OppDist,Buffer.Source,idx(Buffer.Label(idx) == 0));
        if isempty(Pos) || isempty(Neg)
            continue;
        end
        Use = min([MaxPerSide,numel(Pos),numel(Neg)]);
        Add = [Pos(1:Use);Neg(1:Use)];
        AddCount = min(numel(Add),numel(Ranked)-RankedCount);
        if AddCount > 0
            Ranked(RankedCount+1:RankedCount+AddCount) = Add(1:AddCount);
            RankedCount = RankedCount + AddCount;
        end
    end
    Ranked = Ranked(1:RankedCount);
    if isempty(Ranked)
        Ranked = BuildBalancedReplayBatch( ...
            find(Buffer.Label == 1),find(Buffer.Label == 0),Buffer.OppDist,Buffer.Source);
    else
        Ranked = unique(Ranked,'stable');
        Ranked = BuildBalancedReplayBatch( ...
            Ranked(Buffer.Label(Ranked) == 1), ...
            Ranked(Buffer.Label(Ranked) == 0),Buffer.OppDist,Buffer.Source);
    end
    Dataset.Dec = Buffer.Dec(Ranked,:);
    Dataset.Label = Buffer.Label(Ranked);
end

function BatchIdx = BuildBalancedReplayBatch(PosIdx,NegIdx,OppDist,Source)
    PosIdx = SortByDistanceThenSource(OppDist,Source,PosIdx);
    NegIdx = SortByDistanceThenSource(OppDist,Source,NegIdx);
    Count = min(numel(PosIdx),numel(NegIdx));
    if Count <= 0
        BatchIdx = zeros(0,1);
        return;
    end
    BatchIdx = reshape([PosIdx(1:Count)';NegIdx(1:Count)'],[],1);
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

    Model = struct('Mu',Mu,'Sigma',Sigma,'W1',W1,'b1',b1,'W2',W2,'b2',b2);
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
    Prob = min(max(1./(1+exp(-Z)),1e-6),1-1e-6);
end

%% ========== Boundary offspring ==========

function Sample = SelectLocalOppositeSample(AnchorDec,Sector,Population,W)
    Sample = Population([]);
    if isempty(Population)
        return;
    end

    Pool = Population;
    RefObj = ComputeIdealPoint(Pool);
    PoolSector = AssociateSectorsLocal(Pool.objs,W,RefObj);
    Neighbors = BuildSectorNeighbors(W,min(3,max(size(W,1)-1,0)));
    Local = ResolveLocalSectorSet(Sector,Neighbors);
    idx = find(ismember(PoolSector,Local));
    if isempty(idx)
        return;
    end

    Dist = sqrt(sum((double(Pool(idx).decs) - double(AnchorDec)).^2,2));
    [~,ord] = sort(Dist,'ascend');
    Sample = Pool(idx(ord(1)));
end

function Offspring = GenerateBoundaryOffspring(Problem,B,PopulationC,PopulationU,W,Model,rho)
    if isempty(Model) || isempty(B)
        Offspring = B([]);
        return;
    end

    Budget = min(max(0,floor(rho*Problem.N)),max(0,Problem.maxFE-Problem.FE));
    if Budget <= 0
        Offspring = B([]);
        return;
    end

    Support = KeepUniquePopulation([B,PopulationC,PopulationU]);
    SupportBase = BuildOppositeSupportBase(Support,W);
    ParentPool = Support;
    ParentMeta = BuildBoundaryMeta(ParentPool,W,Model,PopulationC,PopulationU,SupportBase);
    AnchorIdx = SelectBoundaryAnchors(ParentPool,B,ParentMeta,W,max(Problem.N,2*Budget));
    if isempty(AnchorIdx)
        Offspring = B([]);
        return;
    end

    CandidateDec = zeros(Budget,size(ParentPool.decs,2));
    CandidateAnchorObjScore = zeros(Budget,1);
    CandidateAnchorOppSupport = zeros(Budget,1);
    CandidateCount = 0;

    SupportRefObj = ComputeIdealPoint(ParentPool,PopulationC,PopulationU);
    SupportCache = ResolveOppositeSupportCache(SupportBase,W,SupportRefObj);
    StablePool = KeepUniquePopulation([PopulationC,PopulationU]);
    StablePoolBase = BuildOppositeSupportBase(StablePool,W);
    StablePoolMeta = BuildBoundaryMeta(StablePool,W,[],PopulationC,PopulationU,StablePoolBase);
    for i = 1 : numel(AnchorIdx)
        if CandidateCount >= Budget
            break;
        end
        BaseIdx = AnchorIdx(i);
        Parent2 = SelectOppositeParent(ParentPool,ParentMeta,BaseIdx,PopulationC,PopulationU,W);
        Parent3 = SelectStableParentFast(StablePool,StablePoolMeta,ParentMeta.sector(BaseIdx));
        if isempty(Parent2) || isempty(Parent3)
            continue;
        end

        Child = OperatorDE(Problem,ParentPool(BaseIdx).decs,Parent2.decs,Parent3.decs);
        if isempty(Child)
            continue;
        end
        Child = Child(1,:);
        DuplicateCandidate = CandidateCount > 0 && ismember(Child,CandidateDec(1:CandidateCount,:),'rows');
        if ismember(Child,ParentPool.decs,'rows') || DuplicateCandidate
            continue;
        end

        ProxySector = ParentMeta.sector(BaseIdx);
        Prob = PredictBoundaryMLP(Model,Child);
        Feasible = double(Prob >= 0.5);
        OppSupport = ComputeOppositeSupportFromCache(Child,Feasible,ProxySector,SupportCache);
        CandidateMeta = ComposeBoundaryMeta(ProxySector,Feasible,Prob,OppSupport,0);
        AnchorMargin = ParentMeta.margin(BaseIdx);
        AnchorOppSupport = ParentMeta.oppSupport(BaseIdx);
        AnchorObjScore = ParentMeta.objScore(BaseIdx);
        Improved = CandidateMeta.margin < AnchorMargin;
        Supported = CandidateMeta.oppSupport >= AnchorOppSupport;
        if ~(Improved && Supported)
            continue;
        end

        CandidateCount = CandidateCount + 1;
        CandidateDec(CandidateCount,:) = Child;
        CandidateAnchorObjScore(CandidateCount,1) = AnchorObjScore;
        CandidateAnchorOppSupport(CandidateCount,1) = AnchorOppSupport;
    end

    if CandidateCount <= 0
        Offspring = B([]);
        return;
    end
    CandidateDec = CandidateDec(1:CandidateCount,:);
    CandidateAnchorObjScore = CandidateAnchorObjScore(1:CandidateCount);
    CandidateAnchorOppSupport = CandidateAnchorOppSupport(1:CandidateCount);

    EvaluatedCandidate = Problem.Evaluation(CandidateDec);
    CandidateMeta = BuildBoundaryMeta(EvaluatedCandidate,W,Model,PopulationC,PopulationU,SupportBase);
    Relevant = CandidateMeta.objScore <= CandidateAnchorObjScore + 0.10 & ...
        CandidateMeta.oppSupport >= CandidateAnchorOppSupport;
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

function AnchorIdx = SelectBoundaryAnchors(ParentPool,B,ParentMeta,W,MaxCount)
    AnchorIdx = find(ismember(ParentPool.decs,B.decs,'rows'));
    if isempty(AnchorIdx)
        return;
    end

    if isempty(W)
        AnchorIdx = SortByBoundaryKey(ParentMeta,AnchorIdx);
        AnchorIdx = AnchorIdx(1:min(MaxCount,numel(AnchorIdx)));
        return;
    end

    Ranked = cell(size(W,1),1);
    for s = 1 : size(W,1)
        idx = AnchorIdx(ParentMeta.sector(AnchorIdx) == s);
        if isempty(idx)
            continue;
        end
        Ranked{s} = SortByBoundaryKey(ParentMeta,idx);
    end
    Order = SectorRoundRobinPick(Ranked,MaxCount);
    AnchorIdx = Order(:,2);
end

function Parent2 = SelectOppositeParent(ParentPool,ParentMeta,BaseIdx,PopulationC,PopulationU,W)
    if ParentMeta.feasible(BaseIdx) > 0
        Pool = PopulationU(any(PopulationU.cons>0,2));
    else
        Pool = PopulationC(all(PopulationC.cons<=0,2));
    end
    Parent2 = SelectLocalOppositeSample( ...
        ParentPool(BaseIdx).decs,ParentMeta.sector(BaseIdx),Pool,W);
end

function Parent3 = SelectStableParentFast(StablePool,StablePoolMeta,SectorId)
    Parent3 = StablePool([]);
    if isempty(StablePool)
        return;
    end

    idx = find(StablePoolMeta.sector == SectorId);
    if isempty(idx)
        return;
    end
    ord = SortBySelectionKey(StablePoolMeta,idx);
    Parent3 = StablePool(ord(1));
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
    if Need > 0 && ~isempty(Model) && ~isempty(Infeasible)
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
    Trusted = FindSupportedBoundaryCandidates(Meta,W);
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

function Trusted = FindSupportedBoundaryCandidates(Meta,W)
    Trusted = false(numel(Meta.sector),1);
    if isempty(Meta.sector)
        return;
    end
    Threshold = ResolveOppSupportGate(Meta,W);
    Trusted = Meta.oppSupport >= Threshold;
end

function Threshold = ResolveOppSupportGate(Meta,W)
    Positive = Meta.oppSupport(Meta.oppSupport > 0);
    if isempty(Positive)
        Threshold = inf;
        return;
    end
    Base = median(Positive);
    if isempty(W)
        Threshold = max(0.05,0.50*Base);
    else
        Threshold = max(0.05,0.35*Base);
    end
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
        'objScore',zeros(Count,1), ...
        'score',ones(Count,1));
end

function Meta = ComposeBoundaryMeta(Sector,Feasible,Prob,OppSupport,ObjScore)
    Count = max([numel(Sector),numel(Feasible),numel(Prob),numel(OppSupport),numel(ObjScore),0]);
    Meta = InitBoundaryMeta(Count);
    if Count == 0
        return;
    end

    Sector = MatchLength(ColumnVector(Sector),Count,0);
    Feasible = MatchLength(ColumnVector(Feasible),Count,0);
    Prob = MatchLength(ColumnVector(Prob),Count,0.5);
    OppSupport = MatchLength(ColumnVector(OppSupport),Count,0);
    ObjScore = MatchLength(ColumnVector(ObjScore),Count,0);
    Margin = 2*abs(Prob - 0.5);

    Meta.sector     = Sector;
    Meta.feasible   = Feasible;
    Meta.prob       = Prob;
    Meta.margin     = Margin;
    Meta.oppSupport = OppSupport;
    Meta.objScore   = ObjScore;
    Meta.score      = 0.50*Margin + 0.30*(1-OppSupport) + 0.20*ObjScore;
end

function Order = SortByBoundaryKey(Meta,idx)
    idx = ColumnVector(idx);
    if isempty(idx)
        Order = idx;
        return;
    end
    Key = [ColumnVector(Meta.margin(idx)),ColumnVector(1-Meta.oppSupport(idx)),ColumnVector(idx)];
    [~,ord] = sortrows(Key,[1 2 3]);
    Order = idx(ord);
end

function Order = SortBySelectionKey(Meta,idx)
    idx = ColumnVector(idx);
    if isempty(idx)
        Order = idx;
        return;
    end
    Key = [ColumnVector(Meta.margin(idx)),ColumnVector(1-Meta.oppSupport(idx)),ColumnVector(Meta.objScore(idx)),ColumnVector(idx)];
    [~,ord] = sortrows(Key,[1 2 3 4]);
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
        if ~any(Mask)
            Mask = ColumnVector(SupportCache.Label ~= CandidateLabel(i));
        end
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
