function [Pool,Diag] = GenerateBoundaryCandidates(Problem,FeasibleAnchors,PopulationU,W,RuntimeOptions)
% Build bridge records for active feasible-infeasible sector pairs.

    if nargin < 5 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    Pool = InitBridgePool(Problem);
    Diag = InitBridgeGenerationDiag();

    FeasibleC   = FilterFeasiblePopulation(FeasibleAnchors);
    InfeasibleU = PopulationU(~all(PopulationU.cons<=0,2));
    Diag.feasibleAnchorCount = numel(FeasibleC);
    Diag.infeasibleHelperCount = numel(InfeasibleU);
    Diag.deltaG = ResolveBridgeActivationGap(RuntimeOptions);
    if isempty(FeasibleC) || isempty(InfeasibleU)
        return;
    end

    RefObj  = [FeasibleC.objs;InfeasibleU.objs];
    SectorF = AssociateSectors(FeasibleC.objs,W,RefObj);
    SectorU = AssociateSectors(InfeasibleU.objs,W,RefObj);
    ScalarF = ComputeSectorScalar(FeasibleC.objs,W,RefObj,SectorF);
    ScalarU = ComputeSectorScalar(InfeasibleU.objs,W,RefObj,SectorU);
    DeltaG  = Diag.deltaG;
    PairMode = ResolveBridgePairMode(RuntimeOptions);
    PairTopK = ResolveBridgePairTopK(RuntimeOptions);
    PairKeepM = ResolveBridgePairKeepM(RuntimeOptions);
    DistanceWeight = ResolveBridgePairDistanceWeight(RuntimeOptions);
    GateMode = ResolveBridgeGateMode(RuntimeOptions);

    FeasibleSector = unique(SectorF(:),'stable');
    InfeasibleSector = unique(SectorU(:),'stable');
    Diag.feasibleSectorCount = numel(FeasibleSector);
    Diag.infeasibleSectorCount = numel(InfeasibleSector);
    SharedSector = intersect(FeasibleSector,InfeasibleSector,'stable');
    Diag.sharedSectorCount = numel(SharedSector);
    if isempty(SharedSector)
        return;
    end

    switch PairMode
        case 1
            [BestFIdx,BestUIdx,RawMargin,SharedSector] = SelectCurrentBridgePairs( ...
                SectorF,ScalarF,SectorU,ScalarU,SharedSector);
            Margin = RawMargin - DeltaG;
            StrictActive = Margin > 0;
            WeakActive = RawMargin > 0;
            UseWeakGate = GateMode == 1 && ~any(StrictActive) && any(WeakActive);
            if UseWeakGate
                Active = WeakActive;
            else
                Active = StrictActive;
            end
            Diag.strictActiveSectorCount = nnz(StrictActive);
            Diag.weakActiveSectorCount = nnz(WeakActive);
            Diag.usedWeakGate = UseWeakGate;
            Diag.minRawMargin = SafeMarginStat(RawMargin,@min);
            Diag.medianRawMargin = SafeMarginStat(RawMargin,@median);
            Diag.maxRawMargin = SafeMarginStat(RawMargin,@max);
            Diag.minActivationMargin = SafeMarginStat(Margin,@min);
            Diag.medianActivationMargin = SafeMarginStat(Margin,@median);
            Diag.maxActivationMargin = SafeMarginStat(Margin,@max);
            Diag.activeSectorCount = nnz(Active);
            Diag.positiveMarginPairCount = nnz(WeakActive);
            Diag.strictPositivePairCount = nnz(StrictActive);
            Diag.gatePassedPairCount = nnz(Active);
            SharedSector = SharedSector(Active);
            Count = numel(SharedSector);
            if Count == 0
                return;
            end

            Pool.source    = ones(Count,1);
            Pool.sector    = SharedSector(:);
            Pool.anchorDec = FeasibleC(BestFIdx(Active)).decs;
            Pool.anchorObj = FeasibleC(BestFIdx(Active)).objs;
            Pool.helperDec = InfeasibleU(BestUIdx(Active)).decs;
            Pool.helperObj = InfeasibleU(BestUIdx(Active)).objs;
        otherwise
            [PairPool,PairDiag] = BuildTopKBridgePairs( ...
                Problem,FeasibleC,InfeasibleU,SectorF,ScalarF,SectorU,ScalarU, ...
                SharedSector,PairTopK,PairKeepM,DistanceWeight,DeltaG,GateMode);
            Diag.strictActiveSectorCount = PairDiag.strictActiveSectorCount;
            Diag.weakActiveSectorCount = PairDiag.weakActiveSectorCount;
            Diag.usedWeakGate = PairDiag.usedWeakGate;
            Diag.minRawMargin = SafeMarginStat(PairDiag.rawMargin,@min);
            Diag.medianRawMargin = SafeMarginStat(PairDiag.rawMargin,@median);
            Diag.maxRawMargin = SafeMarginStat(PairDiag.rawMargin,@max);
            Diag.minActivationMargin = SafeMarginStat(PairDiag.activationMargin,@min);
            Diag.medianActivationMargin = SafeMarginStat(PairDiag.activationMargin,@median);
            Diag.maxActivationMargin = SafeMarginStat(PairDiag.activationMargin,@max);
            Diag.activeSectorCount = PairDiag.activeSectorCount;
            Diag.positiveMarginPairCount = PairDiag.positiveMarginPairCount;
            Diag.strictPositivePairCount = PairDiag.strictPositivePairCount;
            Diag.gatePassedPairCount = PairDiag.gatePassedPairCount;
            Pool = PairPool;
    end
end

function Diag = InitBridgeGenerationDiag()
    Diag = struct( ...
        'feasibleAnchorCount',0, ...
        'infeasibleHelperCount',0, ...
        'feasibleSectorCount',0, ...
        'infeasibleSectorCount',0, ...
        'sharedSectorCount',0, ...
        'activeSectorCount',0, ...
        'strictActiveSectorCount',0, ...
        'weakActiveSectorCount',0, ...
        'usedWeakGate',false, ...
        'deltaG',0, ...
        'positiveMarginPairCount',0, ...
        'strictPositivePairCount',0, ...
        'gatePassedPairCount',0, ...
        'minRawMargin',NaN, ...
        'medianRawMargin',NaN, ...
        'maxRawMargin',NaN, ...
        'minActivationMargin',NaN, ...
        'medianActivationMargin',NaN, ...
        'maxActivationMargin',NaN);
end

function Value = SafeMarginStat(Data,Func)
    Value = NaN;
    if isempty(Data)
        return;
    end
    Data = Data(isfinite(Data));
    if isempty(Data)
        return;
    end
    Value = Func(Data);
end

function Population = FilterFeasiblePopulation(Population)
    if isempty(Population)
        return;
    end
    Population = Population(all(Population.cons<=0,2));
end

function Pool = InitBridgePool(Problem)
    Pool.source    = zeros(0,1);
    Pool.sector    = zeros(0,1);
    Pool.anchorDec = zeros(0,Problem.D);
    Pool.anchorObj = zeros(0,Problem.M);
    Pool.helperDec = zeros(0,Problem.D);
    Pool.helperObj = zeros(0,Problem.M);
end

function DeltaG = ResolveBridgeActivationGap(RuntimeOptions)
    DeltaG = 0.01;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgeActivationGap') ...
            && ~isempty(RuntimeOptions.BridgeActivationGap)
        DeltaG = max(RuntimeOptions.BridgeActivationGap,0);
    end
end

function Mode = ResolveBridgePairMode(RuntimeOptions)
    Mode = 2;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'PairMode') ...
            && ~isempty(RuntimeOptions.PairMode)
        Mode = max(1,min(2,round(RuntimeOptions.PairMode)));
    end
end

function TopK = ResolveBridgePairTopK(RuntimeOptions)
    TopK = 3;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgePairTopK') ...
            && ~isempty(RuntimeOptions.BridgePairTopK)
        TopK = max(1,round(RuntimeOptions.BridgePairTopK));
    end
end

function KeepM = ResolveBridgePairKeepM(RuntimeOptions)
    KeepM = 2;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgePairKeepM') ...
            && ~isempty(RuntimeOptions.BridgePairKeepM)
        KeepM = max(1,round(RuntimeOptions.BridgePairKeepM));
    end
end

function Weight = ResolveBridgePairDistanceWeight(RuntimeOptions)
    Weight = 0.05;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgePairDistanceWeight') ...
            && ~isempty(RuntimeOptions.BridgePairDistanceWeight)
        Weight = max(RuntimeOptions.BridgePairDistanceWeight,0);
    end
end

function Mode = ResolveBridgeGateMode(RuntimeOptions)
    Mode = 1;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'GateMode') ...
            && ~isempty(RuntimeOptions.GateMode)
        Mode = max(1,min(2,round(RuntimeOptions.GateMode)));
    end
end

function [BestFIdx,BestUIdx,RawMargin,SharedSector] = SelectCurrentBridgePairs( ...
    SectorF,ScalarF,SectorU,ScalarU,SharedSector)

    [BestFIdxAll,BestFSector,BestFValue] = SelectSectorBest(SectorF,ScalarF);
    [BestUIdxAll,BestUSector,BestUValue] = SelectSectorBest(SectorU,ScalarU);
    [SharedSector,FLoc,ULoc] = intersect(BestFSector,BestUSector,'stable');
    BestFIdx = BestFIdxAll(FLoc);
    BestUIdx = BestUIdxAll(ULoc);
    RawMargin = BestFValue(FLoc) - BestUValue(ULoc);
end

function [BestFIdx,BestUIdx,RawMargin] = SelectTopKBridgePairs( ...
    Problem,FeasibleC,InfeasibleU,SectorF,ScalarF,SectorU,ScalarU,SharedSector,TopK,DistanceWeight)

    Count = numel(SharedSector);
    BestFIdx = zeros(Count,1);
    BestUIdx = zeros(Count,1);
    RawMargin = -inf(Count,1);
    FeasibleNorm = NormalizeDecisionVectors(Problem,FeasibleC.decs);
    InfeasibleNorm = NormalizeDecisionVectors(Problem,InfeasibleU.decs);

    for i = 1 : Count
        SectorID = SharedSector(i);
        FIdx = SelectSectorTopK(SectorF,ScalarF,SectorID,TopK);
        UIdx = SelectSectorTopK(SectorU,ScalarU,SectorID,TopK);
        [LocalF,LocalU,LocalMargin] = SelectBestBridgePair( ...
            ScalarF(FIdx),ScalarU(UIdx),FeasibleNorm(FIdx,:),InfeasibleNorm(UIdx,:),DistanceWeight);
        BestFIdx(i) = FIdx(LocalF);
        BestUIdx(i) = UIdx(LocalU);
        RawMargin(i) = LocalMargin;
    end
end

function [Pool,Diag] = BuildTopKBridgePairs( ...
    Problem,FeasibleC,InfeasibleU,SectorF,ScalarF,SectorU,ScalarU,SharedSector, ...
    TopK,KeepM,DistanceWeight,DeltaG,GateMode)

    Count = numel(SharedSector);
    Pool = InitBridgePool(Problem);
    Diag = struct( ...
        'strictActiveSectorCount',0, ...
        'weakActiveSectorCount',0, ...
        'activeSectorCount',0, ...
        'usedWeakGate',false, ...
        'positiveMarginPairCount',0, ...
        'strictPositivePairCount',0, ...
        'gatePassedPairCount',0, ...
        'rawMargin',-inf(Count,1), ...
        'activationMargin',-inf(Count,1));
    if Count == 0
        return;
    end

    FeasibleNorm = NormalizeDecisionVectors(Problem,FeasibleC.decs);
    InfeasibleNorm = NormalizeDecisionVectors(Problem,InfeasibleU.decs);
    SectorPair = repmat(struct( ...
        'sectorID',NaN, ...
        'FIdx',zeros(0,1), ...
        'UIdx',zeros(0,1), ...
        'rawMargin',zeros(0,0), ...
        'distance',zeros(0,0), ...
        'score',zeros(0,0), ...
        'strictMask',false(0,0), ...
        'weakMask',false(0,0)),Count,1);
    StrictActive = false(Count,1);
    WeakActive = false(Count,1);
    for i = 1 : Count
        SectorID = SharedSector(i);
        FIdx = SelectSectorTopK(SectorF,ScalarF,SectorID,TopK);
        UIdx = SelectSectorTopK(SectorU,ScalarU,SectorID,TopK);
        [RawMarginMatrix,DistanceMatrix,PairScore] = ComputeBridgePairMatrices( ...
            ScalarF(FIdx),ScalarU(UIdx),FeasibleNorm(FIdx,:),InfeasibleNorm(UIdx,:),DistanceWeight);
        StrictMask = RawMarginMatrix - DeltaG > 0;
        WeakMask = RawMarginMatrix > 0;
        SectorPair(i).sectorID = SectorID;
        SectorPair(i).FIdx = FIdx;
        SectorPair(i).UIdx = UIdx;
        SectorPair(i).rawMargin = RawMarginMatrix;
        SectorPair(i).distance = DistanceMatrix;
        SectorPair(i).score = PairScore;
        SectorPair(i).strictMask = StrictMask;
        SectorPair(i).weakMask = WeakMask;
        Diag.rawMargin(i) = SafeMarginStat(RawMarginMatrix(:),@max);
        Diag.activationMargin(i) = SafeMarginStat((RawMarginMatrix(:) - DeltaG),@max);
        StrictActive(i) = any(StrictMask(:));
        WeakActive(i) = any(WeakMask(:));
        Diag.strictPositivePairCount = Diag.strictPositivePairCount + nnz(StrictMask);
        Diag.positiveMarginPairCount = Diag.positiveMarginPairCount + nnz(WeakMask);
    end

    Diag.strictActiveSectorCount = nnz(StrictActive);
    Diag.weakActiveSectorCount = nnz(WeakActive);
    UseWeakGate = GateMode == 1 && ~any(StrictActive) && any(WeakActive);
    Diag.usedWeakGate = UseWeakGate;
    if UseWeakGate
        ActiveSector = WeakActive;
    else
        ActiveSector = StrictActive;
    end
    Diag.activeSectorCount = nnz(ActiveSector);
    if ~any(ActiveSector)
        return;
    end

    for i = find(ActiveSector(:))'
        if UseWeakGate
            ValidMask = SectorPair(i).weakMask;
        else
            ValidMask = SectorPair(i).strictMask;
        end
        LinearIdx = SelectTopBridgeLinearIndices( ...
            SectorPair(i).score,SectorPair(i).rawMargin,SectorPair(i).distance,KeepM,ValidMask);
        if isempty(LinearIdx)
            continue;
        end
        Diag.gatePassedPairCount = Diag.gatePassedPairCount + numel(LinearIdx);
        [LocalF,LocalU] = ind2sub(size(SectorPair(i).score),LinearIdx);
        CountAdd = numel(LocalF);
        Pool.source = [Pool.source;ones(CountAdd,1)]; %#ok<AGROW>
        Pool.sector = [Pool.sector;repmat(SectorPair(i).sectorID,CountAdd,1)]; %#ok<AGROW>
        Pool.anchorDec = [Pool.anchorDec;FeasibleC(SectorPair(i).FIdx(LocalF)).decs]; %#ok<AGROW>
        Pool.anchorObj = [Pool.anchorObj;FeasibleC(SectorPair(i).FIdx(LocalF)).objs]; %#ok<AGROW>
        Pool.helperDec = [Pool.helperDec;InfeasibleU(SectorPair(i).UIdx(LocalU)).decs]; %#ok<AGROW>
        Pool.helperObj = [Pool.helperObj;InfeasibleU(SectorPair(i).UIdx(LocalU)).objs]; %#ok<AGROW>
    end
end

function [BestIdx,BestSector,BestValue] = SelectSectorBest(Sector,Scalar)
    if isempty(Sector)
        BestIdx = zeros(0,1);
        BestSector = zeros(0,1);
        BestValue = zeros(0,1);
        return;
    end

    RankTable = [(1:numel(Sector))',Sector(:),Scalar(:)];
    RankTable = sortrows(RankTable,[2 3 1]);
    [~,First] = unique(RankTable(:,2),'stable');
    BestIdx    = RankTable(First,1);
    BestSector = RankTable(First,2);
    BestValue  = RankTable(First,3);
end

function Idx = SelectSectorTopK(Sector,Scalar,SectorID,TopK)
    Idx = find(Sector == SectorID);
    if isempty(Idx)
        return;
    end
    [~,Order] = sort(Scalar(Idx),'ascend');
    Idx = Idx(Order(1:min(TopK,numel(Order))));
end

function [BestF,BestU,BestMargin] = SelectBestBridgePair(FValue,UValue,FNorm,UNorm,DistanceWeight)
    [RawMarginMatrix,DistanceMatrix,PairScore] = ComputeBridgePairMatrices( ...
        FValue,UValue,FNorm,UNorm,DistanceWeight);
    BestLinear = BreakBridgePairTie(PairScore,RawMarginMatrix,DistanceMatrix);
    [BestF,BestU] = ind2sub(size(PairScore),BestLinear);
    BestMargin = RawMarginMatrix(BestF,BestU);
end

function [RawMarginMatrix,DistanceMatrix,PairScore] = ComputeBridgePairMatrices( ...
    FValue,UValue,FNorm,UNorm,DistanceWeight)
    RawMarginMatrix = bsxfun(@minus,FValue(:),UValue(:)');
    DistanceMatrix = pdist2(FNorm,UNorm);
    PairScore = max(RawMarginMatrix,0) - DistanceWeight*DistanceMatrix;
end

function LinearIdx = SelectTopBridgeLinearIndices(PairScore,RawMargin,Distance,KeepM,ValidMask)
    [RowIdx,ColIdx] = find(ValidMask);
    if isempty(RowIdx)
        LinearIdx = zeros(0,1);
        return;
    end
    Count = numel(RowIdx);
    RankTable = zeros(Count,5);
    for i = 1 : Count
        RankTable(i,1) = -PairScore(RowIdx(i),ColIdx(i));
        RankTable(i,2) = -RawMargin(RowIdx(i),ColIdx(i));
        RankTable(i,3) = Distance(RowIdx(i),ColIdx(i));
        RankTable(i,4) = RowIdx(i);
        RankTable(i,5) = ColIdx(i);
    end
    RankTable = sortrows(RankTable,[1 2 3 4 5]);
    Keep = min(KeepM,size(RankTable,1));
    LinearIdx = sub2ind(size(PairScore),RankTable(1:Keep,4),RankTable(1:Keep,5));
end

function Index = BreakBridgePairTie(PairScore,RawMargin,Distance)
    BestValue = max(PairScore(:));
    [RowIdx,ColIdx] = find(PairScore == BestValue);
    if numel(RowIdx) <= 1
        Index = sub2ind(size(PairScore),RowIdx(1),ColIdx(1));
        return;
    end

    CandidateMargin = zeros(numel(RowIdx),1);
    CandidateDistance = zeros(numel(RowIdx),1);
    for i = 1 : numel(RowIdx)
        CandidateMargin(i) = RawMargin(RowIdx(i),ColIdx(i));
        CandidateDistance(i) = Distance(RowIdx(i),ColIdx(i));
    end
    BestMask = CandidateMargin == max(CandidateMargin);
    CandidateIdx = find(BestMask);
    if numel(CandidateIdx) > 1
        [~,Local] = min(CandidateDistance(CandidateIdx));
        BestPos = CandidateIdx(Local(1));
    else
        BestPos = CandidateIdx(1);
    end
    Index = sub2ind(size(PairScore),RowIdx(BestPos),ColIdx(BestPos));
end

function DecNorm = NormalizeDecisionVectors(Problem,Dec)
    if isempty(Dec)
        DecNorm = zeros(0,Problem.D);
        return;
    end
    Lower = Problem.lower;
    Upper = Problem.upper;
    Span = Upper - Lower;
    Span(Span<1e-12) = 1;
    DecNorm = (double(Dec) - repmat(Lower,size(Dec,1),1))./repmat(Span,size(Dec,1),1);
end
