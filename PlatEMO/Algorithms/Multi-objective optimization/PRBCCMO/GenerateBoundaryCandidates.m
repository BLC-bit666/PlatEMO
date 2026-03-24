function [Pool,Diag] = GenerateBoundaryCandidates(Problem,FeasibleAnchors,PopulationU,W,RuntimeOptions)
% Build one bridge record for each active feasible-infeasible sector pair.

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

    [BestFIdx,BestFSector,BestFValue] = SelectSectorBest(SectorF,ScalarF);
    [BestUIdx,BestUSector,BestUValue] = SelectSectorBest(SectorU,ScalarU);
    Diag.feasibleSectorCount = numel(BestFSector);
    Diag.infeasibleSectorCount = numel(BestUSector);
    [SharedSector,FLoc,ULoc] = intersect(BestFSector,BestUSector,'stable');
    Diag.sharedSectorCount = numel(SharedSector);
    if isempty(SharedSector)
        return;
    end

    RawMargin = BestFValue(FLoc) - BestUValue(ULoc);
    Margin = RawMargin - DeltaG;
    StrictActive = Margin > 0;
    WeakActive = RawMargin > 0;
    UseWeakGate = ~any(StrictActive) && any(WeakActive);
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
    SharedSector = SharedSector(Active);
    FLoc = FLoc(Active);
    ULoc = ULoc(Active);
    Count = numel(SharedSector);
    if Count == 0
        return;
    end

    Pool.source    = ones(Count,1);
    Pool.sector    = SharedSector(:);
    Pool.anchorDec = FeasibleC(BestFIdx(FLoc)).decs;
    Pool.anchorObj = FeasibleC(BestFIdx(FLoc)).objs;
    Pool.helperDec = InfeasibleU(BestUIdx(ULoc)).decs;
    Pool.helperObj = InfeasibleU(BestUIdx(ULoc)).objs;
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

function DeltaG = ResolveBridgeActivationGap(RuntimeOptions)
    DeltaG = 0.01;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgeActivationGap') ...
            && ~isempty(RuntimeOptions.BridgeActivationGap)
        DeltaG = max(RuntimeOptions.BridgeActivationGap,0);
    end
end
