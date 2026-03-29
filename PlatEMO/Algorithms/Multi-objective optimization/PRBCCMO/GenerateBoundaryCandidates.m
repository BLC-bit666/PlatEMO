function Pool = GenerateBoundaryCandidates(Problem,FeasibleAnchors,PopulationU,W,RuntimeOptions)
% Build bridge records for active feasible-infeasible sector pairs.

    if nargin < 5 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    Pool = InitBridgePool(Problem);

    FeasibleC   = FilterFeasiblePopulation(FeasibleAnchors);
    InfeasibleU = PopulationU(~all(PopulationU.cons<=0,2));
    if isempty(FeasibleC) || isempty(InfeasibleU)
        return;
    end

    RefObj  = [FeasibleC.objs;InfeasibleU.objs];
    SectorF = AssociateSectors(FeasibleC.objs,W,RefObj);
    SectorU = AssociateSectors(InfeasibleU.objs,W,RefObj);
    ScalarF = ComputeSectorScalar(FeasibleC.objs,W,RefObj,SectorF);
    ScalarU = ComputeSectorScalar(InfeasibleU.objs,W,RefObj,SectorU);
    DeltaG  = ResolveBridgeActivationGap(RuntimeOptions);

    SharedSector = intersect(unique(SectorF(:),'stable'),unique(SectorU(:),'stable'),'stable');
    if isempty(SharedSector)
        return;
    end

    [BestFIdx,BestUIdx,RawMargin,SharedSector] = SelectTopKBridgePairs( ...
        SectorF,ScalarF,SectorU,ScalarU,SharedSector,ResolveBridgeTopK(RuntimeOptions));
    Margin = RawMargin - DeltaG;
    StrictActive = Margin > 0;
    WeakActive = RawMargin > 0;
    if ~any(StrictActive) && any(WeakActive)
        Active = WeakActive;
    else
        Active = StrictActive;
    end

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

function [BestFIdx,BestUIdx,RawMargin,SharedSector] = SelectTopKBridgePairs( ...
    SectorF,ScalarF,SectorU,ScalarU,SharedSector,TopK)

    Count = numel(SharedSector);
    BestFIdx = zeros(Count,1);
    BestUIdx = zeros(Count,1);
    RawMargin = zeros(Count,1);
    Keep = false(Count,1);
    TopK = max(1,round(TopK));
    for i = 1 : Count
        SectorID = SharedSector(i);
        FeasibleIdx = find(SectorF == SectorID);
        InfeasibleIdx = find(SectorU == SectorID);
        if isempty(FeasibleIdx) || isempty(InfeasibleIdx)
            continue;
        end

        FRank = sortrows([FeasibleIdx(:),ScalarF(FeasibleIdx(:))],[2 1]);
        URank = sortrows([InfeasibleIdx(:),ScalarU(InfeasibleIdx(:))],[2 1]);
        FTop = FRank(1:min(TopK,size(FRank,1)),1);
        UTop = URank(1:min(TopK,size(URank,1)),1);
        MarginTable = zeros(numel(FTop)*numel(UTop),4);
        Row = 0;
        for f = 1 : numel(FTop)
            for u = 1 : numel(UTop)
                Row = Row + 1;
                MarginTable(Row,1) = FTop(f);
                MarginTable(Row,2) = UTop(u);
                MarginTable(Row,3) = ScalarF(FTop(f)) - ScalarU(UTop(u));
                MarginTable(Row,4) = abs(ScalarF(FTop(f)) - ScalarU(UTop(u)));
            end
        end
        MarginTable = sortrows(MarginTable,[-3 4 1 2]);
        BestFIdx(i) = MarginTable(1,1);
        BestUIdx(i) = MarginTable(1,2);
        RawMargin(i) = MarginTable(1,3);
        Keep(i) = true;
    end
    BestFIdx = BestFIdx(Keep);
    BestUIdx = BestUIdx(Keep);
    RawMargin = RawMargin(Keep);
    SharedSector = SharedSector(Keep);
end

function TopK = ResolveBridgeTopK(RuntimeOptions)
    TopK = 5;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgeTopK') ...
            && ~isempty(RuntimeOptions.BridgeTopK)
        TopK = RuntimeOptions.BridgeTopK;
    end
end
