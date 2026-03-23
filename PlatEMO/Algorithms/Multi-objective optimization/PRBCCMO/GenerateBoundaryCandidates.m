function Pool = GenerateBoundaryCandidates(Problem,PopulationC,PopulationU,W,RuntimeOptions)
% Build one bridge record for each active feasible-infeasible sector pair.

    if nargin < 5 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    Pool = InitBridgePool(Problem);

    FeasibleC   = PopulationC(all(PopulationC.cons<=0,2));
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

    [BestFIdx,BestFSector,BestFValue] = SelectSectorBest(SectorF,ScalarF);
    [BestUIdx,BestUSector,BestUValue] = SelectSectorBest(SectorU,ScalarU);
    [SharedSector,FLoc,ULoc] = intersect(BestFSector,BestUSector,'stable');
    if isempty(SharedSector)
        return;
    end

    Active = BestUValue(ULoc) + DeltaG < BestFValue(FLoc);
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
