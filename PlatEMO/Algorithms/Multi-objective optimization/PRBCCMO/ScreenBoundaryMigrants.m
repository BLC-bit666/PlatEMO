function MigrationPool = ScreenBoundaryMigrants(PopulationC,CandidatePool,W)
% Keep only sector-champion-improving feasible boundary gains.

    BoundaryFeasiblePool = FilterBoundaryFeasiblePool(CandidatePool);
    MigrationPool = [];
    if isempty(BoundaryFeasiblePool)
        return;
    end

    BaseFeasible = FilterFeasiblePopulation(PopulationC);
    if isempty(W)
        W = ones(1,size(BoundaryFeasiblePool.objs,2));
    end

    RefObj = BoundaryFeasiblePool.objs;
    if ~isempty(BaseFeasible)
        RefObj = [BaseFeasible.objs;BoundaryFeasiblePool.objs];
    end

    SectorPool = AssociateSectors(BoundaryFeasiblePool.objs,W,RefObj);
    PoolValue  = ComputeSectorScalar(BoundaryFeasiblePool.objs,W,RefObj,SectorPool);
    DeltaMig   = 0;

    if isempty(BaseFeasible)
        Improve = inf(size(PoolValue));
        Keep = SelectBestPerSector(SectorPool,PoolValue,Improve,DeltaMig);
        SelectIdx = find(Keep);
        [~,Order] = sort(PoolValue(SelectIdx),'ascend');
        MigrationPool = BoundaryFeasiblePool(SelectIdx(Order));
        return;
    end

    SectorBase = AssociateSectors(BaseFeasible.objs,W,RefObj);
    Improve = -inf(size(PoolValue));
    for s = unique(SectorPool(:))'
        CandIdx = find(SectorPool == s);
        BaseIdx = find(SectorBase == s);
        if isempty(CandIdx)
            continue;
        end
        if isempty(BaseIdx)
            Improve(CandIdx) = inf;
            continue;
        end
        ChampionValue = min(ComputeSectorScalar( ...
            BaseFeasible(BaseIdx).objs,W,RefObj,repmat(s,numel(BaseIdx),1)));
        Improve(CandIdx) = ChampionValue - PoolValue(CandIdx);
    end

    Keep = SelectBestPerSector(SectorPool,PoolValue,Improve,DeltaMig);
    SelectIdx = find(Keep);
    [~,Order] = sort(Improve(SelectIdx),'descend');
    MigrationPool = BoundaryFeasiblePool(SelectIdx(Order));
end

function Pool = FilterBoundaryFeasiblePool(Pool)
    if isempty(Pool)
        return;
    end
    Pool = Pool(all(Pool.cons<=0,2));
    Pool = KeepUniquePopulation(Pool);
end

function Keep = SelectBestPerSector(SectorPool,PoolValue,Improve,DeltaMig)
    Keep = false(numel(SectorPool),1);
    for s = unique(SectorPool(:))'
        CandIdx = find(SectorPool == s);
        CandIdx = CandIdx(Improve(CandIdx) > DeltaMig);
        if isempty(CandIdx)
            continue;
        end
        [~,Best] = min(PoolValue(CandIdx));
        Keep(CandIdx(Best)) = true;
    end
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
    Keep = KeepLatestDecisionRows(Population.decs);
    Population = Population(Keep);
end
