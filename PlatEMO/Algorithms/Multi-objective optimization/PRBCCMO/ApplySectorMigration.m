function Population = ApplySectorMigration(Population,MigrationPool,W)
% Replace only the weakest solutions in sectors improved by feasible boundary expansions.

    if isempty(Population) || isempty(MigrationPool)
        return;
    end

    FeasibleMask = all(Population.cons<=0,2);
    FeasiblePop  = Population(FeasibleMask);
    if isempty(FeasiblePop)
        return;
    end

    RefObj  = [FeasiblePop.objs;MigrationPool.objs];
    SectorP = AssociateSectors(FeasiblePop.objs,W,RefObj);
    SectorM = AssociateSectors(MigrationPool.objs,W,RefObj);
    RemoveMask = false(1,numel(FeasiblePop));

    for s = unique(SectorM(:))'
        Cand = MigrationPool(SectorM==s);
        if isempty(Cand)
            continue;
        end
        SectorIdx = find(SectorP==s);
        if isempty(SectorIdx)
            continue;
        end

        BestCand = SelectBestByScalar(Cand,RefObj);
        BestInc  = SelectBestByScalar(FeasiblePop(SectorIdx),RefObj);
        if ~ImprovesSector(BestCand.obj,BestInc.obj,RefObj)
            continue;
        end

        WorstCount = max(1,ceil(0.2*numel(SectorIdx)));
        ScalarValue = ComputeScalarValue(FeasiblePop(SectorIdx).objs,RefObj);
        [~,Rank] = sort(ScalarValue,'descend');
        RemoveMask(SectorIdx(Rank(1:WorstCount))) = true;
    end

    KeepFeasible = ~RemoveMask;
    Population = [FeasiblePop(KeepFeasible),Population(~FeasibleMask)];
end

function Best = SelectBestByScalar(Population,RefObj)
    Value = ComputeScalarValue(Population.objs,RefObj);
    [~,BestIdx] = min(Value);
    Best = Population(BestIdx);
end

function flag = ImprovesSector(CandObj,IncObj,RefObj)
    flag = all(CandObj<=IncObj) && any(CandObj<IncObj);
    if flag
        return;
    end
    CandValue = ComputeScalarValue(CandObj,RefObj);
    IncValue  = ComputeScalarValue(IncObj,RefObj);
    flag = CandValue < IncValue;
end

function Value = ComputeScalarValue(Obj,RefObj)
    MinObj = min(RefObj,[],1);
    Range  = max(RefObj,[],1) - MinObj;
    Range(Range<1e-12) = 1;
    Value = sum((Obj-MinObj)./Range,2);
end
