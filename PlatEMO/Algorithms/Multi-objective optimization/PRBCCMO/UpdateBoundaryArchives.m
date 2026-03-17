function [ArchiveF,ArchiveI] = UpdateBoundaryArchives(ArchiveF,ArchiveI,BoundaryOffspring,PopulationC,Model,W,ArchiveFMax,ArchiveIMax)
% Update feasible and infeasible boundary archives.

    if isempty(BoundaryOffspring) && isempty(ArchiveF) && isempty(ArchiveI)
        return;
    end

    CurrentFeasible = PopulationC(all(PopulationC.cons<=0,2));
    if isempty(BoundaryOffspring)
        BoundaryFeasible   = [];
        BoundaryInfeasible = [];
    else
        BoundaryFeasible   = BoundaryOffspring(all(BoundaryOffspring.cons<=0,2));
        BoundaryInfeasible = BoundaryOffspring(~all(BoundaryOffspring.cons<=0,2));
    end

    ArchiveF = UpdateFeasibleArchive(ArchiveF,BoundaryFeasible,Model,W,ArchiveFMax);
    ArchiveI = UpdateInfeasibleArchive(ArchiveI,BoundaryInfeasible,CurrentFeasible,Model,W,ArchiveIMax);
end

function Archive = UpdateFeasibleArchive(Archive,NewSolutions,Model,W,MaxSize)
    if MaxSize <= 0
        Archive = [];
        return;
    end
    Pool = [Archive,NewSolutions];
    if isempty(Pool)
        Archive = [];
        return;
    end
    Keep = KeepLatestDecisionRows(Pool.decs);
    Pool = Pool(Keep);
    Score   = 1 - 2*abs(PredictBoundaryMLP(Model,Pool.decs)-0.5);
    FrontNo = NDSort(Pool.objs,size(Pool,2));
    Crowd   = CrowdingDistance(Pool.objs,FrontNo);
    Sector  = AssociateSectors(Pool.objs,W,Pool.objs);
    Count   = accumarray(Sector,1,[size(W,1),1]);
    Sparsity = Count(Sector);
    RankMat = [-Score(:),Sparsity(:),-Crowd(:),FrontNo(:)];
    [~,Rank] = sortrows(RankMat,[1 2 3 4]);
    Archive = Pool(Rank(1:min(MaxSize,length(Rank))));
end

function Archive = UpdateInfeasibleArchive(Archive,NewSolutions,CurrentFeasible,Model,W,MaxSize)
    if MaxSize <= 0
        Archive = [];
        return;
    end
    Pool = [Archive,NewSolutions];
    if isempty(Pool)
        Archive = [];
        return;
    end
    Keep = KeepLatestDecisionRows(Pool.decs);
    Pool = Pool(Keep);

    if isempty(CurrentFeasible)
        CountF  = zeros(size(W,1),1);
        SectorI = AssociateSectors(Pool.objs,W,Pool.objs);
    else
        RefObj = [CurrentFeasible.objs;Pool.objs];
        [SectorF,CountF] = AssociateSectors(CurrentFeasible.objs,W,RefObj);
        SectorI          = AssociateSectors(Pool.objs,W,RefObj);
        CandMask         = true(1,numel(Pool));
        for i = 1 : numel(Pool)
            if CountF(SectorI(i)) <= 1
                continue;
            end
            SameSectorObj = CurrentFeasible(SectorF==SectorI(i)).objs;
            if AnyDominates(SameSectorObj,Pool(i).objs)
                CandMask(i) = false;
            end
        end
        if any(~CandMask)
            Pool    = Pool(CandMask);
            SectorI = SectorI(CandMask);
        end
    end

    if isempty(Pool)
        Archive = [];
        return;
    end

    Score   = 1 - 2*abs(PredictBoundaryMLP(Model,Pool.decs)-0.5);
    FrontNo = NDSort(Pool.objs,size(Pool,2));
    Crowd   = CrowdingDistance(Pool.objs,FrontNo);
    if isempty(CurrentFeasible)
        CountF = accumarray(SectorI,1,[size(W,1),1]);
    end
    Sparsity = CountF(SectorI);
    RankMat  = [-Score(:),Sparsity(:),-Crowd(:),FrontNo(:)];
    [~,Rank] = sortrows(RankMat,[1 2 3 4]);
    Archive  = Pool(Rank(1:min(MaxSize,length(Rank))));
end

function flag = AnyDominates(PopObj,obj)
    flag = false;
    if isempty(PopObj)
        return;
    end
    LessEqual = all(PopObj<=repmat(obj,size(PopObj,1),1),2);
    Less      = any(PopObj<repmat(obj,size(PopObj,1),1),2);
    flag      = any(LessEqual & Less);
end
