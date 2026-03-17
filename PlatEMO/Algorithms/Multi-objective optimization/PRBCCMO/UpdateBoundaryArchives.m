function [ArchiveF,ArchiveI] = UpdateBoundaryArchives(Problem,ArchiveF,ArchiveI,BoundaryOffspring,PopulationC,Model,W,HardNegativeArchive,ArchiveFMax,ArchiveIMax)
% Update feasible and infeasible boundary archives with utility-aware ranking.

    if isempty(BoundaryOffspring) && isempty(ArchiveF) && isempty(ArchiveI)
        return;
    end

    CurrentFeasible = PopulationC(all(PopulationC.cons<=0,2));
    if isempty(CurrentFeasible)
        FeasibleObj = zeros(0,Problem.M);
    else
        FeasibleObj = CurrentFeasible.objs;
    end
    if isempty(BoundaryOffspring)
        BoundaryFeasible   = [];
        BoundaryInfeasible = [];
    else
        BoundaryFeasible   = BoundaryOffspring(all(BoundaryOffspring.cons<=0,2));
        BoundaryInfeasible = BoundaryOffspring(~all(BoundaryOffspring.cons<=0,2));
    end

    ArchiveF = UpdateArchiveByUtility( ...
        Problem,ArchiveF,BoundaryFeasible,FeasibleObj,Model,W,HardNegativeArchive,ArchiveFMax);
    ArchiveI = UpdateArchiveByUtility( ...
        Problem,ArchiveI,BoundaryInfeasible,FeasibleObj,Model,W,HardNegativeArchive,ArchiveIMax);
end

function Archive = UpdateArchiveByUtility(Problem,Archive,NewSolutions,FeasibleObj,Model,W,HardNegativeArchive,MaxSize)
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
    if isempty(Pool)
        Archive = [];
        return;
    end

    Detail = ScoreBoundaryCandidates( ...
        Problem,Pool.decs,Pool.objs,FeasibleObj,Model,W,HardNegativeArchive);
    [~,Rank] = sort(Detail.utility(:),'descend');
    Archive = Pool(Rank(1:min(MaxSize,length(Rank))));
end
