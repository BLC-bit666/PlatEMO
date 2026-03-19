function [Archive,Added] = UpdateExternalArchive(Archive,NewSolutions)
% Update the cumulative feasible external archive and return newly added points.

    if nargin < 1 || isempty(Archive)
        Archive = [];
    end
    Added = [];
    if nargin < 2 || isempty(NewSolutions)
        return;
    end

    NewSolutions = NewSolutions(all(NewSolutions.cons<=0,2));
    if isempty(NewSolutions)
        return;
    end

    Previous = Archive;
    Pool = [Archive,NewSolutions];
    Keep = KeepLatestDecisionRows(Pool.decs);
    Pool = Pool(Keep);
    if isempty(Pool)
        Archive = [];
        return;
    end

    FrontNo = NDSort(Pool.objs,1);
    Archive = Pool(FrontNo==1);
    if isempty(Previous)
        Added = Archive;
        return;
    end

    Added = ExtractArchiveAdditions(Archive,Previous);
end

function Added = ExtractArchiveAdditions(Archive,Previous)
    Added = [];
    if isempty(Archive)
        return;
    end
    if isempty(Previous)
        Added = Archive;
        return;
    end

    ArchiveCount = numel(Archive);
    Keep = KeepLatestDecisionRows([Archive.decs;Previous.decs]);
    AddedIdx = Keep(Keep <= ArchiveCount);
    if isempty(AddedIdx)
        return;
    end
    Added = Archive(AddedIdx);
end
