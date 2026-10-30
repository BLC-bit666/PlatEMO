function [Population,Fitness] = ...
        GANCMO_V5Selection(Population,N,isConstrained)
%GANCMO_V5SELECTION CCMO/SPEA2 survival for both populations.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if isConstrained
        Fitness = GANCMO_V5Fitness(Population.objs,Population.cons);
    else
        Fitness = GANCMO_V5Fitness(Population.objs);
    end
    Next = Fitness < 1;
    selectedCount = sum(Next);
    if selectedCount < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif selectedCount > N
        Del = truncateByObjectiveDistance( ...
            Population(Next).objs,selectedCount-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end

    Population = Population(Next);
    Fitness = Fitness(Next);
    [Fitness,Rank] = sort(Fitness);
    Population = Population(Rank);
end

function Del = truncateByObjectiveDistance(PopObj,K)
%TRUNCATEBYOBJECTIVEDISTANCE Remove crowded nondominated solutions.

    Distance = pdist2(PopObj,PopObj);
    count = size(Distance,1);
    Distance(1:count+1:end) = inf;
    Del = false(1,count); Remain = 1:count;
    % Removing a point only deletes a neighbor; the remaining order stays sorted.
    [Sorted,Neighbors] = sort(Distance,2);
    for deletedCount = 1:K
        [~,Rank] = sortrows(Sorted); row = Rank(1); removed = Remain(row);
        Del(removed) = true; Remain(row) = [];
        if deletedCount==K; break; end
        Sorted(row,:) = []; Neighbors(row,:) = [];
        keep = Neighbors'~=removed;
        values = Sorted'; neighbors = Neighbors';
        Sorted = reshape(values(keep),numel(Remain),numel(Remain))';
        Neighbors = reshape(neighbors(keep),numel(Remain),numel(Remain))';
    end
end
