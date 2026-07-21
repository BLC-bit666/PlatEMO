function [Population,Fitness,selIdx,unionFitness] = ...
        EnvironmentalSelection_CBS(Population,N,isConstrained,protectIdx)
%ENVIRONMENTALSELECTION_CBS Environmental selection for the two CCMO populations.
%   The constrained and unconstrained populations share the CCMO/SPEA2
%   survival rule. CrowdingDistance implements a different selection rule
%   and therefore is not interchangeable with this algorithm-specific step.
%   The optional PROTECTIDX rows are excluded from the competitive pool and
%   appended with guaranteed slots; their fitness values are carried out of
%   the single union-level fitness computation rather than recomputed, so
%   the mating order of the competitively selected members is unchanged.
%   SELIDX maps the returned population back to rows of the input union.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate fitness
    if isConstrained
        Fitness = CalFitness_CBS(Population.objs,Population.cons);
    else
        Fitness = CalFitness_CBS(Population.objs);
    end
    unionFitness = Fitness;
    if nargin < 4
        protectIdx = [];
    end
    total = numel(Population);
    protectIdx = reshape(unique(round(double(protectIdx)),'stable'),1,[]);
    if any(protectIdx < 1 | protectIdx > total)
        error('CBSRegionGAN:BadProtectIndex', ...
            'Protected indices must address rows of the input union.');
    end

    if isempty(protectIdx)
        %% Select the next population (validated mainline path)
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
        pick = reshape(find(Next),1,[]);
    else
        %% Protected path: competitive slots exclude the protected rows
        candidate = setdiff(1:total,protectIdx);
        slots = max(0,N-numel(protectIdx));
        CandFitness = Fitness(candidate);
        Next = reshape(CandFitness < 1,1,[]);
        selectedCount = sum(Next);
        if selectedCount < slots
            [~,Rank] = sort(CandFitness);
            Next(Rank(1:min(slots,numel(candidate)))) = true;
        elseif selectedCount > slots
            CandPop = Population(candidate);
            Del = truncateByObjectiveDistance( ...
                CandPop(Next).objs,selectedCount-slots);
            Temp = find(Next);
            Next(Temp(Del)) = false;
        end
        pick = [candidate(Next),protectIdx];
    end

    %% Sort survivors by fitness
    Population = Population(pick);
    Fitness = Fitness(pick);
    [Fitness,Rank] = sort(Fitness);
    Population = Population(Rank);
    selIdx = reshape(pick(Rank),1,[]);
end

function Del = truncateByObjectiveDistance(PopObj,K)
%TRUNCATEBYOBJECTIVEDISTANCE Remove crowded nondominated solutions.

    Distance = pairDistance(PopObj,PopObj);
    count = size(Distance,1);
    Distance(1:count+1:end) = inf;
    Del = false(1,size(PopObj,1));
    deletedCount = 0;
    while deletedCount < K
        Remain = find(~Del);
        Sorted = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Sorted);
        Del(Remain(Rank(1))) = true;
        deletedCount = deletedCount + 1;
    end
end

function D = pairDistance(A,B)
%PAIRDISTANCE Calculate Euclidean distances without an extra toolbox.

    D2 = max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'),0);
    D = sqrt(D2);
end
