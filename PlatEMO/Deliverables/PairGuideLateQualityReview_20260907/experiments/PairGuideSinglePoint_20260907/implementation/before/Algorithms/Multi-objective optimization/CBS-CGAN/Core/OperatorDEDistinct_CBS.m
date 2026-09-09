function [Offspring,ParentIndices] = OperatorDEDistinct_CBS( ...
        Problem,Population,Fitness,count,Parameter)
%OPERATORDEDISTINCT_CBS Ordinary CBS DE with distinct parent rows.
%   The base follows the historical CBS rule. Both difference-vector
%   parents retain fitness-tournament selection, but neither may equal the
%   base or each other.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    populationSize = numel(Population);
    count = max(0,min(populationSize,round(double(count))));
    Fitness = reshape(double(Fitness),[],1);
    if numel(Fitness) ~= populationSize
        error('CBSRegionGAN:DEFitnessSize', ...
            'Fitness must contain one value per population member.');
    end
    if count == 0
        Offspring = Population([]);
        ParentIndices = zeros(0,3);
        return;
    elseif populationSize < 3
        error('CBSRegionGAN:DEPopulationTooSmall', ...
            'Ordinary DE requires at least three solutions.');
    end

    if count == populationSize
        base = (1:populationSize)';
    else
        base = reshape(randperm(populationSize,count),[],1);
    end
    [~,order] = sort(Fitness,'ascend');
    rank = zeros(populationSize,1);
    rank(order) = 1:populationSize;
    ParentIndices = zeros(count,3);
    ParentIndices(:,1) = base;
    allRows = 1:populationSize;
    for child = 1 : count
        candidates = allRows(allRows ~= base(child));
        pick = TournamentSelection(2,1,rank(candidates));
        donor1 = candidates(pick);
        candidates = candidates(candidates ~= donor1);
        pick = TournamentSelection(2,1,rank(candidates));
        donor2 = candidates(pick);
        ParentIndices(child,2:3) = [donor1,donor2];
    end

    if nargin < 5 || isempty(Parameter)
        Offspring = OperatorDE(Problem, ...
            Population(ParentIndices(:,1)), ...
            Population(ParentIndices(:,2)), ...
            Population(ParentIndices(:,3)));
    else
        Offspring = OperatorDE(Problem, ...
            Population(ParentIndices(:,1)), ...
            Population(ParentIndices(:,2)), ...
            Population(ParentIndices(:,3)),Parameter);
    end
end
