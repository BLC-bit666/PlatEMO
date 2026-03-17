function [Population,Fitness] = EnvironmentalSelectionU(Population,N)
% Environmental selection for the unconstrained helper population.

    [Population,~,~] = SelectByNSGA2(Population,N,false);
    Fitness = CalFitness(Population.objs);
end
