function [Population,Fitness] = EnvironmentalSelection_CBS(Population,N,isConstrained)
%ENVIRONMENTALSELECTION_CBS Environmental selection for the two CCMO populations.

    if isConstrained
        Fitness = CalFitness_CBS(Population.objs,Population.cons);
    else
        Fitness = CalFitness_CBS(Population.objs);
    end

    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del = truncateByObjectiveDistance(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end

    Population = Population(Next);
    Fitness = Fitness(Next);
    [Fitness,Rank] = sort(Fitness);
    Population = Population(Rank);
end

function Del = truncateByObjectiveDistance(PopObj,K)
    Distance = pairDistance(PopObj,PopObj);
    Distance(logical(eye(size(Distance,1)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain = find(~Del);
        Sorted = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Sorted);
        Del(Remain(Rank(1))) = true;
    end
end

function D = pairDistance(A,B)
    D2 = max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'),0);
    D = sqrt(D2);
end
