function [Population,Fitness] = EnvironmentalSelectionU(Population,N)
% Environmental selection for the unconstrained helper population.

    Population = SelectByNSGA2Local(Population,N);
    Fitness = CalFitness(Population.objs);
end

function Population = SelectByNSGA2Local(Population,N)
    if isempty(Population)
        return;
    end

    if numel(Population) <= N
        [FrontNo,~] = NDSort(Population.objs,numel(Population));
        CrowdDis = CrowdingDistance(Population.objs,FrontNo);
        [~,Order] = sortrows([FrontNo(:),-CrowdDis(:)],[1 2]);
        Population = Population(Order);
        return;
    end

    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next     = FrontNo < MaxFNo;
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    Last     = find(FrontNo==MaxFNo);
    Need     = N - sum(Next);
    if Need > 0
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:Need))) = true;
    end

    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    [~,Order]  = sortrows([FrontNo(:),-CrowdDis(:)],[1 2]);
    Population = Population(Order);
end
