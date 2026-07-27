function Population = NAEMTMainSelection(Population,N,Model,epsilon)
% Environmental selection of the main task using CDPPV and crowding.

    Value      = ones(numel(Population),1);
    infeasible = ~all(Population.cons<=0,2);
    if any(infeasible)
        Value(infeasible) = NAEMTPredict(Model,Population(infeasible).decs);
    end

    FrontNo  = NAEMTCDPPVNDSort(Population.objs,Value,epsilon,N);
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    maxFront = max(FrontNo(FrontNo<inf));
    Next     = FrontNo < maxFront;
    Last     = find(FrontNo==maxFront);
    [~,rank] = sort(CrowdDis(Last),'descend');
    Next(Last(rank(1:N-sum(Next)))) = true;
    Population = Population(Next);
end
