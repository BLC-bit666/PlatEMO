function Population = NAEMTAuxiliarySelection(Population,N)
% Standard unconstrained NSGA-II selection for the auxiliary task.

    [FrontNo,maxFront] = NDSort(Population.objs,N);
    Next               = FrontNo < maxFront;
    CrowdDis           = CrowdingDistance(Population.objs,FrontNo);
    Last               = find(FrontNo==maxFront);
    [~,rank]           = sort(CrowdDis(Last),'descend');
    Next(Last(rank(1:N-sum(Next)))) = true;
    Population = Population(Next);
end
