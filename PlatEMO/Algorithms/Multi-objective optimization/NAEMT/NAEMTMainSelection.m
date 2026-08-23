function [Population,Value] = NAEMTMainSelection(Population,N,Value,epsilon)
% Environmental selection of the main task using CDPPV and crowding.

    FrontNo  = NAEMTCDPPVNDSort(Population.objs,Value,epsilon,N);
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    maxFront = max(FrontNo(FrontNo<inf));
    Next     = FrontNo < maxFront;
    Last     = find(FrontNo==maxFront);
    [~,rank] = sort(CrowdDis(Last),'descend');
    Next(Last(rank(1:N-sum(Next)))) = true;
    Population = Population(Next);
    Value      = Value(Next);
end
