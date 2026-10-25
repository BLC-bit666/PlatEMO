function [Population,Value] = NAEMTMainSelection(Population,N,Value,epsilon)
% Environmental selection of the main task using CDPPV and crowding.

    PopObj   = Population.objs;
    FrontNo  = NAEMTCDPPVNDSort(PopObj,Value,epsilon,N);
    CrowdDis = CrowdingDistance(PopObj,FrontNo);
    maxFront = max(FrontNo(FrontNo<inf));
    Next     = FrontNo < maxFront;
    Last     = find(FrontNo==maxFront);
    [~,rank] = sort(CrowdDis(Last),'descend');
    Next(Last(rank(1:N-sum(Next)))) = true;
    Population = Population(Next);
    Value      = Value(Next);
end
