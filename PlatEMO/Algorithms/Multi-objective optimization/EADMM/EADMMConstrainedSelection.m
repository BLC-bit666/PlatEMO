function [Population,FrontNo,CrowdDis] = EADMMConstrainedSelection(Population,N)
% NSGA-II selection using the number of violated constraints instead of CV.

    FrontNo  = EADMMConstraintNDSort(Population.objs,Population.cons,N);
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    maxFront = max(FrontNo(FrontNo<inf));
    Next     = FrontNo < maxFront;
    Last     = find(FrontNo==maxFront);
    [~,rank] = sort(CrowdDis(Last),'descend');
    Next(Last(rank(1:N-sum(Next)))) = true;
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end
