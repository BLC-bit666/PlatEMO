function [Population,FrontNo,CrowdDis] = SelectByNSGA2(Population,N,UseConstraint)
% Select a population using NSGA-II style ranking and crowding.

    if nargin < 3
        UseConstraint = false;
    end
    if isempty(Population)
        FrontNo = [];
        CrowdDis = [];
        return;
    end

    if numel(Population) <= N
        if UseConstraint
            [FrontNo,~] = NDSort(Population.objs,Population.cons,numel(Population));
        else
            [FrontNo,~] = NDSort(Population.objs,numel(Population));
        end
        CrowdDis = CrowdingDistance(Population.objs,FrontNo);
        return;
    end

    if UseConstraint
        [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    else
        [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    end
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
    FrontNo    = FrontNo(Order);
    CrowdDis   = CrowdDis(Order);
end
