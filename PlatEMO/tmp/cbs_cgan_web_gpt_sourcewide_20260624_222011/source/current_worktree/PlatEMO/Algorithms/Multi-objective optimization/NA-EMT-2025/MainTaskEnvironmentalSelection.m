function Population = MainTaskEnvironmentalSelection(Population,N,Model,epsilon)
% Environmental selection for the main task based on CDPPV

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Value      = ones(numel(Population),1);
    Infeasible = any(Population.cons>0,2);
    if any(Infeasible)
        Value(Infeasible) = PredictISVPS(Model,Population(Infeasible).decs);
    end

    FrontNo     = CDPPVNDSort(Population.objs,Value,epsilon,N);
    CrowdDis    = CrowdingDistance(Population.objs,FrontNo);
    Assigned    = FrontNo(FrontNo<inf);
    MaxFNo      = max(Assigned);
    Last        = find(FrontNo==MaxFNo);
    Next        = FrontNo < MaxFNo;
    [~,Rank]    = sort(CrowdDis(Last),'descend');
    Remaining   = N - sum(Next);
    if Remaining > 0
        Next(Last(Rank(1:Remaining))) = true;
    end
    Population = Population(Next);
end
