function Population = AuxiliaryTaskEnvironmentalSelection(Population,N)
% Environmental selection for the unconstrained auxiliary task

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next             = FrontNo < MaxFNo;
    Last             = find(FrontNo==MaxFNo);
    CrowdDis         = CrowdingDistance(Population.objs,FrontNo);
    [~,Rank]         = sort(CrowdDis(Last),'descend');
    Remaining        = N - sum(Next);
    if Remaining > 0
        Next(Last(Rank(1:Remaining))) = true;
    end
    Population       = Population(Next);
end
