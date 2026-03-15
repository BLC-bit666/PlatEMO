function FrontNo = CDPPVNDSort(PopObj,Value,epsilon,nSort)
% Non-dominated sorting under the CDPPV rule

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N              = size(PopObj,1);
    FrontNo        = inf(1,N);
    DominateList   = cell(1,N);
    DominatedCount = zeros(1,N);

    for p = 1 : N-1
        for q = p+1 : N
            relation = CDPPVRelation(PopObj(p,:),Value(p),PopObj(q,:),Value(q),epsilon);
            if relation == 1
                DominateList{p}(end+1) = q;
                DominatedCount(q)      = DominatedCount(q) + 1;
            elseif relation == -1
                DominateList{q}(end+1) = p;
                DominatedCount(p)      = DominatedCount(p) + 1;
            end
        end
    end

    CurrentFront = find(DominatedCount==0);
    FrontIndex   = 1;
    Assigned     = 0;
    while ~isempty(CurrentFront) && Assigned < min(nSort,N)
        FrontNo(CurrentFront) = FrontIndex;
        Assigned              = Assigned + numel(CurrentFront);
        NextFront             = [];
        for i = 1 : numel(CurrentFront)
            Dominated = DominateList{CurrentFront(i)};
            for j = 1 : numel(Dominated)
                DominatedCount(Dominated(j)) = DominatedCount(Dominated(j)) - 1;
                if DominatedCount(Dominated(j)) == 0
                    NextFront(end+1) = Dominated(j); %#ok<AGROW>
                end
            end
        end
        CurrentFront = unique(NextFront,'stable');
        FrontIndex   = FrontIndex + 1;
    end
end

function relation = CDPPVRelation(Obj1,Val1,Obj2,Val2,epsilon)
    relation = 0;
    if Val1 >= epsilon && Val2 < epsilon
        relation = 1;
        return;
    elseif Val2 >= epsilon && Val1 < epsilon
        relation = -1;
        return;
    end

    if Val1 > epsilon && Val2 > epsilon
        if ParetoDominates(Obj1,Obj2)
            relation = 1;
        elseif ParetoDominates(Obj2,Obj1)
            relation = -1;
        end
    elseif Val1 < epsilon && Val2 < epsilon
        if Val1 > Val2
            relation = 1;
        elseif Val2 > Val1
            relation = -1;
        end
    end
end

function Flag = ParetoDominates(Obj1,Obj2)
    Flag = all(Obj1<=Obj2) && any(Obj1<Obj2);
end
