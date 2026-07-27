function FrontNo = NAEMTCDPPVNDSort(PopObj,Value,epsilon,nSort)
% Nondominated sorting under the CDPPV relation.

    N              = size(PopObj,1);
    FrontNo        = inf(1,N);
    dominateList   = cell(1,N);
    dominatedCount = zeros(1,N);

    for p = 1 : N-1
        for q = p+1 : N
            relation = CDPPVRelation(PopObj(p,:),Value(p),PopObj(q,:),Value(q),epsilon);
            if relation == 1
                dominateList{p}(end+1) = q;
                dominatedCount(q)      = dominatedCount(q) + 1;
            elseif relation == -1
                dominateList{q}(end+1) = p;
                dominatedCount(p)      = dominatedCount(p) + 1;
            end
        end
    end

    current  = find(dominatedCount==0);
    front    = 1;
    assigned = 0;
    while ~isempty(current) && assigned < min(nSort,N)
        FrontNo(current) = front;
        assigned         = assigned + numel(current);
        next             = [];
        for i = 1 : numel(current)
            dominated = dominateList{current(i)};
            for j = 1 : numel(dominated)
                dominatedCount(dominated(j)) = dominatedCount(dominated(j)) - 1;
                if dominatedCount(dominated(j)) == 0
                    next(end+1) = dominated(j); %#ok<AGROW>
                end
            end
        end
        current = unique(next,'stable');
        front   = front + 1;
    end
end

function relation = CDPPVRelation(Obj1,Val1,Obj2,Val2,epsilon)
% Return 1 if the first solution dominates, -1 for the reverse, and 0 else.

    if Val1 >= epsilon && Val2 < epsilon
        relation = 1;
    elseif Val2 >= epsilon && Val1 < epsilon
        relation = -1;
    elseif Val1 > epsilon && Val2 > epsilon
        relation = ParetoRelation(Obj1,Obj2);
    elseif Val1 < epsilon && Val2 < epsilon
        if Val1 > Val2
            relation = 1;
        elseif Val2 > Val1
            relation = -1;
        else
            relation = 0;
        end
    else
        % The paper does not define dominance between Val=epsilon and
        % Val>epsilon, nor between two values both equal to epsilon.
        relation = 0;
    end
end

function relation = ParetoRelation(Obj1,Obj2)
    if all(Obj1<=Obj2) && any(Obj1<Obj2)
        relation = 1;
    elseif all(Obj2<=Obj1) && any(Obj2<Obj1)
        relation = -1;
    else
        relation = 0;
    end
end
