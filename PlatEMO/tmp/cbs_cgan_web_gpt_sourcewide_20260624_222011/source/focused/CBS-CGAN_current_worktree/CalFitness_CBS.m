function Fitness = CalFitness_CBS(PopObj,PopCon)
%CALFITNESS_CBS SPEA2-style fitness with optional constraint dominance.

    N = size(PopObj,1);
    if nargin < 2 || isempty(PopCon)
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end

    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                better = any(PopObj(i,:) < PopObj(j,:));
                worse  = any(PopObj(i,:) > PopObj(j,:));
                if better && ~worse
                    Dominate(i,j) = true;
                elseif worse && ~better
                    Dominate(j,i) = true;
                end
            end
        end
    end

    Strength = sum(Dominate,2);
    Raw = zeros(1,N);
    for i = 1 : N
        Raw(i) = sum(Strength(Dominate(:,i)));
    end

    Distance = pairDistance(PopObj,PopObj);
    Distance(logical(eye(N))) = inf;
    Distance = sort(Distance,2);
    kth = max(1,min(size(Distance,2),floor(sqrt(N))));
    Density = 1./(Distance(:,kth)+2);
    Fitness = Raw + Density';
end

function D = pairDistance(A,B)
    D2 = max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'),0);
    D = sqrt(D2);
end
