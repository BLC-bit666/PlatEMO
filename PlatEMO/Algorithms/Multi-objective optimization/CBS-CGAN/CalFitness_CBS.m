function Fitness = CalFitness_CBS(PopObj,PopCon)
%CALFITNESS_CBS SPEA2-style fitness with optional constraint dominance.

    N = size(PopObj,1);
    if nargin < 2 || isempty(PopCon)
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end

    left  = reshape(PopObj,N,1,[]);
    right = reshape(PopObj,1,N,[]);
    objectiveDominates = all(left <= right,3) & any(left < right,3);
    Dominate = CV < CV' | (CV == CV' & objectiveDominates);
    Dominate(1:N+1:end) = false;

    Strength = sum(Dominate,2);
    Raw = Strength'*double(Dominate);

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
