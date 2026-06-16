function [keep,spread,k] = BoundaryConditionKNNKeepMask_BDG( ...
        X,C,conditionKNNK,retainRatio,minCount,lower,upper)
%BoundaryConditionKNNKeepMask_BDG Keep low-spread target triples by condition kNN.

    n = size(X,1);
    keep = true(n,1);
    spread = nan(n,1);
    k = min(max(1,round(double(conditionKNNK))),max(0,n-1));
    minCount = max(2,round(double(minCount)));
    retainRatio = min(max(double(retainRatio),0),1);
    if n <= minCount || isempty(C) || k <= 0
        return;
    end
    C = double(C);
    Xn = NormalizeDecision_BDG(double(X),lower,upper);
    dCond = pdist2(C,C);
    dCond(1:n+1:end) = Inf;
    for i = 1 : n
        [~,ord] = sort(dCond(i,:),'ascend');
        ord = ord(1:min(k,numel(ord)));
        d = sqrt(sum((Xn(ord,:) - Xn(i,:)).^2,2));
        spread(i) = MeanFinite_BDG(d);
    end
    if ~any(isfinite(spread))
        return;
    end
    nKeep = max(minCount,ceil(retainRatio * n));
    nKeep = min(n,max(2,nKeep));
    [~,ord] = sort(spread,'ascend','MissingPlacement','last');
    keep = false(n,1);
    keep(ord(1:nKeep)) = true;
end

function Xn = NormalizeDecision_BDG(X,lower,upper)
    D = size(X,2);
    if nargin < 2 || isempty(lower) || numel(lower) ~= D
        lower = zeros(1,D);
    end
    if nargin < 3 || isempty(upper) || numel(upper) ~= D
        upper = ones(1,D);
    end
    lower = double(lower(:)');
    upper = double(upper(:)');
    Xn = (double(X) - lower) ./ (upper - lower + 1e-12);
    Xn = min(max(Xn,0),1);
end

function value = MeanFinite_BDG(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
end
