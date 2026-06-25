function tau = PairLocalTau_CBS(Y,Yf,Yi,objMin,objSpan)
%PAIRLOCALTAU_CBS Project objective points onto feasible-infeasible pairs.

    Y = double(Y);
    Yf = double(Yf);
    Yi = double(Yi);
    n = size(Y,1);
    if n == 0
        tau = zeros(0,1);
        return;
    end
    M = size(Y,2);
    if size(Yf,1) ~= n || size(Yi,1) ~= n || ...
            size(Yf,2) ~= M || size(Yi,2) ~= M
        error('CBSCGAN:BadTauSupport', ...
            'Tau support rows must match the target objective rows.');
    end
    if nargin < 4
        objMin = [];
    end
    if nargin < 5
        objSpan = [];
    end

    P = normalizeObjectiveRows(Y,objMin,objSpan,M);
    A = normalizeObjectiveRows(Yf,objMin,objSpan,M);
    B = normalizeObjectiveRows(Yi,objMin,objSpan,M);
    AB = B - A;
    denom = sum(AB.^2,2);
    tau = zeros(n,1);
    valid = denom > eps;
    if any(valid)
        tau(valid) = sum((P(valid,:) - A(valid,:)).*AB(valid,:),2)./ ...
            denom(valid);
    end
    tau = max(0,min(1,tau));
    tau(~isfinite(tau)) = 0;
end

function Xn = normalizeObjectiveRows(X,objMin,objSpan,M)
    objMin = double(objMin(:)');
    objSpan = double(objSpan(:)');
    if numel(objMin) ~= M || numel(objSpan) ~= M
        objMin = zeros(1,M);
        objSpan = ones(1,M);
    end
    objSpan(objSpan <= eps) = 1;
    Xn = (X - objMin)./objSpan;
    Xn(~isfinite(Xn)) = 0;
end
