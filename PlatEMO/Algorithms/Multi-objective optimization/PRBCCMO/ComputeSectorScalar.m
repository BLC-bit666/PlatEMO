function Value = ComputeSectorScalar(Obj,W,RefObj,Sector)
% Compute the normalized weighted-sum scalar value for one or more sectors.

    if isempty(Obj)
        Value = zeros(0,1);
        return;
    end
    if nargin < 2 || isempty(W)
        W = ones(1,size(Obj,2));
    end
    if nargin < 3 || isempty(RefObj)
        RefObj = Obj;
    end
    if nargin < 4 || isempty(Sector)
        if size(W,1) == 1
            Weight = repmat(W,size(Obj,1),1);
        elseif size(W,1) == size(Obj,1)
            Weight = W;
        else
            Weight = repmat(W(1,:),size(Obj,1),1);
        end
    else
        Weight = W(Sector,:);
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range  = MaxObj - MinObj;
    Range(Range<1e-12) = 1;
    NormObj = (Obj - MinObj)./Range;
    Value   = sum(NormObj.*Weight,2);
end
