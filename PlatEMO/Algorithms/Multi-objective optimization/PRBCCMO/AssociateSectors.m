function [Sector,Count] = AssociateSectors(PopObj,W,RefObj)
% Associate objective vectors to the nearest reference sector.

    if nargin < 3 || isempty(RefObj)
        RefObj = PopObj;
    end
    if isempty(PopObj)
        Sector = zeros(0,1);
        Count  = zeros(size(W,1),1);
        return;
    end
    if isempty(W)
        Sector = ones(size(PopObj,1),1);
        Count  = size(PopObj,1);
        return;
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range  = MaxObj - MinObj;
    Range(Range<1e-12) = 1;

    Obj = (PopObj-MinObj)./Range;
    RowNorm = sqrt(sum(Obj.^2,2));
    ZeroRow = RowNorm < 1e-12;
    Obj(ZeroRow,:) = 1;
    RowNorm(ZeroRow) = sqrt(size(Obj,2));
    Obj = Obj./repmat(RowNorm,1,size(Obj,2));

    WNorm = sqrt(sum(W.^2,2));
    WNorm(WNorm<1e-12) = 1;
    Wn = W./repmat(WNorm,1,size(W,2));

    Cosine = Obj*Wn';
    [~,Sector] = max(Cosine,[],2);
    Count = accumarray(Sector,1,[size(W,1),1]);
end
