function Match = MatchPartnersBySector(AnchorObj,CandidateObj,W,Exclude)
% Match each anchor to a nearest candidate, preferring the same sector.

    N = size(AnchorObj,1);
    Match = zeros(1,N);
    if N == 0 || isempty(CandidateObj)
        Match = zeros(1,0);
        return;
    end
    if nargin < 4 || isempty(Exclude)
        Exclude = zeros(N,1);
    else
        Exclude = Exclude(:);
    end

    RefObj = [AnchorObj;CandidateObj];
    SectorA = AssociateSectors(AnchorObj,W,RefObj);
    SectorC = AssociateSectors(CandidateObj,W,RefObj);
    MinObj  = min(RefObj,[],1);
    Range   = max(RefObj,[],1) - MinObj;
    Range(Range<1e-12) = 1;

    for i = 1 : N
        SameSector = find(SectorC==SectorA(i));
        SameSector = RemoveExcluded(SameSector,Exclude(i));
        if isempty(SameSector)
            SameSector = RemoveExcluded((1:size(CandidateObj,1))',Exclude(i));
        end
        if isempty(SameSector)
            SameSector = Exclude(i);
        end
        AnchorNorm = (AnchorObj(i,:)-MinObj)./Range;
        CandNorm   = (CandidateObj(SameSector,:)-MinObj)./Range;
        Dist       = sum((CandNorm-AnchorNorm).^2,2);
        [~,Best]   = min(Dist);
        Match(i)   = SameSector(Best);
    end
end

function Pool = RemoveExcluded(Pool,Exclude)
    if isempty(Pool) || Exclude <= 0 || numel(Pool) <= 1
        return;
    end
    Pool = Pool(Pool~=Exclude);
end
