function Selected = RerankBoundaryCandidates(CandidateObj,Score,Source,FeasibleObj,W,Budget)
% Re-rank near-boundary candidates after uncertainty pre-screen.

    if nargin < 2 || isempty(Score)
        Score = zeros(size(CandidateObj,1),1);
    end
    if nargin < 3 || isempty(Source)
        Source = ones(size(CandidateObj,1),1);
    end
    if nargin < 4 || isempty(FeasibleObj)
        FeasibleObj = zeros(0,size(CandidateObj,2));
    end
    if nargin < 6 || isempty(Budget) || Budget <= 0
        Budget = size(CandidateObj,1);
    end

    Score = Score(:);
    Source = Source(:);
    Total  = size(CandidateObj,1);
    Budget = min(Budget,Total);
    if Budget <= 0 || Total == 0
        Selected = zeros(1,0);
        return;
    end

    LocalIdx = (1:Total)';
    [Keep,Sector,SectorLoad] = FilterSectorDominated(CandidateObj,FeasibleObj,W);
    if ~any(Keep)
        Selected = zeros(1,0);
        return;
    end

    LocalIdx      = LocalIdx(Keep);
    CandidateObj  = CandidateObj(Keep,:);
    Score         = Score(Keep);
    Source        = Source(Keep);
    Sector        = Sector(Keep);
    FrontNo       = NDSort(CandidateObj,size(CandidateObj,1));
    Crowd         = CrowdingDistance(CandidateObj,FrontNo);
    SourceLoad    = zeros(max(Source),1);
    Active        = true(length(LocalIdx),1);
    SelectedLocal = zeros(1,min(Budget,length(LocalIdx)));
    Count         = 0;

    while Count < length(SelectedLocal) && any(Active)
        Remain = find(Active);
        RankMat = [SourceLoad(Source(Remain)),SectorLoad(Sector(Remain)), ...
                   FrontNo(Remain)',-Crowd(Remain)',-Score(Remain)];
        [~,Rank] = sortrows(RankMat,[1 2 3 4 5]);
        Pick = Remain(Rank(1));

        Count = Count + 1;
        SelectedLocal(Count) = LocalIdx(Pick);
        Active(Pick) = false;
        SourceLoad(Source(Pick)) = SourceLoad(Source(Pick)) + 1;
        SectorLoad(Sector(Pick)) = SectorLoad(Sector(Pick)) + 1;
    end
    Selected = SelectedLocal(1:Count);
end

function [Keep,Sector,SectorLoad] = FilterSectorDominated(CandidateObj,FeasibleObj,W)
    Keep = true(size(CandidateObj,1),1);
    if isempty(CandidateObj)
        Sector     = zeros(0,1);
        SectorLoad = zeros(size(W,1),1);
        return;
    end

    if isempty(FeasibleObj)
        Sector     = AssociateSectors(CandidateObj,W,CandidateObj);
        SectorLoad = zeros(size(W,1),1);
        return;
    end

    RefObj = [FeasibleObj;CandidateObj];
    [SectorF,SectorLoad] = AssociateSectors(FeasibleObj,W,RefObj);
    Sector = AssociateSectors(CandidateObj,W,RefObj);
    for i = 1 : size(CandidateObj,1)
        if SectorLoad(Sector(i)) == 0
            continue;
        end
        SameSectorObj = FeasibleObj(SectorF==Sector(i),:);
        if AnyDominates(SameSectorObj,CandidateObj(i,:))
            Keep(i) = false;
        end
    end
end
