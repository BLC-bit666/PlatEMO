function Detail = ScoreBoundaryCandidates(Problem,CandidateDec,CandidateObj,FeasibleObj,Model,W,HardNegativeArchive)
% Score boundary candidates by uncertainty, Pareto value, novelty, and hard-negative penalty.

    Total = size(CandidateDec,1);
    Detail.prob          = zeros(Total,1);
    Detail.entropy       = zeros(Total,1);
    Detail.hvGain        = zeros(Total,1);
    Detail.sectorNovelty = zeros(Total,1);
    Detail.penaltyFactor = ones(Total,1);
    Detail.utility       = zeros(Total,1);
    Detail.sector        = zeros(Total,1);
    if Total == 0
        return;
    end

    Prob    = PredictBoundaryMLP(Model,CandidateDec);
    Entropy = -(Prob.*log(Prob) + (1-Prob).*log(1-Prob))./log(2);
    [SectorNovelty,Sector] = ComputeSectorNovelty(CandidateObj,FeasibleObj,W);
    HVGain = EstimatePositiveHVPotential(CandidateObj,FeasibleObj);
    PenaltyFactor = ComputeHardNegativeFactor(Problem,CandidateDec,HardNegativeArchive);
    Utility = Entropy .* max(HVGain,1e-12) .* SectorNovelty .* PenaltyFactor;

    Detail.prob          = Prob(:);
    Detail.entropy       = Entropy(:);
    Detail.hvGain        = HVGain(:);
    Detail.sectorNovelty = SectorNovelty(:);
    Detail.penaltyFactor = PenaltyFactor(:);
    Detail.utility       = Utility(:);
    Detail.sector        = Sector(:);
end

function [Novelty,Sector] = ComputeSectorNovelty(CandidateObj,FeasibleObj,W)
    Total = size(CandidateObj,1);
    Novelty = ones(Total,1);
    Sector  = ones(Total,1);
    if Total == 0
        return;
    end

    RefObj = CandidateObj;
    if ~isempty(FeasibleObj)
        RefObj = [FeasibleObj;CandidateObj];
    end
    Sector = AssociateSectors(CandidateObj,W,RefObj);
    SectorCount = max(size(W,1),1);
    ExistingLoad = zeros(SectorCount,1);
    if ~isempty(FeasibleObj)
        [~,ExistingLoad] = AssociateSectors(FeasibleObj,W,RefObj);
    end
    CandidateLoad = accumarray(Sector,1,[SectorCount,1]);
    Novelty = 1./(1 + ExistingLoad(Sector) + max(CandidateLoad(Sector)-1,0));
end

function HVGain = EstimatePositiveHVPotential(CandidateObj,FeasibleObj)
    Total = size(CandidateObj,1);
    if Total == 0
        HVGain = zeros(0,1);
        return;
    end
    if isempty(FeasibleObj)
        HVGain = ones(Total,1);
        return;
    end

    FrontMask = NDSort(FeasibleObj,1) == 1;
    FrontObj  = FeasibleObj(FrontMask,:);
    RefObj    = [FrontObj;CandidateObj];
    MinObj    = min(RefObj,[],1);
    Range     = max(RefObj,[],1) - MinObj;
    Range(Range<1e-12) = 1;
    RefPoint  = max((RefObj-MinObj)./Range,[],1) + 0.1;
    CandNorm  = (CandidateObj-MinObj)./Range;

    HVGain = zeros(Total,1);
    for i = 1 : Total
        if IsDominatedByFront(FrontObj,CandidateObj(i,:))
            continue;
        end
        HVGain(i) = prod(max(0,RefPoint-CandNorm(i,:)));
    end

    HVGain = HVGain./max(max(HVGain),1e-12);
    HVGain(HVGain<1e-12) = 1e-12;
end

function PenaltyFactor = ComputeHardNegativeFactor(Problem,CandidateDec,HardNegativeArchive)
    Total = size(CandidateDec,1);
    PenaltyFactor = ones(Total,1);
    if Total == 0 || isempty(HardNegativeArchive) || ~isstruct(HardNegativeArchive)
        return;
    end
    if ~isfield(HardNegativeArchive,'Dec') || isempty(HardNegativeArchive.Dec)
        return;
    end

    Range = Problem.upper - Problem.lower;
    Range(Range<1e-12) = 1;
    CandNorm   = (CandidateDec - repmat(Problem.lower,Total,1))./repmat(Range,Total,1);
    CenterNorm = (HardNegativeArchive.Dec - repmat(Problem.lower,size(HardNegativeArchive.Dec,1),1))./repmat(Range,size(HardNegativeArchive.Dec,1),1);
    Dist       = pdist2(CandNorm,CenterNorm);
    Radius     = HardNegativeArchive.Radius(:)';
    Radius(Radius<1e-6) = 1e-6;
    ScaledDist = Dist./repmat(Radius,size(Dist,1),1);
    Penalty    = max(exp(-(ScaledDist.^2)),[],2);
    PenaltyFactor = 1 - 0.8*Penalty;
end

function flag = IsDominatedByFront(PopObj,obj)
    flag = false;
    if isempty(PopObj)
        return;
    end
    LessEqual = all(PopObj<=repmat(obj,size(PopObj,1),1),2);
    Less      = any(PopObj<repmat(obj,size(PopObj,1),1),2);
    flag      = any(LessEqual & Less);
end
