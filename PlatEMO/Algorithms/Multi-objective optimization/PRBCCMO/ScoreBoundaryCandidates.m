function Detail = ScoreBoundaryCandidates(Problem,CandidateDec,CandidateObj,FeasibleObj,Model,W,HardNegativeArchive)
% Score boundary candidates by calibrated uncertainty, committee disagreement, Pareto value, novelty, and hard-negative penalty.

    Total = size(CandidateDec,1);
    Detail.prob          = zeros(Total,1);
    Detail.disagreement  = zeros(Total,1);
    Detail.entropy       = zeros(Total,1);
    Detail.hvGain        = zeros(Total,1);
    Detail.sectorNovelty = zeros(Total,1);
    Detail.penaltyFactor = ones(Total,1);
    Detail.uncertaintyUtility = zeros(Total,1);
    Detail.fullUtility   = zeros(Total,1);
    Detail.utility       = zeros(Total,1);
    Detail.sector        = zeros(Total,1);
    if Total == 0
        return;
    end

    [Prob,PredictStats] = PredictBoundaryMLP(Model,CandidateDec);
    Entropy = -(Prob.*log(Prob) + (1-Prob).*log(1-Prob))./log(2);
    Disagreement = PredictStats.std(:);
    [SectorNovelty,Sector] = ComputeSectorNovelty(CandidateObj,FeasibleObj,W);
    HVGain = EstimatePositiveHVPotential(CandidateObj,FeasibleObj);
    PenaltyFactor = ComputeHardNegativeFactor(Problem,CandidateDec,HardNegativeArchive);

    LambdaDisagreement = ResolveDisagreementWeight(Model);
    UncertaintyUtility = Entropy .* (1 + LambdaDisagreement*Disagreement) ...
        .* PenaltyFactor;
    FullUtility = UncertaintyUtility .* max(HVGain,1e-12) .* SectorNovelty;

    Detail.prob          = Prob(:);
    Detail.disagreement  = Disagreement;
    Detail.entropy       = Entropy(:);
    Detail.hvGain        = HVGain(:);
    Detail.sectorNovelty = SectorNovelty(:);
    Detail.penaltyFactor = PenaltyFactor(:);
    Detail.uncertaintyUtility = UncertaintyUtility(:);
    Detail.fullUtility   = FullUtility(:);
    Detail.utility       = FullUtility(:);
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
    HVGain = ones(Total,1);
    if Total == 0 || isempty(FeasibleObj)
        return;
    end

    Ref = max([FeasibleObj;CandidateObj],[],1) + 1;
    try
        ExistingHV = HV(FeasibleObj,Ref);
        HVGain = zeros(Total,1);
        for i = 1 : Total
            HVGain(i) = max(HV([FeasibleObj;CandidateObj(i,:)],Ref) - ExistingHV,0);
        end
    catch
        Dist = pdist2(CandidateObj,FeasibleObj);
        HVGain = 1./(1 + min(Dist,[],2));
    end
end

function PenaltyFactor = ComputeHardNegativeFactor(Problem,CandidateDec,HardNegativeArchive)
    Total = size(CandidateDec,1);
    PenaltyFactor = ones(Total,1);
    if Total == 0 || isempty(HardNegativeArchive) || ~isfield(HardNegativeArchive,'Dec') ...
            || isempty(HardNegativeArchive.Dec)
        return;
    end

    Range = Problem.upper - Problem.lower;
    Range(Range<1e-12) = 1;
    CandNorm   = (CandidateDec - repmat(Problem.lower,Total,1))./repmat(Range,Total,1);
    CenterNorm = (HardNegativeArchive.Dec - repmat(Problem.lower,size(HardNegativeArchive.Dec,1),1)) ...
        ./repmat(Range,size(HardNegativeArchive.Dec,1),1);
    Dist       = pdist2(CandNorm,CenterNorm);
    Radius     = HardNegativeArchive.Radius(:)';
    Radius(Radius<1e-6) = 1e-6;
    ScaledDist = Dist./repmat(Radius,size(Dist,1),1);
    Penalty    = max(exp(-(ScaledDist.^2)),[],2);
    PenaltyFactor = 1 - 0.8*Penalty;
end

function Weight = ResolveDisagreementWeight(Model)
    Weight = 1;
    if isempty(Model)
        return;
    end
    if isfield(Model,'DisagreementWeight') && ~isempty(Model.DisagreementWeight)
        Weight = max(Model.DisagreementWeight,0);
    elseif isfield(Model,'Members') && numel(Model.Members) > 1
        Weight = 1;
    end
end
