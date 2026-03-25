function Detail = ScoreBoundaryCandidates(Problem,CandidateDec,CandidateObj,FeasibleObj,Model,W,HardNegativeArchive)
% Compute lean soft-trust scores for Pareto-bridge boundary candidates.

    Total = size(CandidateDec,1);
    Detail = InitBoundaryCandidateDetail(Total);
    if Total == 0
        return;
    end

    [Prob,PredictStats] = PredictBoundaryMLP(Model,CandidateDec);
    QueryScore   = max(0,1 - 2*abs(Prob(:)-0.5));
    Disagreement = PredictStats.std(:);
    [ParetoValue,Sector] = ComputeParetoRelevance(CandidateObj,FeasibleObj,W);
    Reliability = ResolveReliability(Model,Prob);
    Eligible    = IsOutsideHardNegativeRegion(Problem,CandidateDec,HardNegativeArchive);
    TrustGate   = ResolveTrustGate(Model);
    TrustWeight = ResolveTrustWeight(Model);
    LambdaSigma = ResolveDisagreementWeight(Model);

    BoundaryTrust = Reliability .* QueryScore .* (1 + LambdaSigma*Disagreement);
    SoftWeight    = max(0,(1-TrustWeight) + TrustWeight.*BoundaryTrust);
    Utility       = (1e-6 + ParetoValue(:)) .* SoftWeight;
    UncertainOnly = QueryScore(:);
    HighProb      = Prob(:);

    Utility(~Eligible)       = -inf;
    UncertainOnly(~Eligible) = -inf;
    HighProb(~Eligible)      = -inf;
    QueryScore(~Eligible)    = -inf;
    ParetoValue(~Eligible)   = -inf;
    BoundaryTrust(~Eligible) = 0;

    Detail.prob               = Prob(:);
    Detail.disagreement       = Disagreement;
    Detail.queryScore         = QueryScore;
    Detail.paretoValue        = ParetoValue(:);
    Detail.reliability        = Reliability(:);
    Detail.boundaryTrust      = BoundaryTrust(:);
    Detail.uncertaintyUtility = UncertainOnly(:);
    Detail.highProbUtility    = HighProb(:);
    Detail.utility            = Utility(:);
    Detail.utilityGateOff     = ParetoValue(:);
    Detail.sector             = Sector(:);
    Detail.trustGate          = repmat(TrustGate,Total,1);
    Detail.eligible           = Eligible(:);
end

function Detail = InitBoundaryCandidateDetail(Total)
    Detail = struct();
    Detail.prob               = zeros(Total,1);
    Detail.disagreement       = zeros(Total,1);
    Detail.queryScore         = zeros(Total,1);
    Detail.paretoValue        = zeros(Total,1);
    Detail.reliability        = ones(Total,1);
    Detail.boundaryTrust      = zeros(Total,1);
    Detail.uncertaintyUtility = zeros(Total,1);
    Detail.highProbUtility    = zeros(Total,1);
    Detail.utility            = zeros(Total,1);
    Detail.utilityGateOff     = zeros(Total,1);
    Detail.sector             = zeros(Total,1);
    Detail.trustGate          = false(Total,1);
    Detail.eligible           = true(Total,1);
end

function [ParetoValue,Sector] = ComputeParetoRelevance(CandidateObj,FeasibleObj,W)
    Total = size(CandidateObj,1);
    ParetoValue = zeros(Total,1);
    Sector = ones(Total,1);
    if Total == 0
        return;
    end
    if isempty(W)
        W = ones(1,size(CandidateObj,2));
    end

    RefObj = CandidateObj;
    if ~isempty(FeasibleObj)
        RefObj = [FeasibleObj;CandidateObj];
    end
    Sector = AssociateSectors(CandidateObj,W,RefObj);
    if isempty(FeasibleObj)
        return;
    end

    FeasibleSector = AssociateSectors(FeasibleObj,W,RefObj);
    CandidateValue = ComputeSectorScalar(CandidateObj,W,RefObj,Sector);
    for s = unique(Sector(:))'
        FeasibleIdx = find(FeasibleSector == s);
        CandidateIdx = find(Sector == s);
        if isempty(FeasibleIdx) || isempty(CandidateIdx)
            continue;
        end
        ChampionValue = min(ComputeSectorScalar( ...
            FeasibleObj(FeasibleIdx,:),W,RefObj,repmat(s,numel(FeasibleIdx),1)));
        ParetoValue(CandidateIdx) = max(0,ChampionValue - CandidateValue(CandidateIdx));
    end
end

function Reliability = ResolveReliability(Model,Prob)
    Reliability = ones(numel(Prob),1);
    if isempty(Model)
        return;
    end
    if ~isfield(Model,'ReliabilityBinEdges') || isempty(Model.ReliabilityBinEdges) ...
            || ~isfield(Model,'ReliabilityScore') || isempty(Model.ReliabilityScore)
        return;
    end

    Edges = Model.ReliabilityBinEdges(:)';
    Score = Model.ReliabilityScore(:);
    for i = 1 : numel(Prob)
        p = Prob(i);
        Bin = find(p >= Edges(1:end-1) & p <= Edges(2:end),1,'first');
        if isempty(Bin)
            if p < Edges(1)
                Bin = 1;
            else
                Bin = numel(Score);
            end
        elseif Bin > 1 && p == Edges(Bin)
            Bin = Bin - 1;
        end
        Reliability(i) = Score(Bin);
    end
end

function Eligible = IsOutsideHardNegativeRegion(Problem,CandidateDec,HardNegativeArchive)
    Total = size(CandidateDec,1);
    Eligible = true(Total,1);
    if Total == 0 || isempty(HardNegativeArchive) || ~isfield(HardNegativeArchive,'Dec') ...
            || isempty(HardNegativeArchive.Dec)
        return;
    end

    Range = Problem.upper - Problem.lower;
    Range(Range<1e-12) = 1;
    CandNorm = (CandidateDec - repmat(Problem.lower,Total,1))./repmat(Range,Total,1);
    CenterNorm = (HardNegativeArchive.Dec - repmat(Problem.lower,size(HardNegativeArchive.Dec,1),1)) ...
        ./repmat(Range,size(HardNegativeArchive.Dec,1),1);
    Dist = pdist2(CandNorm,CenterNorm);
    Radius = max(HardNegativeArchive.Radius(:)',1e-6);
    Eligible = all(Dist./repmat(Radius,Total,1) >= 1,2);
end

function Flag = ResolveTrustGate(Model)
    Flag = false;
    if isempty(Model)
        return;
    end
    if isfield(Model,'TrustGate') && ~isempty(Model.TrustGate)
        Flag = logical(Model.TrustGate);
    end
end

function Weight = ResolveTrustWeight(Model)
    Weight = 0;
    if isempty(Model)
        return;
    end
    if isfield(Model,'TrustWeight') && ~isempty(Model.TrustWeight)
        Weight = min(max(Model.TrustWeight,0),1);
    end
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
