function [Offspring,Info] = SelectBoundaryCandidates(Problem,Pool,FeasibleObj,Model,W,HardNegativeArchive,Budget,RuntimeOptions)
% Select bridge queries using the PRBCCMO-BoundaryCore pipeline.

    Offspring = [];
    Info = InitBoundarySeedInfo(Problem);

    if nargin < 8 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    if isempty(Pool.sector)
        return;
    end
    if nargin < 3 || isempty(FeasibleObj)
        FeasibleObj = zeros(0,Problem.M);
    end

    Placement = LocalizeBridgeBoundaryPoints(Problem,Pool,Model,Budget,RuntimeOptions);
    Detail = ScoreBridgeCandidates( ...
        Problem,Pool,FeasibleObj,W,HardNegativeArchive,Placement,Budget,RuntimeOptions);
    SelectorMode = ResolveSelectorMode(RuntimeOptions);
    RankUtility = ResolveCandidateUtility(Detail,SelectorMode);

    if Budget <= 0 || ~any(Detail.valid)
        return;
    end

    Accept = SelectAcceptedCandidates(Detail,Budget,RuntimeOptions);
    if isempty(Accept)
        return;
    end

    DecsSel = Placement.dec(Accept,:);
    Offspring = Problem.Evaluation(DecsSel);

    Info.source        = Pool.source(Accept);
    Info.score         = RankUtility(Accept);
    Info.prob          = Placement.prob(Accept);
    Info.queryScore    = Detail.boundaryScore(Accept);
    Info.disagreement  = Placement.disagreement(Accept);
    Info.paretoValue   = Detail.paretoValue(Accept);
    Info.reliability   = Detail.reliability(Accept);
    Info.boundaryTrust = Detail.boundaryTrust(Accept);
    Info.utility       = RankUtility(Accept);
    Info.sector        = Pool.sector(Accept);
    Info.eligible      = Detail.eligible(Accept);
    Info.boundaryLocal = false(numel(Accept),1);
    Info.trainKeep     = true(numel(Accept),1);
    Info.proxyObjs     = Placement.proxyObj(Accept,:);
    Info.anchorDec     = Pool.anchorDec(Accept,:);
    Info.anchorObj     = Pool.anchorObj(Accept,:);
    Info.helperDec     = Pool.helperDec(Accept,:);
    Info.helperObj     = Pool.helperObj(Accept,:);
end

function Info = InitBoundarySeedInfo(Problem)
    Info = struct();
    Info.source        = zeros(0,1);
    Info.score         = zeros(0,1);
    Info.prob          = zeros(0,1);
    Info.queryScore    = zeros(0,1);
    Info.disagreement  = zeros(0,1);
    Info.paretoValue   = zeros(0,1);
    Info.reliability   = zeros(0,1);
    Info.boundaryTrust = zeros(0,1);
    Info.utility       = zeros(0,1);
    Info.sector        = zeros(0,1);
    Info.eligible      = false(0,1);
    Info.boundaryLocal = false(0,1);
    Info.trainKeep     = false(0,1);
    Info.proxyObjs     = zeros(0,Problem.M);
    Info.anchorDec     = zeros(0,Problem.D);
    Info.anchorObj     = zeros(0,Problem.M);
    Info.helperDec     = zeros(0,Problem.D);
    Info.helperObj     = zeros(0,Problem.M);
end

function Placement = LocalizeBridgeBoundaryPoints(Problem,Pool,Model,Budget,RuntimeOptions)
    Total = size(Pool.anchorDec,1);
    Placement = struct();
    Placement.dec          = zeros(Total,Problem.D);
    Placement.proxyObj     = zeros(Total,size(Pool.anchorObj,2));
    Placement.lambda0      = 0.5*ones(Total,1);
    Placement.lambda       = 0.5*ones(Total,1);
    Placement.prob0        = 0.5*ones(Total,1);
    Placement.prob         = 0.5*ones(Total,1);
    Placement.disagreement = zeros(Total,1);
    Placement.localTrustWeight = ResolveTrustWeight(Model,RuntimeOptions);
    Placement.refineQuota  = 0;
    Placement.refinedMask  = false(Total,1);
    Placement.refineGain   = zeros(Total,1);
    Placement.localTrust   = false;
    if Total == 0
        return;
    end

    ProbeLambda = ResolveProbeLambda(RuntimeOptions);
    RefineStep  = ResolveProbeRefineStep(RuntimeOptions);
    for i = 1 : Total
        [ProbeDec,ProbeObj] = BuildBridgeSamples( ...
            Problem,Pool.anchorDec(i,:),Pool.helperDec(i,:), ...
            Pool.anchorObj(i,:),Pool.helperObj(i,:),ProbeLambda);
        [ProbeProb,ProbeDis] = PredictBoundaryStatistics(Model,ProbeDec);
        Best0 = SelectBestBoundaryProbe(ProbeLambda,ProbeProb);

        Placement.lambda0(i) = ProbeLambda(Best0);
        Placement.prob0(i) = ProbeProb(Best0);
        Placement.lambda(i) = ProbeLambda(Best0);
        Placement.prob(i) = ProbeProb(Best0);
        Placement.disagreement(i) = ProbeDis(Best0);
        Placement.dec(i,:) = ProbeDec(Best0,:);
        Placement.proxyObj(i,:) = ProbeObj(Best0,:);
    end

    Placement.refineQuota = ResolvePlacementRefineQuota( ...
        Total,Budget,Placement.localTrustWeight,RuntimeOptions);
    Placement.localTrust = Placement.refineQuota > 0 && RefineStep > 0;
    if ~Placement.localTrust
        return;
    end

    RefineIdx = SelectPlacementRefineIdx( ...
        Placement.prob0,Placement.disagreement,Placement.refineQuota);
    for k = 1 : numel(RefineIdx)
        i = RefineIdx(k);
        RefineLambda = unique([ ...
            max(0,Placement.lambda0(i)-RefineStep), ...
            Placement.lambda0(i), ...
            min(1,Placement.lambda0(i)+RefineStep)],'stable');
        [RefineDec,RefineObj] = BuildBridgeSamples( ...
            Problem,Pool.anchorDec(i,:),Pool.helperDec(i,:), ...
            Pool.anchorObj(i,:),Pool.helperObj(i,:),RefineLambda);
        [RefineProb,RefineDis] = PredictBoundaryStatistics(Model,RefineDec);
        Best1 = SelectBestBoundaryProbe(RefineLambda,RefineProb);

        Placement.lambda(i) = RefineLambda(Best1);
        Placement.prob(i) = RefineProb(Best1);
        Placement.disagreement(i) = RefineDis(Best1);
        Placement.dec(i,:) = RefineDec(Best1,:);
        Placement.proxyObj(i,:) = RefineObj(Best1,:);
        Placement.refinedMask(i) = true;
        Placement.refineGain(i) = abs(Placement.prob0(i)-0.5) - abs(Placement.prob(i)-0.5);
    end
end

function [Decs,Objs] = BuildBridgeSamples(Problem,AnchorDec,HelperDec,AnchorObj,HelperObj,LambdaSet)
    Count = numel(LambdaSet);
    Decs = zeros(Count,Problem.D);
    Objs = zeros(Count,numel(AnchorObj));
    for j = 1 : Count
        Lambda = LambdaSet(j);
        Decs(j,:) = InterpolateBridgePoint(Problem,AnchorDec,HelperDec,Lambda);
        Objs(j,:) = (1-Lambda)*AnchorObj + Lambda*HelperObj;
    end
end

function [Prob,Disagreement] = PredictBoundaryStatistics(Model,Decs)
    [Prob,Stats] = PredictBoundaryMLP(Model,Decs);
    Disagreement = zeros(size(Prob));
    if isstruct(Stats) && isfield(Stats,'memberProb') && ~isempty(Stats.memberProb)
        Disagreement = var(Stats.memberProb,0,2);
    elseif isstruct(Stats) && isfield(Stats,'std') && ~isempty(Stats.std)
        Disagreement = Stats.std(:).^2;
    end
end

function Best = SelectBestBoundaryProbe(Lambda,Prob)
    Distance = abs(Prob(:)-0.5);
    MinDistance = min(Distance);
    Candidate = find(Distance <= MinDistance + 1e-12);
    if numel(Candidate) <= 1
        Best = Candidate(1);
        return;
    end
    [~,LocalBest] = min(abs(Lambda(Candidate)-0.5));
    Best = Candidate(LocalBest);
end

function Lambda = ResolveProbeLambda(RuntimeOptions)
    Lambda = [0.25,0.50,0.75];
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgeScanLambda') ...
            && ~isempty(RuntimeOptions.BridgeScanLambda)
        Lambda = RuntimeOptions.BridgeScanLambda(:)';
    end
    Lambda = min(max(Lambda,0),1);
    Lambda = unique(Lambda,'stable');
    if isempty(Lambda)
        Lambda = [0.25,0.50,0.75];
    end
end

function Step = ResolveProbeRefineStep(RuntimeOptions)
    Step = 0.125;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgeRefineStep') ...
            && ~isempty(RuntimeOptions.BridgeRefineStep)
        Step = RuntimeOptions.BridgeRefineStep;
    end
    Step = min(max(Step,0),0.5);
end

function Quota = ResolvePlacementRefineQuota(Total,Budget,TrustWeight,RuntimeOptions)
    Quota = 0;
    if Total <= 0 || Budget <= 0
        return;
    end
    if nargin >= 4 && isstruct(RuntimeOptions) ...
            && isfield(RuntimeOptions,'ForcePlacementRefine') ...
            && logical(RuntimeOptions.ForcePlacementRefine)
        Quota = Total;
        return;
    end
    if nargin >= 4 && isstruct(RuntimeOptions) ...
            && isfield(RuntimeOptions,'DisableTrust') ...
            && logical(RuntimeOptions.DisableTrust)
        return;
    end
    Quota = min(Total,max(0,floor(Budget*TrustWeight)));
end

function Idx = SelectPlacementRefineIdx(Prob0,Disagreement,Quota)
    if nargin < 3 || Quota <= 0 || isempty(Prob0)
        Idx = zeros(0,1);
        return;
    end
    Total = numel(Prob0);
    SortKey = [ ...
        abs(Prob0(:)-0.5), ...
        -Disagreement(:), ...
        (1:Total)'];
    [~,Order] = sortrows(SortKey,[1 2 3]);
    Idx = Order(1:min(Quota,Total));
end

function Detail = ScoreBridgeCandidates( ...
    Problem,Pool,FeasibleObj,W,HardNegativeArchive,Placement,Budget,RuntimeOptions)

    Total = size(Placement.dec,1);
    Detail = struct();
    Detail.paretoValue   = zeros(Total,1);
    Detail.boundaryScore = max(0,1 - 2*abs(Placement.prob(:)-0.5));
    Detail.disagreement  = Placement.disagreement(:);
    Detail.boundaryTrust = Detail.boundaryScore;
    Detail.reliability   = ones(Total,1);
    Detail.eligible      = IsOutsideHardNegativeRegion(Problem,Placement.dec,HardNegativeArchive);
    Detail.shortlisted   = false(Total,1);
    Detail.valid         = false(Total,1);
    Detail.sector        = Pool.sector(:);
    if Total == 0
        return;
    end

    Detail.paretoValue = ComputeBridgeParetoValue(Pool,FeasibleObj,W);
    EligibleIdx = find(Detail.eligible);
    if isempty(EligibleIdx) || Budget <= 0
        return;
    end

    ShortlistFactor = ResolveShortlistFactor(RuntimeOptions);
    ShortlistCount = min(numel(EligibleIdx),max(Budget,round(ShortlistFactor*Budget)));
    StageAOrder = SortCandidates(EligibleIdx,Detail.paretoValue);
    ShortlistedIdx = StageAOrder(1:ShortlistCount);
    Detail.shortlisted(ShortlistedIdx) = true;
    Detail.valid = Detail.shortlisted & Detail.eligible;
end

function ParetoValue = ComputeBridgeParetoValue(Pool,FeasibleObj,W)
    Total = size(Pool.anchorObj,1);
    ParetoValue = zeros(Total,1);
    if Total == 0
        return;
    end
    if isempty(W)
        W = ones(1,size(Pool.anchorObj,2));
    end
    RefObj = Pool.anchorObj;
    if ~isempty(FeasibleObj)
        RefObj = [FeasibleObj;Pool.anchorObj];
    end

    AnchorSector = Pool.sector(:);
    AnchorValue = ComputeSectorScalar(Pool.anchorObj,W,RefObj,AnchorSector);
    if isempty(FeasibleObj)
        return;
    end

    FeasibleSector = AssociateSectors(FeasibleObj,W,RefObj);
    FeasibleValue = ComputeSectorScalar(FeasibleObj,W,RefObj,FeasibleSector);
    for s = unique(AnchorSector(:))'
        FeasibleIdx = find(FeasibleSector == s);
        AnchorIdx = find(AnchorSector == s);
        if isempty(FeasibleIdx) || isempty(AnchorIdx)
            continue;
        end
        ChampionValue = min(FeasibleValue(FeasibleIdx));
        ParetoValue(AnchorIdx) = max(0,ChampionValue - AnchorValue(AnchorIdx));
    end
end

function Order = SortCandidates(Idx,ParetoValue)
    SortKey = [ ...
        -ParetoValue(Idx), ...
        Idx(:)];
    [~,LocalOrder] = sortrows(SortKey,[1 2]);
    Order = Idx(LocalOrder);
end

function Accept = SelectAcceptedCandidates(Detail,Budget,RuntimeOptions)
    EligibleIdx = find(Detail.eligible);
    ShortlistedIdx = find(Detail.valid);
    if Budget <= 0 || isempty(EligibleIdx)
        Accept = zeros(0,1);
        return;
    end

    Mode = ResolveSelectorMode(RuntimeOptions);
    switch Mode
        case 'pareto_then_boundary'
            if isempty(ShortlistedIdx)
                Accept = zeros(0,1);
                return;
            end
            Order = SortShortlistedCandidates( ...
                ShortlistedIdx,Detail.boundaryScore,Detail.disagreement);
        case 'pareto_only'
            Order = SortCandidates(EligibleIdx,Detail.paretoValue);
        case 'boundary_only'
            Order = SortShortlistedCandidates( ...
                EligibleIdx,Detail.boundaryScore,Detail.disagreement);
        case 'random_within_shortlist'
            if isempty(ShortlistedIdx)
                Accept = zeros(0,1);
                return;
            end
            Order = ShortlistedIdx(randperm(numel(ShortlistedIdx)));
        otherwise
            error('PRBCCMO:UnsupportedSelectorMode', ...
                'Unsupported selector mode ''%s''.',Mode);
    end
    Accept = Order(1:min(Budget,numel(Order)));
end

function Order = SortShortlistedCandidates(Idx,BoundaryScore,Disagreement)
    SortKey = [ ...
        -BoundaryScore(Idx), ...
        -Disagreement(Idx), ...
        Idx(:)];
    [~,LocalOrder] = sortrows(SortKey,[1 2 3]);
    Order = Idx(LocalOrder);
end

function Factor = ResolveShortlistFactor(RuntimeOptions)
    Factor = 3;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BoundaryShortlistFactor') ...
            && ~isempty(RuntimeOptions.BoundaryShortlistFactor)
        Factor = RuntimeOptions.BoundaryShortlistFactor;
    end
    Factor = max(1,round(Factor));
end

function Mode = ResolveSelectorMode(RuntimeOptions)
    Mode = 'pareto_then_boundary';
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'SelectorMode') ...
            && ~isempty(RuntimeOptions.SelectorMode)
        Mode = lower(strtrim(char(RuntimeOptions.SelectorMode)));
    end
    switch Mode
        case {'pareto_then_boundary','pareto-boundary','boundary_core','boundarycore'}
            Mode = 'pareto_then_boundary';
        case {'pareto_only','pareto-only'}
            Mode = 'pareto_only';
        case {'boundary_only','boundary-only'}
            Mode = 'boundary_only';
        case {'random_within_shortlist','random-shortlist','shortlist_random'}
            Mode = 'random_within_shortlist';
        otherwise
            error('PRBCCMO:UnsupportedSelectorMode', ...
                'Unsupported selector mode ''%s''.',char(RuntimeOptions.SelectorMode));
    end
end

function Utility = ResolveCandidateUtility(Detail,SelectorMode)
    switch SelectorMode
        case 'pareto_then_boundary'
            Utility = Detail.boundaryScore(:);
        case 'pareto_only'
            Utility = Detail.paretoValue(:);
        case 'boundary_only'
            Utility = Detail.boundaryScore(:);
        case 'random_within_shortlist'
            Utility = zeros(size(Detail.boundaryScore(:)));
        otherwise
            error('PRBCCMO:UnsupportedSelectorMode', ...
                'Unsupported selector mode ''%s''.',SelectorMode);
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

function Weight = ResolveTrustWeight(Model,RuntimeOptions)
    Weight = 0;
    if nargin >= 2 && isstruct(RuntimeOptions) && isfield(RuntimeOptions,'DisableTrust') ...
            && logical(RuntimeOptions.DisableTrust)
        return;
    end
    if isempty(Model)
        return;
    end
    if isfield(Model,'TrustWeight') && ~isempty(Model.TrustWeight)
        Weight = min(max(Model.TrustWeight,0),1);
    end
end

function Dec = InterpolateBridgePoint(Problem,AnchorDec,HelperDec,Lambda)
    Dec = InterpolateBoundaryPoint(Problem,AnchorDec,HelperDec,Lambda);
end
