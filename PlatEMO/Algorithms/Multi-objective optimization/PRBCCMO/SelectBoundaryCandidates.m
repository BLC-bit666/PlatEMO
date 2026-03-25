function [Offspring,Info,Diag,CandidateAudit] = SelectBoundaryCandidates(Problem,Pool,FeasibleObj,Model,W,HardNegativeArchive,Budget,RuntimeOptions)
% Select and evaluate Pareto-bridge boundary queries with trusted semantics.

    Offspring = [];
    Info = InitBoundarySeedInfo(Problem);
    Diag = InitBoundarySelectionDiag();
    CandidateAudit = repmat(InitBoundaryCandidateAuditRow(Problem),0,1);

    if nargin < 8 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    Diag.selectionMode = ResolveSelectionMode(RuntimeOptions);
    Diag.budget = max(0,Budget);
    Diag.hasModel = ~isempty(Model);
    if Budget <= 0 || isempty(Pool.sector)
        Diag.candidateCount = numel(Pool.sector);
        return;
    end

    [CandidateDec,ProxySel] = ResolveBridgePlacement(Problem,Pool,Model,RuntimeOptions);
    if nargin < 3 || isempty(FeasibleObj)
        FeasibleObj = zeros(0,Problem.M);
    end

    Detail = ScoreBoundaryCandidates( ...
        Problem,CandidateDec,ProxySel,FeasibleObj,Model,W,HardNegativeArchive);
    SelectionMode = Diag.selectionMode;
    RankScore = ResolveSelectionScore(Detail,SelectionMode);
    Valid = Detail.eligible(:) & isfinite(RankScore(:));
    CandidateAudit = BuildBoundaryCandidateAudit(Problem,Pool,CandidateDec,Detail,SelectionMode,[]);
    Diag = UpdateBoundarySelectionDiag(Diag,Detail,RankScore,Valid);
    if ~any(Valid)
        return;
    end

    ValidIdx = find(Valid);
    switch SelectionMode
        case 3
            Accept = ValidIdx(randperm(numel(ValidIdx),min(Budget,numel(ValidIdx))));
        otherwise
            [~,Order] = sort(RankScore(ValidIdx),'descend');
            Accept = ValidIdx(Order(1:min(Budget,numel(Order))));
    end
    if isempty(Accept)
        return;
    end
    CandidateAudit = BuildBoundaryCandidateAudit(Problem,Pool,CandidateDec,Detail,SelectionMode,Accept);
    Diag.selectedCount = numel(Accept);

    DecsSel   = CandidateDec(Accept,:);
    ProxySel  = ProxySel(Accept,:);
    SourceSel = Pool.source(Accept);

    Offspring = Problem.Evaluation(DecsSel);
    Info.source        = SourceSel(:);
    Info.score         = RankScore(Accept);
    Info.prob          = Detail.prob(Accept);
    Info.queryScore    = Detail.queryScore(Accept);
    Info.disagreement  = Detail.disagreement(Accept);
    Info.paretoValue   = Detail.paretoValue(Accept);
    Info.reliability   = Detail.reliability(Accept);
    Info.boundaryTrust = Detail.boundaryTrust(Accept);
    Info.utility       = RankScore(Accept);
    Info.sector        = Detail.sector(Accept);
    Info.eligible      = Detail.eligible(Accept);
    Info.proxyObjs     = ProxySel;
    Info.anchorDec     = Pool.anchorDec(Accept,:);
    Info.anchorObj     = Pool.anchorObj(Accept,:);
    Info.helperDec     = Pool.helperDec(Accept,:);
    Info.helperObj     = Pool.helperObj(Accept,:);
end

function Diag = InitBoundarySelectionDiag()
    Diag = struct( ...
        'budget',0, ...
        'selectionMode',1, ...
        'hasModel',false, ...
        'candidateCount',0, ...
        'eligibleCount',0, ...
        'ineligibleCount',0, ...
        'finiteScoreCount',0, ...
        'validCount',0, ...
        'selectedCount',0, ...
        'positiveParetoCount',0, ...
        'trustGate',false, ...
        'maxRankScore',NaN, ...
        'maxParetoValue',NaN, ...
        'maxQueryScore',NaN, ...
        'maxBoundaryTrust',NaN);
end

function Diag = UpdateBoundarySelectionDiag(Diag,Detail,RankScore,Valid)
    Diag.candidateCount = numel(Detail.eligible);
    Diag.eligibleCount = sum(Detail.eligible(:));
    Diag.ineligibleCount = Diag.candidateCount - Diag.eligibleCount;
    Diag.finiteScoreCount = sum(isfinite(RankScore(:)));
    Diag.validCount = sum(Valid(:));
    Diag.positiveParetoCount = nnz(Detail.paretoValue(:) > 0);
    Diag.trustGate = any(Detail.trustGate(:));
    Diag.maxRankScore = SafeFiniteMax(RankScore);
    Diag.maxParetoValue = SafeFiniteMax(Detail.paretoValue);
    Diag.maxQueryScore = SafeFiniteMax(Detail.queryScore);
    Diag.maxBoundaryTrust = SafeFiniteMax(Detail.boundaryTrust);
end

function Value = SafeFiniteMax(Data)
    Value = NaN;
    if isempty(Data)
        return;
    end
    Data = Data(isfinite(Data));
    if isempty(Data)
        return;
    end
    Value = max(Data);
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
    Info.proxyObjs     = zeros(0,Problem.M);
    Info.anchorDec     = zeros(0,Problem.D);
    Info.anchorObj     = zeros(0,Problem.M);
    Info.helperDec     = zeros(0,Problem.D);
    Info.helperObj     = zeros(0,Problem.M);
end

function Row = InitBoundaryCandidateAuditRow(Problem)
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'source',NaN, ...
        'selectionMode',1, ...
        'sector',NaN, ...
        'eligible',false, ...
        'selected',false, ...
        'prob',NaN, ...
        'queryScore',NaN, ...
        'disagreement',NaN, ...
        'reliability',NaN, ...
        'paretoValue',NaN, ...
        'boundaryTrust',NaN, ...
        'utility',NaN, ...
        'candidateDec',zeros(1,Problem.D), ...
        'anchorDec',zeros(1,Problem.D), ...
        'helperDec',zeros(1,Problem.D));
end

function Rows = BuildBoundaryCandidateAudit(Problem,Pool,CandidateDec,Detail,SelectionMode,Accept)
    Count = size(CandidateDec,1);
    Rows = repmat(InitBoundaryCandidateAuditRow(Problem),Count,1);
    Selected = false(Count,1);
    if nargin >= 6 && ~isempty(Accept)
        Selected(Accept) = true;
    end
    for i = 1 : Count
        Rows(i).source = SafeVectorValue(Pool.source,i,NaN);
        Rows(i).selectionMode = SelectionMode;
        Rows(i).sector = SafeVectorValue(Detail.sector,i,NaN);
        Rows(i).eligible = logical(SafeVectorValue(Detail.eligible,i,false));
        Rows(i).selected = Selected(i);
        Rows(i).prob = SafeVectorValue(Detail.prob,i,NaN);
        Rows(i).queryScore = SafeVectorValue(Detail.queryScore,i,NaN);
        Rows(i).disagreement = SafeVectorValue(Detail.disagreement,i,NaN);
        Rows(i).reliability = SafeVectorValue(Detail.reliability,i,NaN);
        Rows(i).paretoValue = SafeVectorValue(Detail.paretoValue,i,NaN);
        Rows(i).boundaryTrust = SafeVectorValue(Detail.boundaryTrust,i,NaN);
        Rows(i).utility = SafeVectorValue(ResolveSelectionScore(Detail,SelectionMode),i,NaN);
        Rows(i).candidateDec = CandidateDec(i,:);
        if isfield(Pool,'anchorDec') && size(Pool.anchorDec,1) >= i
            Rows(i).anchorDec = Pool.anchorDec(i,:);
        end
        if isfield(Pool,'helperDec') && size(Pool.helperDec,1) >= i
            Rows(i).helperDec = Pool.helperDec(i,:);
        end
    end
end

function Value = SafeVectorValue(Data,Index,Default)
    Value = Default;
    if isempty(Data) || numel(Data) < Index
        return;
    end
    Value = Data(Index);
end

function SelectionMode = ResolveSelectionMode(RuntimeOptions)
    SelectionMode = 1;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'SelectionMode') && ~isempty(RuntimeOptions.SelectionMode)
        SelectionMode = max(1,min(4,round(RuntimeOptions.SelectionMode)));
    end
end

function Score = ResolveSelectionScore(Detail,SelectionMode)
    switch SelectionMode
        case 2
            Score = Detail.uncertaintyUtility(:);
        case 4
            Score = Detail.highProbUtility(:);
        otherwise
            Score = Detail.utility(:);
    end
end

function [Decs,ProxyObjs] = ResolveBridgePlacement(Problem,Pool,Model,RuntimeOptions)
    Total = size(Pool.anchorDec,1);
    Decs      = zeros(Total,Problem.D);
    ProxyObjs = 0.5*(Pool.anchorObj + Pool.helperObj);
    for i = 1 : Total
        Decs(i,:) = InterpolateBridgePoint(Problem,Pool.anchorDec(i,:),Pool.helperDec(i,:),0.5);
    end
    SelectionMode = ResolveSelectionMode(RuntimeOptions);
    if Total == 0 || isempty(Model) || ~HasTrustSignal(Model) || SelectionMode == 3
        return;
    end

    LambdaSet = ResolveBridgeScanLambda(RuntimeOptions);
    ScanCount = numel(LambdaSet);
    ScanDec   = zeros(Total*ScanCount,Problem.D);
    ScanProxy = zeros(Total*ScanCount,size(Pool.anchorObj,2));
    for i = 1 : Total
        Offset = (i-1)*ScanCount;
        for j = 1 : ScanCount
            Index = Offset + j;
            ScanDec(Index,:)   = InterpolateBridgePoint(Problem,Pool.anchorDec(i,:),Pool.helperDec(i,:),LambdaSet(j));
            ScanProxy(Index,:) = (1-LambdaSet(j))*Pool.anchorObj(i,:) + LambdaSet(j)*Pool.helperObj(i,:);
        end
    end

    Prob = PredictBoundaryMLP(Model,ScanDec);
    Prob = reshape(Prob(:),ScanCount,Total)';
    for i = 1 : Total
        switch SelectionMode
            case 4
                [~,Best] = max(Prob(i,:));
            otherwise
                [~,Best] = min(abs(Prob(i,:)-0.5));
        end
        Decs(i,:)      = ScanDec((i-1)*ScanCount + Best,:);
        ProxyObjs(i,:) = ScanProxy((i-1)*ScanCount + Best,:);
    end
end

function LambdaSet = ResolveBridgeScanLambda(RuntimeOptions)
    LambdaSet = [0.20,0.35,0.50,0.65,0.80];
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BridgeScanLambda') ...
            && ~isempty(RuntimeOptions.BridgeScanLambda)
        LambdaSet = RuntimeOptions.BridgeScanLambda(:)';
    end
end

function Flag = HasTrustSignal(Model)
    Flag = false;
    if ~isempty(Model) && isfield(Model,'TrustWeight') && ~isempty(Model.TrustWeight)
        Flag = Model.TrustWeight > 0;
    end
end

function Dec = InterpolateBridgePoint(Problem,AnchorDec,HelperDec,Lambda)
    Dec = AnchorDec;
    RealIdx = find(Problem.encoding<=2);
    if ~isempty(RealIdx)
        Dec(RealIdx) = AnchorDec(RealIdx) + Lambda*(HelperDec(RealIdx)-AnchorDec(RealIdx));
    end
    OtherIdx = setdiff(1:Problem.D,RealIdx);
    if ~isempty(OtherIdx)
        Mask = rand(1,numel(OtherIdx)) < Lambda;
        Dec(OtherIdx(Mask)) = HelperDec(OtherIdx(Mask));
    end
    Dec = Problem.CalDec(Dec);
end
