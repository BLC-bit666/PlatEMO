function [Offspring,Info] = SelectBoundaryCandidates(Problem,Pool,PopulationC,Model,W,HardNegativeArchive,Budget,RuntimeOptions)
% Select and evaluate Pareto-bridge boundary queries with trusted semantics.

    Offspring = [];
    Info = InitBoundarySeedInfo(Problem);

    if nargin < 8 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    if Budget <= 0 || isempty(Pool.sector)
        return;
    end

    [CandidateDec,ProxySel] = ResolveBridgePlacement(Problem,Pool,Model,RuntimeOptions);
    FeasibleC = PopulationC(all(PopulationC.cons<=0,2));
    if isempty(FeasibleC)
        FeasibleObj = zeros(0,Problem.M);
    else
        FeasibleObj = FeasibleC.objs;
    end

    Detail = ScoreBoundaryCandidates( ...
        Problem,CandidateDec,ProxySel,FeasibleObj,Model,W,HardNegativeArchive);
    SelectionMode = ResolveSelectionMode(RuntimeOptions);
    RankScore = ResolveSelectionScore(Detail,SelectionMode);
    Valid = Detail.eligible(:) & isfinite(RankScore(:));
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

function SelectionMode = ResolveSelectionMode(RuntimeOptions)
    SelectionMode = 1;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'SelectionMode') && ~isempty(RuntimeOptions.SelectionMode)
        SelectionMode = max(1,min(3,round(RuntimeOptions.SelectionMode)));
    end
end

function Score = ResolveSelectionScore(Detail,SelectionMode)
    TrustGate = any(Detail.trustGate);
    switch SelectionMode
        case 2
            if TrustGate
                Score = Detail.uncertaintyUtility(:);
            else
                Score = Detail.utilityGateOff(:);
            end
        otherwise
            if TrustGate
                Score = Detail.utility(:);
            else
                Score = Detail.utilityGateOff(:);
            end
    end
end

function [Decs,ProxyObjs] = ResolveBridgePlacement(Problem,Pool,Model,RuntimeOptions)
    Total = size(Pool.anchorDec,1);
    Decs      = zeros(Total,Problem.D);
    ProxyObjs = 0.5*(Pool.anchorObj + Pool.helperObj);
    for i = 1 : Total
        Decs(i,:) = InterpolateBridgePoint(Problem,Pool.anchorDec(i,:),Pool.helperDec(i,:),0.5);
    end
    if Total == 0 || isempty(Model) || ~HasTrustGate(Model) || ResolveSelectionMode(RuntimeOptions) == 3
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
    Diff = reshape(abs(Prob(:)-0.5),ScanCount,Total)';
    for i = 1 : Total
        [~,Best] = min(Diff(i,:));
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

function Flag = HasTrustGate(Model)
    Flag = false;
    if ~isempty(Model) && isfield(Model,'TrustGate') && ~isempty(Model.TrustGate)
        Flag = logical(Model.TrustGate);
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
