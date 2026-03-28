function Metric = EvaluateBoundaryCalibration(Model,CalDec,CalLabel,BoundaryLocalMask,BinCount)
% Evaluate calibration quality on the boundary-local held-out subset.

    if nargin < 5 || isempty(BinCount)
        BinCount = 10;
    end
    if nargin < 4 || isempty(BoundaryLocalMask)
        BoundaryLocalMask = zeros(0,1);
    end

    Metric = SummarizeCalibrationProbabilities([],[],BinCount);
    Metric.brier     = inf;
    Metric.ece       = inf;
    Metric.nearGap   = inf;
    Metric.nearCount = 0;
    Metric.coreNearGap    = inf;
    Metric.coreNearCount  = 0;
    Metric.relaxedNearGap   = inf;
    Metric.relaxedNearCount = 0;
    Metric.trustGate = false;
    Metric.trustAuditPass = false;
    Metric.trustWeight = 0;
    Metric.trustWeightRaw = 0;
    Metric.calibrator = 'raw';
    Metric.classCount = 0;
    Metric.singleClass = false;
    Metric.invalidReason = 'empty_buffer';
    Metric.boundaryCount = 0;
    Metric.boundaryGap = inf;
    Metric.boundaryEce = inf;
    Metric.boundaryLocalMaskCount = 0;
    Metric.predNearCount = 0;
    Metric.globalCount = 0;
    Metric.globalClassCount = 0;
    Metric.globalSingleClass = false;
    Metric.globalEce = inf;
    Metric.globalGap = inf;
    Metric.globalMeanProb = NaN;
    Metric.globalFeasibleRate = NaN;
    if isempty(Model)
        Metric.invalidReason = 'missing_model';
        return;
    end
    if isempty(CalDec) || isempty(CalLabel)
        return;
    end

    CalLabel = double(CalLabel(:));
    Prob = PredictBoundaryMLP(Model,CalDec);
    Global = SummarizeCalibrationProbabilities(Prob,CalLabel,BinCount);
    GlobalGap = abs(Global.meanProb - Global.feasibleRate);
    Metric.globalCount = Global.count;
    Metric.globalClassCount = Global.classCount;
    Metric.globalSingleClass = Global.classCount < 2;
    Metric.globalEce = Global.ece;
    Metric.globalGap = GlobalGap;
    Metric.globalMeanProb = Global.meanProb;
    Metric.globalFeasibleRate = Global.feasibleRate;

    BoundaryLocalMask = NormalizeBoundaryLocalMask(BoundaryLocalMask,numel(CalLabel));
    Delta = ResolveBoundaryLocalDelta(Model);
    PredNearMask = abs(Prob(:)-0.5) <= Delta;
    BoundaryMask = PredNearMask | BoundaryLocalMask;
    Boundary = SummarizeCalibrationProbabilities(Prob(BoundaryMask),CalLabel(BoundaryMask),BinCount);
    BoundaryGap = abs(Boundary.meanProb - Boundary.feasibleRate);
    if ~isfinite(BoundaryGap)
        BoundaryGap = inf;
    end

    Metric = CopySummaryFields(Metric,Boundary);
    Metric.boundaryCount = Boundary.count;
    Metric.boundaryGap = BoundaryGap;
    Metric.boundaryEce = Boundary.ece;
    Metric.boundaryLocalMaskCount = sum(BoundaryLocalMask);
    Metric.predNearCount = sum(PredNearMask);
    Metric.trustGate = false;
    Metric.trustWeight = 0;
    Metric.calibrator = 'raw';
    Metric.classCount = Boundary.classCount;
    Metric.singleClass = Boundary.classCount < 2;
    Metric.invalidReason = '';
    Metric.valid = Boundary.count > 0;
    if Boundary.count == 0
        Metric.valid = false;
        Metric.invalidReason = 'empty_boundary_local';
    elseif Metric.singleClass
        Metric.valid = false;
        Metric.invalidReason = 'invalid_single_class';
    end

    Metric.nearCount = Boundary.count;
    Metric.nearMeanProb = Boundary.meanProb;
    Metric.nearFeasibleRate = Boundary.feasibleRate;
    Metric.nearGap = BoundaryGap;
    Metric.coreNearCount = Boundary.count;
    Metric.coreNearMeanProb = Boundary.meanProb;
    Metric.coreNearFeasibleRate = Boundary.feasibleRate;
    Metric.coreNearGap = BoundaryGap;
    Metric.relaxedNearCount = Boundary.count;
    Metric.relaxedNearMeanProb = Boundary.meanProb;
    Metric.relaxedNearFeasibleRate = Boundary.feasibleRate;
    Metric.relaxedNearGap = BoundaryGap;

    if isfield(Model,'TrustGate') && ~isempty(Model.TrustGate)
        Metric.trustGate = logical(Model.TrustGate);
    end
    if isfield(Model,'TrustAuditPass') && ~isempty(Model.TrustAuditPass)
        Metric.trustAuditPass = logical(Model.TrustAuditPass);
    end
    if isfield(Model,'TrustWeight') && ~isempty(Model.TrustWeight)
        Metric.trustWeight = Model.TrustWeight;
    end
    if isfield(Model,'TrustWeightRaw') && ~isempty(Model.TrustWeightRaw)
        Metric.trustWeightRaw = Model.TrustWeightRaw;
    end
    if isfield(Model,'CalibratorType') && ~isempty(Model.CalibratorType)
        Metric.calibrator = Model.CalibratorType;
    end
end

function Mask = NormalizeBoundaryLocalMask(Mask,Count)
    if isempty(Mask)
        Mask = false(Count,1);
        return;
    end
    Mask = logical(Mask(:));
    if numel(Mask) < Count
        Mask = [Mask;false(Count-numel(Mask),1)];
    elseif numel(Mask) > Count
        Mask = Mask(1:Count);
    end
end

function Delta = ResolveBoundaryLocalDelta(Model)
    Delta = 0.10;
    if isstruct(Model) && isfield(Model,'BoundaryLocalDelta') && ~isempty(Model.BoundaryLocalDelta)
        Delta = Model.BoundaryLocalDelta;
    end
    Delta = min(max(Delta,0),0.5);
end

function Metric = CopySummaryFields(Metric,Summary)
    Fields = fieldnames(Summary);
    for i = 1 : numel(Fields)
        Metric.(Fields{i}) = Summary.(Fields{i});
    end
end
