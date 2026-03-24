function Metric = EvaluateBoundaryCalibration(Model,CalDec,CalLabel,BinCount)
% Evaluate calibration quality on a held-out labeled evaluation buffer.

    if nargin < 4 || isempty(BinCount)
        BinCount = 10;
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
    Metric.calibrator = 'raw';
    Metric.classCount = 0;
    Metric.singleClass = false;
    Metric.invalidReason = 'empty_buffer';
    if isempty(Model)
        Metric.invalidReason = 'missing_model';
        return;
    end
    if isempty(CalDec) || isempty(CalLabel)
        return;
    end

    CalLabel = double(CalLabel(:));
    Prob = PredictBoundaryMLP(Model,CalDec);
    Metric = SummarizeCalibrationProbabilities(Prob,CalLabel,BinCount);
    Metric.classCount = numel(unique(CalLabel));
    Metric.singleClass = Metric.classCount < 2;
    Metric.invalidReason = '';
    if Metric.singleClass
        Metric.valid = false;
        Metric.invalidReason = 'invalid_single_class';
    end
    if isfield(Model,'TrustGate') && ~isempty(Model.TrustGate)
        Metric.trustGate = logical(Model.TrustGate);
    end
    if isfield(Model,'CalibratorType') && ~isempty(Model.CalibratorType)
        Metric.calibrator = Model.CalibratorType;
    end
end
