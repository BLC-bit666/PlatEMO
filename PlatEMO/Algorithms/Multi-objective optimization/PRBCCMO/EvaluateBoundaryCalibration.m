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
    Metric.trustGate = false;
    Metric.calibrator = 'raw';
    if isempty(CalDec) || isempty(CalLabel) || isempty(Model)
        return;
    end

    CalLabel = double(CalLabel(:));
    if numel(unique(CalLabel)) < 2
        return;
    end

    Prob = PredictBoundaryMLP(Model,CalDec);
    Metric = SummarizeCalibrationProbabilities(Prob,CalLabel,BinCount);
    if isfield(Model,'TrustGate') && ~isempty(Model.TrustGate)
        Metric.trustGate = logical(Model.TrustGate);
    end
    if isfield(Model,'CalibratorType') && ~isempty(Model.CalibratorType)
        Metric.calibrator = Model.CalibratorType;
    end
end
