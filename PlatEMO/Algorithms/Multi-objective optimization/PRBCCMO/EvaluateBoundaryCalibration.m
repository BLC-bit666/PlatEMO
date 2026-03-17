function Metric = EvaluateBoundaryCalibration(Model,CalDec,CalLabel)
% Evaluate calibration quality on the held-out boundary calibration buffer.

    Metric.brier     = inf;
    Metric.ece       = inf;
    Metric.nearGap   = inf;
    Metric.nearCount = 0;
    if isempty(CalDec) || isempty(CalLabel) || isempty(Model)
        return;
    end

    CalLabel = double(CalLabel(:));
    if numel(unique(CalLabel)) < 2
        return;
    end

    Prob = PredictBoundaryMLP(Model,CalDec);
    Metric.brier = mean((Prob-CalLabel).^2);
    Metric.ece   = ComputeECE(Prob,CalLabel,10);

    NearMask = Prob>=0.4 & Prob<=0.6;
    Metric.nearCount = sum(NearMask);
    if Metric.nearCount > 0
        Metric.nearGap = abs(mean(CalLabel(NearMask)) - 0.5);
    else
        Metric.nearGap = 0;
    end
end

function ECE = ComputeECE(Prob,Label,BinCount)
    Edges = linspace(0,1,BinCount+1);
    Total = numel(Label);
    ECE   = 0;
    for i = 1 : BinCount
        if i < BinCount
            Mask = Prob>=Edges(i) & Prob<Edges(i+1);
        else
            Mask = Prob>=Edges(i) & Prob<=Edges(i+1);
        end
        if ~any(Mask)
            continue;
        end
        Acc  = mean(Label(Mask));
        Conf = mean(Prob(Mask));
        ECE  = ECE + (sum(Mask)/Total)*abs(Acc-Conf);
    end
end
