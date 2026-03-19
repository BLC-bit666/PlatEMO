function Summary = SummarizeCalibrationProbabilities(Prob,Label,BinCount)
% Summarize calibration quality from predicted probabilities and labels.

    if nargin < 3 || isempty(BinCount)
        BinCount = 10;
    end
    BinCount = max(1,round(BinCount));
    Edges = linspace(0,1,BinCount+1);
    Summary = InitCalibrationSummary(Edges);

    if isempty(Prob) || isempty(Label)
        return;
    end

    Prob  = double(Prob(:));
    Label = double(Label(:));
    Valid = isfinite(Prob) & isfinite(Label);
    Prob  = Prob(Valid);
    Label = double(Label(Valid) > 0);
    if isempty(Prob)
        return;
    end

    Prob = min(max(Prob,0),1);
    Summary.valid        = true;
    Summary.count        = numel(Label);
    Summary.feasibleRate = mean(Label);
    Summary.meanProb     = mean(Prob);
    Summary.brier        = mean((Prob-Label).^2);
    ProbClip             = min(max(Prob,1e-12),1-1e-12);
    Summary.logLoss      = -mean(Label.*log(ProbClip) + (1-Label).*log(1-ProbClip));
    Summary.ece          = ComputeECE(Prob,Label,Edges);
    Summary.prob         = Prob;
    Summary.label        = Label;
    Summary.bin          = SummarizeBins(Prob,Label,Edges);

    ValidBins = Summary.bin.count > 0;
    Summary.reliabilityX      = Summary.bin.meanProb(ValidBins);
    Summary.reliabilityY      = Summary.bin.feasibleRate(ValidBins);
    Summary.reliabilityWeight = Summary.bin.weight(ValidBins);

    NearMask = Prob>=0.4 & Prob<=0.6;
    Summary.nearCount = sum(NearMask);
    if Summary.nearCount > 0
        Summary.nearMeanProb     = mean(Prob(NearMask));
        Summary.nearFeasibleRate = mean(Label(NearMask));
        Summary.nearGap          = abs(Summary.nearFeasibleRate - 0.5);
    end
end

function Summary = InitCalibrationSummary(Edges)
    BinCount = numel(Edges) - 1;
    Summary.valid            = false;
    Summary.count            = 0;
    Summary.feasibleRate     = NaN;
    Summary.meanProb         = NaN;
    Summary.brier            = NaN;
    Summary.logLoss          = NaN;
    Summary.ece              = NaN;
    Summary.nearCount        = 0;
    Summary.nearMeanProb     = NaN;
    Summary.nearFeasibleRate = NaN;
    Summary.nearGap          = NaN;
    Summary.binEdges         = Edges(:)';
    Summary.prob             = zeros(0,1);
    Summary.label            = zeros(0,1);
    Summary.reliabilityX     = zeros(0,1);
    Summary.reliabilityY     = zeros(0,1);
    Summary.reliabilityWeight = zeros(0,1);
    Summary.bin.left         = Edges(1:end-1)';
    Summary.bin.right        = Edges(2:end)';
    Summary.bin.center       = 0.5*(Summary.bin.left + Summary.bin.right);
    Summary.bin.count        = zeros(BinCount,1);
    Summary.bin.weight       = zeros(BinCount,1);
    Summary.bin.meanProb     = NaN(BinCount,1);
    Summary.bin.feasibleRate = NaN(BinCount,1);
    Summary.bin.gap          = NaN(BinCount,1);
end

function Bin = SummarizeBins(Prob,Label,Edges)
    Bin = InitCalibrationSummary(Edges).bin;
    Total = max(numel(Label),1);
    for i = 1 : numel(Bin.left)
        if i < numel(Bin.left)
            Mask = Prob>=Edges(i) & Prob<Edges(i+1);
        else
            Mask = Prob>=Edges(i) & Prob<=Edges(i+1);
        end
        Count = sum(Mask);
        Bin.count(i)  = Count;
        Bin.weight(i) = Count/Total;
        if Count == 0
            continue;
        end
        Bin.meanProb(i)     = mean(Prob(Mask));
        Bin.feasibleRate(i) = mean(Label(Mask));
        Bin.gap(i)          = Bin.feasibleRate(i) - Bin.meanProb(i);
    end
end

function ECE = ComputeECE(Prob,Label,Edges)
    ECE = 0;
    Total = max(numel(Label),1);
    for i = 1 : numel(Edges)-1
        if i < numel(Edges)-1
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
