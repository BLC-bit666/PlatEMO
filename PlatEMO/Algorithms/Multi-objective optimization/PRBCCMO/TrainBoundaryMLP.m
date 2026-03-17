function Model = TrainBoundaryMLP(X,Y,Hidden,Epoch,LR,PrevModel,CalDec,CalLabel,Options)
% Train or warm-start a lightweight one-hidden-layer MLP for boundary probability.

    if nargin < 6
        PrevModel = [];
    end
    if nargin < 7
        CalDec = [];
    end
    if nargin < 8
        CalLabel = [];
    end
    if nargin < 9
        Options = struct();
    end
    if isstruct(CalDec) && nargin == 7
        Options = CalDec;
        CalDec  = [];
        CalLabel = [];
    end

    WarmStart = isfield(Options,'WarmStart') && Options.WarmStart;

    Model = PrevModel;
    if isempty(X) || size(X,1) < 4
        return;
    end

    X = double(X);
    Y = double(Y(:) > 0);
    if numel(unique(Y)) < 2
        return;
    end

    [~,D] = size(X);
    Hidden = max(2,round(Hidden));
    Epoch  = max(1,round(Epoch));
    LR     = max(LR,1e-4);
    Lambda = 1e-4;

    XTrain = X;
    YTrain = Y;
    if (isempty(CalDec) || isempty(CalLabel)) && size(X,1) >= 10
        [TrainIdx,CalIdx] = SplitCalibrationData(Y);
        XTrain = X(TrainIdx,:);
        YTrain = Y(TrainIdx);
        if numel(unique(YTrain)) < 2
            XTrain = X;
            YTrain = Y;
            CalIdx = zeros(0,1);
        end
        CalDec = X(CalIdx,:);
        CalLabel = Y(CalIdx);
    end

    if WarmStart && IsCompatibleWarmStart(PrevModel,D,Hidden)
        Mu    = PrevModel.Mu;
        Sigma = PrevModel.Sigma;
        W1    = PrevModel.W1;
        b1    = PrevModel.b1;
        W2    = PrevModel.W2;
        b2    = PrevModel.b2;
    else
        Mu    = mean(XTrain,1);
        Sigma = std(XTrain,0,1);
        Sigma(Sigma<1e-12) = 1;
        W1 = 0.1*randn(D,Hidden);
        b1 = zeros(1,Hidden);
        W2 = 0.1*randn(Hidden,1);
        b2 = 0;
    end
    Xn = (XTrain-Mu)./Sigma;
    [Weight,NormWeight] = BuildClassWeights(YTrain);
    NT = size(Xn,1);

    for e = 1 : Epoch
        H = tanh(Xn*W1 + repmat(b1,NT,1));
        Z = H*W2 + b2;
        P = 1./(1+exp(-Z));
        D2 = Weight.*(P - YTrain);

        dW2 = (H'*D2)./NormWeight + Lambda*W2;
        db2 = sum(D2)./NormWeight;
        D1  = (D2*W2').*(1-H.^2);
        dW1 = (Xn'*D1)./NormWeight + Lambda*W1;
        db1 = sum(D1,1)./NormWeight;

        Step = LR/sqrt(e);
        W1 = W1 - Step*dW1;
        b1 = b1 - Step*db1;
        W2 = W2 - Step*dW2;
        b2 = b2 - Step*db2;
    end

    Model.Mu    = Mu;
    Model.Sigma = Sigma;
    Model.W1    = W1;
    Model.b1    = b1;
    Model.W2    = W2;
    Model.b2    = b2;
    Model.Temp  = FitTemperature(Model,CalDec,CalLabel);
end

function tf = IsCompatibleWarmStart(Model,D,Hidden)
    tf = ~isempty(Model) && isfield(Model,'W1') && isfield(Model,'W2') ...
        && size(Model.W1,1) == D && size(Model.W1,2) == Hidden ...
        && size(Model.W2,1) == Hidden;
end

function [TrainIdx,CalIdx] = SplitCalibrationData(Y)
    N = numel(Y);
    TrainIdx = (1:N)';
    CalIdx   = zeros(0,1);
    if N < 10
        return;
    end

    CalCount = min(max(2,round(0.2*N)),N-4);
    CalIdx   = (N-CalCount+1:N)';
    TrainIdx = (1:N-CalCount)';
    if numel(unique(Y(CalIdx))) < 2
        CalIdx = zeros(0,1);
        TrainIdx = (1:N)';
    end
end

function [Weight,NormWeight] = BuildClassWeights(Y)
    N = numel(Y);
    Pos = sum(Y==1);
    Neg = N - Pos;
    WPos = N/(2*max(1,Pos));
    WNeg = N/(2*max(1,Neg));
    Weight = WNeg + (WPos-WNeg).*Y;
    NormWeight = max(sum(Weight),1);
end

function Temp = FitTemperature(Model,CalDec,CalLabel)
    Temp = 1;
    if isempty(CalDec) || isempty(CalLabel)
        return;
    end

    CalDec = double(CalDec);
    CalLabel = double(CalLabel(:));
    if numel(unique(CalLabel)) < 2
        return;
    end
    Xn = (CalDec-Model.Mu)./Model.Sigma;
    H  = tanh(Xn*Model.W1 + repmat(Model.b1,size(Xn,1),1));
    Z  = H*Model.W2 + Model.b2;
    Yc = CalLabel;
    Theta = 0;
    for iter = 1 : 40
        TempNow = exp(Theta);
        P = 1./(1+exp(-Z./TempNow));
        GradT = mean((P-Yc).*(-Z./(TempNow^2)));
        Theta = Theta - 0.05*(GradT*TempNow);
        Theta = min(max(Theta,log(0.5)),log(5));
    end
    Temp = exp(Theta);
end
