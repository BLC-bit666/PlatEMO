function Model = TrainBoundaryMLP(X,Y,Hidden,Epoch,LR,PrevModel,CalDec,CalLabel,Options)
% Train a single calibrated boundary MLP or a small committee with structured losses.

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

    Hidden = max(2,round(Hidden));
    Epoch  = max(1,round(Epoch));
    LR     = max(LR,1e-4);
    EnsembleSize = max(1,round(GetOption(Options,'EnsembleSize',1)));
    CalibratorType = ResolveCalibratorType(GetOption(Options,'Calibrator','temperature'));

    if EnsembleSize <= 1
        Model = TrainSingleBoundaryModel( ...
            X,Y,Hidden,Epoch,LR,PrevModel,CalDec,CalLabel,Options,CalibratorType);
        return;
    end

    Members = cell(1,EnsembleSize);
    for k = 1 : EnsembleSize
        PrevMember = ExtractPreviousMember(PrevModel,k);
        [BootX,BootY] = BootstrapTrainingSubset(X,Y);
        MemberOptions = Options;
        MemberOptions.EnsembleSize = 1;
        MemberOptions.MemberIndex  = k;
        Members{k} = TrainSingleBoundaryModel( ...
            BootX,BootY,Hidden,Epoch,LR,PrevMember,CalDec,CalLabel,MemberOptions,CalibratorType);
    end

    Model.Members        = Members;
    Model.EnsembleSize   = EnsembleSize;
    Model.CalibratorType = CalibratorType;
    Model.DisagreementWeight = max(GetOption(Options,'DisagreementWeight',1),0);
end

function Model = TrainSingleBoundaryModel(X,Y,Hidden,Epoch,LR,PrevModel,CalDec,CalLabel,Options,CalibratorType)
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

    PairFeasible = NormalizeDecisionMatrix(GetOption(Options,'PairFeasibleDec',[]),size(X,2));
    PairInfeasible = NormalizeDecisionMatrix(GetOption(Options,'PairInfeasibleDec',[]),size(X,2));
    MidDec = NormalizeDecisionMatrix(GetOption(Options,'MidDec',[]),size(X,2));
    HardNegDec = NormalizeDecisionMatrix(GetOption(Options,'HardNegDec',[]),size(X,2));
    PairMargin = max(0,GetOption(Options,'PairMargin',0.05));
    LambdaPair = max(0,GetOption(Options,'LambdaPair',1));
    LambdaMid  = max(0,GetOption(Options,'LambdaMid',1));
    LambdaHard = max(0,GetOption(Options,'LambdaHardNeg',1));

    [~,D] = size(X);
    LambdaReg = 1e-4;

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
        Delta2 = Weight.*(P - YTrain)./NormWeight;

        dW2 = H'*Delta2 + LambdaReg*W2;
        db2 = sum(Delta2);
        D1  = (Delta2*W2').*(1-H.^2);
        dW1 = Xn'*D1 + LambdaReg*W1;
        db1 = sum(D1,1);

        if LambdaPair > 0 && ~isempty(PairFeasible) && ~isempty(PairInfeasible)
            [PairdW1,Pairdb1,PairdW2,Pairdb2] = ComputePairGradients( ...
                PairFeasible,PairInfeasible,PairMargin,Mu,Sigma,W1,b1,W2,b2);
            dW1 = dW1 + LambdaPair*PairdW1;
            db1 = db1 + LambdaPair*Pairdb1;
            dW2 = dW2 + LambdaPair*PairdW2;
            db2 = db2 + LambdaPair*Pairdb2;
        end

        if LambdaMid > 0 && ~isempty(MidDec)
            [MiddW1,Middb1,MiddW2,Middb2] = ComputeMidpointGradients( ...
                MidDec,Mu,Sigma,W1,b1,W2,b2);
            dW1 = dW1 + LambdaMid*MiddW1;
            db1 = db1 + LambdaMid*Middb1;
            dW2 = dW2 + LambdaMid*MiddW2;
            db2 = db2 + LambdaMid*Middb2;
        end

        if LambdaHard > 0 && ~isempty(HardNegDec)
            [HarddW1,Harddb1,HarddW2,Harddb2] = ComputeHardNegativeGradients( ...
                HardNegDec,Mu,Sigma,W1,b1,W2,b2);
            dW1 = dW1 + LambdaHard*HarddW1;
            db1 = db1 + LambdaHard*Harddb1;
            dW2 = dW2 + LambdaHard*HarddW2;
            db2 = db2 + LambdaHard*Harddb2;
        end

        Step = LR/sqrt(e);
        W1 = W1 - Step*dW1;
        b1 = b1 - Step*db1;
        W2 = W2 - Step*dW2;
        b2 = b2 - Step*db2;
    end

    Model.Mu             = Mu;
    Model.Sigma          = Sigma;
    Model.W1             = W1;
    Model.b1             = b1;
    Model.W2             = W2;
    Model.b2             = b2;
    Model.CalibratorType = CalibratorType;
    Model.DisagreementWeight = max(GetOption(Options,'DisagreementWeight',1),0);
    Model.Temp           = 1;
    Model.SigmoidA       = 1;
    Model.SigmoidB       = 0;
    [Model.Temp,Model.SigmoidA,Model.SigmoidB] = FitCalibrator(Model,CalDec,CalLabel,CalibratorType);
end

function [dW1,db1,dW2,db2] = ComputePairGradients(FeasibleDec,InfeasibleDec,Margin,Mu,Sigma,W1,b1,W2,b2)
    Count = min(size(FeasibleDec,1),size(InfeasibleDec,1));
    dW1 = zeros(size(W1));
    db1 = zeros(size(b1));
    dW2 = zeros(size(W2));
    db2 = 0;
    if Count <= 0
        return;
    end

    FeasibleDec   = FeasibleDec(1:Count,:);
    InfeasibleDec = InfeasibleDec(1:Count,:);
    [Pf,Zf,Hf,Xf] = ForwardBoundary(FeasibleDec,Mu,Sigma,W1,b1,W2,b2);
    [Pi,Zi,Hi,Xi] = ForwardBoundary(InfeasibleDec,Mu,Sigma,W1,b1,W2,b2);
    Active = (Margin - Pf + Pi) > 0;
    ActiveCount = sum(Active);
    if ActiveCount == 0
        return;
    end

    dZf = -(Pf.*(1-Pf))./ActiveCount;
    dZi = +(Pi.*(1-Pi))./ActiveCount;
    dZf(~Active) = 0;
    dZi(~Active) = 0;
    [dW1f,db1f,dW2f,db2f] = BackwardBoundary(Xf,Hf,dZf,W2);
    [dW1i,db1i,dW2i,db2i] = BackwardBoundary(Xi,Hi,dZi,W2);
    dW1 = dW1f + dW1i;
    db1 = db1f + db1i;
    dW2 = dW2f + dW2i;
    db2 = db2f + db2i;
end

function [dW1,db1,dW2,db2] = ComputeMidpointGradients(MidDec,Mu,Sigma,W1,b1,W2,b2)
    dW1 = zeros(size(W1));
    db1 = zeros(size(b1));
    dW2 = zeros(size(W2));
    db2 = 0;
    if isempty(MidDec)
        return;
    end

    [P,~,H,Xn] = ForwardBoundary(MidDec,Mu,Sigma,W1,b1,W2,b2);
    Count = size(MidDec,1);
    dZ = (2*(P-0.5).*(P.*(1-P)))./max(Count,1);
    [dW1,db1,dW2,db2] = BackwardBoundary(Xn,H,dZ,W2);
end

function [dW1,db1,dW2,db2] = ComputeHardNegativeGradients(HardNegDec,Mu,Sigma,W1,b1,W2,b2)
    dW1 = zeros(size(W1));
    db1 = zeros(size(b1));
    dW2 = zeros(size(W2));
    db2 = 0;
    if isempty(HardNegDec)
        return;
    end

    [P,~,H,Xn] = ForwardBoundary(HardNegDec,Mu,Sigma,W1,b1,W2,b2);
    Active = P > 0.5;
    ActiveCount = sum(Active);
    if ActiveCount == 0
        return;
    end
    dZ = zeros(size(P));
    dZ(Active) = (2*(P(Active)-0.5).*(P(Active).*(1-P(Active))))./ActiveCount;
    [dW1,db1,dW2,db2] = BackwardBoundary(Xn,H,dZ,W2);
end

function [P,Z,H,Xn] = ForwardBoundary(X,Mu,Sigma,W1,b1,W2,b2)
    Xn = (double(X)-Mu)./Sigma;
    H  = tanh(Xn*W1 + repmat(b1,size(Xn,1),1));
    Z  = H*W2 + b2;
    P  = 1./(1+exp(-Z));
end

function [dW1,db1,dW2,db2] = BackwardBoundary(Xn,H,dZ,W2)
    dW2 = H'*dZ;
    db2 = sum(dZ);
    D1  = (dZ*W2').*(1-H.^2);
    dW1 = Xn'*D1;
    db1 = sum(D1,1);
end

function [Temp,A,B] = FitCalibrator(Model,CalDec,CalLabel,CalibratorType)
    Temp = 1;
    A = 1;
    B = 0;
    if isempty(CalDec) || isempty(CalLabel)
        return;
    end

    CalDec = double(CalDec);
    CalLabel = double(CalLabel(:) > 0);
    if numel(unique(CalLabel)) < 2
        return;
    end

    switch CalibratorType
        case 'raw'
            return;
        case 'temperature'
            Temp = FitTemperature(Model,CalDec,CalLabel);
        case 'sigmoid'
            [A,B] = FitSigmoid(Model,CalDec,CalLabel);
    end
end

function Temp = FitTemperature(Model,CalDec,CalLabel)
    Temp = 1;
    Z = RawBoundaryLogit(Model,CalDec);
    Theta = 0;
    for iter = 1 : 60
        TempNow = exp(Theta);
        P = 1./(1+exp(-Z./TempNow));
        GradT = mean((P-CalLabel).*(-Z./(TempNow^2)));
        Theta = Theta - 0.05*(GradT*TempNow);
        Theta = min(max(Theta,log(0.25)),log(8));
    end
    Temp = exp(Theta);
end

function [A,B] = FitSigmoid(Model,CalDec,CalLabel)
    Z = RawBoundaryLogit(Model,CalDec);
    A = 1;
    B = 0;
    for iter = 1 : 80
        Score = A*Z + B;
        P = 1./(1+exp(-Score));
        GradA = mean((P-CalLabel).*Z);
        GradB = mean(P-CalLabel);
        Step = 0.05/sqrt(iter);
        A = A - Step*GradA;
        B = B - Step*GradB;
        A = min(max(A,-10),10);
        B = min(max(B,-10),10);
    end
end

function Z = RawBoundaryLogit(Model,X)
    Xn = (double(X)-Model.Mu)./Model.Sigma;
    H  = tanh(Xn*Model.W1 + repmat(Model.b1,size(Xn,1),1));
    Z  = H*Model.W2 + Model.b2;
end

function tf = IsCompatibleWarmStart(Model,D,Hidden)
    if isempty(Model) || isfield(Model,'Members')
        tf = false;
        return;
    end
    tf = isfield(Model,'W1') && isfield(Model,'W2') ...
        && size(Model.W1,1) == D && size(Model.W1,2) == Hidden ...
        && size(Model.W2,1) == Hidden;
end

function PrevMember = ExtractPreviousMember(PrevModel,Index)
    PrevMember = [];
    if isempty(PrevModel)
        return;
    end
    if isfield(PrevModel,'Members') && numel(PrevModel.Members) >= Index
        if iscell(PrevModel.Members)
            PrevMember = PrevModel.Members{Index};
        else
            PrevMember = PrevModel.Members(Index);
        end
    elseif Index == 1 && ~isfield(PrevModel,'Members')
        PrevMember = PrevModel;
    end
end

function [BootX,BootY] = BootstrapTrainingSubset(X,Y)
    BootX = double(X);
    BootY = double(Y(:) > 0);
    N = size(BootX,1);
    if N < 4
        return;
    end

    for iter = 1 : 5
        Idx = randi(N,N,1);
        CandidateY = BootY(Idx);
        if numel(unique(CandidateY)) >= 2
            BootX = BootX(Idx,:);
            BootY = CandidateY;
            return;
        end
    end
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

function Value = GetOption(Options,Name,Default)
    if isstruct(Options) && isfield(Options,Name) && ~isempty(Options.(Name))
        Value = Options.(Name);
    else
        Value = Default;
    end
end

function Dec = NormalizeDecisionMatrix(Dec,D)
    if nargin < 2
        D = [];
    end
    if isempty(Dec)
        Dec = zeros(0,max(D,0));
        return;
    end
    Dec = double(Dec);
    if isvector(Dec)
        Dec = reshape(Dec,1,[]);
    end
end

function Type = ResolveCalibratorType(Type)
    if isnumeric(Type)
        switch round(Type)
            case 1
                Type = 'raw';
            case 2
                Type = 'temperature';
            case 3
                Type = 'sigmoid';
            otherwise
                Type = 'temperature';
        end
        return;
    end

    Type = lower(char(Type));
    switch Type
        case {'raw','none'}
            Type = 'raw';
        case {'temperature','temp'}
            Type = 'temperature';
        case {'sigmoid','platt'}
            Type = 'sigmoid';
        otherwise
            Type = 'temperature';
    end
end
