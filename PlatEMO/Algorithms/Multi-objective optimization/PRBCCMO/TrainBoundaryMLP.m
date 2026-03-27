function Model = TrainBoundaryMLP(X,Y,Hidden,Epoch,LR,PrevModel,CalDec,CalLabel,Options)
% Train a shallow committee with boundary-semantics losses on held-out data.

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
    CalibratorCandidates = ResolveCalibratorCandidates(Options);

    if EnsembleSize <= 1
        Model = TrainSingleBoundaryModel( ...
            X,Y,Hidden,Epoch,LR,PrevModel,CalDec,CalLabel,Options);
        Model = FinalizeBoundaryModel(Model,CalDec,CalLabel,CalibratorCandidates,Options);
        return;
    end

    Members = cell(1,EnsembleSize);
    for k = 1 : EnsembleSize
        PrevMember = ExtractPreviousMember(PrevModel,k);
        [BootX,BootY] = BootstrapTrainingSubset(X,Y,Options);
        MemberOptions = Options;
        MemberOptions.EnsembleSize = 1;
        MemberOptions.MemberIndex  = k;
        Members{k} = TrainSingleBoundaryModel( ...
            BootX,BootY,Hidden,Epoch,LR,PrevMember,CalDec,CalLabel,MemberOptions);
    end

    Model.Members = Members;
    Model.EnsembleSize = EnsembleSize;
    Model.DisagreementWeight = max(GetOption(Options,'DisagreementWeight',1),0);
    Model = FinalizeBoundaryModel(Model,CalDec,CalLabel,CalibratorCandidates,Options);
end

function Model = TrainSingleBoundaryModel(X,Y,Hidden,Epoch,LR,PrevModel,CalDec,CalLabel,Options)
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
    PairMargin = max(0,GetOption(Options,'PairMargin',0.05));
    LambdaPair = max(0,GetOption(Options,'LambdaPair',1));
    LambdaMid  = max(0,GetOption(Options,'LambdaMid',1));
    LambdaBrier = max(0,GetOption(Options,'LambdaBrier',0.5));

    [~,D] = size(X);
    LambdaReg = 1e-4;

    XTrain = X;
    YTrain = Y;
    if (isempty(CalDec) || isempty(CalLabel)) && size(X,1) >= 10
        [TrainIdx,~] = SplitCalibrationData(Y);
        XTrain = X(TrainIdx,:);
        YTrain = Y(TrainIdx);
        if numel(unique(YTrain)) < 2
            XTrain = X;
            YTrain = Y;
        end
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
    SampleBoost = BuildBracketSampleBoost(XTrain,PairFeasible,PairInfeasible, ...
        max(1,GetOption(Options,'BracketOversampleFactor',3)));
    [Weight,NormWeight] = BuildClassWeights(YTrain,SampleBoost);
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

        if LambdaBrier > 0
            DeltaBrier = Weight.*(2*(P-YTrain).*(P.*(1-P)))./NormWeight;
            dW2 = dW2 + LambdaBrier*(H'*DeltaBrier);
            db2 = db2 + LambdaBrier*sum(DeltaBrier);
            D1B = (DeltaBrier*W2').*(1-H.^2);
            dW1 = dW1 + LambdaBrier*(Xn'*D1B);
            db1 = db1 + LambdaBrier*sum(D1B,1);
        end

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
    Model.DisagreementWeight = max(GetOption(Options,'DisagreementWeight',1),0);
    Model.CalibratorType = 'raw';
    Model.Temp           = 1;
    Model.BetaA          = 1;
    Model.BetaB          = -1;
    Model.BetaC          = 0;
end

function Model = FinalizeBoundaryModel(Model,CalDec,CalLabel,Candidates,Options)
    if isempty(Model)
        return;
    end
    Model.DisagreementWeight = max(GetOption(Options,'DisagreementWeight',1),0);
    Model.Temp  = 1;
    Model.BetaA = 1;
    Model.BetaB = -1;
    Model.BetaC = 0;
    Model.CalibratorType = 'raw';
    Model.CalibrationObjective = inf;

    if isempty(CalDec) || isempty(CalLabel)
        return;
    end
    CalLabel = double(CalLabel(:) > 0);
    if numel(unique(CalLabel)) < 2
        return;
    end

    [RawProb,RawLogit] = PredictRawBoundaryModel(Model,CalDec);
    [Type,Temp,A,B,C,Objective] = SelectCalibrator(RawProb,RawLogit,CalLabel,Candidates);
    Model.CalibratorType = Type;
    Model.Temp  = Temp;
    Model.BetaA = A;
    Model.BetaB = B;
    Model.BetaC = C;
    Model.CalibrationObjective = Objective;
end

function [Type,Temp,A,B,C,Objective] = SelectCalibrator(RawProb,RawLogit,CalLabel,Candidates)
    Type = 'raw';
    Temp = 1;
    A = 1;
    B = -1;
    C = 0;
    Objective = inf;
    if isempty(Candidates)
        Candidates = {'raw'};
    end

    for i = 1 : numel(Candidates)
        Candidate = ResolveCalibratorType(Candidates{i});
        TempNow = 1;
        ANow = 1;
        BNow = -1;
        CNow = 0;
        switch Candidate
            case 'raw'
                Prob = RawProb;
            case 'temperature'
                TempNow = FitTemperatureFromLogit(RawLogit,CalLabel);
                Prob = 1./(1+exp(-RawLogit./TempNow));
            case 'beta'
                [ANow,BNow,CNow] = FitBetaFromProb(RawProb,CalLabel);
                Prob = ApplyBetaCalibration(RawProb,ANow,BNow,CNow);
            otherwise
                continue;
        end
        Metric = SummarizeCalibrationProbabilities(Prob,CalLabel);
        Score = Metric.brier + Metric.ece + 2*Metric.nearGap;
        if Score < Objective
            Type = Candidate;
            Temp = TempNow;
            A = ANow;
            B = BNow;
            C = CNow;
            Objective = Score;
        end
    end
end

function [RawProb,RawLogit] = PredictRawBoundaryModel(Model,X)
    if isempty(X) || isempty(Model)
        RawProb = zeros(0,1);
        RawLogit = zeros(0,1);
        return;
    end
    if isfield(Model,'Members') && ~isempty(Model.Members)
        K = numel(Model.Members);
        Count = size(X,1);
        MemberProb = zeros(Count,K);
        MemberLogit = zeros(Count,K);
        for k = 1 : K
            if iscell(Model.Members)
                Member = Model.Members{k};
            else
                Member = Model.Members(k);
            end
            [MemberProb(:,k),MemberLogit(:,k)] = PredictRawBoundaryMember(Member,X);
        end
        RawProb  = mean(MemberProb,2);
        RawLogit = mean(MemberLogit,2);
    else
        [RawProb,RawLogit] = PredictRawBoundaryMember(Model,X);
    end
    RawProb = min(max(RawProb,1e-6),1-1e-6);
end

function [P,Z] = PredictRawBoundaryMember(Model,X)
    if isempty(Model) || ~isfield(Model,'Mu')
        P = 0.5*ones(size(X,1),1);
        Z = zeros(size(X,1),1);
        return;
    end

    Xn = (double(X)-Model.Mu)./Model.Sigma;
    H  = tanh(Xn*Model.W1 + repmat(Model.b1,size(Xn,1),1));
    Z  = H*Model.W2 + Model.b2;
    P  = 1./(1+exp(-Z));
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
    [Pf,~,Hf,Xf] = ForwardBoundary(FeasibleDec,Mu,Sigma,W1,b1,W2,b2);
    [Pi,~,Hi,Xi] = ForwardBoundary(InfeasibleDec,Mu,Sigma,W1,b1,W2,b2);
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

function Temp = FitTemperatureFromLogit(Z,CalLabel)
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

function [A,B,C] = FitBetaFromProb(RawProb,CalLabel)
    RawProb = min(max(RawProb,1e-6),1-1e-6);
    LogP = log(RawProb);
    Log1P = log(1-RawProb);
    A = 1;
    B = -1;
    C = 0;
    for iter = 1 : 100
        Score = A*LogP + B*Log1P + C;
        P = 1./(1+exp(-Score));
        GradA = mean((P-CalLabel).*LogP);
        GradB = mean((P-CalLabel).*Log1P);
        GradC = mean(P-CalLabel);
        Step = 0.05/sqrt(iter);
        A = A - Step*GradA;
        B = B - Step*GradB;
        C = C - Step*GradC;
        A = min(max(A,-10),10);
        B = min(max(B,-10),10);
        C = min(max(C,-10),10);
    end
end

function P = ApplyBetaCalibration(RawProb,A,B,C)
    RawProb = min(max(RawProb,1e-6),1-1e-6);
    Score = A*log(RawProb) + B*log(1-RawProb) + C;
    P = 1./(1+exp(-Score));
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

function [BootX,BootY] = BootstrapTrainingSubset(X,Y,Options)
    BootX = double(X);
    BootY = double(Y(:) > 0);
    N = size(BootX,1);
    if N < 4
        return;
    end

    PairFeasible = NormalizeDecisionMatrix(GetOption(Options,'PairFeasibleDec',[]),size(BootX,2));
    PairInfeasible = NormalizeDecisionMatrix(GetOption(Options,'PairInfeasibleDec',[]),size(BootX,2));
    SampleBoost = BuildBracketSampleBoost(BootX,PairFeasible,PairInfeasible, ...
        max(1,GetOption(Options,'BracketOversampleFactor',3)));
    SampleProb = SampleBoost./max(sum(SampleBoost),1);
    for iter = 1 : 5
        Idx = SampleWeightedRows(SampleProb,N);
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

function [Weight,NormWeight] = BuildClassWeights(Y,ExtraWeight)
    N = numel(Y);
    Pos = sum(Y==1);
    Neg = N - Pos;
    WPos = N/(2*max(1,Pos));
    WNeg = N/(2*max(1,Neg));
    if nargin < 2 || isempty(ExtraWeight)
        ExtraWeight = ones(N,1);
    else
        ExtraWeight = max(double(ExtraWeight(:)),1);
        ExtraWeight = ExtraWeight./max(mean(ExtraWeight),1e-12);
    end
    Weight = (WNeg + (WPos-WNeg).*Y).*ExtraWeight;
    NormWeight = max(sum(Weight),1);
end

function Boost = BuildBracketSampleBoost(X,PairFeasible,PairInfeasible,Factor)
    Boost = ones(size(X,1),1);
    if nargin < 4 || Factor <= 1 || isempty(X)
        return;
    end

    TightDec = [PairFeasible;PairInfeasible];
    if isempty(TightDec)
        return;
    end
    TightDec = unique(double(TightDec),'rows','stable');
    Match = ismember(double(X),TightDec,'rows');
    Boost(Match) = Factor;
end

function Idx = SampleWeightedRows(Prob,Count)
    if isempty(Prob)
        Idx = zeros(0,1);
        return;
    end
    Prob = double(Prob(:));
    Prob(~isfinite(Prob) | Prob < 0) = 0;
    if sum(Prob) <= 0
        Prob = ones(size(Prob))/numel(Prob);
    else
        Prob = Prob/sum(Prob);
    end
    CDF = cumsum(Prob);
    CDF(end) = 1;
    R = rand(Count,1);
    Idx = zeros(Count,1);
    for i = 1 : Count
        Idx(i) = find(CDF >= R(i),1,'first');
    end
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

function Candidates = ResolveCalibratorCandidates(Options)
    if isstruct(Options) && isfield(Options,'CalibratorCandidates') && ~isempty(Options.CalibratorCandidates)
        Source = Options.CalibratorCandidates;
    elseif isstruct(Options) && isfield(Options,'Calibrator') && ~isempty(Options.Calibrator)
        Source = Options.Calibrator;
    else
        Source = {'raw','beta'};
    end

    if ~iscell(Source)
        Source = {Source};
    end
    Candidates = cell(1,0);
    for i = 1 : numel(Source)
        Type = ResolveCalibratorType(Source{i});
        if isempty(Type)
            continue;
        end
        if ~any(strcmp(Candidates,Type))
            Candidates{end+1} = Type; %#ok<AGROW>
        end
    end
    if isempty(Candidates)
        Candidates = {'raw'};
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
                Type = 'beta';
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
        case {'beta','sigmoid','platt'}
            Type = 'beta';
        otherwise
            Type = '';
    end
end
