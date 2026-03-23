function [P,Stats] = PredictBoundaryMLP(Model,X)
% Predict calibrated boundary probabilities from a shallow committee.

    if nargin < 2 || isempty(X)
        P = zeros(0,1);
        Stats = InitPredictStats(0,0);
        return;
    end

    Count = size(X,1);
    if isempty(Model)
        P = 0.5*ones(Count,1);
        Stats = InitPredictStats(Count,1);
        Stats.memberProb = P;
        return;
    end

    if isfield(Model,'Members') && ~isempty(Model.Members)
        K = numel(Model.Members);
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
        RawProb = mean(MemberProb,2);
        RawLogit = mean(MemberLogit,2);
        Stats = InitPredictStats(Count,K);
        Stats.memberProb = MemberProb;
        Stats.std = std(MemberProb,0,2);
    else
        [RawProb,RawLogit] = PredictRawBoundaryMember(Model,X);
        Stats = InitPredictStats(Count,1);
        Stats.memberProb = RawProb;
    end

    P = ApplyBoundaryCalibration(Model,RawLogit,RawProb);
    P = min(max(P,1e-6),1-1e-6);
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

function P = ApplyBoundaryCalibration(Model,Z,RawProb)
    CalibratorType = ResolveCalibratorType(Model);
    switch CalibratorType
        case 'raw'
            P = RawProb;
        case 'temperature'
            Temp = 1;
            if isfield(Model,'Temp') && ~isempty(Model.Temp)
                Temp = max(Model.Temp,1e-6);
            end
            Score = Z./Temp;
            P = 1./(1+exp(-Score));
        case 'beta'
            A = 1;
            B = -1;
            C = 0;
            if isfield(Model,'BetaA') && ~isempty(Model.BetaA)
                A = Model.BetaA;
            end
            if isfield(Model,'BetaB') && ~isempty(Model.BetaB)
                B = Model.BetaB;
            end
            if isfield(Model,'BetaC') && ~isempty(Model.BetaC)
                C = Model.BetaC;
            end
            RawProb = min(max(RawProb,1e-6),1-1e-6);
            Score = A*log(RawProb) + B*log(1-RawProb) + C;
            P = 1./(1+exp(-Score));
        otherwise
            P = RawProb;
    end
end

function Type = ResolveCalibratorType(Model)
    Type = 'raw';
    if isfield(Model,'CalibratorType') && ~isempty(Model.CalibratorType)
        Type = lower(char(Model.CalibratorType));
    end
    switch Type
        case {'raw','temperature','beta'}
        otherwise
            Type = 'raw';
    end
end

function Stats = InitPredictStats(Count,EnsembleSize)
    Stats.std = zeros(Count,1);
    Stats.memberProb = zeros(Count,max(1,EnsembleSize));
    Stats.ensembleSize = EnsembleSize;
end
