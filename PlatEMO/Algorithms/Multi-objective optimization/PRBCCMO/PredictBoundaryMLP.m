function [P,Stats] = PredictBoundaryMLP(Model,X)
% Predict boundary probabilities from a single MLP or a small calibrated committee.

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
        for k = 1 : K
            if iscell(Model.Members)
                Member = Model.Members{k};
            else
                Member = Model.Members(k);
            end
            MemberProb(:,k) = PredictSingleBoundaryMember(Member,X);
        end
        P = mean(MemberProb,2);
        Stats = InitPredictStats(Count,K);
        Stats.memberProb = MemberProb;
        Stats.std = std(MemberProb,0,2);
    else
        P = PredictSingleBoundaryMember(Model,X);
        Stats = InitPredictStats(Count,1);
        Stats.memberProb = P;
    end

    P = min(max(P,1e-6),1-1e-6);
end

function P = PredictSingleBoundaryMember(Model,X)
    if isempty(Model) || ~isfield(Model,'Mu')
        P = 0.5*ones(size(X,1),1);
        return;
    end

    Xn = (double(X)-Model.Mu)./Model.Sigma;
    H  = tanh(Xn*Model.W1 + repmat(Model.b1,size(Xn,1),1));
    Z  = H*Model.W2 + Model.b2;
    P  = ApplyBoundaryCalibration(Model,Z);
    P  = min(max(P,1e-6),1-1e-6);
end

function P = ApplyBoundaryCalibration(Model,Z)
    CalibratorType = ResolveCalibratorType(Model);
    switch CalibratorType
        case 'raw'
            Score = Z;
        case 'temperature'
            Temp = 1;
            if isfield(Model,'Temp') && ~isempty(Model.Temp)
                Temp = max(Model.Temp,1e-6);
            end
            Score = Z./Temp;
        case 'sigmoid'
            A = 1;
            B = 0;
            if isfield(Model,'SigmoidA') && ~isempty(Model.SigmoidA)
                A = Model.SigmoidA;
            end
            if isfield(Model,'SigmoidB') && ~isempty(Model.SigmoidB)
                B = Model.SigmoidB;
            end
            Score = A*Z + B;
        otherwise
            Score = Z;
    end
    P = 1./(1+exp(-Score));
end

function Type = ResolveCalibratorType(Model)
    Type = 'temperature';
    if isfield(Model,'CalibratorType') && ~isempty(Model.CalibratorType)
        Type = lower(char(Model.CalibratorType));
    end
    switch Type
        case {'raw','temperature','sigmoid'}
        otherwise
            Type = 'temperature';
    end
end

function Stats = InitPredictStats(Count,EnsembleSize)
    Stats.std = zeros(Count,1);
    Stats.memberProb = zeros(Count,max(1,EnsembleSize));
    Stats.ensembleSize = EnsembleSize;
end
