function [Score,Meta] = ComputeBoundarySelectorUtility(VariantName,Detail,TrustContext,Budget,RuntimeOptions)
% Compute offline/runtime selector utilities on a shared candidate pool.

    if nargin < 2 || ~isstruct(Detail)
        Detail = struct();
    end
    if nargin < 3 || ~isstruct(TrustContext)
        TrustContext = struct();
    end
    if nargin < 4 || isempty(Budget)
        Budget = 0;
    end
    if nargin < 5 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end

    VariantName = CanonicalBoundarySelectorVariant(VariantName);
    Eligible = ResolveDetailVector(Detail,'eligible',true) ~= 0;
    ParetoValue = ResolveDetailVector(Detail,'paretoValue',NaN);
    BoundaryTrust = ResolveDetailVector(Detail,'boundaryTrust',NaN);
    Total = numel(Eligible);

    Score = -inf(Total,1);
    Meta = struct( ...
        'variant',VariantName, ...
        'shortlisted',false(Total,1), ...
        'alpha',NaN, ...
        'alphaSoft',NaN, ...
        'trustGate',ResolveTrustGate(Detail,TrustContext), ...
        'trustWeight',ResolveTrustWeight(Detail,TrustContext));

    Budget = round(double(Budget));
    if Total == 0 || Budget <= 0
        return;
    end

    EligibleIdx = find(Eligible & isfinite(ParetoValue));
    if isempty(EligibleIdx)
        return;
    end

    ShortlistFactor = ResolvePositiveRuntimeOption(RuntimeOptions,'FullShortlistFactor',3,1);
    ShortlistCount = min(numel(EligibleIdx),max(Budget,ceil(ShortlistFactor*Budget)));
    RankTable = [-ParetoValue(EligibleIdx),EligibleIdx];
    RankTable = sortrows(RankTable,[1 2]);
    ShortlistedIdx = RankTable(1:ShortlistCount,2);
    Meta.shortlisted(ShortlistedIdx) = true;

    switch VariantName
        case 'full_v2_current'
            LocalScore = ComputeCurrentLocalScore( ...
                ParetoValue(ShortlistedIdx),BoundaryTrust(ShortlistedIdx),Meta.trustWeight,RuntimeOptions);
        case 'full_no_trust'
            LocalScore = NormalizeDescendingRank(BoundaryTrust(ShortlistedIdx));
        case {'full_soft_trust','full_v2'}
            [LocalScore,Meta.alpha,Meta.alphaSoft] = ComputeSoftLocalScore( ...
                ParetoValue(ShortlistedIdx),BoundaryTrust(ShortlistedIdx), ...
                Meta.trustGate,Detail,TrustContext,RuntimeOptions);
        otherwise
            error('PRBCCMO:BoundarySelectorVariant', ...
                'Unsupported boundary selector variant ''%s''.',VariantName);
    end

    Score(ShortlistedIdx) = LocalScore;
end

function Name = CanonicalBoundarySelectorVariant(Name)
    Name = lower(strtrim(char(Name)));
    switch Name
        case {'full','fullv2','full_v2','full-v2'}
            Name = 'full_v2';
        case {'fullv2current','full_v2_current','full-v2-current','current'}
            Name = 'full_v2_current';
        case {'fullnotrust','full_no_trust','full-no-trust','notrust'}
            Name = 'full_no_trust';
        case {'fullsofttrust','full_soft_trust','full-soft-trust','softtrust'}
            Name = 'full_soft_trust';
        otherwise
            error('PRBCCMO:BoundarySelectorVariant', ...
                'Unsupported boundary selector variant ''%s''.',char(Name));
    end
end

function LocalScore = ComputeCurrentLocalScore(ParetoValue,BoundaryTrust,TrustWeight,RuntimeOptions)
    VNorm = NormalizeUnitInterval(ParetoValue);
    TrustTau = ResolveNonnegativeRuntimeOption(RuntimeOptions,'FullTrustTau',0.10);
    if TrustWeight < TrustTau
        LocalScore = VNorm;
        return;
    end
    Alpha = min(TrustWeight,ResolveUnitRuntimeOption(RuntimeOptions,'FullAlphaMax',0.50));
    BNorm = NormalizeUnitInterval(BoundaryTrust);
    LocalScore = (1-Alpha).*VNorm + Alpha.*BNorm;
end

function [LocalScore,Alpha,AlphaSoft] = ComputeSoftLocalScore( ...
    ParetoValue,BoundaryTrust,TrustGate,Detail,TrustContext,RuntimeOptions)

    VNorm = NormalizeDescendingRank(ParetoValue);
    BNorm = NormalizeDescendingRank(BoundaryTrust);
    TauE = ResolvePositiveRuntimeOption(RuntimeOptions,'TrustTauE',0.10,eps);
    TauN = ResolvePositiveRuntimeOption(RuntimeOptions,'TrustTauN',0.10,eps);
    ECE = ResolveTrustMetric(Detail,TrustContext,'ece',inf);
    CoreNearGap = ResolveTrustMetric(Detail,TrustContext,'coreNearGap',inf);

    if ~isfinite(ECE) || ~isfinite(CoreNearGap)
        AlphaSoft = 0;
    else
        AlphaSoft = 1/(1 + max(ECE,0)/TauE) * 1/(1 + max(CoreNearGap,0)/TauN);
    end
    if TrustGate
        AlphaCap = ResolveUnitRuntimeOption(RuntimeOptions,'FullAlphaHighMax',0.35);
    else
        AlphaCap = ResolveUnitRuntimeOption(RuntimeOptions,'FullAlphaLowMax',0.15);
    end
    Alpha = min(AlphaCap,max(AlphaSoft,0));
    LocalScore = (1-Alpha).*VNorm + Alpha.*BNorm;
end

function Value = ResolveDetailVector(Detail,Field,Default)
    if isfield(Detail,Field) && ~isempty(Detail.(Field))
        Value = Detail.(Field)(:);
        return;
    end
    if isscalar(Default)
        Count = 0;
        if isfield(Detail,'eligible') && ~isempty(Detail.eligible)
            Count = numel(Detail.eligible);
        elseif isfield(Detail,'paretoValue') && ~isempty(Detail.paretoValue)
            Count = numel(Detail.paretoValue);
        elseif isfield(Detail,'boundaryTrust') && ~isempty(Detail.boundaryTrust)
            Count = numel(Detail.boundaryTrust);
        end
        Value = repmat(Default,Count,1);
    else
        Value = Default(:);
    end
end

function Value = ResolveTrustMetric(Detail,TrustContext,Field,Default)
    Value = Default;
    switch lower(Field)
        case 'ece'
            DetailField = 'trustECE';
            ContextField = 'ece';
        case {'coreneargap','core_near_gap'}
            DetailField = 'trustCoreNearGap';
            ContextField = 'coreNearGap';
        otherwise
            error('PRBCCMO:BoundarySelectorTrustMetric', ...
                'Unsupported trust metric ''%s''.',Field);
    end
    if isfield(Detail,DetailField) && ~isempty(Detail.(DetailField))
        Data = Detail.(DetailField);
        Data = Data(isfinite(Data));
        if ~isempty(Data)
            Value = Data(1);
            return;
        end
    end
    if isfield(TrustContext,ContextField) && ~isempty(TrustContext.(ContextField))
        Value = TrustContext.(ContextField);
    end
end

function Value = ResolveTrustWeight(Detail,TrustContext)
    Value = 0;
    if isfield(Detail,'trustWeight') && ~isempty(Detail.trustWeight)
        Data = Detail.trustWeight(:);
        Data = Data(isfinite(Data));
        if ~isempty(Data)
            Value = min(max(Data(1),0),1);
            return;
        end
    end
    if isfield(TrustContext,'trustWeight') && ~isempty(TrustContext.trustWeight)
        Value = min(max(TrustContext.trustWeight,0),1);
    end
end

function Flag = ResolveTrustGate(Detail,TrustContext)
    Flag = false;
    if isfield(Detail,'trustGate') && ~isempty(Detail.trustGate)
        Flag = logical(Detail.trustGate(1));
        return;
    end
    if isfield(TrustContext,'trustGate') && ~isempty(TrustContext.trustGate)
        Flag = logical(TrustContext.trustGate);
    end
end

function Value = ResolvePositiveRuntimeOption(RuntimeOptions,Field,Default,FloorValue)
    Value = Default;
    if nargin < 4 || isempty(FloorValue)
        FloorValue = 0;
    end
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,Field) && ~isempty(RuntimeOptions.(Field))
        Value = RuntimeOptions.(Field);
    end
    Value = max(double(Value),FloorValue);
end

function Value = ResolveNonnegativeRuntimeOption(RuntimeOptions,Field,Default)
    Value = ResolvePositiveRuntimeOption(RuntimeOptions,Field,Default,0);
end

function Value = ResolveUnitRuntimeOption(RuntimeOptions,Field,Default)
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,Field) && ~isempty(RuntimeOptions.(Field))
        Value = RuntimeOptions.(Field);
    else
        Value = Default;
    end
    Value = min(max(double(Value),0),1);
end

function Value = NormalizeUnitInterval(Data)
    Data = Data(:);
    FiniteMask = isfinite(Data);
    Value = zeros(size(Data));
    if ~any(FiniteMask)
        return;
    end
    MinValue = min(Data(FiniteMask));
    MaxValue = max(Data(FiniteMask));
    if MaxValue <= MinValue + 1e-12
        Value(FiniteMask) = 1;
        return;
    end
    Value(FiniteMask) = (Data(FiniteMask) - MinValue) ./ (MaxValue - MinValue);
end

function Value = NormalizeDescendingRank(Data)
    Data = Data(:);
    FiniteMask = isfinite(Data);
    Value = zeros(size(Data));
    if ~any(FiniteMask)
        return;
    end

    FiniteData = Data(FiniteMask);
    Count = numel(FiniteData);
    if Count == 1
        Value(FiniteMask) = 1;
        return;
    end

    [SortedData,Order] = sort(FiniteData,'descend');
    Ranks = zeros(Count,1);
    StartIdx = 1;
    Tolerance = max(1,max(abs(SortedData))) * 1e-12;
    while StartIdx <= Count
        EndIdx = StartIdx;
        while EndIdx < Count && abs(SortedData(EndIdx+1) - SortedData(StartIdx)) <= Tolerance
            EndIdx = EndIdx + 1;
        end
        AvgRank = mean(StartIdx:EndIdx);
        Ranks(Order(StartIdx:EndIdx)) = AvgRank;
        StartIdx = EndIdx + 1;
    end
    Value(FiniteMask) = 1 - (Ranks - 1) ./ (Count - 1);
end
