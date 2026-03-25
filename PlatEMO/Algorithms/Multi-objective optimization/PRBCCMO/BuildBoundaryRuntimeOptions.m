function Options = BuildBoundaryRuntimeOptions( ...
    SelectionMode,LocalMode,TraceFlag,PairMode,PairTopK,PairDistanceWeight,GateMode,TraceProbLabel, ...
    PairKeepM,TrustTauE,TrustTauN,TrustMinCoreCount,TrustAdmissionStreak,TrustFallbackCap)
% Build runtime options for trust-aware Pareto-bridge querying.

    if nargin < 1 || isempty(SelectionMode)
        SelectionMode = 1;
    end
    if nargin < 2 || isempty(LocalMode)
        LocalMode = 1;
    end
    if nargin < 3 || isempty(TraceFlag)
        TraceFlag = false;
    end
    if nargin < 4 || isempty(PairMode)
        PairMode = 2;
    end
    if nargin < 5 || isempty(PairTopK)
        PairTopK = 3;
    end
    if nargin < 6 || isempty(PairDistanceWeight)
        PairDistanceWeight = 0.05;
    end
    if nargin < 7 || isempty(GateMode)
        GateMode = 1;
    end
    if nargin < 8 || isempty(TraceProbLabel)
        TraceProbLabel = false;
    end
    if nargin < 9 || isempty(PairKeepM)
        PairKeepM = 2;
    end
    if nargin < 10 || isempty(TrustTauE)
        TrustTauE = 0.10;
    end
    if nargin < 11 || isempty(TrustTauN)
        TrustTauN = 0.10;
    end
    if nargin < 12 || isempty(TrustMinCoreCount)
        TrustMinCoreCount = 20;
    end
    if nargin < 13 || isempty(TrustAdmissionStreak)
        TrustAdmissionStreak = 3;
    end
    if nargin < 14 || isempty(TrustFallbackCap)
        TrustFallbackCap = 0.25;
    end

    SelectionMode = max(1,min(4,round(SelectionMode)));
    LocalMode     = max(1,min(2,round(LocalMode)));
    PairMode      = max(1,min(2,round(PairMode)));
    GateMode      = max(1,min(2,round(GateMode)));

    Options = struct();
    Options.SelectionMode = SelectionMode;
    Options.LocalMode     = LocalMode;
    Options.PairMode      = PairMode;
    Options.GateMode      = GateMode;
    Options.TraceFlag     = logical(TraceFlag);
    Options.TraceProbLabel = logical(TraceProbLabel);
    Options.SelectionName = ResolveSelectionName(SelectionMode);
    Options.LocalName     = ResolveLocalName(LocalMode);
    Options.PairName      = ResolvePairName(PairMode);
    Options.GateName      = ResolveGateName(GateMode);
    Options.BridgeActivationGap = 0.01;
    Options.BridgePairTopK      = max(1,round(PairTopK));
    Options.BridgePairKeepM     = max(1,round(PairKeepM));
    Options.BridgePairDistanceWeight = max(PairDistanceWeight,0);
    Options.MigrationGap        = 0;
    Options.BridgeScanLambda    = [0.20,0.35,0.50,0.65,0.80];
    Options.TrustTauE           = max(TrustTauE,0);
    Options.TrustTauN           = max(TrustTauN,0);
    Options.TrustMinCoreCount   = max(1,round(TrustMinCoreCount));
    Options.TrustAdmissionStreak = max(1,round(TrustAdmissionStreak));
    Options.TrustFallbackCap    = min(max(TrustFallbackCap,0),1);
end

function Name = ResolveSelectionName(SelectionMode)
    switch SelectionMode
        case 1
            Name = 'trusted_query';
        case 2
            Name = 'uncertain_only';
        case 3
            Name = 'random_bridge';
        case 4
            Name = 'highprob_boundary';
        otherwise
            Name = 'trusted_query';
    end
end

function Name = ResolvePairName(PairMode)
    switch PairMode
        case 1
            Name = 'current_pair';
        case 2
            Name = 'topk_pair';
        otherwise
            Name = 'topk_pair';
    end
end

function Name = ResolveGateName(GateMode)
    switch GateMode
        case 1
            Name = 'two_stage';
        case 2
            Name = 'strict_only';
        otherwise
            Name = 'two_stage';
    end
end

function Name = ResolveLocalName(LocalMode)
    switch LocalMode
        case 1
            Name = 'label_aware';
        case 2
            Name = 'isotropic';
        otherwise
            Name = 'label_aware';
    end
end
