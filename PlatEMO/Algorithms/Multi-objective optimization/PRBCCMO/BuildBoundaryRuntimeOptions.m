function Options = BuildBoundaryRuntimeOptions(SelectionMode,LocalMode,TraceFlag)
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

    SelectionMode = max(1,min(3,round(SelectionMode)));
    LocalMode     = max(1,min(2,round(LocalMode)));

    Options = struct();
    Options.SelectionMode = SelectionMode;
    Options.LocalMode     = LocalMode;
    Options.TraceFlag     = logical(TraceFlag);
    Options.TraceProbLabel = false;
    Options.SelectionName = ResolveSelectionName(SelectionMode);
    Options.LocalName     = ResolveLocalName(LocalMode);
    Options.BridgeActivationGap = 0.01;
    Options.MigrationGap        = 0;
    Options.BridgeScanLambda    = [0.20,0.35,0.50,0.65,0.80];
end

function Name = ResolveSelectionName(SelectionMode)
    switch SelectionMode
        case 1
            Name = 'trusted_query';
        case 2
            Name = 'uncertain_only';
        case 3
            Name = 'random_bridge';
        otherwise
            Name = 'trusted_query';
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
