function Options = BuildBoundaryRuntimeOptions(varargin)
% Build fixed PRBCCMO-Lite runtime options with TopKPair and midpoint probes.

    Options = struct();
    Options.SelectionMode = 1;
    Options.LocalMode     = 1;
    Options.GateMode      = 1;
    Options.TraceFlag     = false;
    Options.TraceProbLabel = false;
    Options.SelectionName = 'pareto_only';
    Options.LocalName     = 'label_aware';
    Options.BridgeName    = 'topk_pair';
    Options.GateName      = 'two_stage';
    Options.BridgeActivationGap = 0.01;
    Options.BridgeScanLambda    = [0.25,0.50,0.75];
    Options.BridgeTopK          = 5;
    Options.FullShortlistFactor = 3;
    Options.FullTrustTau        = 0.10;
    Options.FullAlphaMax        = 0.50;
    Options.FullAlphaLowMax     = 0.15;
    Options.FullAlphaHighMax    = 0.35;
    Options.MigrationGap        = 0;
    Options.TrustTauE           = 0.10;
    Options.TrustTauN           = 0.10;
    Options.TrustMinCoreCount   = 20;
    Options.TrustAdmissionStreak = 3;
    Options.TrustFallbackCap    = 0.25;
    Options.DisableTrust        = false;
    Options.DisableBridgeScan   = true;
    Options.CalibratorCandidates = {};
    Options.Calibrator         = [];

    if nargin == 0
        return;
    end

    Override = struct();
    if all(cellfun(@isstruct,varargin))
        for i = 1 : nargin
            Fields = fieldnames(varargin{i});
            for j = 1 : numel(Fields)
                Field = Fields{j};
                Value = varargin{i}.(Field);
                if isempty(Value)
                    continue;
                end
                Override.(Field) = Value;
            end
        end
    else
        if mod(nargin,2) ~= 0
            error('PRBCCMO:RuntimeOptionsInput', ...
                'Runtime overrides must be a struct or name-value pairs.');
        end
        for i = 1 : 2 : nargin
            Name = varargin{i};
            if ~(ischar(Name) || (isstring(Name) && isscalar(Name)))
                error('PRBCCMO:RuntimeOptionsInput', ...
                    'Runtime override names must be character vectors or scalars.');
            end
            Override.(char(Name)) = varargin{i+1};
        end
    end
    Override = FilterRuntimeOverrides(Override,fieldnames(Options));

    Fields = fieldnames(Override);
    for i = 1 : numel(Fields)
        Field = Fields{i};
        Value = Override.(Field);
        if isempty(Value)
            continue;
        end
        Options.(Field) = Value;
    end
end

function Override = FilterRuntimeOverrides(Override,AllowedFields)
    Fields = fieldnames(Override);
    for i = numel(Fields) : -1 : 1
        Field = Fields{i};
        if ~any(strcmp(Field,AllowedFields))
            Override = rmfield(Override,Field);
        end
    end
end
