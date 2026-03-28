function Options = BuildBoundaryRuntimeOptions(varargin)
% Build PRBCCMO-BoundaryCore runtime options.

    Options = struct();
    Options.TraceFlag      = false;
    Options.TraceProbLabel = false;
    Options.BridgeActivationGap = 0.01;
    Options.BridgeScanLambda    = [0.25,0.50,0.75];
    Options.BridgeRefineStep    = 0.125;
    Options.BridgeTopK          = 5;
    Options.SelectorMode        = 'pareto_then_boundary';
    Options.BoundaryShortlistFactor = 3;
    Options.BoundaryLocalDelta  = 0.10;
    Options.BracketTightGap     = 0.03;
    Options.TrustTauE           = 0.10;
    Options.TrustTauN           = 0.10;
    Options.TrustMinCoreCount   = 20;
    Options.ForwardAlpha        = 0.10;
    Options.ForcePlacementRefine = false;
    Options.DisableFeasibleForward = false;
    Options.DisableInfeasibleShrink = false;
    Options.DisableTrust        = false;

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
