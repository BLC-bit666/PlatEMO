function Options = BuildBoundaryRuntimeOptions(SelectionMode,LocalMode,TraceFlag)
% Build runtime options for Section B experiment variants.

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
    Options.SelectionName = ResolveSelectionName(SelectionMode);
    Options.LocalName     = ResolveLocalName(LocalMode);
    Options.ArchiveSelectionMode = SelectionMode;
    if SelectionMode == 3
        Options.ArchiveSelectionMode = 1;
    end
end

function Name = ResolveSelectionName(SelectionMode)
    switch SelectionMode
        case 1
            Name = 'full';
        case 2
            Name = 'uncertain_only';
        case 3
            Name = 'random_boundary';
        otherwise
            Name = 'full';
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
