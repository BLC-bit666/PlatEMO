function varargout = RegionQueryTracker_RC(action,varargin)
%REGIONQUERYTRACKER_RC Track query-ref conversion and direct GAN BMem entry.
%   This helper is a pure state machine. Ref-level conversion is deliberately
%   independent from decision-level attribution because another search
%   operator may populate an exposed reference vector.

    action = lower(strtrim(string(action)));
    switch action
        case "init"
            varargout{1} = initTracker(varargin{:});
        case "register"
            [varargout{1:nargout}] = registerQueryEvent(varargin{:});
        case "updatebmem"
            [varargout{1:nargout}] = updateBoundaryMemory(varargin{:});
        case "summary"
            varargout{1} = trackerSummary(varargin{:});
        case "export"
            varargout{1} = exportTracker(varargin{:});
        otherwise
            error('CBSRegionQueryTracker:BadAction', ...
                'Unsupported RegionQueryTracker_RC action: %s.',action);
    end
end

function S = initTracker(D)
    D = double(D);
    if ~isscalar(D) || ~isfinite(D) || D ~= fix(D) || D < 1
        error('CBSRegionQueryTracker:BadDimension', ...
            'Decision dimension must be a positive finite integer.');
    end
    S = struct( ...
        'D',D, ...
        'next_sample_id',1, ...
        'ref_exposure',emptyRefExposureRows(), ...
        'samples',emptySampleRows(D), ...
        'ref_conversions',emptyRefConversionRows(), ...
        'direct_entries',emptyDirectEntryRows(D), ...
        'seen_bmem_decisions',zeros(0,D));
end

function [S,Event] = registerQueryEvent(S,gen,FE,SampleRefs,Groups, ...
        GDec,InP1,InP2)
    S = validateTracker(S);
    [gen,FE] = validateEventPosition(gen,FE);
    SampleRefs = double(SampleRefs(:));
    Groups = double(Groups(:));
    GDec = double(GDec);
    n = size(GDec,1);
    if numel(SampleRefs) ~= n || numel(Groups) ~= n || ...
            numel(InP1) ~= n || numel(InP2) ~= n
        error('CBSRegionQueryTracker:BadRegisterRows', ...
            'Refs, groups, decisions, and survival masks must have equal rows.');
    end
    if size(GDec,2) ~= S.D
        error('CBSRegionQueryTracker:BadDecisionColumns', ...
            'Generated decisions must have tracker decision dimension columns.');
    end
    if any(~isfinite(SampleRefs) | SampleRefs ~= fix(SampleRefs) | ...
            SampleRefs < 1)
        error('CBSRegionQueryTracker:BadRefCode', ...
            'Query refs must be positive finite integers.');
    end
    if any(~isfinite(Groups) | Groups ~= fix(Groups) | ...
            Groups < 1 | Groups > 3)
        error('CBSRegionQueryTracker:BadGroupCode', ...
            'Query group codes must be finite integers in 1:3.');
    end
    InP1 = validateMask(InP1,n);
    InP2 = validateMask(InP2,n);

    exposed = Groups == 2 | Groups == 3;
    if any(exposed)
        Keys = unique([SampleRefs(exposed),Groups(exposed)],'rows','stable');
    else
        Keys = zeros(0,2);
    end
    for i = 1 : size(Keys,1)
        ref = Keys(i,1);
        group = Keys(i,2);
        generatedCount = sum(SampleRefs == ref & Groups == group);
        idx = findRefExposure(S.ref_exposure,ref,group);
        if isempty(idx)
            Row = refExposureTemplate();
            Row.ref = ref;
            Row.group = group;
            Row.first_query_gen = gen;
            Row.first_query_fe = FE;
            Row.last_query_gen = gen;
            Row.last_query_fe = FE;
            Row.exposure_count = 1;
            Row.generated_count = generatedCount;
            S.ref_exposure(end+1,1) = Row;
        else
            S.ref_exposure(idx).last_query_gen = gen;
            S.ref_exposure(idx).last_query_fe = FE;
            S.ref_exposure(idx).exposure_count = ...
                S.ref_exposure(idx).exposure_count + 1;
            S.ref_exposure(idx).generated_count = ...
                S.ref_exposure(idx).generated_count + generatedCount;
        end
    end

    eligible = InP1 | InP2;
    eligibleRows = find(eligible);
    sampleIDs = zeros(numel(eligibleRows),1);
    for i = 1 : numel(eligibleRows)
        r = eligibleRows(i);
        Row = sampleTemplate(S.D);
        Row.sample_id = S.next_sample_id;
        Row.query_gen = gen;
        Row.query_fe = FE;
        Row.query_ref = SampleRefs(r);
        Row.query_group = Groups(r);
        Row.survive_P1 = double(InP1(r));
        Row.survive_P2 = double(InP2(r));
        Row.decision = GDec(r,:);
        S.samples(end+1,1) = Row;
        sampleIDs(i) = S.next_sample_id;
        S.next_sample_id = S.next_sample_id + 1;
    end
    Event = struct( ...
        'ref_exposure_count',size(Keys,1), ...
        'ref_exposure_generated_count',sum(exposed), ...
        'eligible_sample_count',numel(eligibleRows), ...
        'sample_id',sampleIDs);
end

function [S,Update] = updateBoundaryMemory(S,gen,FE,BMem)
    S = validateTracker(S);
    [gen,FE] = validateEventPosition(gen,FE);
    [BRefs,BDec] = boundaryMemoryRows(BMem,S.D);
    Update = emptyTrackerUpdate();

    for i = 1 : numel(S.ref_exposure)
        if S.ref_exposure(i).converted ~= 0 || ...
                ~any(BRefs == S.ref_exposure(i).ref)
            continue;
        end
        S.ref_exposure(i).converted = 1;
        S.ref_exposure(i).first_populated_gen = gen;
        S.ref_exposure(i).first_populated_fe = FE;
        S.ref_exposure(i).lag_gen = gen - ...
            S.ref_exposure(i).first_query_gen;
        S.ref_exposure(i).lag_fe = FE - ...
            S.ref_exposure(i).first_query_fe;
        Row = refConversionFromExposure(S.ref_exposure(i));
        S.ref_conversions(end+1,1) = Row;
        Update.ref_conversion_ref(end+1,1) = Row.ref;
        Update.ref_conversion_group(end+1,1) = Row.group;
        Update.ref_conversion_lag_gen(end+1,1) = Row.lag_gen;
        Update.ref_conversion_lag_fe(end+1,1) = Row.lag_fe;
    end
    Update.ref_conversion_count = numel(Update.ref_conversion_ref);

    [newBMemRows,S.seen_bmem_decisions] = newMultisetRows( ...
        BDec,S.seen_bmem_decisions);
    for i = 1 : size(newBMemRows,1)
        sampleIdx = firstUnenteredSample(S.samples,newBMemRows(i,:));
        if isempty(sampleIdx)
            continue;
        end
        S.samples(sampleIdx).entered = 1;
        S.samples(sampleIdx).entry_gen = gen;
        S.samples(sampleIdx).entry_fe = FE;
        S.samples(sampleIdx).lag_gen = gen - S.samples(sampleIdx).query_gen;
        S.samples(sampleIdx).lag_fe = FE - S.samples(sampleIdx).query_fe;
        Row = directEntryFromSample(S.samples(sampleIdx),S.D);
        S.direct_entries(end+1,1) = Row;
        Update.direct_bmem_entry_sample_id(end+1,1) = Row.sample_id;
        Update.direct_bmem_entry_ref(end+1,1) = Row.query_ref;
        Update.direct_bmem_entry_group(end+1,1) = Row.query_group;
        Update.direct_bmem_entry_lag_gen(end+1,1) = Row.lag_gen;
        Update.direct_bmem_entry_lag_fe(end+1,1) = Row.lag_fe;
    end
    Update.direct_bmem_entry_count = ...
        numel(Update.direct_bmem_entry_sample_id);
end

function Summary = trackerSummary(S)
    S = validateTracker(S);
    exposureCount = numel(S.ref_exposure);
    convertedCount = numel(S.ref_conversions);
    eligibleCount = numel(S.samples);
    directCount = numel(S.direct_entries);
    Summary = struct( ...
        'ref_exposure_count',exposureCount, ...
        'ref_frontier_exposure_count',sumExposureGroup(S.ref_exposure,2), ...
        'ref_remote_exposure_count',sumExposureGroup(S.ref_exposure,3), ...
        'ref_exposure_event_count',sumExposureField( ...
            S.ref_exposure,'exposure_count'), ...
        'ref_generated_count',sumExposureField( ...
            S.ref_exposure,'generated_count'), ...
        'ref_converted_count',convertedCount, ...
        'ref_conversion_rate',safeRate(convertedCount,exposureCount), ...
        'ref_conversion_lag_gen50',medianStructField( ...
            S.ref_conversions,'lag_gen'), ...
        'ref_conversion_lag_fe50',medianStructField( ...
            S.ref_conversions,'lag_fe'), ...
        'direct_eligible_sample_count',eligibleCount, ...
        'direct_bmem_entry_count',directCount, ...
        'direct_bmem_entry_rate',safeRate(directCount,eligibleCount), ...
        'direct_bmem_entry_lag_gen50',medianStructField( ...
            S.direct_entries,'lag_gen'), ...
        'direct_bmem_entry_lag_fe50',medianStructField( ...
            S.direct_entries,'lag_fe'));
end

function X = exportTracker(S)
    S = validateTracker(S);
    X = struct( ...
        'ref_exposure',S.ref_exposure, ...
        'ref_conversions',S.ref_conversions, ...
        'samples',S.samples, ...
        'direct_entries',S.direct_entries, ...
        'summary',trackerSummary(S));
end

function S = validateTracker(S)
    required = {'D','next_sample_id','ref_exposure','samples', ...
        'ref_conversions','direct_entries','seen_bmem_decisions'};
    if ~isstruct(S) || ~all(isfield(S,required))
        error('CBSRegionQueryTracker:BadState', ...
            'Tracker state is missing required fields.');
    end
end

function [gen,FE] = validateEventPosition(gen,FE)
    gen = double(gen);
    FE = double(FE);
    if ~isscalar(gen) || ~isfinite(gen) || gen ~= fix(gen) || gen < 0
        error('CBSRegionQueryTracker:BadGeneration', ...
            'Generation must be a nonnegative finite integer.');
    end
    if ~isscalar(FE) || ~isfinite(FE) || FE < 0
        error('CBSRegionQueryTracker:BadFE', ...
            'FE must be a nonnegative finite scalar.');
    end
end

function Mask = validateMask(Mask,n)
    if numel(Mask) ~= n
        error('CBSRegionQueryTracker:BadRegisterRows', ...
            'Survival-mask rows must match generated decisions.');
    end
    Values = double(Mask(:));
    if any(~isfinite(Values) | (Values ~= 0 & Values ~= 1))
        error('CBSRegionQueryTracker:BadSurvivalMask', ...
            'Survival masks must contain only finite zero/one values.');
    end
    Mask = logical(Values);
end

function idx = findRefExposure(Rows,ref,group)
    idx = [];
    for i = 1 : numel(Rows)
        if Rows(i).ref == ref && Rows(i).group == group
            idx = i;
            return;
        end
    end
end

function [Refs,Dec] = boundaryMemoryRows(BMem,D)
    Refs = zeros(0,1);
    Dec = zeros(0,D);
    if isempty(BMem)
        return;
    end
    if ~isstruct(BMem)
        error('CBSRegionQueryTracker:BadBMem', ...
            'BMem must be a struct or empty.');
    end
    if isfield(BMem,'ref') && ~isempty(BMem.ref)
        Refs = double(BMem.ref(:));
        if any(~isfinite(Refs) | Refs ~= fix(Refs) | Refs < 1)
            error('CBSRegionQueryTracker:BadBMemRef', ...
                'BMem refs must be positive finite integers.');
        end
    end
    if isfield(BMem,'x_b') && ~isempty(BMem.x_b)
        Dec = double(BMem.x_b);
        if size(Dec,2) ~= D
            error('CBSRegionQueryTracker:BadBMemDecisionColumns', ...
                'BMem decisions must have tracker decision dimension columns.');
        end
    end
end

function [NewRows,Seen] = newMultisetRows(Rows,Seen)
    NewRows = zeros(0,size(Seen,2));
    if isempty(Rows)
        return;
    end
    used = false(size(Seen,1),1);
    tol = 1e-12;
    for i = 1 : size(Rows,1)
        match = [];
        if ~isempty(Seen)
            dist = max(abs(Seen - Rows(i,:)),[],2);
            match = find(~used & dist <= tol,1,'first');
        end
        if isempty(match)
            NewRows(end+1,:) = Rows(i,:); %#ok<AGROW>
        else
            used(match) = true;
        end
    end
    Seen = [Seen;NewRows];
end

function idx = firstUnenteredSample(Samples,Decision)
    idx = [];
    tol = 1e-12;
    for i = 1 : numel(Samples)
        if Samples(i).entered ~= 0
            continue;
        end
        if max(abs(Samples(i).decision - Decision),[],2) <= tol
            idx = i;
            return;
        end
    end
end

function Row = refConversionFromExposure(Exposure)
    Row = refConversionTemplate();
    names = fieldnames(Row);
    for i = 1 : numel(names)
        Row.(names{i}) = Exposure.(names{i});
    end
end

function Row = directEntryFromSample(Sample,D)
    Row = directEntryTemplate(D);
    names = fieldnames(Row);
    for i = 1 : numel(names)
        Row.(names{i}) = Sample.(names{i});
    end
end

function count = sumExposureGroup(Rows,group)
    count = 0;
    if ~isempty(Rows)
        count = sum([Rows.group] == group);
    end
end

function value = sumExposureField(Rows,name)
    value = 0;
    if ~isempty(Rows)
        value = sum([Rows.(name)]);
    end
end

function value = medianStructField(Rows,name)
    value = NaN;
    if isempty(Rows)
        return;
    end
    values = double([Rows.(name)]);
    values = values(isfinite(values));
    if ~isempty(values)
        value = median(values);
    end
end

function value = safeRate(n,d)
    if d > 0
        value = n/d;
    else
        value = NaN;
    end
end

function Update = emptyTrackerUpdate()
    Update = struct( ...
        'ref_conversion_count',0, ...
        'ref_conversion_ref',zeros(0,1), ...
        'ref_conversion_group',zeros(0,1), ...
        'ref_conversion_lag_gen',zeros(0,1), ...
        'ref_conversion_lag_fe',zeros(0,1), ...
        'direct_bmem_entry_count',0, ...
        'direct_bmem_entry_sample_id',zeros(0,1), ...
        'direct_bmem_entry_ref',zeros(0,1), ...
        'direct_bmem_entry_group',zeros(0,1), ...
        'direct_bmem_entry_lag_gen',zeros(0,1), ...
        'direct_bmem_entry_lag_fe',zeros(0,1));
end

function Rows = emptyRefExposureRows()
    Rows = repmat(refExposureTemplate(),0,1);
end

function Row = refExposureTemplate()
    Row = struct( ...
        'ref',NaN, ...
        'group',NaN, ...
        'first_query_gen',NaN, ...
        'first_query_fe',NaN, ...
        'last_query_gen',NaN, ...
        'last_query_fe',NaN, ...
        'exposure_count',0, ...
        'generated_count',0, ...
        'converted',0, ...
        'first_populated_gen',NaN, ...
        'first_populated_fe',NaN, ...
        'lag_gen',NaN, ...
        'lag_fe',NaN);
end

function Rows = emptySampleRows(D)
    Rows = repmat(sampleTemplate(D),0,1);
end

function Row = sampleTemplate(D)
    Row = struct( ...
        'sample_id',NaN, ...
        'query_gen',NaN, ...
        'query_fe',NaN, ...
        'query_ref',NaN, ...
        'query_group',NaN, ...
        'survive_P1',0, ...
        'survive_P2',0, ...
        'decision',zeros(1,D), ...
        'entered',0, ...
        'entry_gen',NaN, ...
        'entry_fe',NaN, ...
        'lag_gen',NaN, ...
        'lag_fe',NaN);
end

function Rows = emptyRefConversionRows()
    Rows = repmat(refConversionTemplate(),0,1);
end

function Row = refConversionTemplate()
    Row = struct( ...
        'ref',NaN, ...
        'group',NaN, ...
        'first_query_gen',NaN, ...
        'first_query_fe',NaN, ...
        'last_query_gen',NaN, ...
        'last_query_fe',NaN, ...
        'exposure_count',0, ...
        'generated_count',0, ...
        'converted',0, ...
        'first_populated_gen',NaN, ...
        'first_populated_fe',NaN, ...
        'lag_gen',NaN, ...
        'lag_fe',NaN);
end

function Rows = emptyDirectEntryRows(D)
    Rows = repmat(directEntryTemplate(D),0,1);
end

function Row = directEntryTemplate(D)
    Row = struct( ...
        'sample_id',NaN, ...
        'query_gen',NaN, ...
        'query_fe',NaN, ...
        'query_ref',NaN, ...
        'query_group',NaN, ...
        'survive_P1',0, ...
        'survive_P2',0, ...
        'decision',zeros(1,D), ...
        'entered',0, ...
        'entry_gen',NaN, ...
        'entry_fe',NaN, ...
        'lag_gen',NaN, ...
        'lag_fe',NaN);
end
