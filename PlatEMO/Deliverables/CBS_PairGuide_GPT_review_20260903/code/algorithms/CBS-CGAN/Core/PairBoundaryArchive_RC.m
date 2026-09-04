function varargout = PairBoundaryArchive_RC(action,varargin)
%PAIRBOUNDARYARCHIVE_RC Atomic real feasible/infeasible boundary pairs.
%   Pair identity is persistent. Current P1 elites may create or locally
%   support pairs, but a retained feasible endpoint need not remain in P1.
%   Every endpoint replacement must strictly shorten the normalized gap;
%   reference labels never gate archive admission or endpoint replacement.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    switch lower(strtrim(string(action)))
        case "update"
            [varargout{1:nargout}] = updateArchive(varargin{:});
        case "trainingdata"
            [varargout{1:nargout}] = buildTrainingData(varargin{:});
        case "querycontexts"
            [varargout{1:nargout}] = buildQueryContexts(varargin{:});
        case "selectcandidates"
            [varargout{1:nargout}] = selectCandidates(varargin{:});
        case "bmem"
            varargout{1} = archiveAsBoundaryMemory(varargin{:});
        otherwise
            error('CBSPairGuide:BadArchiveAction', ...
                'Unsupported pair archive action: %s.',action);
    end
end

function [Archive,RefScale,Trace] = updateArchive(Archive,Population1, ...
        Evaluated,W,Problem,Options,Feedback,currentFE)
%UPDATEARCHIVE Refresh pairs after environmental selection.

    Options = fillOptions(Options);
    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    Trace = emptyMemoryTrace();
    oldCount = numel(Archive.id);

    [EliteX,EliteY,EliteRef,RefScale,Trace] = currentElites( ...
        Population1,W,Trace);
    [FeedbackX,FeedbackY,FeedbackC,FeedbackIds] = ...
        validFeedbackRows(Feedback,Problem);
    [Archive,FeedbackTrace,activated] = applyGuidedFeedback(Archive, ...
        FeedbackX,FeedbackY,FeedbackC,FeedbackIds,W,RefScale, ...
        Problem,currentFE,Options);
    [FeasX,FeasY,InfX,InfY] = evaluatedRows( ...
        Evaluated,FeedbackX,Problem,Options);
    [Archive,EvaluatedTrace,evaluatedActivated] = tightenRetainedPairs( ...
        Archive,FeasX,FeasY,InfX,InfY,W,RefScale,Problem, ...
        currentFE,Options);
    activated = activated | evaluatedActivated;
    Trace.tightenedFeasible = FeedbackTrace.tightenedFeasible+ ...
        EvaluatedTrace.tightenedFeasible;
    Trace.tightenedInfeasible = FeedbackTrace.tightenedInfeasible+ ...
        EvaluatedTrace.tightenedInfeasible;

    % Current elites are local support evidence, not endpoint membership.
    SupportX = excludeDecisions( ...
        EliteX,FeedbackX,Problem,Options.pairDuplicateTolerance);
    activated = activated | locallySupportedPairs( ...
        Archive,SupportX,Problem,Options);
    if oldCount > 0
        Archive.active(1:oldCount) = activated;
        Archive.age(activated) = 0;
        Archive.age(~activated) = Archive.age(~activated)+1;
    end

    % Every unrepresented current elite may start one new atomic pair.
    for elite = 1 : size(EliteX,1)
        if any(sameDecisionRows(FeedbackX,EliteX(elite,:),Problem, ...
                Options.pairDuplicateTolerance)) || ...
                any(sameDecisionRows(Archive.xf,EliteX(elite,:),Problem, ...
                Options.pairDuplicateTolerance))
            continue;
        end
        candidate = nearestDecisionRow(EliteX(elite,:),InfX,Problem);
        if candidate == 0
            continue;
        end
        Archive = appendPair(Archive,EliteX(elite,:),InfX(candidate,:), ...
            EliteY(elite,:),InfY(candidate,:),EliteRef(elite), ...
            Problem,currentFE);
        Trace.added = Trace.added+1;
    end

    beforePrune = numel(Archive.id);
    Archive = pruneArchive(Archive,Options,size(W,1),Problem);
    Trace.removed = beforePrune-numel(Archive.id);
    active = Archive.active;
    Trace.afterCap = nnz(active);
    Trace.capDropped = max(0,beforePrune-numel(Archive.id));
    Trace.retained = numel(Archive.id);
    Trace.paired = nnz(active);
    Trace.active = nnz(active);
    Trace.inactive = nnz(~active);
    Trace.pairedBeforeMAD = Trace.paired;
    Trace.trueFeasible = size(EliteX,1);
    Trace.afterFront = size(EliteX,1);
    Trace.pairGapMedian = finitePercentile(Archive.gap(active),0.5);
    Trace.pairGapP90 = finitePercentile(Archive.gap(active),0.9);
    Trace.archiveChanged = Trace.added+Trace.tightenedFeasible+ ...
        Trace.tightenedInfeasible+Trace.removed;
    Trace.previousCount = oldCount;
end

function [EliteX,EliteY,EliteRef,RefScale,Trace] = currentElites( ...
        Population,W,Trace)
%CURRENTELITES Return feasible Fitness<1 rows after P1 selection.

    if isempty(Population)
        D = 0;
    else
        D = size(Population.decs,2);
    end
    EliteX = zeros(0,D);
    EliteY = zeros(0,size(W,2));
    EliteRef = zeros(0,1);
    RefScale = struct('minimum',zeros(1,size(W,2)), ...
        'span',ones(1,size(W,2)));
    if isempty(Population)
        return;
    end
    X = double(Population.decs);
    Y = double(Population.objs);
    C = double(Population.cons);
    valid = all(isfinite(X),2) & all(isfinite(Y),2) & ...
        all(isfinite(C),2);
    feasible = valid & constraintViolation(C) <= 0;
    Trace.trueFeasible = nnz(feasible);
    if ~any(feasible)
        return;
    end
    Fitness = CalFitness_CBS(Y,C);
    elite = feasible & reshape(double(Fitness),[],1) < 1;
    if ~any(elite)
        return;
    end
    EliteX = X(elite,:);
    EliteY = Y(elite,:);
    [EliteRef,RefScale] = AssignReferenceVectors_CBS(EliteY,W);
    [~,uniqueRows] = unique(EliteX,'rows','stable');
    uniqueRows = sort(uniqueRows);
    EliteX = EliteX(uniqueRows,:);
    EliteY = EliteY(uniqueRows,:);
    EliteRef = EliteRef(uniqueRows);
end

function [X,Y,C,ids] = validFeedbackRows(Feedback,Problem)
%VALIDFEEDBACKROWS Keep only attributed, real evaluated guided children.

    X = zeros(0,Problem.D);
    Y = zeros(0,Problem.M);
    C = zeros(0,0);
    ids = zeros(0,1);
    required = {'childDecs','childObjs','childCons','matchedPairIds'};
    if isempty(Feedback) || ~isstruct(Feedback) || ...
            ~all(isfield(Feedback,required))
        return;
    end
    allX = double(Feedback.childDecs);
    allY = double(Feedback.childObjs);
    allC = double(Feedback.childCons);
    allIds = reshape(double(Feedback.matchedPairIds),[],1);
    count = min([size(allX,1),size(allY,1),size(allC,1),numel(allIds)]);
    if count == 0
        return;
    end
    allX = allX(1:count,:);
    allY = allY(1:count,:);
    allC = allC(1:count,:);
    allIds = allIds(1:count);
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    valid = all(isfinite(allX),2) & all(isfinite(allY),2) & ...
        all(isfinite(allC),2) & isfinite(allIds) & ...
        allIds == fix(allIds) & allIds > 0 & ...
        all(allX >= lower-1e-12,2) & all(allX <= upper+1e-12,2);
    X = allX(valid,:);
    Y = allY(valid,:);
    C = allC(valid,:);
    ids = allIds(valid);
end

function [FeasX,FeasY,InfX,InfY] = evaluatedRows( ...
        Population,Excluded,Problem,Options)
%EVALUATEDROWS Split real evaluated, unattributed rows by feasibility.

    FeasX = zeros(0,Problem.D);
    FeasY = zeros(0,Problem.M);
    InfX = zeros(0,Problem.D);
    InfY = zeros(0,Problem.M);
    if isempty(Population)
        return;
    end
    X = double(Population.decs);
    Y = double(Population.objs);
    C = double(Population.cons);
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    valid = all(isfinite(X),2) & all(isfinite(Y),2) & ...
        all(isfinite(C),2) & all(X >= lower-1e-12,2) & ...
        all(X <= upper+1e-12,2);
    if ~isempty(Excluded)
        valid = valid & minimumNormalizedDistance( ...
            X,Excluded,Problem) > Options.pairDuplicateTolerance;
    end
    X = X(valid,:);
    Y = Y(valid,:);
    C = C(valid,:);
    if isempty(X)
        return;
    end
    [X,rows] = unique(X,'rows','stable');
    Y = Y(rows,:);
    C = C(rows,:);
    feasible = constraintViolation(C) <= 0;
    FeasX = X(feasible,:);
    FeasY = Y(feasible,:);
    InfX = X(~feasible,:);
    InfY = Y(~feasible,:);
end

function [Archive,Trace,activated] = applyGuidedFeedback(Archive, ...
        X,Y,C,ids,W,RefScale,Problem,currentFE,Options)
%APPLYGUIDEDFEEDBACK Tighten only each child's attributed persistent pair.

    Trace = struct('tightenedFeasible',0,'tightenedInfeasible',0);
    activated = false(numel(Archive.id),1);
    for id = reshape(unique(ids,'stable'),1,[])
        row = find(Archive.id == id,1);
        if isempty(row)
            continue;
        end
        local = ids == id;
        feasible = constraintViolation(C(local,:)) <= 0;
        localX = X(local,:);
        localY = Y(local,:);
        [Archive,changedF,changedI] = tightenOnePair(Archive,row, ...
            localX(feasible,:),localY(feasible,:), ...
            localX(~feasible,:),localY(~feasible,:),W,RefScale, ...
            Problem,currentFE,Options);
        Trace.tightenedFeasible = Trace.tightenedFeasible+changedF;
        Trace.tightenedInfeasible = Trace.tightenedInfeasible+changedI;
        activated(row) = changedF+changedI > 0;
    end
end

function [Archive,Trace,activated] = tightenRetainedPairs(Archive, ...
        FeasX,FeasY,InfX,InfY,W,RefScale,Problem,currentFE,Options)
%TIGHTENRETAINEDPAIRS Let unattributed real rows revive retained pairs.

    Trace = struct('tightenedFeasible',0,'tightenedInfeasible',0);
    activated = false(numel(Archive.id),1);
    feasiblePair = nearestPairRows(FeasX,Archive.xf,Problem);
    infeasiblePair = nearestPairRows(InfX,Archive.xi,Problem);
    for row = 1 : numel(Archive.id)
        useF = feasiblePair == row;
        useI = infeasiblePair == row;
        [Archive,changedF,changedI] = tightenOnePair(Archive,row, ...
            FeasX(useF,:),FeasY(useF,:),InfX(useI,:),InfY(useI,:), ...
            W,RefScale,Problem,currentFE,Options);
        Trace.tightenedFeasible = Trace.tightenedFeasible+changedF;
        Trace.tightenedInfeasible = Trace.tightenedInfeasible+changedI;
        activated(row) = changedF+changedI > 0;
    end
end

function rows = nearestPairRows(X,Opposite,Problem)
%NEARESTPAIRROWS Attribute each ordinary row to one closest retained pair.

    rows = zeros(size(X,1),1);
    if isempty(X) || isempty(Opposite)
        return;
    end
    lower = double(Problem.lower);
    span = double(Problem.upper)-lower;
    span(span <= eps) = 1;
    Xn = (double(X)-lower)./span;
    On = (double(Opposite)-lower)./span;
    distance2 = max(0,sum(Xn.^2,2)+sum(On.^2,2)'-2*(Xn*On'));
    [~,rows] = min(distance2,[],2);
end

function [Archive,changedF,changedI] = tightenOnePair(Archive,row, ...
        FeasX,FeasY,InfX,InfY,W,RefScale,Problem,currentFE,Options)
%TIGHTENONEPAIR Alternate best real endpoints until no gap improvement.

    changedF = 0;
    changedI = 0;
    while true
        improved = false;
        candidate = nearestDecisionRow(Archive.xi(row,:),FeasX,Problem);
        if candidate > 0
            gap = normalizedDistance(FeasX(candidate,:), ...
                Archive.xi(row,:),Problem);
            if gap+Options.pairImprovementTolerance < Archive.gap(row)
                Archive.xf(row,:) = FeasX(candidate,:);
                Archive.yf(row,:) = FeasY(candidate,:);
                Archive.ref(row) = AssignReferenceVectors_CBS( ...
                    FeasY(candidate,:),W,RefScale);
                Archive.gap(row) = gap;
                Archive.lastFE(row) = double(currentFE);
                changedF = changedF+1;
                improved = true;
            end
        end
        candidate = nearestDecisionRow(Archive.xf(row,:),InfX,Problem);
        if candidate > 0
            gap = normalizedDistance(Archive.xf(row,:), ...
                InfX(candidate,:),Problem);
            if gap+Options.pairImprovementTolerance < Archive.gap(row)
                Archive.xi(row,:) = InfX(candidate,:);
                Archive.yi(row,:) = InfY(candidate,:);
                Archive.gap(row) = gap;
                Archive.lastFE(row) = double(currentFE);
                changedI = changedI+1;
                improved = true;
            end
        end
        if ~improved
            return;
        end
    end
end

function supported = locallySupportedPairs(Archive,EliteX,Problem,Options)
%LOCALLYSUPPORTEDPAIRS Decision-space support without reference gating.

    supported = false(numel(Archive.id),1);
    if isempty(EliteX) || isempty(Archive.id)
        return;
    end
    lower = double(Problem.lower);
    span = double(Problem.upper)-lower;
    span(span <= eps) = 1;
    EliteNorm = (double(EliteX)-lower)./span;
    h = robustLocalScale(EliteNorm,Options.pairDuplicateTolerance);
    validH = isfinite(h) & h > Options.pairDuplicateTolerance;
    if any(validH)
        h(~validH) = median(h(validH));
        validH(:) = true;
    end
    XfNorm = (double(Archive.xf)-lower)./span;
    for row = 1 : numel(Archive.id)
        distance = sqrt(sum((EliteNorm-XfNorm(row,:)).^2,2));
        exact = distance <= Options.pairDuplicateTolerance;
        local = validH & distance <= Options.pairSupportMultiplier*h+ ...
            Options.pairDuplicateTolerance;
        supported(row) = any(exact | local);
    end
end

function h = robustLocalScale(X,tolerance)
%ROBUSTLOCALSCALE Median of up to three nearest positive distances.

    count = size(X,1);
    h = nan(count,1);
    if count < 2
        return;
    end
    distance = sqrt(max(0,sum(X.^2,2)+sum(X.^2,2)'-2*(X*X')));
    for row = 1 : count
        positive = sort(distance(row,distance(row,:) > tolerance));
        if ~isempty(positive)
            h(row) = median(positive(1:min(3,numel(positive))));
        end
    end
end

function X = excludeDecisions(X,Excluded,Problem,tolerance)
%EXCLUDEDECISIONS Prevent attributed children from changing another pair.

    if isempty(X) || isempty(Excluded)
        return;
    end
    X = X(minimumNormalizedDistance(X,Excluded,Problem) > tolerance,:);
end

function [Data,Gate,TrainC,QueryRefs,BMem] = buildTrainingData( ...
        Archive,W,Problem,Options)
%BUILDTRAININGDATA Build complete absolute endpoint pairs.

    Options = fillOptions(Options);
    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    rows = find(Archive.active & all(isfinite(Archive.xf),2) & ...
        all(isfinite(Archive.xi),2) & Archive.ref >= 1 & ...
        Archive.ref <= size(W,1));
    Data = emptyTrainingData(Problem.D,Problem.M);
    if ~isempty(rows)
        lower = double(Problem.lower);
        span = double(Problem.upper)-lower;
        span(span <= eps) = 1;
        Data.xF = (Archive.xf(rows,:)-lower)./span;
        Data.xI = (Archive.xi(rows,:)-lower)./span;
        Data.w = double(W(Archive.ref(rows),:));
        Data.ref = Archive.ref(rows);
        Data.id = Archive.id(rows);
        Data.lastFE = Archive.lastFE(rows);
        Data.delta = Data.xI-Data.xF;
        Data.count = numel(rows);
        Data.cF = [Data.w,ones(Data.count,1)];
        Data.cI = [Data.w,zeros(Data.count,1)];
    end
    TrainC = [Data.cF;Data.cI];
    QueryRefs = unique(Data.ref,'stable');
    Gate = struct('effective',Data.count,'active',Data.count, ...
        'regions',numel(QueryRefs),'pairs',Data.count, ...
        'eligible',Data.count >= Options.pairMinPairs && ...
        numel(QueryRefs) >= Options.pairMinRegions);
    BMem = archiveAsBoundaryMemory(Archive,Problem);
end

function [QueryC,Info] = buildQueryContexts(Archive,W,Options,totalBudget)
%BUILDQUERYCONTEXTS Query the infeasible side over supported references.

    Options = fillOptions(Options); %#ok<NASGU>
    if isempty(Archive) || ~isstruct(Archive)
        QueryC = zeros(0,size(W,2)+1);
        Info = struct('refs',zeros(0,1));
        return;
    end
    activeRefs = unique(Archive.ref(Archive.active),'stable');
    activeRefs = activeRefs(activeRefs >= 1 & activeRefs <= size(W,1));
    totalBudget = max(0,round(double(totalBudget)));
    Info = struct('refs',zeros(0,1));
    QueryC = zeros(0,size(W,2)+1);
    if isempty(activeRefs) || totalBudget == 0
        return;
    end
    refs = balancedRows(activeRefs,totalBudget);
    QueryC = [double(W(refs,:)),zeros(numel(refs),1)];
    Info.refs = refs;
end

function [SelectedDecs,SelectedRefs,MatchedIds,Trace] = ...
        selectCandidates(RawDec,SampleInfo,Archive,Population,~,W,~, ...
        Problem,Options)
%SELECTCANDIDATES Apply validity, duplicate, attribution, and support gates.
%   No critic ranking and no 500-to-200 truncation are performed here.

    Options = fillOptions(Options);
    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    RawDec = double(RawDec);
    refs = reshape(double(SampleInfo.refs),[],1);
    rowCount = min(size(RawDec,1),numel(refs));
    RawDec = RawDec(1:rowCount,:);
    refs = refs(1:rowCount);
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    span = upper-lower;
    span(span <= eps) = 1;
    Xn = (RawDec-lower)./span;
    tolerance = Options.pairDuplicateTolerance;
    valid = all(isfinite(RawDec),2) & isfinite(refs) & ...
        refs == fix(refs) & refs >= 1 & refs <= size(W,1) & ...
        all(RawDec >= lower-1e-12,2) & all(RawDec <= upper+1e-12,2);

    base = [double(Population.decs);Archive.xf;Archive.xi];
    if ~isempty(base)
        base = (base-lower)./span;
        valid = valid & minimumDistance(Xn,base) > tolerance;
    end
    rows = find(valid);
    if ~isempty(rows)
        signature = round(Xn(rows,:)/max(tolerance,eps));
        [~,uniqueRows] = unique(signature,'rows','stable');
        uniqueMask = false(rowCount,1);
        uniqueMask(rows(sort(uniqueRows))) = true;
        valid = valid & uniqueMask;
    end

    matched = zeros(rowCount,1);
    matchFailures = 0;
    supportFailures = 0;
    activeRows = find(Archive.active);
    Xi = (Archive.xi-lower)./span;
    for raw = reshape(find(valid),1,[])
        local = activeRows(ismember(Archive.ref(activeRows), ...
            neighborRefs(W,refs(raw),Options.pairNeighborRefCount)));
        if isempty(local)
            valid(raw) = false;
            matchFailures = matchFailures+1;
            continue;
        end
        distance = sqrt(sum((Xi(local,:)-Xn(raw,:)).^2,2));
        [matchDistance,pick] = min(distance);
        pairRow = local(pick);
        radius = supportRadius(Archive,pairRow,W,Problem,Options);
        if ~isfinite(radius) || matchDistance > ...
                Options.pairSupportMultiplier*max(radius,tolerance)
            valid(raw) = false;
            supportFailures = supportFailures+1;
            continue;
        end
        matched(raw) = Archive.id(pairRow);
    end

    selected = find(valid);
    SelectedDecs = min(max(RawDec(selected,:),lower),upper);
    SelectedRefs = refs(selected);
    MatchedIds = matched(selected);
    Trace = emptyPoolTrace();
    Trace.active = rowCount > 0;
    Trace.rawCount = rowCount;
    Trace.keptCount = numel(selected);
    Trace.rawConditions = numel(unique(refs));
    Trace.keptConditions = numel(unique(SelectedRefs));
    Trace.keepIdx = selected;
    Trace.percentile = nan(rowCount,1);
    Trace.trainConditions = numel(unique(Archive.ref(Archive.active)));
    Trace.validCount = numel(selected);
    Trace.rawNearDuplicateRate = nearDuplicateRate( ...
        Xn(all(isfinite(Xn),2),:),tolerance);
    Trace.keptNearDuplicateRate = nearDuplicateRate(Xn(valid),tolerance);
    Trace.matchFailures = matchFailures;
    Trace.supportFailures = supportFailures;
    Trace.matchedPairIds = MatchedIds;
end

function row = nearestDecisionRow(x,X,Problem)
%NEARESTDECISIONROW Nearest real row in normalized decision space.

    row = 0;
    if isempty(X)
        return;
    end
    lower = double(Problem.lower);
    span = double(Problem.upper)-lower;
    span(span <= eps) = 1;
    distance = sqrt(sum(((double(X)-double(x))./span).^2,2));
    [~,row] = min(distance);
end

function Archive = appendPair(Archive,xf,xi,yf,yi,ref,Problem,currentFE)
%APPENDPAIR Add one complete pair with a never-reused ID.

    Archive.id(end+1,1) = Archive.nextId;
    Archive.nextId = Archive.nextId+1;
    Archive.xf(end+1,:) = xf;
    Archive.xi(end+1,:) = xi;
    Archive.yf(end+1,:) = yf;
    Archive.yi(end+1,:) = yi;
    Archive.ref(end+1,1) = ref;
    Archive.gap(end+1,1) = normalizedDistance(xf,xi,Problem);
    Archive.age(end+1,1) = 0;
    Archive.lastFE(end+1,1) = double(currentFE);
    Archive.active(end+1,1) = true;
end

function Archive = pruneArchive(Archive,Options,refCount,Problem)
%PRUNEARCHIVE Remove stale/invalid rows and enforce archive capacities.

    keep = validArchiveRows(Archive,refCount,Problem) & ...
        (Archive.active | Archive.age < Options.pairInactiveMaxAge);
    candidates = find(keep);
    selected = zeros(0,1);
    for ref = reshape(unique(Archive.ref(candidates),'stable'),1,[])
        local = candidates(Archive.ref(candidates) == ref);
        key = [-double(Archive.active(local)),Archive.gap(local), ...
            Archive.age(local),-Archive.lastFE(local),Archive.id(local)];
        [~,order] = sortrows(key,[1 2 3 4 5]);
        selected = [selected;local(order(1:min( ...
            Options.pairArchivePerRef,numel(order))))]; %#ok<AGROW>
    end
    selected = sort(selected);
    Archive = subsetArchive(Archive,selected);
    totalCap = Options.pairArchivePerRef*refCount;
    if numel(Archive.id) > totalCap
        error('CBSPairGuide:ArchiveCapacityExceeded', ...
            'Pair archive exceeds per-reference total capacity.');
    end
end

function valid = validArchiveRows(Archive,refCount,Problem)
%VALIDARCHIVEROWS Reject corrupt retained pairs before age/cap pruning.

    lower = double(Problem.lower);
    upper = double(Problem.upper);
    valid = isfinite(Archive.id) & Archive.id == fix(Archive.id) & ...
        Archive.id > 0 & isfinite(Archive.ref) & ...
        Archive.ref == fix(Archive.ref) & Archive.ref >= 1 & ...
        Archive.ref <= refCount & isfinite(Archive.gap) & ...
        Archive.gap >= 0 & isfinite(Archive.age) & Archive.age >= 0 & ...
        isfinite(Archive.lastFE) & all(isfinite(Archive.xf),2) & ...
        all(isfinite(Archive.xi),2) & all(isfinite(Archive.yf),2) & ...
        all(isfinite(Archive.yi),2) & ...
        all(Archive.xf >= lower-1e-12,2) & ...
        all(Archive.xf <= upper+1e-12,2) & ...
        all(Archive.xi >= lower-1e-12,2) & ...
        all(Archive.xi <= upper+1e-12,2);
end

function radius = supportRadius(Archive,row,W,Problem,Options)
%SUPPORTRADIUS Robust local Xi scale, never a pair-admission threshold.

    lower = double(Problem.lower);
    span = double(Problem.upper)-lower;
    span(span <= eps) = 1;
    Xi = (Archive.xi-lower)./span;
    local = Archive.active & ismember(Archive.ref, ...
        neighborRefs(W,Archive.ref(row),Options.pairNeighborRefCount));
    distance = sqrt(sum((Xi(local,:)-Xi(row,:)).^2,2));
    distance = sort(distance(distance > Options.pairDuplicateTolerance));
    if isempty(distance)
        radius = Archive.gap(row);
    else
        radius = median(distance(1:min(3,numel(distance))));
        radius = max(radius,Archive.gap(row));
    end
end

function BMem = archiveAsBoundaryMemory(Archive,Problem)
%ARCHIVEASBOUNDARYMEMORY Provide diagnostics and pair attribution state.

    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    BMem = struct('id',Archive.id,'ref',Archive.ref,'gap',Archive.gap, ...
        'x_b',Archive.xf,'y_b',Archive.yf,'x_i',Archive.xi, ...
        'y_i',Archive.yi,'active',Archive.active,'lastFE',Archive.lastFE);
end

function Archive = ensureArchive(Archive,D,M)
%ENSUREARCHIVE Normalize empty and earlier archive values.

    Empty = emptyArchive(D,M);
    if isempty(Archive) || ~isstruct(Archive)
        Archive = Empty;
        return;
    end
    names = fieldnames(Empty);
    for i = 1 : numel(names)
        if ~isfield(Archive,names{i})
            Archive.(names{i}) = Empty.(names{i});
        end
    end
    count = size(Archive.xf,1);
    vectorFields = {'id','ref','gap','age','lastFE','active'};
    for i = 1 : numel(vectorFields)
        name = vectorFields{i};
        if size(Archive.(name),1) ~= count
            Archive.(name) = Empty.(name);
        end
    end
    if ~isscalar(Archive.nextId) || ~isfinite(Archive.nextId)
        Archive.nextId = max([0;Archive.id])+1;
    end
end

function Archive = emptyArchive(D,M)
%EMPTYARCHIVE Construct the unified pair schema.

    Archive = struct('id',zeros(0,1),'xf',zeros(0,D), ...
        'xi',zeros(0,D),'yf',zeros(0,M),'yi',zeros(0,M), ...
        'ref',zeros(0,1),'gap',zeros(0,1),'age',zeros(0,1), ...
        'lastFE',zeros(0,1),'active',false(0,1),'nextId',1);
end

function Data = emptyTrainingData(D,M)
%EMPTYTRAININGDATA Complete-pair endpoint data; pair ID is metadata only.

    Data = struct('xF',zeros(0,D),'xI',zeros(0,D), ...
        'delta',zeros(0,D),'w',zeros(0,M),'ref',zeros(0,1), ...
        'id',zeros(0,1),'lastFE',zeros(0,1), ...
        'cF',zeros(0,M+1),'cI',zeros(0,M+1),'count',0);
end

function Trace = emptyMemoryTrace()
%EMPTYMEMORYTRACE Retain common audit fields and add pair lifecycle fields.

    Trace = struct('trueFeasible',0,'afterFront',0,'frontDropped',0, ...
        'frontOpportunityRefs',0,'afterCap',0,'capDropped',0, ...
        'retained',0,'pairedBeforeMAD',0,'unpairedBeforeMAD',0, ...
        'paired',0,'unpaired',0,'madDropped',0,'legalWithin5',0, ...
        'legalWithin10',0,'legalAny',0,'dominanceRejected',0, ...
        'pairRank1To5',0,'pairRank6To10',0,'pairRankOver10',0, ...
        'previousUnpaired',0,'previousUnpairedPaired',0, ...
        'pairGapMedian',NaN,'pairGapP90',NaN, ...
        'pairAngleMedian',NaN,'pairAngleP90',NaN, ...
        'added',0,'strong',0,'weak',0,'generatedWeak',0, ...
        'tightenedFeasible',0,'tightenedInfeasible',0,'removed',0, ...
        'active',0,'inactive',0,'archiveChanged',0,'previousCount',0);
end

function Trace = emptyPoolTrace()
%EMPTYPOOLTRACE Match the shared generation audit contract.

    Trace = struct('active',false,'rawCount',0,'keptCount',0, ...
        'rawConditions',0,'keptConditions',0,'keepIdx',zeros(0,1), ...
        'percentile',zeros(0,1),'rawOracleCount',0, ...
        'rawOracleFeasible',0,'trainConditions',0, ...
        'rawSupportedCount',0,'rawSupportedFeasible',0, ...
        'rawUnsupportedCount',0,'rawUnsupportedFeasible',0, ...
        'rawReferenceMatch',0,'rawSupportedReferenceMatch',0, ...
        'rawUnsupportedReferenceMatch',0, ...
        'rawBoundaryDistance',zeros(0,1), ...
        'rawBoundaryBand',zeros(0,1), ...
        'rawBoundarySupported',false(0,1), ...
        'keptBoundaryDistance',zeros(0,1), ...
        'keptBoundaryBand',zeros(0,1), ...
        'keptBoundarySupported',false(0,1), ...
        'rejectedBoundaryDistance',zeros(0,1), ...
        'rejectedBoundaryBand',zeros(0,1), ...
        'rejectedBoundarySupported',false(0,1), ...
        'criticBoundarySpearman',NaN,'criticBoundaryPairCount',0, ...
        'rawDirectionCoverage',NaN,'rawDirectionEntropy',NaN, ...
        'rawNearDuplicateRate',NaN,'keptDirectionCoverage',NaN, ...
        'keptDirectionEntropy',NaN,'keptNearDuplicateRate',NaN, ...
        'validCount',0, ...
        'matchFailures',0,'supportFailures',0, ...
        'matchedPairIds',zeros(0,1));
end

function Archive = subsetArchive(Archive,rows)
%SUBSETARCHIVE Subset row fields without changing the next ID.

    rows = reshape(rows,[],1);
    fields = {'id','xf','xi','yf','yi','ref','gap','age','lastFE','active'};
    for i = 1 : numel(fields)
        name = fields{i};
        Archive.(name) = Archive.(name)(rows,:);
    end
end

function same = sameDecisionRows(X,x,Problem,tolerance)
%SAMEDECISIONROWS Vectorized normalized equality.

    if isempty(X)
        same = false(0,1);
        return;
    end
    span = double(Problem.upper)-double(Problem.lower);
    span(span <= eps) = 1;
    same = sqrt(sum(((double(X)-double(x))./span).^2,2)) <= tolerance;
end

function distance = minimumNormalizedDistance(A,B,Problem)
%MINIMUMNORMALIZEDDISTANCE Minimum decision-box distance for every A row.

    if isempty(A)
        distance = zeros(0,1);
    elseif isempty(B)
        distance = inf(size(A,1),1);
    else
        lower = double(Problem.lower);
        span = double(Problem.upper)-lower;
        span(span <= eps) = 1;
        An = (double(A)-lower)./span;
        Bn = (double(B)-lower)./span;
        distance2 = max(0,sum(An.^2,2)+sum(Bn.^2,2)'-2*(An*Bn'));
        distance = sqrt(min(distance2,[],2));
    end
end

function distance = normalizedDistance(a,b,Problem)
%NORMALIZEDDISTANCE Decision-box normalized Euclidean distance.

    span = double(Problem.upper)-double(Problem.lower);
    span(span <= eps) = 1;
    distance = sqrt(sum(((double(a)-double(b))./span).^2,2));
end

function value = constraintViolation(C)
%CONSTRAINTVIOLATION Sum positive constraints rowwise.

    if isempty(C)
        value = zeros(size(C,1),1);
    else
        value = sum(max(0,double(C)),2);
    end
end

function refs = neighborRefs(W,ref,count)
%NEIGHBORREFS Reference plus the nearest requested directions.

    distance = sqrt(sum((double(W)-double(W(ref,:))).^2,2));
    [~,order] = sortrows([distance,(1:size(W,1))'],[1 2]);
    total = min(size(W,1),1+max(0,round(double(count))));
    refs = order(1:total);
end

function rows = balancedRows(values,total)
%BALANCEDROWS Uniform randomized allocation over supported references.

    values = reshape(values,[],1);
    base = floor(total/numel(values));
    rows = repelem(values,base,1);
    remainder = mod(total,numel(values));
    if remainder > 0
        order = randperm(numel(values),remainder);
        rows = [rows;values(order)];
    end
    rows = rows(randperm(numel(rows)));
end

function distance = minimumDistance(A,B)
%MINIMUMDISTANCE Minimum Euclidean row distance.

    if isempty(A)
        distance = zeros(0,1);
    elseif isempty(B)
        distance = inf(size(A,1),1);
    else
        distance2 = max(0,sum(A.^2,2)+sum(B.^2,2)'-2*(A*B'));
        distance = sqrt(min(distance2,[],2));
    end
end

function value = finitePercentile(X,p)
%FINITEPERCENTILE Linear percentile without Statistics Toolbox.

    X = sort(double(X(isfinite(X))));
    if isempty(X)
        value = NaN;
        return;
    end
    position = 1+(numel(X)-1)*min(1,max(0,double(p)));
    lo = floor(position);
    hi = ceil(position);
    value = X(lo)+(position-lo)*(X(hi)-X(lo));
end

function rate = nearDuplicateRate(X,tolerance)
%NEARDUPLICATERATE Fraction collapsed by normalized rounding.

    if isempty(X)
        rate = NaN;
        return;
    end
    signature = round(double(X)/max(double(tolerance),eps));
    rate = 1-size(unique(signature,'rows'),1)/size(X,1);
end

function Options = fillOptions(Options)
%FILLOPTIONS Fixed unified archive/query choices.

    Options = defaultOption(Options,'pairArchivePerRef',5);
    Options = defaultOption(Options,'pairInactiveMaxAge',10);
    Options = defaultOption(Options,'pairNeighborRefCount',5);
    Options = defaultOption(Options,'pairMinPairs',32);
    Options = defaultOption(Options,'pairMinRegions',4);
    Options = defaultOption(Options,'pairDuplicateTolerance',1e-6);
    Options = defaultOption(Options,'pairImprovementTolerance',1e-12);
    Options = defaultOption(Options,'pairSupportMultiplier',2);
    integer = {'pairArchivePerRef','pairInactiveMaxAge', ...
        'pairNeighborRefCount','pairMinPairs','pairMinRegions'};
    for i = 1 : numel(integer)
        name = integer{i};
        Options.(name) = max(0,round(double(Options.(name))));
    end
    Options.pairArchivePerRef = max(1,Options.pairArchivePerRef);
    Options.pairMinPairs = max(1,Options.pairMinPairs);
    Options.pairMinRegions = max(1,Options.pairMinRegions);
    Options.pairDuplicateTolerance = max(eps,double( ...
        Options.pairDuplicateTolerance));
    Options.pairImprovementTolerance = max(0,double( ...
        Options.pairImprovementTolerance));
    Options.pairSupportMultiplier = max(0,double( ...
        Options.pairSupportMultiplier));
end

function S = defaultOption(S,name,value)
%DEFAULTOPTION Fill one missing structure field.

    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end
