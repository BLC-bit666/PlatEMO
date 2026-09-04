function varargout = PairBoundaryArchive_RC(action,varargin)
%PAIRBOUNDARYARCHIVE_RC Maintain one legal endpoint pair per reference.
%   Conditions are always (w,1) for a truly evaluated feasible endpoint
%   and (w,0) for a truly evaluated infeasible endpoint. Pair IDs are
%   attribution metadata only and never enter the generator condition.
%   resumeEligible distinguishes a legal pause from invalidated history.

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
%UPDATEARCHIVE Refresh legal pairs from all truly evaluated information.

    Options = fillOptions(Options);
    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    Trace = emptyMemoryTrace();
    oldCount = numel(Archive.id);
    Historical = Archive;

    [EliteX,EliteY,EliteFitness,Trace] = currentElites( ...
        Population1,Problem,Trace);
    [FeedbackX,FeedbackY,FeedbackC,FeedbackIds] = ...
        validFeedbackRows(Feedback,Problem);
    [FeasX,FeasY,InfX,InfY] = evaluatedRows( ...
        Evaluated,FeedbackX,Problem,Options);
    feedbackInfeasible = constraintViolation(FeedbackC) > 0;
    scaleObjectives = [EliteY;InfY;FeedbackY(feedbackInfeasible,:); ...
        Historical.yi];
    [~,RefScale] = AssignReferenceVectors_CBS( ...
        finiteObjectiveRows(scaleObjectives,Problem.M),W);
    EliteRef = AssignReferenceVectors_CBS(EliteY,W,RefScale);

    % General candidates exclude attributed children in this generation.
    % Those children may update only their persistent matched pair first.
    [ItX,ItY,ItRef,Trace.dominanceRejected] = legalInfeasiblePool( ...
        [InfX;Historical.xi],[InfY;Historical.yi],EliteY,W, ...
        RefScale,Problem,Options);

    Archive = refreshPairMetadata( ...
        Archive,EliteY,EliteFitness,W,RefScale,Problem);
    initiallyLegal = legalPairRows( ...
        Archive,EliteY,W,RefScale,Problem,Options);
    % A changed condition or lost legality breaks lifecycle continuity.
    sameReference = Archive.ref == Historical.ref;
    Archive.resumeEligible = Archive.resumeEligible & ...
        initiallyLegal & sameReference;
    continuable = Archive.resumeEligible;
    [Archive,feedbackTrace,activated,forcedInactive,NewPairs] = ...
        applyGuidedFeedback(Archive,FeedbackX,FeedbackY,FeedbackC, ...
        FeedbackIds,continuable,ItX,ItY,ItRef,EliteY,W, ...
        RefScale,Problem,currentFE,Options);
    [Archive,evaluatedTrace,evaluatedActivated] = ...
        tightenRetainedPairs(Archive,FeasX,FeasY,InfX,InfY, ...
        EliteY,W,RefScale,Problem,currentFE,Options, ...
        continuable & ~forcedInactive);
    activated = activated | evaluatedActivated;
    Archive.resumeEligible(forcedInactive) = false;
    Trace.tightenedFeasible = feedbackTrace.tightenedFeasible+ ...
        evaluatedTrace.tightenedFeasible;
    Trace.tightenedInfeasible = feedbackTrace.tightenedInfeasible+ ...
        evaluatedTrace.tightenedInfeasible;

    Archive = refreshPairMetadata( ...
        Archive,EliteY,EliteFitness,W,RefScale,Problem);
    currentlyLegal = legalPairRows( ...
        Archive,EliteY,W,RefScale,Problem,Options);
    forcedInactive = forcedInactive(1:min(numel(forcedInactive), ...
        numel(currentlyLegal)));
    if numel(forcedInactive) < numel(currentlyLegal)
        forcedInactive(end+1:numel(currentlyLegal),1) = false;
    end
    Archive.resumeEligible = Archive.resumeEligible & ...
        currentlyLegal & ~forcedInactive;
    supported = locallySupportedPairs(Archive,EliteX,Problem,Options);
    Archive.active = currentlyLegal & ~forcedInactive & ...
        (activated | (supported & Archive.resumeEligible));
    Archive.age(Archive.active) = 0;
    Archive.age(~Archive.active) = Archive.age(~Archive.active)+1;

    % A supported old pair is already present as a candidate, so selecting
    % the canonical pair below cannot worsen its gap without a better pair.
    for row = 1 : size(NewPairs.xf,1)
        [Archive,added] = addOrActivatePair(Archive, ...
            NewPairs.xf(row,:),NewPairs.xi(row,:), ...
            NewPairs.yf(row,:),NewPairs.yi(row,:), ...
            NewPairs.ref(row),NewPairs.fitness(row),W,RefScale, ...
            Problem,currentFE,Options);
        Trace.added = Trace.added+added;
    end
    for elite = 1 : size(EliteX,1)
        [candidate,rank] = nearestLegalXi(EliteX(elite,:), ...
            EliteY(elite,:),EliteRef(elite),ItX,ItY,ItRef,W, ...
            RefScale,Problem,Options);
        if candidate == 0
            continue;
        end
        [Archive,added] = addOrActivatePair(Archive, ...
            EliteX(elite,:),ItX(candidate,:),EliteY(elite,:), ...
            ItY(candidate,:),EliteRef(elite),EliteFitness(elite), ...
            W,RefScale,Problem,currentFE,Options,rank);
        Trace.added = Trace.added+added;
    end

    Archive = refreshPairMetadata( ...
        Archive,EliteY,EliteFitness,W,RefScale,Problem);
    legal = legalPairRows(Archive,EliteY,W,RefScale,Problem,Options);
    Archive.resumeEligible = Archive.resumeEligible & legal;
    Archive.active = Archive.active & legal;
    beforePrune = numel(Archive.id);
    Archive = pruneArchive(Archive,Options,size(W,1),Problem);
    Trace.removed = max(0,beforePrune-numel(Archive.id));
    active = Archive.active;
    Trace.afterCap = nnz(active);
    Trace.capDropped = Trace.removed;
    Trace.retained = numel(Archive.id);
    Trace.paired = nnz(active);
    Trace.active = nnz(active);
    Trace.inactive = nnz(~active);
    Trace.strong = Trace.active;
    Trace.weak = Trace.inactive;
    Trace.pairedBeforeMAD = Trace.paired;
    Trace.trueFeasible = size(EliteX,1);
    Trace.afterFront = size(EliteX,1);
    Trace.pairGapMedian = finitePercentile(Archive.gap(active),0.5);
    Trace.pairGapP90 = finitePercentile(Archive.gap(active),0.9);
    Trace.archiveChanged = Trace.added+Trace.tightenedFeasible+ ...
        Trace.tightenedInfeasible+Trace.removed;
    Trace.previousCount = oldCount;
end

function [EliteX,EliteY,EliteFitness,Trace] = currentElites( ...
        Population,Problem,Trace)
%CURRENTELITES Return current P1 feasible rows with Fitness strictly < 1.

    EliteX = zeros(0,Problem.D);
    EliteY = zeros(0,Problem.M);
    EliteFitness = zeros(0,1);
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
    Fitness = reshape(double(CalFitness_CBS(Y,C)),[],1);
    elite = feasible & Fitness < 1;
    if ~any(elite)
        return;
    end
    EliteX = X(elite,:);
    EliteY = Y(elite,:);
    EliteFitness = Fitness(elite);
    [~,rows] = unique(EliteX,'rows','stable');
    rows = sort(rows);
    EliteX = EliteX(rows,:);
    EliteY = EliteY(rows,:);
    EliteFitness = EliteFitness(rows);
end

function [X,Y,C,ids] = validFeedbackRows(Feedback,Problem)
%VALIDFEEDBACKROWS Keep attributed, truly evaluated PairGuide children.

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
%EVALUATEDROWS Split all ordinary Union rows by true feasibility.

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

function [ItX,ItY,ItRef,dominanceRejected] = legalInfeasiblePool( ...
        X,Y,EliteY,W,RefScale,Problem,Options)
%LEGALINFEASIBLEPOOL Merge current and historical globally legal xi rows.

    ItX = zeros(0,Problem.D);
    ItY = zeros(0,Problem.M);
    ItRef = zeros(0,1);
    dominanceRejected = 0;
    if isempty(X)
        return;
    end
    X = double(X);
    Y = double(Y);
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    valid = all(isfinite(X),2) & all(isfinite(Y),2) & ...
        all(X >= lower-1e-12,2) & all(X <= upper+1e-12,2);
    X = X(valid,:);
    Y = Y(valid,:);
    if isempty(X)
        return;
    end
    legal = ~dominatedByAny(Y,EliteY,RefScale,1e-12);
    dominanceRejected = nnz(~legal);
    X = X(legal,:);
    Y = Y(legal,:);
    if isempty(X)
        return;
    end
    signature = round(((X-lower)./decisionSpan(Problem))/ ...
        Options.pairDuplicateTolerance);
    [~,rows] = unique(signature,'rows','stable');
    rows = sort(rows);
    ItX = X(rows,:);
    ItY = Y(rows,:);
    ItRef = AssignReferenceVectors_CBS(ItY,W,RefScale);
end

function [Archive,Trace,activated,forcedInactive,NewPairs] = ...
        applyGuidedFeedback(Archive,X,Y,C,ids,initiallyLegal, ...
        ItX,ItY,ItRef,EliteY,W,RefScale,Problem,currentFE,Options)
%APPLYGUIDEDFEEDBACK A child can modify only its attributed original pair.

    Trace = struct('tightenedFeasible',0,'tightenedInfeasible',0);
    activated = false(numel(Archive.id),1);
    forcedInactive = false(numel(Archive.id),1);
    NewPairs = emptyPairCandidates(Problem.D,Problem.M);
    for item = 1 : numel(ids)
        row = find(Archive.id == ids(item),1);
        if isempty(row)
            continue;
        end
        isFeasible = constraintViolation(C(item,:)) <= 0;
        if isFeasible
            newRef = AssignReferenceVectors_CBS(Y(item,:),W,RefScale);
            invalidOldXi = pairDominatesRows( ...
                Y(item,:),Archive.yi(row,:),RefScale,1e-12);
            if invalidOldXi || ~initiallyLegal(row)
                forcedInactive(row) = true;
                [candidate,rank] = nearestLegalXi(X(item,:),Y(item,:), ...
                    newRef,ItX,ItY,ItRef,W,RefScale,Problem,Options);
                if candidate > 0
                    NewPairs = appendPairCandidate(NewPairs,X(item,:), ...
                        ItX(candidate,:),Y(item,:),ItY(candidate,:), ...
                        newRef,pairFitness(Y(item,:),EliteY),rank);
                end
                continue;
            end
            gap = normalizedDistance(X(item,:),Archive.xi(row,:),Problem);
            xiRef = AssignReferenceVectors_CBS( ...
                Archive.yi(row,:),W,RefScale);
            legal = gap+Options.pairImprovementTolerance < ...
                Archive.gap(row) && ~invalidOldXi && ...
                ismember(xiRef,neighborRefs( ...
                W,newRef,Options.pairNeighborRefCount));
            if legal
                Archive.xf(row,:) = X(item,:);
                Archive.yf(row,:) = Y(item,:);
                Archive.ref(row) = newRef;
                Archive.gap(row) = gap;
                Archive.lastFE(row) = double(currentFE);
                Archive.fitness(row) = pairFitness(Y(item,:),EliteY);
                Archive.rank(row) = referenceRank(W,newRef,xiRef);
                activated(row) = true;
                Trace.tightenedFeasible = Trace.tightenedFeasible+1;
            end
        else
            if ~initiallyLegal(row) || ...
                    dominatedByAny(Y(item,:),EliteY,RefScale,1e-12) || ...
                    pairDominatesRows(Archive.yf(row,:),Y(item,:), ...
                    RefScale,1e-12)
                continue;
            end
            newXiRef = AssignReferenceVectors_CBS(Y(item,:),W,RefScale);
            gap = normalizedDistance(Archive.xf(row,:),X(item,:),Problem);
            legal = gap+Options.pairImprovementTolerance < ...
                Archive.gap(row) && ismember(newXiRef,neighborRefs( ...
                W,Archive.ref(row),Options.pairNeighborRefCount));
            if legal
                Archive.xi(row,:) = X(item,:);
                Archive.yi(row,:) = Y(item,:);
                Archive.gap(row) = gap;
                Archive.lastFE(row) = double(currentFE);
                Archive.rank(row) = referenceRank( ...
                    W,Archive.ref(row),newXiRef);
                activated(row) = true;
                Trace.tightenedInfeasible = ...
                    Trace.tightenedInfeasible+1;
            end
        end
    end
end

function [Archive,Trace,activated] = tightenRetainedPairs(Archive, ...
        FeasX,FeasY,InfX,InfY,EliteY,W,RefScale,Problem,currentFE, ...
        Options,eligibleRows)
%TIGHTENRETAINEDPAIRS Let every ordinary local row try every legal pair.

    Trace = struct('tightenedFeasible',0,'tightenedInfeasible',0);
    activated = false(numel(Archive.id),1);
    if isempty(Archive.id)
        return;
    end
    FeasRef = AssignReferenceVectors_CBS(FeasY,W,RefScale);
    legalInf = ~dominatedByAny(InfY,EliteY,RefScale,1e-12);
    InfX = InfX(legalInf,:);
    InfY = InfY(legalInf,:);
    InfRef = AssignReferenceVectors_CBS(InfY,W,RefScale);
    maxPasses = max(1,2*(size(FeasX,1)+size(InfX,1))+2);
    for row = reshape(find(eligibleRows),1,[])
        pass = 0;
        changed = true;
        while changed && pass < maxPasses
            pass = pass+1;
            changed = false;
            if ~isempty(FeasX)
                xiRef = AssignReferenceVectors_CBS( ...
                    Archive.yi(row,:),W,RefScale);
                gaps = normalizedDistances(FeasX,Archive.xi(row,:),Problem);
                legal = gaps+Options.pairImprovementTolerance < ...
                    Archive.gap(row) & ~pairDominatesRows(FeasY, ...
                    repmat(Archive.yi(row,:),size(FeasY,1),1), ...
                    RefScale,1e-12);
                for i = reshape(find(legal),1,[])
                    legal(i) = ismember(xiRef,neighborRefs( ...
                        W,FeasRef(i),Options.pairNeighborRefCount));
                end
                ranks = arrayfun(@(r)referenceRank(W,r,xiRef),FeasRef);
                candidate = bestGapCandidate(gaps,legal,ranks);
                if candidate > 0
                    Archive.xf(row,:) = FeasX(candidate,:);
                    Archive.yf(row,:) = FeasY(candidate,:);
                    Archive.ref(row) = FeasRef(candidate);
                    Archive.gap(row) = gaps(candidate);
                    Archive.lastFE(row) = double(currentFE);
                    Archive.fitness(row) = pairFitness( ...
                        FeasY(candidate,:),EliteY);
                    Archive.rank(row) = referenceRank( ...
                        W,Archive.ref(row),xiRef);
                    activated(row) = true;
                    Trace.tightenedFeasible = ...
                        Trace.tightenedFeasible+1;
                    changed = true;
                end
            end
            if ~isempty(InfX)
                gaps = normalizedDistances(InfX,Archive.xf(row,:),Problem);
                legal = gaps+Options.pairImprovementTolerance < ...
                    Archive.gap(row) & ismember(InfRef,neighborRefs( ...
                    W,Archive.ref(row),Options.pairNeighborRefCount)) & ...
                    ~pairDominatesRows(repmat(Archive.yf(row,:), ...
                    size(InfY,1),1),InfY,RefScale,1e-12);
                ranks = arrayfun(@(r)referenceRank( ...
                    W,Archive.ref(row),r),InfRef);
                candidate = bestGapCandidate(gaps,legal,ranks);
                if candidate > 0
                    Archive.xi(row,:) = InfX(candidate,:);
                    Archive.yi(row,:) = InfY(candidate,:);
                    Archive.gap(row) = gaps(candidate);
                    Archive.lastFE(row) = double(currentFE);
                    Archive.rank(row) = referenceRank( ...
                        W,Archive.ref(row),InfRef(candidate));
                    activated(row) = true;
                    Trace.tightenedInfeasible = ...
                        Trace.tightenedInfeasible+1;
                    changed = true;
                end
            end
        end
    end
end

function candidate = bestGapCandidate(gaps,legal,ranks)
%BESTGAPCANDIDATE Deterministic gap then angular-rank selection.

    rows = find(legal);
    candidate = 0;
    if isempty(rows)
        return;
    end
    [~,order] = sortrows([gaps(rows),ranks(rows),rows],[1 2 3]);
    candidate = rows(order(1));
end

function [candidate,rank] = nearestLegalXi( ...
        xf,yf,xfRef,ItX,ItY,ItRef,W,RefScale,Problem,Options)
%NEARESTLEGALXI Find the decision-nearest legal xi in the 5-ref corridor.

    candidate = 0;
    rank = Inf;
    if isempty(ItX)
        return;
    end
    legal = ismember(ItRef,neighborRefs( ...
        W,xfRef,Options.pairNeighborRefCount)) & ...
        ~pairDominatesRows(repmat(yf,size(ItY,1),1),ItY, ...
        RefScale,1e-12);
    rows = find(legal);
    if isempty(rows)
        return;
    end
    gaps = normalizedDistances(ItX(rows,:),xf,Problem);
    ranks = arrayfun(@(r)referenceRank(W,xfRef,r),ItRef(rows));
    [~,order] = sortrows([gaps,ranks(:),rows],[1 2 3]);
    candidate = rows(order(1));
    rank = ranks(order(1));
end

function supported = locallySupportedPairs(Archive,EliteX,Problem,Options)
%LOCALLYSUPPORTEDPAIRS Use the pair gap as the parameter-free local radius.

    supported = false(numel(Archive.id),1);
    if isempty(EliteX) || isempty(Archive.id)
        return;
    end
    for row = 1 : numel(Archive.id)
        distance = normalizedDistances(EliteX,Archive.xf(row,:),Problem);
        supported(row) = any(distance <= Archive.gap(row)+ ...
            Options.pairDuplicateTolerance);
    end
end

function Archive = refreshPairMetadata( ...
        Archive,EliteY,EliteFitness,W,RefScale,Problem)
%REFRESHPAIRMETADATA Recompute gap, angular rank, and support fitness.

    if isempty(Archive.id)
        return;
    end
    Archive.ref = AssignReferenceVectors_CBS(Archive.yf,W,RefScale);
    xiRef = AssignReferenceVectors_CBS(Archive.yi,W,RefScale);
    for row = 1 : numel(Archive.id)
        Archive.gap(row) = normalizedDistance( ...
            Archive.xf(row,:),Archive.xi(row,:),Problem);
        Archive.rank(row) = referenceRank(W,Archive.ref(row),xiRef(row));
        Archive.fitness(row) = pairFitness( ...
            Archive.yf(row,:),EliteY,EliteFitness);
    end
end

function valid = legalPairRows(Archive,EliteY,W,RefScale,Problem,Options)
%LEGALPAIRROWS Enforce box, dominance, and angular-neighborhood legality.

    valid = validArchiveRows(Archive,size(W,1),Problem);
    if isempty(Archive.id)
        return;
    end
    valid = valid & ~dominatedByAny( ...
        Archive.yi,EliteY,RefScale,1e-12) & ...
        ~pairDominatesRows(Archive.yf,Archive.yi,RefScale,1e-12);
    xiRef = AssignReferenceVectors_CBS(Archive.yi,W,RefScale);
    for row = reshape(find(valid),1,[])
        valid(row) = ismember(xiRef(row),neighborRefs( ...
            W,Archive.ref(row),Options.pairNeighborRefCount));
    end
end

function [Archive,added] = addOrActivatePair(Archive,xf,xi,yf,yi,ref, ...
        fitness,W,RefScale,Problem,currentFE,Options,varargin)
%ADDORACTIVATEPAIR Resume an eligible exact pair or allocate a fresh ID.

    added = 0;
    same = Archive.resumeEligible & Archive.ref == ref & ...
        sameDecisionRows(Archive.xf,xf,Problem, ...
        Options.pairDuplicateTolerance) & ...
        sameDecisionRows(Archive.xi,xi,Problem, ...
        Options.pairDuplicateTolerance);
    row = find(same,1);
    xiRef = AssignReferenceVectors_CBS(yi,W,RefScale);
    if nargin >= 13 && ~isempty(varargin)
        rank = double(varargin{1});
    else
        rank = referenceRank(W,ref,xiRef);
    end
    if ~isempty(row)
        Archive.active(row) = true;
        Archive.age(row) = 0;
        % Mere current-P1 support keeps a pair active but is not a new
        % endpoint observation. Preserve lastFE so it cannot spuriously
        % trigger warm-start retraining.
        Archive.rank(row) = rank;
        Archive.fitness(row) = fitness;
        return;
    end
    Archive.id(end+1,1) = Archive.nextId;
    Archive.nextId = Archive.nextId+1;
    Archive.xf(end+1,:) = xf;
    Archive.xi(end+1,:) = xi;
    Archive.yf(end+1,:) = yf;
    Archive.yi(end+1,:) = yi;
    Archive.ref(end+1,1) = ref;
    Archive.gap(end+1,1) = normalizedDistance(xf,xi,Problem);
    Archive.rank(end+1,1) = rank;
    Archive.fitness(end+1,1) = fitness;
    Archive.age(end+1,1) = 0;
    Archive.lastFE(end+1,1) = double(currentFE);
    Archive.active(end+1,1) = true;
    Archive.resumeEligible(end+1,1) = true;
    added = 1;
end

function Candidates = emptyPairCandidates(D,M)
%EMPTYPAIRCANDIDATES New pairs requested by attributed feasible feedback.

    Candidates = struct('xf',zeros(0,D),'xi',zeros(0,D), ...
        'yf',zeros(0,M),'yi',zeros(0,M),'ref',zeros(0,1), ...
        'fitness',zeros(0,1),'rank',zeros(0,1));
end

function Candidates = appendPairCandidate(Candidates,xf,xi,yf,yi,ref, ...
        fitness,rank)
%APPENDPAIRCANDIDATE Append one prospective fresh-ID pair.

    Candidates.xf(end+1,:) = xf;
    Candidates.xi(end+1,:) = xi;
    Candidates.yf(end+1,:) = yf;
    Candidates.yi(end+1,:) = yi;
    Candidates.ref(end+1,1) = ref;
    Candidates.fitness(end+1,1) = fitness;
    Candidates.rank(end+1,1) = rank;
end

function fitness = pairFitness(y,EliteY,varargin)
%PAIRFITNESS Deterministic SPEA2 fitness used only as a late tie-break.

    if ~isempty(EliteY) && ~isempty(varargin)
        eliteFitness = reshape(double(varargin{1}),[],1);
        exact = all(abs(double(EliteY)-double(y)) <= 1e-12,2);
        if any(exact) && numel(eliteFitness) == size(EliteY,1)
            fitness = min(eliteFitness(exact));
            return;
        end
    end
    objectives = [double(EliteY);double(y)];
    values = reshape(double(CalFitness_CBS(objectives)),[],1);
    fitness = values(end);
end

function [Data,Gate,TrainC,QueryRefs,BMem] = buildTrainingData( ...
        Archive,W,Problem,Options)
%BUILDTRAININGDATA Build complete normalized endpoint pairs.

    Options = fillOptions(Options);
    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    rows = find(Archive.active & all(isfinite(Archive.xf),2) & ...
        all(isfinite(Archive.xi),2) & Archive.ref >= 1 & ...
        Archive.ref <= size(W,1));
    Data = emptyTrainingData(Problem.D,Problem.M);
    if ~isempty(rows)
        lower = double(Problem.lower);
        span = decisionSpan(Problem);
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
    if numel(unique(Data.ref)) ~= Data.count
        error('CBSPairGuide:NonUniqueTrainingReference', ...
            'Every active reference must own exactly one training pair.');
    end
    TrainC = [Data.cF;Data.cI];
    QueryRefs = Data.ref;
    Gate = struct('effective',Data.count,'active',Data.count, ...
        'regions',Data.count,'pairs',Data.count, ...
        'eligible',Data.count >= Options.pairMinPairs);
    BMem = archiveAsBoundaryMemory(Archive,Problem);
end

function [QueryC,Info] = buildQueryContexts(Archive,W,Options,totalBudget)
%BUILDQUERYCONTEXTS Stratify randomized s=0 queries over active pairs.

    Options = fillOptions(Options); %#ok<NASGU>
    Info = struct('refs',zeros(0,1),'pairIds',zeros(0,1));
    QueryC = zeros(0,size(W,2)+1);
    if isempty(Archive) || ~isstruct(Archive)
        return;
    end
    activeRows = find(Archive.active & Archive.ref >= 1 & ...
        Archive.ref <= size(W,1));
    totalBudget = max(0,round(double(totalBudget)));
    if isempty(activeRows) || totalBudget == 0
        return;
    end
    rows = balancedRows(activeRows,totalBudget);
    refs = Archive.ref(rows);
    QueryC = [double(W(refs,:)),zeros(numel(refs),1)];
    Info.refs = refs;
    Info.pairIds = Archive.id(rows);
end

function [SelectedDecs,SelectedRefs,MatchedIds,Trace] = ...
        selectCandidates(RawDec,SampleInfo,Archive,Population,Fitness,W, ...
        RefScale,Problem,Options)
%SELECTCANDIDATES Rho gate, one raw per pair, then objective-only corridor.

    Options = fillOptions(Options);
    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    RawDec = double(RawDec);
    refs = reshape(double(SampleInfo.refs),[],1);
    if isfield(SampleInfo,'pairIds')
        pairIds = reshape(double(SampleInfo.pairIds),[],1);
    else
        pairIds = zeros(0,1);
    end
    rowCount = min([size(RawDec,1),numel(refs),numel(pairIds)]);
    RawDec = RawDec(1:rowCount,:);
    refs = refs(1:rowCount);
    pairIds = pairIds(1:rowCount);
    Trace = emptyPoolTrace();
    Trace.active = rowCount > 0;
    Trace.rawCount = rowCount;
    Trace.percentile = nan(rowCount,1);
    Trace.rawConditions = numel(unique(refs));
    Trace.trainConditions = numel(unique(Archive.ref(Archive.active)));
    SelectedDecs = zeros(0,Problem.D);
    SelectedRefs = zeros(0,1);
    MatchedIds = zeros(0,1);
    if rowCount == 0 || isempty(Population)
        return;
    end

    lower = double(Problem.lower);
    upper = double(Problem.upper);
    span = decisionSpan(Problem);
    Xn = (RawDec-lower)./span;
    validRaw = all(isfinite(RawDec),2) & isfinite(refs) & ...
        refs == fix(refs) & refs >= 1 & refs <= size(W,1) & ...
        isfinite(pairIds) & pairIds == fix(pairIds) & pairIds > 0 & ...
        all(RawDec >= lower-1e-12,2) & all(RawDec <= upper+1e-12,2);

    PopX = double(Population.decs);
    PopY = double(Population.objs);
    PopC = double(Population.cons);
    Fitness = reshape(double(Fitness),[],1);
    populationCount = min([size(PopX,1),size(PopY,1), ...
        size(PopC,1),numel(Fitness)]);
    PopX = PopX(1:populationCount,:);
    PopY = PopY(1:populationCount,:);
    PopC = PopC(1:populationCount,:);
    Fitness = Fitness(1:populationCount);
    validPopulation = all(isfinite(PopX),2) & all(isfinite(PopY),2) & ...
        all(isfinite(PopC),2);
    elite = find(validPopulation & constraintViolation(PopC) <= 0 & ...
        Fitness < 1);
    if isempty(elite)
        Trace.supportFailures = nnz(validRaw);
        return;
    end
    eliteRefs = AssignReferenceVectors_CBS(PopY(elite,:),W,RefScale);
    EliteNorm = (PopX(elite,:)-lower)./span;

    rawPairRows = zeros(rowCount,1);
    rawRho = inf(rowCount,1);
    rawGap = nan(rowCount,1);
    rawParentError = nan(rowCount,1);
    rawGuideError = nan(rowCount,1);
    archiveCount = numel(Archive.id);
    supportState = zeros(archiveCount,1);
    cachedPairGap = nan(archiveCount,1);
    cachedParentError = nan(archiveCount,1);
    cachedXiNorm = zeros(archiveCount,Problem.D);
    matchFailures = 0;
    supportFailures = 0;
    for raw = reshape(find(validRaw),1,[])
        pair = find(Archive.id == pairIds(raw) & Archive.active,1);
        if isempty(pair) || refs(raw) ~= Archive.ref(pair)
            matchFailures = matchFailures+1;
            continue;
        end
        if supportState(pair) == 0
            neighborhood = neighborRefs( ...
                W,Archive.ref(pair),Options.pairNeighborRefCount);
            local = find(ismember(eliteRefs,neighborhood));
            if isempty(local)
                supportState(pair) = -1;
            else
                xfNorm = (Archive.xf(pair,:)-lower)./span;
                xiNorm = (Archive.xi(pair,:)-lower)./span;
                distance = sqrt(sum((EliteNorm(local,:)-xfNorm).^2,2));
                nearest = local(distance <= min(distance)+1e-12);
                [~,order] = sortrows( ...
                    [Fitness(elite(nearest)),elite(nearest)],[1 2]);
                parentLocal = nearest(order(1));
                cachedPairGap(pair) = norm(xiNorm-xfNorm);
                cachedParentError(pair) = ...
                    norm(EliteNorm(parentLocal,:)-xfNorm);
                cachedXiNorm(pair,:) = xiNorm;
                supportState(pair) = 1;
            end
        end
        if supportState(pair) < 0
            supportFailures = supportFailures+1;
            continue;
        end
        pairGap = cachedPairGap(pair);
        parentError = cachedParentError(pair);
        guideError = norm(Xn(raw,:)-cachedXiNorm(pair,:));
        rho = (parentError+guideError)/(pairGap+eps);
        if ~isfinite(rho) || pairGap <= 1e-12 || rho >= 1
            continue;
        end
        rawPairRows(raw) = pair;
        rawRho(raw) = rho;
        rawGap(raw) = pairGap;
        rawParentError(raw) = parentError;
        rawGuideError(raw) = guideError;
    end

    representatives = zeros(0,1);
    for id = reshape(unique(pairIds(rawPairRows > 0),'stable'),1,[])
        rows = find(rawPairRows > 0 & pairIds == id);
        [~,order] = sortrows([rawRho(rows),rows],[1 2]);
        representatives(end+1,1) = rows(order(1)); %#ok<AGROW>
    end
    Trace.pairRepresentativeCount = numel(representatives);
    Trace.rhoFailures = nnz(validRaw)-numel(find(rawPairRows > 0));
    if isempty(representatives)
        Trace.matchFailures = matchFailures;
        Trace.supportFailures = supportFailures;
        Trace.rawNearDuplicateRate = nearDuplicateRate( ...
            Xn(all(isfinite(Xn),2),:),Options.pairDuplicateTolerance);
        return;
    end

    validObjectives = all(isfinite(PopY),2);
    occupancy = zeros(size(W,1),1);
    if any(validObjectives)
        populationRefs = AssignReferenceVectors_CBS( ...
            PopY(validObjectives,:),W,RefScale);
        occupancy = accumarray(populationRefs,1,[size(W,1),1],@sum,0);
    end
    pairRows = rawPairRows(representatives);
    key = [occupancy(Archive.ref(pairRows)),rawRho(representatives), ...
        -Archive.gap(pairRows),Archive.id(pairRows)];
    [~,priority] = sortrows(key,[1 2 3 4]);
    representatives = representatives(priority);
    quota = Options.guideQuota;
    objectiveLimit = min([numel(representatives),2*quota, ...
        Options.objectiveBudget]);
    shortlist = representatives(1:objectiveLimit);
    pairRows = rawPairRows(shortlist);
    Trace.objectiveCandidateCount = objectiveLimit;
    Trace.objectiveFE = objectiveLimit;
    Trace.ObjFE = objectiveLimit;
    if objectiveLimit == 0
        Trace.matchFailures = matchFailures;
        Trace.supportFailures = supportFailures;
        return;
    end
    GObj = double(Problem.CalObj(RawDec(shortlist,:)));
    if size(GObj,1) ~= objectiveLimit || size(GObj,2) ~= Problem.M
        error('CBSPairGuide:ObjectiveOnlyShapeMismatch', ...
            'Problem.CalObj must return one M-objective row per donor.');
    end
    yF = Archive.yf(pairRows,:);
    yI = Archive.yi(pairRows,:);
    localDominance = ~pairDominatesRows( ...
        GObj,yI,RefScale,1e-12) & ...
        ~pairDominatesRows(yF,GObj,RefScale,1e-12);
    GNorm = normalizeObjectives(GObj,RefScale);
    FNorm = normalizeObjectives(yF,RefScale);
    INorm = normalizeObjectives(yI,RefScale);
    corridor = all(GNorm >= min(FNorm,INorm)-1e-12 & ...
        GNorm <= max(FNorm,INorm)+1e-12,2);
    % Deliberately no global feasible-elite dominance filter for G.
    passed = all(isfinite(GObj),2) & localDominance & corridor;
    accepted = find(passed,quota,'first');
    selectedRaw = shortlist(accepted);
    selectedPairs = rawPairRows(selectedRaw);
    SelectedDecs = RawDec(selectedRaw,:);
    SelectedRefs = Archive.ref(selectedPairs);
    MatchedIds = Archive.id(selectedPairs);

    Trace.keptCount = numel(selectedRaw);
    Trace.keptConditions = numel(unique(SelectedRefs));
    Trace.keepIdx = selectedRaw;
    Trace.validCount = numel(selectedRaw);
    Trace.matchFailures = matchFailures;
    Trace.supportFailures = supportFailures;
    Trace.matchedPairIds = MatchedIds;
    Trace.selectedRho = rawRho(selectedRaw);
    Trace.selectedPairGap = rawGap(selectedRaw);
    Trace.selectedParentError = rawParentError(selectedRaw);
    Trace.selectedGuideError = rawGuideError(selectedRaw);
    Trace.localDominancePass = nnz(localDominance);
    Trace.corridorPass = nnz(corridor);
    Trace.rawNearDuplicateRate = nearDuplicateRate( ...
        Xn(all(isfinite(Xn),2),:),Options.pairDuplicateTolerance);
    Trace.keptNearDuplicateRate = nearDuplicateRate( ...
        Xn(selectedRaw,:),Options.pairDuplicateTolerance);
end

function Archive = pruneArchive(Archive,Options,refCount,Problem)
%PRUNEARCHIVE Enforce stale deletion and one canonical pair per reference.

    keep = validArchiveRows(Archive,refCount,Problem) & ...
        (Archive.active | Archive.age < Options.pairInactiveMaxAge);
    candidates = find(keep);
    selected = zeros(0,1);
    for ref = reshape(unique(Archive.ref(candidates),'stable'),1,[])
        local = candidates(Archive.ref(candidates) == ref);
        active = local(Archive.active(local));
        if ~isempty(active)
            local = active;
        end
        key = [Archive.gap(local),Archive.rank(local), ...
            Archive.fitness(local),-Archive.lastFE(local),Archive.id(local)];
        [~,order] = sortrows(key,[1 2 3 4 5]);
        selected(end+1,1) = local(order(1)); %#ok<AGROW>
    end
    Archive = subsetArchive(Archive,sort(selected));
    if numel(unique(Archive.ref)) ~= numel(Archive.ref) || ...
            numel(Archive.id) > refCount
        error('CBSPairGuide:ArchiveCapacityExceeded', ...
            'The archive must contain at most one pair per reference.');
    end
end

function valid = validArchiveRows(Archive,refCount,Problem)
%VALIDARCHIVEROWS Reject corrupt retained pair rows.

    lower = double(Problem.lower);
    upper = double(Problem.upper);
    valid = isfinite(Archive.id) & Archive.id == fix(Archive.id) & ...
        Archive.id > 0 & isfinite(Archive.ref) & ...
        Archive.ref == fix(Archive.ref) & Archive.ref >= 1 & ...
        Archive.ref <= refCount & isfinite(Archive.gap) & ...
        Archive.gap >= 0 & isfinite(Archive.rank) & Archive.rank >= 1 & ...
        isfinite(Archive.fitness) & isfinite(Archive.age) & ...
        Archive.age >= 0 & isfinite(Archive.lastFE) & ...
        isfinite(Archive.resumeEligible) & ...
        all(isfinite(Archive.xf),2) & all(isfinite(Archive.xi),2) & ...
        all(isfinite(Archive.yf),2) & all(isfinite(Archive.yi),2) & ...
        all(Archive.xf >= lower-1e-12,2) & ...
        all(Archive.xf <= upper+1e-12,2) & ...
        all(Archive.xi >= lower-1e-12,2) & ...
        all(Archive.xi <= upper+1e-12,2);
end

function BMem = archiveAsBoundaryMemory(Archive,Problem)
%ARCHIVEASBOUNDARYMEMORY Expose endpoints and persistent attribution IDs.

    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    BMem = struct('id',Archive.id,'ref',Archive.ref,'gap',Archive.gap, ...
        'x_b',Archive.xf,'y_b',Archive.yf,'x_i',Archive.xi, ...
        'y_i',Archive.yi,'active',Archive.active, ...
        'lastFE',Archive.lastFE);
end

function Archive = ensureArchive(Archive,D,M)
%ENSUREARCHIVE Normalize empty and earlier archive schemas.

    Empty = emptyArchive(D,M);
    if isempty(Archive) || ~isstruct(Archive)
        Archive = Empty;
        return;
    end
    count = size(Archive.xf,1);
    names = fieldnames(Empty);
    for i = 1 : numel(names)
        name = names{i};
        if ~isfield(Archive,name)
            if strcmp(name,'rank')
                Archive.rank = ones(count,1);
            elseif strcmp(name,'fitness')
                Archive.fitness = realmax*ones(count,1);
            elseif strcmp(name,'resumeEligible')
                Archive.resumeEligible = true(count,1);
            else
                Archive.(name) = Empty.(name);
            end
        end
    end
    vectorFields = {'id','ref','gap','rank','fitness','age','lastFE', ...
        'active','resumeEligible'};
    for i = 1 : numel(vectorFields)
        name = vectorFields{i};
        if size(Archive.(name),1) ~= count
            if strcmp(name,'rank')
                Archive.(name) = ones(count,1);
            elseif strcmp(name,'fitness')
                Archive.(name) = realmax*ones(count,1);
            elseif strcmp(name,'resumeEligible')
                Archive.(name) = true(count,1);
            else
                Archive.(name) = Empty.(name);
            end
        end
    end
    if ~isscalar(Archive.nextId) || ~isfinite(Archive.nextId)
        Archive.nextId = max([0;Archive.id])+1;
    end
end

function Archive = emptyArchive(D,M)
%EMPTYARCHIVE Construct the persistent unified pair schema.

    Archive = struct('id',zeros(0,1),'xf',zeros(0,D), ...
        'xi',zeros(0,D),'yf',zeros(0,M),'yi',zeros(0,M), ...
        'ref',zeros(0,1),'gap',zeros(0,1),'rank',zeros(0,1), ...
        'fitness',zeros(0,1),'age',zeros(0,1), ...
        'lastFE',zeros(0,1),'active',false(0,1), ...
        'resumeEligible',false(0,1),'nextId',1);
end

function Data = emptyTrainingData(D,M)
%EMPTYTRAININGDATA Complete-pair endpoint data; ID remains metadata.

    Data = struct('xF',zeros(0,D),'xI',zeros(0,D), ...
        'delta',zeros(0,D),'w',zeros(0,M),'ref',zeros(0,1), ...
        'id',zeros(0,1),'lastFE',zeros(0,1), ...
        'cF',zeros(0,M+1),'cI',zeros(0,M+1),'count',0);
end

function Trace = emptyMemoryTrace()
%EMPTYMEMORYTRACE Pair lifecycle diagnostics with legacy field compatibility.

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
%EMPTYPOOLTRACE Donor filtering plus strict objective-only cost counters.

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
        'validCount',0,'matchFailures',0,'supportFailures',0, ...
        'rhoFailures',0,'pairRepresentativeCount',0, ...
        'objectiveCandidateCount',0,'objectiveFE',0,'ObjFE',0, ...
        'constraintFE',0, ...
        'localDominancePass',0,'corridorPass',0, ...
        'matchedPairIds',zeros(0,1),'selectedRho',zeros(0,1), ...
        'selectedPairGap',zeros(0,1), ...
        'selectedParentError',zeros(0,1), ...
        'selectedGuideError',zeros(0,1));
end

function Archive = subsetArchive(Archive,rows)
%SUBSETARCHIVE Subset row fields while preserving the next unique ID.

    rows = reshape(rows,[],1);
    fields = {'id','xf','xi','yf','yi','ref','gap','rank','fitness', ...
        'age','lastFE','active','resumeEligible'};
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
    same = normalizedDistances(X,x,Problem) <= tolerance;
end

function distance = normalizedDistances(X,x,Problem)
%NORMALIZEDDISTANCES Decision-box Euclidean distances to one row.

    if isempty(X)
        distance = zeros(0,1);
    else
        distance = sqrt(sum(((double(X)-double(x))./ ...
            decisionSpan(Problem)).^2,2));
    end
end

function distance = normalizedDistance(a,b,Problem)
%NORMALIZEDDISTANCE Decision-box Euclidean distance between paired rows.

    distance = sqrt(sum(((double(a)-double(b))./ ...
        decisionSpan(Problem)).^2,2));
end

function distance = minimumNormalizedDistance(A,B,Problem)
%MINIMUMNORMALIZEDDISTANCE Minimum decision-box distance for each A row.

    if isempty(A)
        distance = zeros(0,1);
    elseif isempty(B)
        distance = inf(size(A,1),1);
    else
        lower = double(Problem.lower);
        span = decisionSpan(Problem);
        An = (double(A)-lower)./span;
        Bn = (double(B)-lower)./span;
        distance2 = max(0,sum(An.^2,2)+sum(Bn.^2,2)'-2*(An*Bn'));
        distance = sqrt(min(distance2,[],2));
    end
end

function span = decisionSpan(Problem)
%DECISIONSPAN Positive decision-box scale.

    span = double(Problem.upper)-double(Problem.lower);
    span(span <= eps) = 1;
end

function value = constraintViolation(C)
%CONSTRAINTVIOLATION Sum positive constraints rowwise.

    if isempty(C)
        value = zeros(size(C,1),1);
    else
        value = sum(max(0,double(C)),2);
    end
end

function refs = neighborRefs(W,ref,totalCount)
%NEIGHBORREFS Angular neighborhood whose total includes its own direction.

    order = angularOrder(W,ref);
    total = min(size(W,1),max(1,round(double(totalCount))));
    refs = order(1:total);
end

function rank = referenceRank(W,baseRef,candidateRef)
%REFERENCERANK One-based angular rank with stable reference-ID tie-break.

    order = angularOrder(W,baseRef);
    rank = find(order == candidateRef,1);
    if isempty(rank)
        rank = Inf;
    end
end

function order = angularOrder(W,ref)
%ANGULARORDER Sort unit reference vectors by cosine angle.

    W = double(W);
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    angular = 1-Wn*Wn(ref,:)';
    [~,order] = sortrows([angular,(1:size(W,1))'],[1 2]);
end

function rows = balancedRows(values,total)
%BALANCEDROWS Uniform randomized allocation over active pair rows.

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

function dominated = dominatedByAny(Y,Candidates,Scale,tolerance)
%DOMINATEDBYANY True when any candidate Pareto-dominates each Y row.

    dominated = false(size(Y,1),1);
    if isempty(Y) || isempty(Candidates)
        return;
    end
    Yn = normalizeObjectives(Y,Scale);
    Cn = normalizeObjectives(Candidates,Scale);
    for row = 1 : size(Yn,1)
        dominated(row) = any(all(Cn <= Yn(row,:)+tolerance,2) & ...
            any(Cn < Yn(row,:)-tolerance,2));
    end
end

function dominates = pairDominatesRows(A,B,Scale,tolerance)
%PAIRDOMINATESROWS Rowwise normalized Pareto dominance.

    if isempty(A) || isempty(B)
        dominates = false(max(size(A,1),size(B,1)),1);
        return;
    end
    if size(A,1) ~= size(B,1)
        error('CBSPairGuide:DominanceShapeMismatch', ...
            'Rowwise dominance inputs must have equal row counts.');
    end
    An = normalizeObjectives(A,Scale);
    Bn = normalizeObjectives(B,Scale);
    dominates = all(An <= Bn+tolerance,2) & ...
        any(An < Bn-tolerance,2);
end

function Yn = normalizeObjectives(Y,Scale)
%NORMALIZEOBJECTIVES Apply the shared reference-vector objective scale.

    Yn = (double(Y)-reshape(double(Scale.minimum),1,[]))./ ...
        reshape(double(Scale.span),1,[]);
end

function Y = finiteObjectiveRows(Y,M)
%FINITEOBJECTIVEROWS Preserve objective width while removing invalid rows.

    if isempty(Y)
        Y = zeros(0,M);
    else
        Y = double(Y);
        Y = Y(all(isfinite(Y),2),:);
    end
end

function rate = nearDuplicateRate(X,tolerance)
%NEARDUPLICATERATE Diagnostic only; raw donors are never rejected for it.

    if isempty(X)
        rate = NaN;
        return;
    end
    signature = round(double(X)/max(double(tolerance),eps));
    rate = 1-size(unique(signature,'rows'),1)/size(X,1);
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

function Options = fillOptions(Options)
%FILLOPTIONS Fixed minimal PairGuide archive and donor settings.

    Options = defaultOption(Options,'pairArchivePerRef',1);
    Options = defaultOption(Options,'pairInactiveMaxAge',10);
    Options = defaultOption(Options,'pairNeighborRefCount',5);
    Options = defaultOption(Options,'pairMinPairs',32);
    Options = defaultOption(Options,'pairDuplicateTolerance',1e-6);
    Options = defaultOption(Options,'pairImprovementTolerance',1e-12);
    Options = defaultOption(Options,'guideQuota',20);
    Options = defaultOption(Options,'objectiveBudget',Inf);
    integer = {'pairArchivePerRef','pairInactiveMaxAge', ...
        'pairNeighborRefCount','pairMinPairs','guideQuota'};
    for i = 1 : numel(integer)
        name = integer{i};
        Options.(name) = max(0,round(double(Options.(name))));
    end
    if Options.pairArchivePerRef ~= 1
        error('CBSPairGuide:ArchivePerReferenceMustBeOne', ...
            'PairGuide requires exactly one archive pair per reference.');
    end
    Options.pairNeighborRefCount = max(1,Options.pairNeighborRefCount);
    Options.pairMinPairs = max(1,Options.pairMinPairs);
    Options.pairDuplicateTolerance = max(eps,double( ...
        Options.pairDuplicateTolerance));
    Options.pairImprovementTolerance = max(0,double( ...
        Options.pairImprovementTolerance));
    Options.objectiveBudget = max(0,floor(double(Options.objectiveBudget)));
end

function S = defaultOption(S,name,value)
%DEFAULTOPTION Fill one missing structure field.

    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end
