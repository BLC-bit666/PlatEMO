function varargout = PairBoundaryArchive_RC(action,varargin)
%PAIRBOUNDARYARCHIVE_RC Retain at most 500 admissible real boundary pairs.
%   Conditions are always (w,1) for a truly evaluated feasible endpoint
%   and (w,0) for a truly evaluated infeasible endpoint. Pair IDs are
%   attribution metadata only and never enter the generator condition.
%   Screen the complete feasible pool before matching. Infeasible endpoints
%   dominated by any feasible candidate cannot enter or remain in a pair.
%   Unqualified history is deleted; 1000 endpoint slots is only an upper bound.

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

function [Archive,RefScale,Trace] = updateArchive(Archive,P1, ...
        Evaluated,W,Problem,Options,Feedback,currentFE)
%UPDATEARCHIVE Preserve real evidence; update original pairs before selection.

    Options = fillOptions(Options);
    span = decisionSpan(Problem);
    neighbors = referenceNeighborhoods(W);
    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    Historical = Archive;
    Trace = emptyMemoryTrace();
    Trace.evaluatedInputPoints = numel(Evaluated);
    Trace.candidateInputPoints = 2*numel(Historical.id)+numel(Evaluated);
    Trace.events = struct('pairId',{},'source',{},'sourceRow',{}, ...
        'originalPair',{},'feasible',{},'beforeGap',{},'afterGap',{}, ...
        'decision',{},'objectives',{},'fe',{});
    [fx,fy,ix,iy] = evaluatedRows(Evaluated,zeros(0,Problem.D),Problem,Options);
    [gx,gy,gc,ids] = validFeedbackRows(Feedback,Problem);
    if isempty(gx)
        ofx = fx; ofy = fy; oix = ix; oiy = iy;
    else
        [ofx,ofy,oix,oiy] = evaluatedRows(Evaluated,gx,Problem,Options);
    end
    objectives = [fy;iy;Archive.yf;Archive.yi];
    [~,RefScale] = AssignReferenceVectors_CBS(objectives,W);
    Archive = refreshPairMetadata(Archive,[],[],W,RefScale,Problem);
    oldRefs = Historical.ref;
    ordinaryX = [ofx;oix;Historical.xf;Historical.xi];
    ordinaryY = [ofy;oiy;Historical.yf;Historical.yi];
    ordinaryFeasible = [true(size(ofx,1),1);false(size(oix,1),1); ...
        true(size(Historical.xf,1),1);false(size(Historical.xi,1),1)];
    ordinaryIds = zeros(size(ordinaryX,1),1);
    % Without guided feedback the ordinary shadow equals the actual update.
    if ~isempty(ids)
        Shadow = tightenRows(Archive,ordinaryX,ordinaryY,ordinaryFeasible, ...
            ordinaryIds,"ordinary",W,RefScale,span,neighbors,Options,currentFE,Trace.events);
        shadowGap = Shadow.gap;
        clear Shadow;
    end
    for k = 1:numel(ids)
        row = find(Archive.id == ids(k),1);
        if ~isempty(row)
            [Archive,event] = replaceEndpoint(Archive,row,gx(k,:),gy(k,:), ...
                constraintViolation(gc(k,:)) <= 0,span,Options, ...
                "guided",k,true,currentFE);
            if ~isempty(event); Trace.events(end+1,1) = event; end
        end
    end
    % The same evaluated children may subsequently improve other local pairs.
    [Archive,Trace.events] = tightenRows(Archive,gx,gy, ...
        constraintViolation(gc) <= 0,ids,"guided",W,RefScale, ...
        span,neighbors,Options,currentFE,Trace.events);
    [Archive,Trace.events] = tightenRows(Archive,ordinaryX,ordinaryY,ordinaryFeasible, ...
        ordinaryIds,"ordinary",W,RefScale,span,neighbors,Options,currentFE,Trace.events);
    if isempty(ids); shadowGap = Archive.gap; end
    Trace.netGapReduction = sum(shadowGap-Archive.gap);
    Trace.netGapById = [Archive.id,shadowGap,Archive.gap];
    Archive = refreshPairMetadata(Archive,[],[],W,RefScale,Problem);
    Trace.migrated = nnz(Archive.ref ~= oldRefs);
    % Both retained sides participate in the same local re-pairing pool.
    fx = [fx;Historical.xf]; fy = [fy;Historical.yf];
    ix = [ix;Historical.xi]; iy = [iy;Historical.yi];
    [fx,uniqueRows] = unique(fx,'rows','stable'); fy = fy(uniqueRows,:);
    [ix,uniqueRows] = unique(ix,'rows','stable'); iy = iy(uniqueRows,:);
    % Classify complete pools before nearest-neighbor matching. An inadmissible
    % nearer endpoint must never hide an admissible farther one.
    fronts = zeros(0,1);
    if ~isempty(fy); fronts = reshape(NDSort(fy,inf),[],1); end
    p1Objs = zeros(0,Problem.M);
    if ~isempty(P1)
        feasible = all(P1.cons <= 0,2) & all(isfinite(P1.objs),2);
        p1Objs = double(P1(feasible).objs);
    end
    if Options.archiveFrontDepth > 0
        feasibleEligible = fronts <= Options.archiveFrontDepth;
    else
        feasibleEligible = ~dominatedByAny(fy,p1Objs,RefScale,1e-12);
    end
    infeasibleEligible = ~dominatedByAny(iy,fy(fronts == 1,:),RefScale,1e-12);
    Trace.feasiblePool = size(fx,1);
    Trace.feasiblePoolEligible = nnz(feasibleEligible);
    Trace.infeasiblePool = size(ix,1);
    Trace.infeasiblePoolEligible = nnz(infeasibleEligible);
    frefs = AssignReferenceVectors_CBS(fy,W,RefScale);
    irefs = AssignReferenceVectors_CBS(iy,W,RefScale);
    Candidate = emptyArchive(Problem.D,Problem.M);
    % At most one nearest opposite endpoint per feasible candidate.
    Candidate.ref = zeros(size(fx,1),1);
    Candidate.xf = zeros(size(fx)); Candidate.yf = zeros(size(fy));
    Candidate.xi = zeros(size(fx)); Candidate.yi = zeros(size(fy));
    Candidate.gap = zeros(size(fx,1),1);
    count = 0;
    for ref = reshape(unique(frefs(feasibleEligible)),1,[])
        candidates = find(frefs == ref & feasibleEligible);
        local = find(neighbors(ref,irefs)' & infeasibleEligible);
        for f = reshape(candidates,1,[])
            gaps = sqrt(sum(((ix(local,:)-fx(f,:))./span).^2,2));
            admissible = gaps > Options.pairImprovementTolerance;
            eligible = local(admissible);
            gaps = gaps(admissible);
            if isempty(eligible); continue; end
            [gap,j] = min(gaps);
            count = count+1;
            Candidate.ref(count) = ref;
            Candidate.xf(count,:) = fx(f,:); Candidate.yf(count,:) = fy(f,:);
            Candidate.xi(count,:) = ix(eligible(j),:); Candidate.yi(count,:) = iy(eligible(j),:);
            Candidate.gap(count) = gap;
        end
    end
    % Provisional ordering keys only. Allocate IDs to retained new pairs below.
    firstNewId = Archive.nextId;
    Candidate.id = firstNewId+(0:size(fx,1)-1)';
    Candidate.active = false(size(fx,1),1);
    Candidate = subsetArchive(Candidate,(1:count)');
    fields = {'id','ref','xf','yf','xi','yi','gap','active'};
    for k = 1:numel(fields)
        Archive.(fields{k}) = [Archive.(fields{k});Candidate.(fields{k})];
    end
    Archive = refreshPairMetadata(Archive,[],[],W,RefScale,Problem);
    valid = validArchiveRows(Archive,size(W,1),Problem);
    Trace.invalidRemoved = nnz(~valid);
    Archive = subsetArchive(Archive,find(valid));
    % Identical endpoint pairs retain the oldest surviving lineage ID.
    [~,byId] = sort(Archive.id);
    [~,distinct] = unique([Archive.ref(byId),Archive.xf(byId,:),Archive.xi(byId,:)],'rows','stable');
    Trace.duplicateRemoved = numel(Archive.id)-numel(distinct);
    Archive = subsetArchive(Archive,sort(byId(distinct)));
    % Tightening can merge distinct lineages into the same pair. The original
    % legal pairs are already represented by Historical's 1000 input slots;
    % keep them as alternatives so an occupied slot is not lost by merging.
    Trace.restoredCapacity = 0;
    minimumRetained = min(Options.pairArchiveCapacity,numel(Historical.id));
    if numel(Archive.id) < minimumRetained
        Reserve = refreshPairMetadata(Historical,[],[],W,RefScale,Problem);
        valid = validArchiveRows(Reserve,size(W,1),Problem);
        Reserve = subsetArchive(Reserve,find(valid));
        [~,distinct] = unique([Reserve.ref,Reserve.xf,Reserve.xi],'rows','stable');
        Reserve = subsetArchive(Reserve,distinct);
        duplicate = ismember([Reserve.ref,Reserve.xf,Reserve.xi], ...
            [Archive.ref,Archive.xf,Archive.xi],'rows');
        Reserve = subsetArchive(Reserve,find(~duplicate));
        Trace.restoredCapacity = numel(Reserve.id);
        Reserve.id = firstNewId+numel(Candidate.id)+(0:numel(Reserve.id)-1)';
        Reserve.active(:) = false;
        for k = 1:numel(fields)
            Archive.(fields{k}) = [Archive.(fields{k});Reserve.(fields{k})];
        end
    end
    p1Dominated = dominatedByAny(Archive.yf,p1Objs,RefScale,1e-12);
    [foundF,whereF] = ismember(Archive.xf,fx,'rows');
    [foundI,whereI] = ismember(Archive.xi,ix,'rows');
    assert(all(foundF & foundI),'CBSPairGuide:FrontPool', ...
        'An archive endpoint is missing from the complete evaluated pool.');
    frontRanks = fronts(whereF);
    eligibleF = feasibleEligible(whereF);
    eligibleI = infeasibleEligible(whereI);
    eligible = eligibleF & eligibleI;
    Trace.frontRejectedPairs = nnz(~eligibleF);
    Trace.infeasibleDominatedPairs = nnz(~eligibleI);
    Trace.eligibilityRemoved = nnz(~eligible);
    % Apply the same rule to tightened history and capacity reserves. Do not
    % resurrect a rejected pair just to occupy an otherwise empty slot.
    Archive = subsetArchive(Archive,find(eligible));
    p1Dominated = p1Dominated(eligible);
    frontRanks = frontRanks(eligible);
    eligible = true(numel(Archive.id),1);
    [Archive,Trace.capDropped,kept] = ...
        retainArchive(Archive,W,RefScale,Options.pairArchiveCapacity,eligible);
    Trace.p1DominatedPairs = nnz(p1Dominated(kept));
    Trace.archiveFrontDepth = Options.archiveFrontDepth;
    Trace.retainedFrontRanks = frontRanks(kept);
    Trace.eligiblePairs = nnz(eligible(kept));
    Trace.eligibilityRejectedPairs = nnz(~eligible(kept));
    new = Archive.id >= firstNewId;
    Trace.added = nnz(new);
    Archive.id(new) = firstNewId+(0:Trace.added-1)';
    Archive.nextId = firstNewId+Trace.added;
    Trace.removedIds = Historical.id(~ismember(Historical.id,Archive.id));
    Trace.removed = numel(Trace.removedIds);
    Trace.restored = nnz(ismember(Archive.id(Archive.active), ...
        Historical.id(~Historical.active)));
    events = Trace.events;
    if ~isempty(events)
        guided = [events.source] == "guided";
        feasible = [events.feasible];
        Trace.guidedTightenedFeasible = nnz(guided & feasible);
        Trace.guidedTightenedInfeasible = nnz(guided & ~feasible);
        Trace.ordinaryTightenedFeasible = nnz(~guided & feasible);
        Trace.ordinaryTightenedInfeasible = nnz(~guided & ~feasible);
    end
    Trace.tightenedFeasible = Trace.guidedTightenedFeasible+Trace.ordinaryTightenedFeasible;
    Trace.tightenedInfeasible = Trace.guidedTightenedInfeasible+Trace.ordinaryTightenedInfeasible;
    Trace.active = nnz(Archive.active);
    Trace.inactive = nnz(~Archive.active);
    Trace.retained = numel(Archive.id);
    Trace.afterCap = Trace.retained; Trace.paired = Trace.active;
    Trace.strong = Trace.active; Trace.weak = Trace.inactive;
    Trace.trueFeasible = size(fx,1); Trace.afterFront = nnz(feasibleEligible);
    Trace.pairGapMedian = finitePercentile(Archive.gap(Archive.active),0.5);
    Trace.pairGapP90 = finitePercentile(Archive.gap(Archive.active),0.9);
    Trace.archiveChanged = Trace.added+numel(events)+Trace.removed;
    Trace.previousCount = numel(Historical.id);
end

function [Archive,dropped,keep] = retainArchive(Archive,W,Scale,capacity,eligible)
%RETAINARCHIVE Direction champions first, then ranked alternatives in rounds.
%   Eligibility is computed once against the full current comparison pool.
%   Only qualified pairs reach capacity competition; no age/TTL is used.
    Archive.active(:) = false;
    keep = (1:numel(Archive.id))';
    rank = zeros(numel(Archive.id),1);
    for ref = reshape(unique(Archive.ref),1,[])
        rows = find(Archive.ref == ref);
        quality = directionQuality(Archive.yf(rows,:),W(ref,:),Scale);
        [~,order] = sortrows([~eligible(rows),quality,Archive.gap(rows),Archive.id(rows)]);
        rank(rows(order)) = (1:numel(rows))';
        Archive.active(rows(order(1))) = eligible(rows(order(1)));
    end
    dropped = max(0,numel(Archive.id)-capacity);
    if dropped > 0
        [~,order] = sortrows([~Archive.active,rank,Archive.ref,Archive.id]);
        keep = sort(order(1:capacity));
        Archive = subsetArchive(Archive,keep);
    end
end

function q = directionQuality(y,w,Scale)
%DIRECTIONQUALITY Current-frame weighted Chebyshev value.
    q = max(normalizeObjectives(y,Scale)./(double(w)+1e-12),[],2);
end

function [Archive,events] = tightenRows(Archive,X,Y,feasible,matchedIds, ...
        source,W,Scale,span,neighbors,Options,currentFE,events)
%TIGHTENROWS Every real row may improve pairs only in the five-ref neighborhood.
    if isempty(X) || isempty(Archive.id); return; end
    refs = AssignReferenceVectors_CBS(Y,W,Scale);
    localRows = cell(size(W,1),1);
    for ref = reshape(unique(refs),1,[])
        localRows{ref} = find(neighbors(ref,Archive.ref)');
    end
    recordEvents = nargout > 1;
    for k = 1:size(X,1)
        % Search outward from this evaluated point's reference. The inverse
        % relation can include more than five references near angular edges.
        local = localRows{refs(k)};
        local = local(Archive.id(local) ~= matchedIds(k));
        if feasible(k)
            gap = sqrt(sum(((X(k,:)-Archive.xi(local,:))./span).^2,2));
        else
            gap = sqrt(sum(((Archive.xf(local,:)-X(k,:))./span).^2,2));
        end
        before = Archive.gap(local);
        accepted = ~(gap <= 0 | gap >= before-Options.pairImprovementTolerance);
        rows = local(accepted); gap = gap(accepted); before = before(accepted);
        if feasible(k)
            Archive.xf(rows,:) = repmat(X(k,:),numel(rows),1);
            Archive.yf(rows,:) = repmat(Y(k,:),numel(rows),1);
        else
            Archive.xi(rows,:) = repmat(X(k,:),numel(rows),1);
            Archive.yi(rows,:) = repmat(Y(k,:),numel(rows),1);
        end
        Archive.gap(rows) = gap;
        if recordEvents
            for j = 1:numel(rows)
                events(end+1,1) = struct('pairId',Archive.id(rows(j)),'source',source, ...
                    'sourceRow',k,'originalPair',false,'feasible',feasible(k), ...
                    'beforeGap',before(j),'afterGap',gap(j),'decision',X(k,:), ...
                    'objectives',Y(k,:),'fe',currentFE); %#ok<AGROW>
            end
        end
    end
end

function [Archive,event] = replaceEndpoint(Archive,row,x,y,feasible, ...
        span,Options,source,sourceRow,original,currentFE)
%REPLACEENDPOINT Any real improvement beyond tolerance is accepted.
    event = struct('pairId',{},'source',{},'sourceRow',{}, ...
        'originalPair',{},'feasible',{},'beforeGap',{},'afterGap',{}, ...
        'decision',{},'objectives',{},'fe',{});
    oldGap = Archive.gap(row);
    if feasible
        newGap = sqrt(sum(((x-Archive.xi(row,:))./span).^2,2));
    else
        newGap = sqrt(sum(((Archive.xf(row,:)-x)./span).^2,2));
    end
    if newGap <= 0 || newGap >= oldGap-Options.pairImprovementTolerance
        return;
    end
    if feasible
        Archive.xf(row,:) = x; Archive.yf(row,:) = y;
    else
        Archive.xi(row,:) = x; Archive.yi(row,:) = y;
    end
    Archive.gap(row) = newGap;
    event = struct('pairId',Archive.id(row),'source',source, ...
        'sourceRow',sourceRow,'originalPair',original,'feasible',feasible, ...
        'beforeGap',oldGap,'afterGap',newGap,'decision',x,'objectives',y,'fe',currentFE);
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
        allIds == fix(allIds) & allIds >= 0 & ...
        all(allX >= lower-1e-12,2) & all(allX <= upper+1e-12,2);
    X = allX(valid,:);
    Y = allY(valid,:);
    C = allC(valid,:);
    ids = allIds(valid);
end

function [FeasX,FeasY,InfX,InfY] = evaluatedRows( ...
        Population,Excluded,Problem,~)
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
        valid = valid & ~ismember(X,double(Excluded),'rows');
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

function Archive = refreshPairMetadata(Archive,~,~,W,RefScale,Problem)
%REFRESHPAIRMETADATA Derived caches never control historical validity.
    if isempty(Archive.id); return; end
    Archive.ref = AssignReferenceVectors_CBS(Archive.yf,W,RefScale);
    Archive.gap = normalizedDistance(Archive.xf,Archive.xi,Problem);
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
        Data.yF = double(Archive.yf(rows,:));
        Data.yI = double(Archive.yi(rows,:));
        Data.W = double(W);
        Data.w = double(W(Archive.ref(rows),:));
        Data.ref = Archive.ref(rows);
        Data.id = Archive.id(rows);
        Data.delta = Data.xI-Data.xF;
        Data.count = numel(rows);
        if isfield(Options,'referenceScale')
            scale = Options.referenceScale;
        else
            [~,scale] = AssignReferenceVectors_CBS([Archive.yf;Archive.yi],W);
        end
        refF = AssignReferenceVectors_CBS(Archive.yf(rows,:),W,scale);
        refI = AssignReferenceVectors_CBS(Archive.yi(rows,:),W,scale);
        Data.cF = [double(W(refF,:)),ones(Data.count,1)];
        Data.cI = [double(W(refI,:)),zeros(Data.count,1)];
        Data.referenceScale = scale;
    end
    if numel(unique(Data.ref)) ~= Data.count
        error('CBSPairGuide:NonUniqueTrainingReference', ...
            'Every active reference must own exactly one training pair.');
    end
    TrainC = [Data.cF;Data.cI];
    [~,uniqueRows] = unique([[Data.xF;Data.xI],TrainC(:,end)],'rows','stable');
    TrainC = TrainC(uniqueRows,:);
    QueryRefs = unique(AssignReferenceVectors_CBS(TrainC(:,1:end-1),W, ...
        struct('minimum',zeros(1,size(W,2)),'span',ones(1,size(W,2)))));
    Gate = struct('effective',Data.count,'active',Data.count, ...
        'regions',Data.count,'pairs',Data.count, ...
        'eligible',Data.count >= Options.pairMinPairs);
    BMem = archiveAsBoundaryMemory(Archive,Problem);
end

function [QueryC,Info] = buildQueryContexts(Archive,W,Options,totalBudget)
%BUILDQUERYCONTEXTS Query the complete direction set and both binary sides.

    Options = fillOptions(Options);
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
    if ~isfield(Options,'pairOnly') || ~Options.pairOnly
        % Query labels are independent of archive occupancy and pair IDs.
        refs = balancedRows((1:size(W,1))',totalBudget);
        sides = zeros(totalBudget,1);
        phase = randi(2)-1;
        for ref = 1:size(W,1)
            local = find(refs == ref);
            sides(local) = mod((0:numel(local)-1)'+phase,2);
            phase = mod(phase+numel(local),2);
        end
        QueryC = [double(W(refs,:)),sides];
        Info.refs = refs; Info.pairIds = zeros(totalBudget,1); Info.sides = sides;
        Info.conditions = QueryC;
        return;
    end
    rows = balancedRows(activeRows,totalBudget);
    refs = Archive.ref(rows);
    QueryC = [double(W(refs,:)),zeros(numel(refs),1)];
    Info.refs = refs;
    Info.pairIds = Archive.id(rows);
end

function [Dec,Refs,Ids,T] = selectCandidates(X,Info,Archive,Scale,Problem,O)
%SELECTCANDIDATES Prefer uncovered requests, then in-box points closest to F.
% The box uses ALL archive endpoints, including inactive retained pairs.
% Request coverage is a pre-evaluation proxy, not a generated objective label.
    if isfield(O,'pairOnly') && O.pairOnly
        [Dec,Refs,Ids,T] = selectPairedCandidates(X,Info,Archive,Scale,Problem,O);
        return;
    end
    O = fillOptions(O); Archive = ensureArchive(Archive,Problem.D,Problem.M);
    n = size(X,1); T = emptyPoolTrace();
    T.active = n > 0; T.rawCount = n; T.candidateDecs = double(X); T.percentile = nan(n,1);
    T.rawConditions = numel(unique(Info.refs)); T.trainConditions = nnz(Archive.active);
    span = decisionSpan(Problem); Z = (double(X)-Problem.lower)./span;
    valid = all(isfinite(Z),2) & all(Z >= 0 & Z <= 1,2);
    if isfield(O,'W'); W=double(O.W); else; W=zeros(0,Problem.M); end
    if isempty(W)
        error('CBSPairGuide:MissingQueryDirections','Candidate selection requires the complete W.');
    end
    % Recover metadata from actual finite query vectors when available. An
    % empty or externally supplied direction is not a reason to discard X.
    refs = reshape(double(Info.refs),[],1);
    assert(numel(refs)==n,'CBSPairGuide:BadQueryMetadata','One reference ID is required per candidate.');
    if isfield(Info,'conditions')
        C=double(Info.conditions);
        assert(isequal(size(C),[n,size(W,2)+1]) && all(isfinite(C),'all'), ...
            'CBSPairGuide:BadQueryMetadata','Query conditions must be finite and match the candidate rows.');
        [found,recovered]=ismember(C(:,1:end-1),W,'rows');
        if any(~found)
            [extra,~,index]=unique(C(~found,1:end-1),'rows','stable');
            recovered(~found)=size(W,1)+index; W=[W;extra];
        end
        T.recoveredReferenceCount=nnz(refs~=recovered | ~isfinite(refs));
        refs=recovered;
    else
        T.recoveredReferenceCount=0;
        assert(all(isfinite(refs) & refs==fix(refs) & refs>=1 & refs<=size(W,1)), ...
            'CBSPairGuide:BadQueryMetadata','Unregistered reference IDs require the actual query conditions.');
    end
    T.invalidCount = nnz(~valid); T.validCount = nnz(valid);
    knownY = [Archive.yf;Archive.yi];
    if isfield(O,'p1AllObjs')
        knownY = [knownY;double(O.p1AllObjs)];
    elseif isfield(O,'p1Objs')
        knownY = [knownY;double(O.p1Objs)];
    end
    knownY = knownY(all(isfinite(knownY),2),:);
    knownRefs = unique(AssignReferenceVectors_CBS(knownY,W,Scale));
    uncovered = valid & ~ismember(refs,knownRefs);
    allEndpoints = [Archive.xf;Archive.xi];
    inside = false(n,1); distanceF = inf(n,1);
    lowerBox = nan(1,Problem.D); upperBox = lowerBox;
    if ~isempty(allEndpoints)
        lowerBox = min(double(allEndpoints),[],1);
        upperBox = max(double(allEndpoints),[],1);
        lo = (lowerBox-Problem.lower)./span; hi = (upperBox-Problem.lower)./span;
        inside = valid & all(Z>=lo-1e-12 & Z<=hi+1e-12,2);
        F = (double(Archive.xf)-Problem.lower)./span;
        for j=1:size(F,1)
            distanceF = min(distanceF,sqrt(sum((Z-F(j,:)).^2,2)));
        end
    end
    eligibleKnown = valid & ~uncovered & inside;
    base = zeros(0,Problem.D);
    if isfield(O,'currentDecs'); base = (double(O.currentDecs)-Problem.lower)./span; end
    chosen = zeros(0,1);
    % Unknown requests keep direction diversity without inventing F targets.
    groups = unique(refs(uncovered)); groups = groups(randperm(numel(groups)));
    remaining = uncovered;
    while numel(chosen)<O.guideQuota && any(remaining)
        before = numel(chosen);
        for ref=reshape(groups,1,[])
            for k=reshape(find(remaining & refs==ref),1,[])
                remaining(k)=false;
                if ~isempty(base) && any(vecnorm(base-Z(k,:),2,2)<=O.pairDuplicateTolerance); continue; end
                chosen(end+1,1)=k; base(end+1,:)=Z(k,:); break; %#ok<AGROW>
            end
            if numel(chosen)>=O.guideQuota; break; end
        end
        if numel(chosen)==before; break; end
    end
    % Known requests are globally sorted by distance to the nearest real F.
    local=find(eligibleKnown); [~,order]=sortrows([distanceF(local),local]);
    for k=reshape(local(order),1,[])
        if numel(chosen)>=O.guideQuota; break; end
        if ~isempty(base) && any(vecnorm(base-Z(k,:),2,2)<=O.pairDuplicateTolerance); continue; end
        chosen(end+1,1)=k; base(end+1,:)=Z(k,:); %#ok<AGROW>
    end
    Dec=double(X(chosen,:)); Refs=refs(chosen); Ids=zeros(numel(chosen),1);
    T.keepIdx=chosen; T.keptCount=numel(chosen); T.keptConditions=numel(unique(Refs));
    T.matchedPairIds=Ids; T.rawPairIds=zeros(n,1); T.rawRefs=refs;
    T.rawSides=Info.sides; T.selectedSides=Info.sides(chosen);
    T.selectionPolicy="uncovered-first-archive-box-v1";
    T.queryVectors=W; T.knownRefs=knownRefs; T.uncoveredRefs=setdiff((1:size(W,1))',knownRefs);
    T.requestedUncovered=uncovered; T.insideArchiveBox=inside;
    T.archiveBoxLower=lowerBox; T.archiveBoxUpper=upperBox;
    T.nearestFeasibleDistance=distanceF; T.keptUncoveredCount=nnz(uncovered(chosen));
    T.knownOutsideBoxCount=nnz(valid & ~uncovered & ~inside);
    % A global box is not a pair interval or a certified boundary region.
    T.xf=nan(n,Problem.D); T.xi=T.xf; T.yf=nan(n,Problem.M); T.yi=T.yf;
    T.nativeInBand=nan(n,1); T.gaps=nan(n,1); T.axialBefore=nan(n,1);
    T.perpendicularBefore=nan(n,1); T.boundaryDistanceUpperBoundRMS=nan(n,1);
    T.boundaryCertified=false(n,1); T.coarseInterval=false(n,1);
    T.spherePassCount=nnz(uncovered | eligibleKnown); T.jointPassCount=T.spherePassCount;
    T.rawNearDuplicateRate=nearDuplicateRate(Z(valid,:),O.pairDuplicateTolerance);
    T.keptNearDuplicateRate=nearDuplicateRate(Z(chosen,:),O.pairDuplicateTolerance);
    T.scores=distanceF; T.priority=[];
end


function [SelectedDecs,SelectedRefs,MatchedIds,Trace] = ...
        selectPairedCandidates(RawDec,SampleInfo,Archive,RefScale,Problem,Options)
%SELECTCANDIDATES Select unchanged native proposals without real oracle calls.
    Options = fillOptions(Options);
    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    count = size(RawDec,1);
    Trace = emptyPoolTrace();
    Trace.active = count > 0;
    Trace.rawCount = count;
    Trace.percentile = nan(count,1);
    Trace.rawConditions = numel(unique(SampleInfo.refs));
    Trace.trainConditions = nnz(Archive.active);
    lower = double(Problem.lower); span = decisionSpan(Problem);
    cn = (double(RawDec)-lower)./span;
    Trace.candidateDecs = double(RawDec);
    Trace.axialBefore = nan(count,1);
    Trace.perpendicularBefore = nan(count,1);
    Trace.gaps = nan(count,1);
    Trace.boundaryDistanceUpperBoundRMS = nan(count,1);
    Trace.nativeInBand = false(count,1);
    Trace.xf = nan(count,Problem.D); Trace.xi = nan(count,Problem.D);
    Trace.yf = nan(count,Problem.M); Trace.yi = nan(count,Problem.M);
    scores = inf(count,1); pairRows = zeros(count,1);
    priorities = zeros(numel(Archive.id),1);
    for p = reshape(find(Archive.active),1,[])
        localValue = 1;
        occupancy = 0;
        if isfield(Options,'p1Refs')
            occupancy = nnz(Options.p1Refs == Archive.ref(p));
            local = ismember(Options.p1Refs,neighborRefs(Options.W,Archive.ref(p),5));
            if dominatedByAny(Archive.yi(p,:), ...
                    [Archive.yf(p,:);Options.p1Objs(local,:)],RefScale,1e-12)
                localValue = 0.1;
            end
        elseif pairDominatesRows(Archive.yf(p,:),Archive.yi(p,:),RefScale,1e-12)
            localValue = 0.1;
        end
        gstar = 0.003*sqrt(Problem.D)/sqrt(0.6^2+0.05^2);
        priorities(p) = localValue*Archive.gap(p)/(Archive.gap(p)+gstar)/(1+occupancy);
    end
    for k = 1:count
        if any(~isfinite(cn(k,:))) || any(cn(k,:) < 0 | cn(k,:) > 1)
            Trace.invalidCount = Trace.invalidCount+1;
            continue;
        end
        p = find(Archive.id == SampleInfo.pairIds(k) & Archive.active & ...
            Archive.ref == SampleInfo.refs(k),1);
        if isempty(p)
            Trace.matchFailures = Trace.matchFailures+1;
            continue;
        end
        f = (Archive.xf(p,:)-lower)./span;
        i = (Archive.xi(p,:)-lower)./span;
        d = i-f; g = norm(d);
        if g <= 0; continue; end
        t0 = dot(cn(k,:)-f,d)/(g*g);
        r0 = cn(k,:)-f-t0*d;
        Trace.axialBefore(k) = t0;
        Trace.perpendicularBefore(k) = norm(r0)/g;
        Trace.gaps(k) = g;
        Trace.nativeInBand(k) = t0 >= 0.4 && t0 <= 0.6 && norm(r0) <= 0.05*g;
        % Any boundary crossing on the real segment is within this distance.
        % The former projected-interval bound does not apply to native c.
        Trace.boundaryDistanceUpperBoundRMS(k) = ...
            max(norm(cn(k,:)-f),norm(cn(k,:)-i))/sqrt(Problem.D);
        Trace.xf(k,:) = Archive.xf(p,:); Trace.xi(k,:) = Archive.xi(p,:);
        Trace.yf(k,:) = Archive.yf(p,:); Trace.yi(k,:) = Archive.yi(p,:);
        endpointError = 0;
        if isfield(SampleInfo,'generatedF')
            endpointError = (norm(SampleInfo.generatedF(k,:)-f)+ ...
                norm(SampleInfo.generatedI(k,:)-i))/(g+eps);
        end
        scores(k) = endpointError;
        pairRows(k) = p;
    end
    valid = pairRows > 0;
    % Relative narrowness and an absolute boundary-distance bound are
    % different diagnostics. Neither is an additional selection threshold.
    Trace.boundaryCertified = valid & Trace.boundaryDistanceUpperBoundRMS <= 0.003;
    Trace.coarseInterval = valid & ~Trace.boundaryCertified;
    Trace.spherePassCount = nnz(valid); % Legacy name: finite, in-box, matched.
    Trace.sphereRejectCount = count-Trace.invalidCount-Trace.matchFailures-nnz(valid);
    Trace.jointPassCount = nnz(valid);
    pairs = unique(pairRows(valid));
    [~,order] = sortrows([-priorities(pairs),Archive.ref(pairs),Archive.id(pairs)]);
    pairs = pairs(order);
    chosen = zeros(0,1);
    duplicateBase = zeros(0,Problem.D);
    if isfield(Options,'currentDecs')
        duplicateBase = (Options.currentDecs-lower)./span;
    end
    for round = 1:2
        for p = reshape(pairs,1,[])
            local = find(valid & pairRows == p & ~ismember((1:count)',chosen));
            [~,order] = sortrows([scores(local),local]);
            for k = reshape(local(order),1,[])
                if ~isempty(duplicateBase) && any(sqrt(sum((duplicateBase-cn(k,:)).^2,2)) ...
                        <= Options.pairDuplicateTolerance)
                    continue;
                end
                chosen(end+1,1) = k; %#ok<AGROW>
                duplicateBase(end+1,:) = cn(k,:); %#ok<AGROW>
                break;
            end
            if numel(chosen) >= Options.guideQuota; break; end
        end
        if numel(chosen) >= Options.guideQuota; break; end
    end
    if Options.guideQuota == 0; chosen = zeros(0,1); end
    SelectedDecs = double(RawDec(chosen,:));
    SelectedRefs = Archive.ref(pairRows(chosen));
    MatchedIds = Archive.id(pairRows(chosen));
    Trace.keepIdx = chosen;
    Trace.keptCount = numel(chosen); Trace.validCount = numel(chosen);
    Trace.keptConditions = numel(unique(SelectedRefs));
    Trace.matchedPairIds = MatchedIds;
    Trace.rawNearDuplicateRate = nearDuplicateRate(cn(all(isfinite(cn),2),:), ...
        Options.pairDuplicateTolerance);
    Trace.keptNearDuplicateRate = nearDuplicateRate(cn(chosen,:),Options.pairDuplicateTolerance);
    Trace.scores = scores;
    Trace.rawPairIds = SampleInfo.pairIds;
    Trace.rawRefs = SampleInfo.refs;
    Trace.priority = priorities;
end

function valid = validArchiveRows(Archive,refCount,Problem)
%VALIDARCHIVEROWS Geometry validity; endpoint roles certify opposite real labels.
    lower = double(Problem.lower); upper = double(Problem.upper);
    valid = isfinite(Archive.id) & Archive.id > 0 & Archive.id == fix(Archive.id) & ...
        Archive.ref >= 1 & Archive.ref <= refCount & ...
        isfinite(Archive.gap) & Archive.gap > 0 & ...
        all(isfinite([Archive.xf,Archive.xi,Archive.yf,Archive.yi]),2) & ...
        all(Archive.xf >= lower-1e-12 & Archive.xf <= upper+1e-12,2) & ...
        all(Archive.xi >= lower-1e-12 & Archive.xi <= upper+1e-12,2);
end

function BMem = archiveAsBoundaryMemory(Archive,Problem)
%ARCHIVEASBOUNDARYMEMORY Read-only endpoint view for Pending attribution.
    Archive = ensureArchive(Archive,Problem.D,Problem.M);
    BMem = struct('id',Archive.id,'ref',Archive.ref,'gap',Archive.gap, ...
        'x_b',Archive.xf,'y_b',Archive.yf,'x_i',Archive.xi, ...
        'y_i',Archive.yi,'active',Archive.active);
end

function Archive = ensureArchive(Archive,D,M)
%ENSUREARCHIVE Discard obsolete lifecycle clocks when reading an older archive.
    Empty = emptyArchive(D,M);
    if isempty(Archive) || ~isstruct(Archive)
        Archive = Empty;
        return;
    end
    names = fieldnames(Empty);
    extra = setdiff(fieldnames(Archive),names);
    if ~isempty(extra); Archive = rmfield(Archive,extra); end
    if ~isfield(Archive,'nextId')
        Archive.nextId = max([0;Archive.id])+1;
    end
    Archive.nextId = max(Archive.nextId,max([0;Archive.id])+1);
    if ~isfield(Archive,'active'); Archive.active = true(numel(Archive.id),1); end
end

function Archive = emptyArchive(D,M)
%EMPTYARCHIVE Six semantic fields plus derived gap/active caches and ID allocator.
    Archive = struct('id',zeros(0,1),'ref',zeros(0,1), ...
        'xf',zeros(0,D),'yf',zeros(0,M),'xi',zeros(0,D),'yi',zeros(0,M), ...
        'gap',zeros(0,1),'active',false(0,1),'nextId',1);
end

function Data = emptyTrainingData(D,M)
%EMPTYTRAININGDATA Complete-pair endpoint data; ID remains metadata.

    Data = struct('xF',zeros(0,D),'xI',zeros(0,D), ...
        'delta',zeros(0,D),'w',zeros(0,M),'ref',zeros(0,1), ...
        'id',zeros(0,1), ...
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
        'tightenedFeasible',0,'tightenedInfeasible',0, ...
        'guidedTightenedFeasible',0,'guidedTightenedInfeasible',0, ...
        'ordinaryTightenedFeasible',0,'ordinaryTightenedInfeasible',0, ...
        'removed',0,'active',0,'inactive',0,'resumeEligible',0, ...
        'archiveChanged',0,'previousCount',0);
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
        'validCount',0,'invalidCount',0,'matchFailures',0, ...
        'sphereRejectCount',0,'spherePassCount',0, ...
        'objectiveCandidateCount',0,'objectiveFE',0,'ObjFE',0, ...
        'constraintFE',0, ...
        'localDominancePass',0,'corridorPass',0,'jointPassCount',0, ...
        'matchedPairIds',zeros(0,1));
end

function Archive = subsetArchive(Archive,rows)
%SUBSETARCHIVE Preserve stable IDs and their allocator.
    fields = {'id','ref','xf','yf','xi','yi','gap','active'};
    for k = 1:numel(fields)
        Archive.(fields{k}) = Archive.(fields{k})(rows,:);
    end
end

function distance = normalizedDistance(a,b,Problem)
%NORMALIZEDDISTANCE Decision-box Euclidean distance between paired rows.

    distance = sqrt(sum(((double(a)-double(b))./ ...
        decisionSpan(Problem)).^2,2));
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

function neighbors = referenceNeighborhoods(W)
%REFERENCENEIGHBORHOODS Cache the same scalar angular ordering for fixed W.
    persistent lastW lastNeighbors
    if isempty(lastW) || ~isequal(W,lastW)
        lastW = W;
        lastNeighbors = false(size(W,1));
        for ref = 1:size(W,1)
            lastNeighbors(ref,neighborRefs(W,ref,5)) = true;
        end
    end
    neighbors = lastNeighbors;
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
    Options = defaultOption(Options,'pairArchiveCapacity',500);
    Options = defaultOption(Options,'archiveFrontDepth',1);
    Options = defaultOption(Options,'pairNeighborRefCount',5);
    Options = defaultOption(Options,'pairMinPairs',8);
    Options = defaultOption(Options,'pairDuplicateTolerance',1e-6);
    Options = defaultOption(Options,'pairImprovementTolerance',1e-12);
    Options = defaultOption(Options,'guideQuota',20);
    Options = defaultOption(Options,'objectiveBudget',Inf);
    integer = {'pairArchivePerRef','pairNeighborRefCount','pairMinPairs','guideQuota'};
    for i = 1 : numel(integer)
        name = integer{i};
        Options.(name) = max(0,round(double(Options.(name))));
    end
    if Options.pairArchivePerRef ~= 1
        error('CBSPairGuide:ArchivePerReferenceMustBeOne', ...
            'PairGuide requires exactly one active archive pair per reference.');
    end
    if ~isscalar(Options.pairArchiveCapacity) || Options.pairArchiveCapacity ~= 500
        error('CBSPairGuide:ArchiveCapacity','PairGuide retains 500 pairs (1000 endpoint slots).');
    end
    if ~isnumeric(Options.archiveFrontDepth) || ~isscalar(Options.archiveFrontDepth) || ...
            ~ismember(Options.archiveFrontDepth,[0 1 2])
        error('CBSPairGuide:ArchiveFrontDepth','Front depth must be 0 (P1), 1, or 2.');
    end
    if Options.pairNeighborRefCount ~= 5
        error('CBSPairGuide:NeighborCount','The neighborhood contains exactly five references (including itself).');
    end
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
