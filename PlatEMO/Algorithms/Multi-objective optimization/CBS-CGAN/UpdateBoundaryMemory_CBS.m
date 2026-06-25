function [BMem,Diag] = UpdateBoundaryMemory_CBS(PrevBMem, ...
    Population1,Offspring1,Population2,Offspring2,W,Options)
%UPDATEBOUNDARYMEMORY_CBS Build pair-supported thin boundary memory.

    Samples = [Population1,Offspring1,Population2,Offspring2];
    [D,M] = inferMemoryDimensions(Samples,PrevBMem);
    Stats = emptyBoundaryPipelineStats();
    if isempty(Samples)
        XCurrent = zeros(0,D);
        YCurrent = zeros(0,M);
        CCurrent = zeros(0,1);
    else
        XCurrent = Samples.decs;
        YCurrent = Samples.objs;
        CCurrent = Samples.cons;
        if size(CCurrent,2) == 0
            CCurrent = zeros(size(XCurrent,1),1);
        end
    end
    Stats.boundary_sample_current_count = size(XCurrent,1);

    [X,Y,C,Stats.boundary_sample_prev_pair_count] = ...
        appendPreviousPairedCandidates( ...
        XCurrent,YCurrent,CCurrent,PrevBMem,D,M);
    D = size(X,2);
    M = size(Y,2);
    Stats.boundary_sample_merged_count = size(Y,1);
    valid = all(isfinite(X),2) & all(isfinite(Y),2);
    X = X(valid,:);
    Y = Y(valid,:);
    C = C(valid,:);
    Stats.boundary_sample_finite_count = size(Y,1);
    if isempty(Y)
        BMem = emptyBMem(0,D,M);
        Diag = makeDiag(BMem,Stats);
        return;
    end

    currentValid = all(isfinite(XCurrent),2) & all(isfinite(YCurrent),2);
    XCurrent = XCurrent(currentValid,:);
    YCurrent = YCurrent(currentValid,:);
    CCurrent = CCurrent(currentValid,:);

    CV = sum(max(0,C),2);
    Feasible = CV <= 0;
    CVCurrent = sum(max(0,CCurrent),2);
    CurrentFeasible = CVCurrent <= 0;
    Stats.boundary_feasible_count = sum(Feasible);
    Stats.boundary_infeasible_count = sum(~Feasible);

    [Yn,ObjMin,ObjSpan] = normalizeRows(Y);
    Ref = assignReferences(Yn,W);

    [Candidate,PairStats] = harvestBoundaryCandidates(X,Y,Yn,Feasible,Ref,W, ...
        Options,ObjMin,ObjSpan,XCurrent,YCurrent,CurrentFeasible);
    Stats = mergeStructs(Stats,PairStats);
    [Candidate,FilterStats] = filterBoundaryCandidates(Candidate,Options);
    Stats = mergeStructs(Stats,FilterStats);
    [BMem,CanonicalStats] = canonicalizeSingleBoundary(Candidate,W,Options);
    Stats = mergeStructs(Stats,CanonicalStats);
    Diag = makeDiag(BMem,Stats);
end

function [X,Y,C,prevPairCount] = appendPreviousPairedCandidates( ...
    X,Y,C,PrevBMem,D,M)
    prevPairCount = 0;
    if size(C,2) == 0
        C = zeros(size(X,1),1);
    end
    Prev = ensureMemoryFields(PrevBMem,D,M);
    if isempty(Prev.y_b) || isempty(Prev.x_f) || isempty(Prev.x_i)
        return;
    end
    valid = all(isfinite(Prev.x_f),2) & all(isfinite(Prev.y_f),2) & ...
        all(isfinite(Prev.x_i),2) & all(isfinite(Prev.y_i),2) & ...
        isfinite(Prev.gap(:));
    if ~any(valid)
        return;
    end
    prevPairCount = sum(valid);
    cols = max(1,size(C,2));
    if size(C,2) == 0
        C = zeros(size(X,1),cols);
    end
    X = [X;Prev.x_f(valid,:);Prev.x_i(valid,:)];
    Y = [Y;Prev.y_f(valid,:);Prev.y_i(valid,:)];
    C = [C;zeros(sum(valid),cols);ones(sum(valid),cols)];
end

function [Candidate,Stats] = harvestBoundaryCandidates(X,Y,Yn,Feasible,Ref,W, ...
    Options,ObjMin,ObjSpan,XCurrent,YCurrent,CurrentFeasible)
    D = size(X,2);
    M = size(Y,2);
    Candidate = emptyBMem(0,D,M);
    Stats = emptyPairSearchStats();
    if isempty(Y) || isempty(W)
        return;
    end

    mainFeasible = false(size(Feasible));
    fAll = find(Feasible);
    if ~isempty(fAll)
        front = firstFrontMask(Y(fAll,:));
        mainFeasible(fAll(front)) = true;
    end
    Stats.boundary_main_feasible_count = sum(mainFeasible);

    nRef = size(W,1);
    for r = 1 : nRef
        neigh = neighborRefs(W,r,Options.pairNeighborRefRadius);
        fidx = find(mainFeasible & Ref == r);
        iidx = find(~Feasible & ismember(Ref,neigh));
        if ~isempty(fidx)
            Stats.boundary_ref_with_main_feasible_count = ...
                Stats.boundary_ref_with_main_feasible_count + 1;
        end
        if ~isempty(iidx)
            Stats.boundary_ref_with_neighbor_infeasible_count = ...
                Stats.boundary_ref_with_neighbor_infeasible_count + 1;
        end
        if isempty(fidx) || isempty(iidx)
            continue;
        end
        Stats.boundary_ref_pairable_count = ...
            Stats.boundary_ref_pairable_count + 1;
        Dist = pairDistance(Yn(fidx,:),Yn(iidx,:));
        Stats.boundary_pair_distance_count = ...
            Stats.boundary_pair_distance_count + numel(Dist);
        [gapSorted,order] = sort(Dist(:),'ascend');
        appended = 0;
        for t = 1 : numel(order)
            [fi,ii] = ind2sub(size(Dist),order(t));
            f = fidx(fi);
            i = iidx(ii);
            if feasibleDominatesInfeasible(Y(f,:),Y(i,:))
                Stats.boundary_pair_dominated_skip_count = ...
                    Stats.boundary_pair_dominated_skip_count + 1;
                continue;
            end
            [xb,yb] = selectThinBoundaryTarget( ...
                XCurrent,YCurrent,CurrentFeasible,ObjMin,ObjSpan, ...
                X(f,:),Y(f,:),Y(i,:),Options);
            Candidate.ref(end+1,1) = r;
            Candidate.y_b(end+1,:) = yb;
            Candidate.gap(end+1,1) = gapSorted(t);
            Candidate.x_b(end+1,:) = xb;
            Candidate.x_f(end+1,:) = X(f,:);
            Candidate.y_f(end+1,:) = Y(f,:);
            Candidate.x_i(end+1,:) = X(i,:);
            Candidate.y_i(end+1,:) = Y(i,:);
            appended = appended + 1;
            Stats.boundary_pair_appended_count = ...
                Stats.boundary_pair_appended_count + 1;
            if appended >= Options.maxCandidatePairsPerRef
                break;
            end
        end
    end
end

function [xb,yb] = selectThinBoundaryTarget( ...
    XCurrent,YCurrent,CurrentFeasible,ObjMin,ObjSpan,xf,yf,yi,Options)
    if boundaryTargetModeIsFeasibleEndpoint(Options)
        xb = xf;
        yb = yf;
        return;
    end
    valid = CurrentFeasible(:) & all(isfinite(XCurrent),2) & ...
        all(isfinite(YCurrent),2);
    if any(valid)
        A = normalizeWithScale(yf,ObjMin,ObjSpan);
        B = normalizeWithScale(yi,ObjMin,ObjSpan);
        endpoint = samePointRows(XCurrent,YCurrent,xf,yf);
        [row,foundSupport] = nearestInteriorSupportRow( ...
            YCurrent,valid & ~endpoint,A,B,ObjMin,ObjSpan);
        if ~foundSupport
            row = nearestSegmentSupportRow(YCurrent,valid,A,B, ...
                ObjMin,ObjSpan);
        end
        xb = XCurrent(row,:);
        yb = YCurrent(row,:);
    else
        xb = xf;
        yb = yf;
    end
end

function yes = boundaryTargetModeIsFeasibleEndpoint(Options)
    yes = false;
    if isstruct(Options) && isfield(Options,'boundaryTargetMode')
        mode = lower(strtrim(string(Options.boundaryTargetMode)));
        yes = mode == "feasible_endpoint" || mode == "feasible";
    end
end

function [row,found] = nearestInteriorSupportRow( ...
    YCurrent,valid,A,B,ObjMin,ObjSpan)
    row = NaN;
    found = false;
    idx = find(valid(:));
    if isempty(idx)
        return;
    end
    Yn = normalizeWithScale(YCurrent(idx,:),ObjMin,ObjSpan);
    dist = pointSegmentDistance(Yn,A,B);
    coord = segmentCoordinates(Yn,A,B);
    segmentLength = sqrt(sum((B - A).^2));
    maxDistance = max(1e-12,0.25*segmentLength);
    usable = dist <= maxDistance & coord > 1e-9 & coord < 1 - 1e-9;
    if ~any(usable)
        return;
    end
    candidates = find(usable);
    [~,best] = min(dist(candidates));
    row = idx(candidates(best));
    found = true;
end

function row = nearestSegmentSupportRow(YCurrent,valid,A,B,ObjMin,ObjSpan)
    idx = find(valid(:));
    Yn = normalizeWithScale(YCurrent(idx,:),ObjMin,ObjSpan);
    dist = pointSegmentDistance(Yn,A,B);
    [~,best] = min(dist);
    row = idx(best);
end

function same = samePointRows(X,Y,xf,yf)
    same = sameRows(X,xf) & sameRows(Y,yf);
end

function same = sameRows(X,row)
    same = false(size(X,1),1);
    row = row(:)';
    if size(X,2) ~= numel(row)
        return;
    end
    if isempty(row)
        same(:) = true;
        return;
    end
    scale = max(1,max(abs([X(:);row(:)])));
    tol = 1e-10*scale;
    same = max(abs(X - row),[],2) <= tol;
end

function coord = segmentCoordinates(P,A,B)
    AB = B - A;
    denom = sum(AB.^2);
    if denom <= eps
        coord = zeros(size(P,1),1);
    else
        coord = ((P - A)*AB')/denom;
        coord = max(0,min(1,double(coord)));
    end
    coord(~isfinite(coord)) = 0;
end

function [Candidate,Stats] = filterBoundaryCandidates(Candidate,Options)
    Stats = emptyFilterStats();
    Stats.boundary_candidate_raw_count = memoryRowCount(Candidate);
    if isempty(Candidate) || isempty(Candidate.y_b)
        Stats.boundary_candidate_finite_count = memoryRowCount(Candidate);
        Stats.boundary_candidate_pareto_count = memoryRowCount(Candidate);
        Stats.boundary_candidate_gap_count = memoryRowCount(Candidate);
        return;
    end
    Candidate = subsetMemory(Candidate,isfinite(Candidate.gap));
    Stats.boundary_candidate_finite_count = memoryRowCount(Candidate);
    Candidate = filterParetoMainLayer(Candidate,Options);
    Stats.boundary_candidate_pareto_count = memoryRowCount(Candidate);
    Candidate = filterGapCap(Candidate,Options);
    Stats.boundary_candidate_gap_count = memoryRowCount(Candidate);
end

function Candidate = filterParetoMainLayer(Candidate,Options)
    if isempty(Candidate) || isempty(Candidate.y_b)
        return;
    end
    minCount = boundaryMinCount(Options);
    rank = paretoRankLimited(Candidate.y_b,2);
    keep = rank == 1;
    if sum(keep) < minCount
        keep = rank <= 2;
    end
    Candidate = subsetMemory(Candidate,keep);
end

function Candidate = filterGapCap(Candidate,Options)
    if isempty(Candidate) || isempty(Candidate.y_b)
        return;
    end
    minCount = boundaryMinCount(Options);
    if size(Candidate.y_b,1) <= minCount
        return;
    end
    gaps = Candidate.gap(:);
    cap = adaptiveGapCap(gaps);
    keep = gaps <= cap + 1e-12;
    if sum(keep) < minCount
        [~,ord] = sort(gaps,'ascend');
        keep = false(size(gaps));
        keep(ord(1:min(minCount,numel(ord)))) = true;
    end
    Candidate = subsetMemory(Candidate,keep);
end

function cap = adaptiveGapCap(gaps)
    gaps = gaps(isfinite(gaps));
    if isempty(gaps)
        cap = Inf;
        return;
    end
    med = median(gaps);
    dev = median(abs(gaps - med));
    if dev <= eps
        cap = med*(1 + 0.25) + 1e-12;
    else
        cap = med + 3*1.4826*dev;
    end
end

function minCount = boundaryMinCount(Options)
    minCount = 1;
    if isstruct(Options) && isfield(Options,'minBoundaryLength')
        minCount = max(1,round(double(Options.minBoundaryLength)));
    end
end

function rank = paretoRankLimited(Y,maxRank)
    n = size(Y,1);
    rank = inf(n,1);
    remaining = true(n,1);
    for r = 1 : maxRank
        idx = find(remaining);
        if isempty(idx)
            break;
        end
        front = firstFrontMask(Y(idx,:));
        rank(idx(front)) = r;
        remaining(idx(front)) = false;
    end
    rank(isinf(rank)) = maxRank + 1;
end

function front = firstFrontMask(Y)
    n = size(Y,1);
    front = true(n,1);
    epsTol = 1e-12;
    for i = 1 : n
        for j = 1 : n
            if i == j
                continue;
            end
            if all(Y(j,:) <= Y(i,:) + epsTol) && ...
                    any(Y(j,:) < Y(i,:) - epsTol)
                front(i) = false;
                break;
            end
        end
    end
end

function yes = feasibleDominatesInfeasible(Yf,Yi)
    epsTol = 1e-12;
    yes = all(Yf <= Yi + epsTol,2) & any(Yf < Yi - epsTol,2);
end

function [BMem,Stats] = canonicalizeSingleBoundary(BMem,W,Options)
    Stats = emptyCanonicalStats();
    if isempty(BMem) || isempty(BMem.y_b)
        Stats.boundary_candidate_dedup_count = memoryRowCount(BMem);
        Stats.boundary_candidate_final_count = memoryRowCount(BMem);
        return;
    end
    BMem = deduplicateRef(BMem,Options);
    Stats.boundary_candidate_dedup_count = memoryRowCount(BMem);
    idx = sortedBoundaryNodes(BMem,W);
    BMem = subsetMemory(BMem,idx);
    if size(BMem.y_b,1) < boundaryMinCount(Options)
        BMem = emptyBMem(0,size(BMem.x_b,2),size(BMem.y_b,2));
    end
    Stats.boundary_candidate_final_count = memoryRowCount(BMem);
end

function idx = sortedBoundaryNodes(BMem,W)
    idx = (1:size(BMem.y_b,1))';
    if numel(idx) <= 1
        return;
    end
    RefPoint = W(BMem.ref(idx),:);
    [~,~,coeff] = svd(RefPoint - mean(RefPoint,1),'econ');
    if isempty(coeff)
        [~,ord] = sort(BMem.ref(idx));
    else
        score = RefPoint*coeff(:,1);
        [~,ord] = sortrows([score(:),BMem.ref(idx(:))]);
    end
    idx = idx(ord);
end

function BMem = deduplicateRef(BMem,Options)
    if isempty(BMem) || isempty(BMem.y_b)
        return;
    end
    keepPerRef = canonicalPairLimit(Options);
    refs = unique(BMem.ref(:)','stable');
    keep = false(size(BMem.ref));
    for i = 1 : numel(refs)
        idx = find(BMem.ref == refs(i));
        [~,ord] = sort(BMem.gap(idx),'ascend');
        take = min(keepPerRef,numel(ord));
        keep(idx(ord(1:take))) = true;
    end
    BMem = subsetMemory(BMem,keep);
end

function value = canonicalPairLimit(Options)
    if isstruct(Options) && isfield(Options,'maxCanonicalPairsPerRef')
        value = Options.maxCanonicalPairsPerRef;
    elseif isstruct(Options) && isfield(Options,'maxCandidatePairsPerRef')
        value = Options.maxCandidatePairsPerRef;
    else
        value = 1;
    end
    value = max(1,round(double(value)));
end

function BMem = subsetMemory(BMem,keep)
    keep = keep(:);
    BMem.ref = BMem.ref(keep,:);
    BMem.y_b = BMem.y_b(keep,:);
    BMem.gap = BMem.gap(keep,:);
    BMem.x_b = BMem.x_b(keep,:);
    BMem.x_f = BMem.x_f(keep,:);
    BMem.y_f = BMem.y_f(keep,:);
    BMem.x_i = BMem.x_i(keep,:);
    BMem.y_i = BMem.y_i(keep,:);
end

function BMem = ensureMemoryFields(BMem,D,M)
    if isempty(BMem) || ~isstruct(BMem)
        BMem = emptyBMem(0,D,M);
        return;
    end
    n = 0;
    if isfield(BMem,'y_b')
        n = size(BMem.y_b,1);
    end
    if ~isfield(BMem,'ref'); BMem.ref = zeros(n,1); end
    if ~isfield(BMem,'y_b'); BMem.y_b = zeros(n,M); end
    if ~isfield(BMem,'gap'); BMem.gap = nan(n,1); end
    if ~isfield(BMem,'x_b'); BMem.x_b = nan(n,D); end
    if ~isfield(BMem,'x_f'); BMem.x_f = nan(n,D); end
    if ~isfield(BMem,'y_f'); BMem.y_f = nan(n,M); end
    if ~isfield(BMem,'x_i'); BMem.x_i = nan(n,D); end
    if ~isfield(BMem,'y_i'); BMem.y_i = nan(n,M); end
end

function [D,M] = inferMemoryDimensions(Samples,PrevBMem)
    D = 0;
    M = 0;
    if ~isempty(Samples)
        D = size(Samples.decs,2);
        M = size(Samples.objs,2);
        return;
    end
    if isstruct(PrevBMem)
        if isfield(PrevBMem,'x_b')
            D = size(PrevBMem.x_b,2);
        elseif isfield(PrevBMem,'x_f')
            D = size(PrevBMem.x_f,2);
        end
        if isfield(PrevBMem,'y_b')
            M = size(PrevBMem.y_b,2);
        elseif isfield(PrevBMem,'y_f')
            M = size(PrevBMem.y_f,2);
        end
    end
end

function BMem = emptyBMem(N,D,M)
    BMem = struct( ...
        'ref',zeros(N,1), ...
        'y_b',zeros(N,M), ...
        'gap',zeros(N,1), ...
        'x_b',nan(N,D), ...
        'x_f',nan(N,D), ...
        'y_f',zeros(N,M), ...
        'x_i',nan(N,D), ...
        'y_i',zeros(N,M));
end

function Diag = makeDiag(BMem,Stats)
    if nargin < 2 || isempty(Stats)
        Stats = emptyBoundaryPipelineStats();
    end
    if isempty(BMem) || isempty(BMem.y_b)
        Diag = struct( ...
            'bmem_count',0, ...
            'boundary_count',0, ...
            'finite_gap_count',0, ...
            'inf_gap_count',0, ...
            'median_gap',NaN, ...
            'max_gap',NaN);
    else
        finiteGap = BMem.gap(isfinite(BMem.gap));
        Diag = struct( ...
            'bmem_count',size(BMem.y_b,1), ...
            'boundary_count',1, ...
            'finite_gap_count',numel(finiteGap), ...
            'inf_gap_count',sum(isinf(BMem.gap)), ...
            'median_gap',medianOrNaN(finiteGap), ...
            'max_gap',maxOrNaN(finiteGap));
    end
    Diag = mergeStructs(Diag,Stats);
end

function Stats = emptyBoundaryPipelineStats()
    Stats = struct( ...
        'boundary_sample_current_count',0, ...
        'boundary_sample_prev_pair_count',0, ...
        'boundary_sample_merged_count',0, ...
        'boundary_sample_finite_count',0, ...
        'boundary_feasible_count',0, ...
        'boundary_infeasible_count',0, ...
        'boundary_main_feasible_count',0, ...
        'boundary_ref_with_main_feasible_count',0, ...
        'boundary_ref_with_neighbor_infeasible_count',0, ...
        'boundary_ref_pairable_count',0, ...
        'boundary_pair_distance_count',0, ...
        'boundary_pair_dominated_skip_count',0, ...
        'boundary_pair_appended_count',0, ...
        'boundary_candidate_raw_count',0, ...
        'boundary_candidate_finite_count',0, ...
        'boundary_candidate_pareto_count',0, ...
        'boundary_candidate_gap_count',0, ...
        'boundary_candidate_dedup_count',0, ...
        'boundary_candidate_final_count',0);
end

function Stats = emptyPairSearchStats()
    Base = emptyBoundaryPipelineStats();
    Stats = rmfield(Base,{ ...
        'boundary_sample_current_count', ...
        'boundary_sample_prev_pair_count', ...
        'boundary_sample_merged_count', ...
        'boundary_sample_finite_count', ...
        'boundary_feasible_count', ...
        'boundary_infeasible_count', ...
        'boundary_candidate_raw_count', ...
        'boundary_candidate_finite_count', ...
        'boundary_candidate_pareto_count', ...
        'boundary_candidate_gap_count', ...
        'boundary_candidate_dedup_count', ...
        'boundary_candidate_final_count'});
end

function Stats = emptyFilterStats()
    Stats = struct( ...
        'boundary_candidate_raw_count',0, ...
        'boundary_candidate_finite_count',0, ...
        'boundary_candidate_pareto_count',0, ...
        'boundary_candidate_gap_count',0);
end

function Stats = emptyCanonicalStats()
    Stats = struct( ...
        'boundary_candidate_dedup_count',0, ...
        'boundary_candidate_final_count',0);
end

function N = memoryRowCount(BMem)
    if isempty(BMem) || ~isstruct(BMem) || ~isfield(BMem,'y_b') || ...
            isempty(BMem.y_b)
        N = 0;
    else
        N = size(BMem.y_b,1);
    end
end

function A = mergeStructs(A,B)
    if isempty(B)
        return;
    end
    fields = fieldnames(B);
    for i = 1 : numel(fields)
        A.(fields{i}) = B.(fields{i});
    end
end

function value = medianOrNaN(x)
    if isempty(x)
        value = NaN;
    else
        value = median(x);
    end
end

function value = maxOrNaN(x)
    if isempty(x)
        value = NaN;
    else
        value = max(x);
    end
end

function Ref = assignReferences(Yn,W)
    n = size(Yn,1);
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    NormY = sqrt(sum(Yn.^2,2));
    Yu = Yn./max(NormY,eps);
    Score = Yu*Wn';
    [~,Ref] = max(Score,[],2);
    zeroRows = NormY <= eps;
    if any(zeroRows)
        D = pairDistance(Yn(zeroRows,:),W);
        [~,Ref(zeroRows)] = min(D,[],2);
    end
    Ref = reshape(Ref,n,1);
end

function neigh = neighborRefs(W,r,radius)
    if radius <= 0
        neigh = r;
        return;
    end
    d = sqrt(sum((W - W(r,:)).^2,2));
    [~,ord] = sort(d,'ascend');
    count = min(numel(ord),1 + 2*radius);
    neigh = ord(1:count);
end

function D = pairDistance(A,B)
    if isempty(A) || isempty(B)
        D = zeros(size(A,1),size(B,1));
        return;
    end
    D = sqrt(max(0, ...
        sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B')));
end

function D = pointSegmentDistance(P,A,B)
    AB = B - A;
    denom = sum(AB.^2);
    if denom <= eps
        D = sqrt(sum((P - A).^2,2));
        return;
    end
    T = ((P - A)*AB')/denom;
    T = max(0,min(1,T));
    Projection = A + T.*AB;
    D = sqrt(sum((P - Projection).^2,2));
end

function [Xn,MinV,SpanV] = normalizeRows(X)
    if isempty(X)
        MinV = zeros(1,0);
        SpanV = ones(1,0);
        Xn = X;
        return;
    end
    MinV = min(X,[],1);
    MaxV = max(X,[],1);
    SpanV = MaxV - MinV;
    SpanV(SpanV <= eps) = 1;
    Xn = (X - MinV)./SpanV;
    Xn(~isfinite(Xn)) = 0;
end

function Xn = normalizeWithScale(X,MinV,SpanV)
    if isempty(X)
        Xn = X;
        return;
    end
    SpanV = SpanV(:)';
    SpanV(SpanV <= eps) = 1;
    Xn = (X - MinV(:)')./SpanV;
    Xn(~isfinite(Xn)) = 0;
end
