function [BMem,Diag] = UpdateBoundaryMemory_RC(PrevBMem, ...
    Population1,Offspring1,Population2,Offspring2,W,Options)
%UPDATEBOUNDARYMEMORY_RC Boundary cloud for the region-conditioned WGAN.
%   Feasible anchors come from the first K feasible objective fronts and are
%   capped per reference. Each anchor is paired with its nearest current
%   infeasible neighbor, then filtered by the adaptive gap cap. Feasible
%   anchors from the previous BMem compete in the same current pool and are
%   re-paired; previous infeasible endpoints are never reused.
%   Options.bmemMode="coherent" switches to current-window normalization,
%   strict one-refresh TTL, pair-before-cap selection, and no objective-
%   dominance pair rejection. The default "legacy" path is unchanged.

    Samples = [Population1,Offspring1,Population2,Offspring2];
    [D,M] = inferMemoryDimensions(Samples,PrevBMem);
    if isempty(Samples)
        BMem = emptyBMem(0,D,M);
        Diag = makeDiag(BMem);
        return;
    end

    X = Samples.decs;
    Y = Samples.objs;
    C = Samples.cons;
    if size(C,2) == 0
        C = zeros(size(X,1),1);
    end

    valid = all(isfinite(X),2) & all(isfinite(Y),2);
    X = X(valid,:);
    Y = Y(valid,:);
    C = C(valid,:);
    if isempty(Y)
        BMem = emptyBMem(0,D,M);
        Diag = makeDiag(BMem);
        return;
    end

    if boundaryMemoryMode(Options) == "coherent"
        [BMem,Diag] = updateCoherentBoundaryMemory( ...
            X,Y,C,PrevBMem,W,Options,D,M);
        return;
    end

    Stats = emptyPrevBMemStats();
    currentCount = size(Y,1);
    [X,Y,C,PrevRows,Stats.prev_bmem_candidate_count] = ...
        appendPreviousFeasibleAnchors(X,Y,C,PrevBMem,D,M);
    AnchorSource = [zeros(currentCount,1); ...
        ones(size(Y,1) - currentCount,1)];
    AnchorAge = zeros(size(Y,1),1);
    if size(Y,1) > currentCount
        PrevAge = double(PrevRows.age_f(:));
        PrevAge(~isfinite(PrevAge) | PrevAge < 0) = 0;
        AnchorAge(currentCount+1:end) = PrevAge + 1;
    end
    PairableInfeasible = false(size(Y,1),1);
    PairableInfeasible(1:currentCount) = true;

    CV = sum(max(0,C),2);
    Feasible = CV <= 0;

    [Yn,~,~] = normalizeRows(Y);
    Ref = assignReferences(Yn,W);

    Cloud = harvestBoundaryCloud(X,Y,Yn,Feasible,Ref,W,Options, ...
        PairableInfeasible,AnchorSource,AnchorAge);
    Cloud = filterGapCap(Cloud,Options);
    Stats.prev_bmem_survivor_count = countPreviousSurvivors(Cloud,PrevRows);
    BMem = Cloud;
    Diag = makeDiag(BMem,Stats);
end

function [BMem,Diag] = updateCoherentBoundaryMemory( ...
        CurrentX,CurrentY,CurrentC,PrevBMem,W,Options,D,M)
    Stats = emptyPrevBMemStats();
    currentCount = size(CurrentY,1);
    [X,Y,C,PrevRows,Stats.prev_bmem_candidate_count] = ...
        appendPreviousFeasibleAnchorsStrictTTL( ...
        CurrentX,CurrentY,CurrentC,PrevBMem,D,M);

    AnchorSource = [zeros(currentCount,1); ...
        ones(size(Y,1) - currentCount,1)];
    AnchorAge = zeros(size(Y,1),1);
    if size(Y,1) > currentCount
        AnchorAge(currentCount+1:end) = 1;
    end
    PairableInfeasible = false(size(Y,1),1);
    PairableInfeasible(1:currentCount) = true;

    CV = sum(max(0,C),2);
    Feasible = CV <= 0;
    [~,MinV,SpanV] = normalizeRows(CurrentY);
    Yn = normalizeRowsUsingScale(Y,MinV,SpanV);
    Ref = assignReferences(Yn,W);

    Cloud = harvestBoundaryCloud(X,Y,Yn,Feasible,Ref,W,Options, ...
        PairableInfeasible,AnchorSource,AnchorAge);
    Cloud = filterGapCap(Cloud,Options);
    Cloud = capPairedCloudPerRef(Cloud,Options);
    Stats.prev_bmem_survivor_count = countPreviousSurvivors(Cloud,PrevRows);
    BMem = Cloud;
    Diag = makeDiag(BMem,Stats);
end

function Cloud = harvestBoundaryCloud(X,Y,Yn,Feasible,Ref,W,Options, ...
        PairableInfeasible,AnchorSource,AnchorAge)
    D = size(X,2);
    M = size(Y,2);
    Cloud = emptyBMem(0,D,M);
    if isempty(Y) || isempty(W)
        return;
    end
    if nargin < 8 || isempty(PairableInfeasible) || ...
            numel(PairableInfeasible) ~= numel(Feasible)
        PairableInfeasible = true(size(Feasible));
    else
        PairableInfeasible = logical(PairableInfeasible(:));
    end
    if nargin < 9 || isempty(AnchorSource) || ...
            numel(AnchorSource) ~= numel(Feasible)
        AnchorSource = zeros(size(Feasible));
    else
        AnchorSource = double(AnchorSource(:));
    end
    if nargin < 10 || isempty(AnchorAge) || ...
            numel(AnchorAge) ~= numel(Feasible)
        AnchorAge = zeros(size(Feasible));
    else
        AnchorAge = double(AnchorAge(:));
    end
    frontDepth = boundaryFrontDepth(Options);
    radius = boundaryNeighborRadius(Options);

    % First K objective-space fronts of the FEASIBLE subset are the anchors.
    % Restricting to the feasible subset is essential: in constrained problems
    % infeasible points usually have better objectives and would otherwise
    % dominate and evict the feasible boundary front.
    mainFeasible = false(size(Feasible));
    frontRank = nan(size(Feasible));
    fAll = find(Feasible);
    if ~isempty(fAll)
        rank = paretoRankLimited(Y(fAll,:),frontDepth);
        frontRank(fAll) = rank;
        mainFeasible(fAll(rank <= frontDepth)) = true;
    end
    coherent = boundaryMemoryMode(Options) == "coherent";
    if ~coherent
        mainFeasible = capFeasibleAnchorsPerRef( ...
            mainFeasible,Y,Ref,Options);
    end

    nRef = size(W,1);
    for r = 1 : nRef
        neigh = neighborRefs(W,r,radius);
        fidx = find(mainFeasible & Ref == r);
        iidx = find(~Feasible & PairableInfeasible & ismember(Ref,neigh));
        if isempty(fidx) || isempty(iidx)
            continue;
        end
        % Normalized-objective distance from each feasible anchor to every
        % neighbourhood infeasible point; nearest infeasible defines the gap
        % (boundary proximity). The legacy path skips an anchor that already
        % dominates its nearest infeasible point; coherent mode retains every
        % current feasible/infeasible neighborhood pair and lets gap quality
        % decide after pairing.
        Dfi = pairDistance(Yn(fidx,:),Yn(iidx,:));
        [minGap,nnI] = min(Dfi,[],2);
        for a = 1 : numel(fidx)
            f = fidx(a);
            i = iidx(nnI(a));
            if ~coherent && feasibleDominatesInfeasible(Y(f,:),Y(i,:))
                continue;
            end
            Cloud.ref(end+1,1) = r;
            Cloud.y_b(end+1,:) = Y(f,:);
            Cloud.gap(end+1,1) = minGap(a);
            Cloud.x_b(end+1,:) = X(f,:);
            Cloud.x_f(end+1,:) = X(f,:);
            Cloud.y_f(end+1,:) = Y(f,:);
            Cloud.x_i(end+1,:) = X(i,:);
            Cloud.y_i(end+1,:) = Y(i,:);
            Cloud.source_f(end+1,1) = AnchorSource(f);
            Cloud.age_f(end+1,1) = AnchorAge(f);
            Cloud.front_rank_f(end+1,1) = frontRank(f);
            Cloud.candidate_row_f(end+1,1) = f;
            Cloud.candidate_row_i(end+1,1) = i;
            PairVector = Yn(i,:) - Yn(f,:);
            PairNorm = sqrt(sum(PairVector.^2,2));
            if isfinite(PairNorm) && PairNorm > eps
                Cloud.pair_normal(end+1,:) = PairVector./PairNorm;
            else
                Cloud.pair_normal(end+1,:) = zeros(1,M);
            end
        end
    end
end

function Cloud = capPairedCloudPerRef(Cloud,Options)
    maxPerRef = boundaryMaxAnchorsPerRef(Options);
    if isempty(Cloud) || isempty(Cloud.y_b) || ~isfinite(maxPerRef)
        return;
    end
    refs = unique(Cloud.ref(:),'stable');
    keep = false(size(Cloud.ref));
    for k = 1 : numel(refs)
        idx = find(Cloud.ref == refs(k));
        if numel(idx) <= maxPerRef
            keep(idx) = true;
            continue;
        end
        Fitness = CalFitness_CBS(Cloud.y_f(idx,:));
        Keys = [finiteSortColumn(Cloud.gap(idx)), ...
            finiteSortColumn(Cloud.age_f(idx)), ...
            finiteSortColumn(Cloud.front_rank_f(idx)), ...
            finiteSortColumn(Fitness), ...
            finiteSortColumn(Cloud.candidate_row_f(idx))];
        [~,order] = sortrows(Keys,1:size(Keys,2));
        keep(idx(order(1:maxPerRef))) = true;
    end
    Cloud = subsetMemory(Cloud,keep);
end

function values = finiteSortColumn(values)
    values = double(values(:));
    values(~isfinite(values)) = Inf;
end

function mainFeasible = capFeasibleAnchorsPerRef(mainFeasible,Y,Ref,Options)
    maxPerRef = boundaryMaxAnchorsPerRef(Options);
    if ~isfinite(maxPerRef)
        return;
    end
    refs = unique(Ref(mainFeasible),'stable');
    keep = false(size(mainFeasible));
    for k = 1 : numel(refs)
        idx = find(mainFeasible & Ref == refs(k));
        if numel(idx) <= maxPerRef
            keep(idx) = true;
            continue;
        end
        Fitness = CalFitness_CBS(Y(idx,:));
        [~,ord] = sort(Fitness,'ascend');
        keep(idx(ord(1:maxPerRef))) = true;
    end
    mainFeasible = keep;
end

function [X,Y,C,PrevRows,prevCount] = appendPreviousFeasibleAnchors( ...
    X,Y,C,PrevBMem,D,M)
    PrevRows = emptyBMem(0,D,M);
    prevCount = 0;
    Prev = ensureMemoryFields(PrevBMem,D,M);
    if isempty(Prev.y_f)
        return;
    end
    valid = all(isfinite(Prev.x_f),2) & all(isfinite(Prev.y_f),2);
    if ~any(valid)
        return;
    end
    PrevRows = subsetMemory(Prev,valid);
    prevCount = size(PrevRows.y_f,1);
    X = [X;PrevRows.x_f];
    Y = [Y;PrevRows.y_f];
    C = [C;zeros(prevCount,size(C,2))];
end

function [X,Y,C,PrevRows,prevCount] = ...
        appendPreviousFeasibleAnchorsStrictTTL( ...
        X,Y,C,PrevBMem,D,M)
    PrevRows = emptyBMem(0,D,M);
    prevCount = 0;
    Prev = ensureMemoryFields(PrevBMem,D,M);
    if isempty(Prev.y_f)
        return;
    end
    age = double(Prev.age_f(:));
    age(~isfinite(age) | age < 0) = 0;
    valid = all(isfinite(Prev.x_f),2) & ...
        all(isfinite(Prev.y_f),2) & age < 1;
    if ~any(valid)
        return;
    end
    Prev.age_f = age;
    PrevRows = subsetMemory(Prev,valid);
    prevCount = size(PrevRows.y_f,1);
    X = [X;PrevRows.x_f];
    Y = [Y;PrevRows.y_f];
    C = [C;zeros(prevCount,size(C,2))];
end

function count = countPreviousSurvivors(BMem,PrevRows)
    count = 0;
    if isempty(BMem) || isempty(PrevRows) || isempty(BMem.y_b) || ...
            isempty(PrevRows.y_f)
        return;
    end
    for i = 1 : size(BMem.y_b,1)
        sameY = all(abs(PrevRows.y_f - BMem.y_b(i,:)) <= 1e-12,2);
        sameX = all(abs(PrevRows.x_f - BMem.x_b(i,:)) <= 1e-12,2);
        if any(sameY & sameX)
            count = count + 1;
        end
    end
end

function Cloud = filterGapCap(Cloud,Options)
    if isempty(Cloud) || isempty(Cloud.y_b)
        return;
    end
    minCount = boundaryMinCount(Options);
    if size(Cloud.y_b,1) <= minCount
        return;
    end
    gaps = Cloud.gap(:);
    cap = adaptiveGapCap(gaps);
    keep = gaps <= cap + 1e-12;
    if sum(keep) < minCount
        [~,ord] = sort(gaps,'ascend');
        keep = false(size(gaps));
        keep(ord(1:min(minCount,numel(ord)))) = true;
    end
    Cloud = subsetMemory(Cloud,keep);
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

function k = boundaryFrontDepth(Options)
    k = 2;
    if isstruct(Options) && isfield(Options,'frontDepth') && ...
            ~isempty(Options.frontDepth)
        k = max(1,round(double(Options.frontDepth)));
    end
end

function n = boundaryMaxAnchorsPerRef(Options)
    n = Inf;
    if isstruct(Options) && isfield(Options,'maxAnchorsPerRef') && ...
            ~isempty(Options.maxAnchorsPerRef)
        value = double(Options.maxAnchorsPerRef);
        if isfinite(value) && value > 0
            n = max(1,round(value));
        end
    end
end

function r = boundaryNeighborRadius(Options)
    r = 2;
    if isstruct(Options) && isfield(Options,'pairNeighborRefRadius') && ...
            ~isempty(Options.pairNeighborRefRadius)
        r = max(0,round(double(Options.pairNeighborRefRadius)));
    end
end

function minCount = boundaryMinCount(Options)
    minCount = 2;
    if isstruct(Options) && isfield(Options,'minBoundaryLength') && ...
            ~isempty(Options.minBoundaryLength)
        minCount = max(1,round(double(Options.minBoundaryLength)));
    end
end

function mode = boundaryMemoryMode(Options)
    mode = "legacy";
    if isstruct(Options) && isfield(Options,'bmemMode') && ...
            ~isempty(Options.bmemMode)
        mode = lower(strip(string(Options.bmemMode)));
    end
    if ~isscalar(mode) || ~ismember(mode,["legacy","coherent"])
        error('CBSRegionGAN:BadBMemMode', ...
            'bmemMode must be "legacy" or "coherent".');
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

function Ref = assignReferences(Yn,W)
    n = size(Yn,1);
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    NormY = sqrt(sum(Yn.^2,2));
    Yu = Yn./max(NormY,eps);
    Score = Yu*Wn';
    [~,Ref] = max(Score,[],2);
    zeroRows = NormY <= eps;
    if any(zeroRows)
        Dz = pairDistance(Yn(zeroRows,:),W);
        [~,Ref(zeroRows)] = min(Dz,[],2);
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
    D = sqrt(max(0,sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B')));
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

function Xn = normalizeRowsUsingScale(X,MinV,SpanV)
    SpanV = double(SpanV(:)');
    MinV = double(MinV(:)');
    SpanV(~isfinite(SpanV) | SpanV <= eps) = 1;
    Xn = (double(X) - MinV)./SpanV;
    Xn(~isfinite(Xn)) = 0;
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
    BMem.source_f = BMem.source_f(keep,:);
    BMem.age_f = BMem.age_f(keep,:);
    BMem.front_rank_f = BMem.front_rank_f(keep,:);
    BMem.candidate_row_f = BMem.candidate_row_f(keep,:);
    BMem.candidate_row_i = BMem.candidate_row_i(keep,:);
    BMem.pair_normal = BMem.pair_normal(keep,:);
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
    if ~isfield(BMem,'source_f'); BMem.source_f = zeros(n,1); end
    if ~isfield(BMem,'age_f'); BMem.age_f = zeros(n,1); end
    if ~isfield(BMem,'front_rank_f'); BMem.front_rank_f = nan(n,1); end
    if ~isfield(BMem,'candidate_row_f'); BMem.candidate_row_f = nan(n,1); end
    if ~isfield(BMem,'candidate_row_i'); BMem.candidate_row_i = nan(n,1); end
    if ~isfield(BMem,'pair_normal'); BMem.pair_normal = nan(n,M); end
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
        'y_i',zeros(N,M), ...
        'source_f',zeros(N,1), ...
        'age_f',zeros(N,1), ...
        'front_rank_f',nan(N,1), ...
        'candidate_row_f',nan(N,1), ...
        'candidate_row_i',nan(N,1), ...
        'pair_normal',nan(N,M));
end

function Diag = makeDiag(BMem,Stats)
    if nargin < 2 || isempty(Stats)
        Stats = emptyPrevBMemStats();
    end
    if isempty(BMem) || isempty(BMem.y_b)
        Diag = struct( ...
            'bmem_count',0, ...
            'region_count',0, ...
            'finite_gap_count',0, ...
            'median_gap',NaN, ...
            'max_gap',NaN, ...
            'points_per_region_median',0, ...
            'points_per_region_max',0, ...
            'prev_bmem_candidate_count',Stats.prev_bmem_candidate_count, ...
            'prev_bmem_survivor_count',Stats.prev_bmem_survivor_count);
        return;
    end
    finiteGap = BMem.gap(isfinite(BMem.gap));
    refs = BMem.ref(:);
    u = unique(refs);
    counts = arrayfun(@(k)sum(refs == k),u);
    Diag = struct( ...
        'bmem_count',size(BMem.y_b,1), ...
        'region_count',numel(u), ...
        'finite_gap_count',numel(finiteGap), ...
        'median_gap',medianOrNaN(finiteGap), ...
        'max_gap',maxOrNaN(finiteGap), ...
        'points_per_region_median',median(counts), ...
        'points_per_region_max',max(counts), ...
        'prev_bmem_candidate_count',Stats.prev_bmem_candidate_count, ...
        'prev_bmem_survivor_count',Stats.prev_bmem_survivor_count);
end

function Stats = emptyPrevBMemStats()
    Stats = struct( ...
        'prev_bmem_candidate_count',0, ...
        'prev_bmem_survivor_count',0);
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
