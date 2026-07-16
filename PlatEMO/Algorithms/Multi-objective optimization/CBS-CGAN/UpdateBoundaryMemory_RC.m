function BMem = UpdateBoundaryMemory_RC(PrevBMem, ...
        Population1,Offspring1,Population2,Offspring2,W,Options)
%UPDATEBOUNDARYMEMORY_RC Update the fixed legacy boundary memory.

    Samples = [Population1,Offspring1,Population2,Offspring2];
    [D,M] = inferDimensions(Samples,PrevBMem);
    if isempty(Samples)
        BMem = emptyBMem(0,D,M);
        return;
    end
    X = Samples.decs;
    Y = Samples.objs;
    C = Samples.cons;
    if isempty(C)
        C = zeros(size(X,1),1);
    end
    valid = all(isfinite(X),2) & all(isfinite(Y),2);
    X = X(valid,:);
    Y = Y(valid,:);
    C = C(valid,:);
    if isempty(Y)
        BMem = emptyBMem(0,D,M);
        return;
    end

    currentCount = size(Y,1);
    [X,Y,C,PrevRows] = appendPreviousAnchors(X,Y,C,PrevBMem,D,M);
    source = [zeros(currentCount,1);ones(size(Y,1)-currentCount,1)];
    age = zeros(size(Y,1),1);
    if size(Y,1) > currentCount
        previousAge = double(PrevRows.age_f(:));
        previousAge(~isfinite(previousAge) | previousAge < 0) = 0;
        age(currentCount+1:end) = previousAge + 1;
    end
    pairableInfeasible = false(size(Y,1),1);
    pairableInfeasible(1:currentCount) = true;
    feasible = sum(max(0,C),2) <= 0;
    Yn = normalizeRows(Y);
    Ref = assignReferences(Yn,W);
    BMem = harvestCloud(X,Y,Yn,feasible,Ref,W,Options, ...
        pairableInfeasible,source,age,D,M);
    BMem = filterGapCap(BMem,Options);
end

function Cloud = harvestCloud(X,Y,Yn,Feasible,Ref,W,Options, ...
        PairableInfeasible,Source,Age,D,M)
    Cloud = emptyBMem(0,D,M);
    if isempty(Y) || isempty(W)
        return;
    end
    frontDepth = optionInteger(Options,'frontDepth',2,1);
    radius = optionInteger(Options,'pairNeighborRefRadius',2,0);
    mainFeasible = false(size(Feasible));
    feasibleRows = find(Feasible);
    if ~isempty(feasibleRows)
        rank = paretoRankLimited(Y(feasibleRows,:),frontDepth);
        mainFeasible(feasibleRows(rank <= frontDepth)) = true;
    end
    mainFeasible = capAnchorsPerRef(mainFeasible,Y,Ref,Options);

    for r = 1 : size(W,1)
        neighborhood = neighborRefs(W,r,radius);
        feasibleIdx = find(mainFeasible & Ref == r);
        infeasibleIdx = find(~Feasible & PairableInfeasible & ...
            ismember(Ref,neighborhood));
        if isempty(feasibleIdx) || isempty(infeasibleIdx)
            continue;
        end
        distance = pairDistance(Yn(feasibleIdx,:),Yn(infeasibleIdx,:));
        [minGap,nearest] = min(distance,[],2);
        for a = 1 : numel(feasibleIdx)
            f = feasibleIdx(a);
            i = infeasibleIdx(nearest(a));
            if feasibleDominatesInfeasible(Y(f,:),Y(i,:))
                continue;
            end
            Cloud.ref(end+1,1) = r;
            Cloud.gap(end+1,1) = minGap(a);
            Cloud.x_b(end+1,:) = X(f,:);
            Cloud.y_b(end+1,:) = Y(f,:);
            Cloud.x_f(end+1,:) = X(f,:);
            Cloud.y_f(end+1,:) = Y(f,:);
            Cloud.x_i(end+1,:) = X(i,:);
            Cloud.y_i(end+1,:) = Y(i,:);
            Cloud.source_f(end+1,1) = Source(f);
            Cloud.age_f(end+1,1) = Age(f);
        end
    end
end

function keep = capAnchorsPerRef(keep,Y,Ref,Options)
    maxPerRef = optionInteger(Options,'maxAnchorsPerRef',Inf,1);
    if ~isfinite(maxPerRef)
        return;
    end
    refs = unique(Ref(keep),'stable');
    capped = false(size(keep));
    for k = 1 : numel(refs)
        idx = find(keep & Ref == refs(k));
        if numel(idx) <= maxPerRef
            capped(idx) = true;
        else
            Fitness = CalFitness_CBS(Y(idx,:));
            [~,order] = sort(Fitness,'ascend');
            capped(idx(order(1:maxPerRef))) = true;
        end
    end
    keep = capped;
end

function [X,Y,C,Prev] = appendPreviousAnchors(X,Y,C,PrevBMem,D,M)
    Prev = ensureMemoryFields(PrevBMem,D,M);
    if isempty(Prev.y_f)
        return;
    end
    valid = all(isfinite(Prev.x_f),2) & all(isfinite(Prev.y_f),2);
    Prev = subsetMemory(Prev,valid);
    count = size(Prev.y_f,1);
    X = [X;Prev.x_f];
    Y = [Y;Prev.y_f];
    C = [C;zeros(count,size(C,2))];
end

function Cloud = filterGapCap(Cloud,Options)
    if isempty(Cloud.y_b)
        return;
    end
    minCount = optionInteger(Options,'minBoundaryLength',2,1);
    if size(Cloud.y_b,1) <= minCount
        return;
    end
    gaps = Cloud.gap(:);
    finiteGaps = gaps(isfinite(gaps));
    if isempty(finiteGaps)
        cap = Inf;
    else
        med = median(finiteGaps);
        deviation = median(abs(finiteGaps-med));
        if deviation <= eps
            cap = med*1.25 + 1e-12;
        else
            cap = med + 3*1.4826*deviation;
        end
    end
    keep = gaps <= cap + 1e-12;
    if sum(keep) < minCount
        [~,order] = sort(gaps,'ascend');
        keep = false(size(gaps));
        keep(order(1:min(minCount,numel(order)))) = true;
    end
    Cloud = subsetMemory(Cloud,keep);
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
    tolerance = 1e-12;
    for i = 1 : n
        for j = 1 : n
            if i ~= j && all(Y(j,:) <= Y(i,:) + tolerance) && ...
                    any(Y(j,:) < Y(i,:) - tolerance)
                front(i) = false;
                break;
            end
        end
    end
end

function yes = feasibleDominatesInfeasible(Yf,Yi)
    tolerance = 1e-12;
    yes = all(Yf <= Yi+tolerance,2) & any(Yf < Yi-tolerance,2);
end

function Ref = assignReferences(Yn,W)
    n = size(Yn,1);
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    NormY = sqrt(sum(Yn.^2,2));
    Yu = Yn./max(NormY,eps);
    [~,Ref] = max(Yu*Wn',[],2);
    zeroRows = NormY <= eps;
    if any(zeroRows)
        distance = pairDistance(Yn(zeroRows,:),W);
        [~,Ref(zeroRows)] = min(distance,[],2);
    end
    Ref = reshape(Ref,n,1);
end

function refs = neighborRefs(W,r,radius)
    if radius <= 0
        refs = r;
        return;
    end
    distance = sqrt(sum((W-W(r,:)).^2,2));
    [~,order] = sort(distance,'ascend');
    refs = order(1:min(numel(order),1+2*radius));
end

function D = pairDistance(A,B)
    if isempty(A) || isempty(B)
        D = zeros(size(A,1),size(B,1));
    else
        D = sqrt(max(0,sum(A.^2,2)+sum(B.^2,2)'-2*(A*B')));
    end
end

function Xn = normalizeRows(X)
    minimum = min(X,[],1);
    span = max(X,[],1)-minimum;
    span(span <= eps) = 1;
    Xn = (X-minimum)./span;
    Xn(~isfinite(Xn)) = 0;
end

function BMem = subsetMemory(BMem,keep)
    names = fieldnames(BMem);
    for i = 1 : numel(names)
        BMem.(names{i}) = BMem.(names{i})(keep,:);
    end
end

function BMem = ensureMemoryFields(BMem,D,M)
    if isempty(BMem) || ~isstruct(BMem)
        BMem = emptyBMem(0,D,M);
        return;
    end
    n = size(BMem.y_f,1);
    if ~isfield(BMem,'ref'); BMem.ref = zeros(n,1); end
    if ~isfield(BMem,'gap'); BMem.gap = nan(n,1); end
    if ~isfield(BMem,'x_b'); BMem.x_b = nan(n,D); end
    if ~isfield(BMem,'y_b'); BMem.y_b = nan(n,M); end
    if ~isfield(BMem,'x_f'); BMem.x_f = nan(n,D); end
    if ~isfield(BMem,'y_f'); BMem.y_f = nan(n,M); end
    if ~isfield(BMem,'x_i'); BMem.x_i = nan(n,D); end
    if ~isfield(BMem,'y_i'); BMem.y_i = nan(n,M); end
    if ~isfield(BMem,'source_f'); BMem.source_f = zeros(n,1); end
    if ~isfield(BMem,'age_f'); BMem.age_f = zeros(n,1); end
end

function [D,M] = inferDimensions(Samples,PrevBMem)
    if ~isempty(Samples)
        D = size(Samples.decs,2);
        M = size(Samples.objs,2);
    elseif isstruct(PrevBMem) && isfield(PrevBMem,'x_f')
        D = size(PrevBMem.x_f,2);
        M = size(PrevBMem.y_f,2);
    else
        D = 0;
        M = 0;
    end
end

function BMem = emptyBMem(N,D,M)
    BMem = struct( ...
        'ref',zeros(N,1), ...
        'gap',zeros(N,1), ...
        'x_b',nan(N,D), ...
        'y_b',zeros(N,M), ...
        'x_f',nan(N,D), ...
        'y_f',zeros(N,M), ...
        'x_i',nan(N,D), ...
        'y_i',zeros(N,M), ...
        'source_f',zeros(N,1), ...
        'age_f',zeros(N,1));
end

function value = optionInteger(Options,name,defaultValue,minimum)
    value = defaultValue;
    if isstruct(Options) && isfield(Options,name) && ...
            ~isempty(Options.(name))
        value = double(Options.(name));
    end
    if isfinite(value)
        value = max(minimum,round(value));
    end
end
