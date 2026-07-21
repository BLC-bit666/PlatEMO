function BMem = UpdateBoundaryMemory_RC(PrevBMem, ...
        Population1,Offspring1,Population2,Offspring2,W,Options)
%UPDATEBOUNDARYMEMORY_RC Update the fixed boundary-anchor memory.
%   The memory pairs nondominated feasible anchors with nearby infeasible
%   solutions under reference-vector neighborhoods. Repeated rows are
%   intentionally retained because they define the training weights.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Collect and validate current evaluated solutions
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

    %% Append retained feasible anchors from the previous memory
    currentCount = size(Y,1);
    [X,Y,C] = appendPreviousAnchors(X,Y,C,PrevBMem,D,M);
    pairableInfeasible = false(size(Y,1),1);
    pairableInfeasible(1:currentCount) = true;
    %% Associate solutions with reference vectors and harvest pairs
    feasible = sum(max(0,C),2) <= 0;
    Yn = normalizeRows(Y);
    Ref = assignReferences(Yn,W);
    BMem = harvestCloud(X,Y,Yn,feasible,Ref,W,Options, ...
        pairableInfeasible,D,M);
    BMem = filterGapCap(BMem,Options);
end

function Cloud = harvestCloud(X,Y,Yn,Feasible,Ref,W,Options, ...
        PairableInfeasible,D,M)
%HARVESTCLOUD Pair feasible anchors with nearby infeasible solutions.

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
    Cloud = emptyBMem(nnz(mainFeasible),D,M);
    row = 0;
    Neighborhoods = referenceNeighborhoods(W,radius);

    for r = 1 : size(W,1)
        neighborhood = Neighborhoods{r};
        feasibleIdx = find(mainFeasible & Ref == r);
        infeasibleIdx = find(~Feasible & PairableInfeasible & ...
            ismember(Ref,neighborhood));
        if isempty(feasibleIdx) || isempty(infeasibleIdx)
            continue;
        end
        distance = pairDistance(Yn(feasibleIdx,:),Yn(infeasibleIdx,:));
        for a = 1 : numel(feasibleIdx)
            f = feasibleIdx(a);
            legal = ~feasibleDominatesInfeasible( ...
                Y(f,:),Y(infeasibleIdx,:));
            if ~any(legal)
                continue;
            end
            [minGap,which] = min(distance(a,legal));
            legalIdx = infeasibleIdx(legal);
            row = row + 1;
            Cloud.ref(row,1) = r;
            Cloud.gap(row,1) = minGap;
            Cloud.x_b(row,:) = X(f,:);
            Cloud.y_b(row,:) = Y(f,:);
            Cloud.x_i(row,:) = X(legalIdx(which),:);
        end
    end
    Cloud = subsetMemory(Cloud,1:row);
end

function keep = capAnchorsPerRef(keep,Y,Ref,Options)
%CAPANCHORSPERREF Limit feasible anchors retained by each reference vector.

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

function [X,Y,C] = appendPreviousAnchors(X,Y,C,PrevBMem,D,M)
%APPENDPREVIOUSANCHORS Carry feasible anchors into the current event.

    Prev = ensureMemoryFields(PrevBMem,D,M);
    if isempty(Prev.y_b)
        return;
    end
    valid = all(isfinite(Prev.x_b),2) & all(isfinite(Prev.y_b),2);
    Prev = subsetMemory(Prev,valid);
    count = size(Prev.y_b,1);
    X = [X;Prev.x_b];
    Y = [Y;Prev.y_b];
    C = [C;zeros(count,size(C,2))];
end

function Cloud = filterGapCap(Cloud,Options)
%FILTERGAPCAP Remove anomalously distant boundary pairs using a MAD cap.

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
%PARETORANKLIMITED Rank only the requested number of Pareto fronts.
%   The boundary rule uses a 1e-12 dominance tolerance, so the exact-ranking
%   platform NDSort routine is not semantically interchangeable here.

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
%FIRSTFRONTMASK Identify the first Pareto front with a small tolerance.

    n = size(Y,1);
    tolerance = 1e-12;
    target = reshape(Y,n,1,[]);
    candidate = reshape(Y,1,n,[]);
    dominated = all(candidate <= target+tolerance,3) & ...
        any(candidate < target-tolerance,3);
    front = ~any(dominated,2);
end

function yes = feasibleDominatesInfeasible(Yf,Yi)
%FEASIBLEDOMINATESINFEASIBLE Test objective dominance for a pair.

    tolerance = 1e-12;
    yes = all(Yf <= Yi+tolerance,2) & any(Yf < Yi-tolerance,2);
end

function Ref = assignReferences(Yn,W)
%ASSIGNREFERENCES Associate normalized objectives with reference vectors.

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
%NEIGHBORREFS Return the local reference-vector neighborhood.

    if radius <= 0
        refs = r;
        return;
    end
    distance = sqrt(sum((W-W(r,:)).^2,2));
    [~,order] = sort(distance,'ascend');
    refs = order(1:min(numel(order),1+2*radius));
end

function Neighborhoods = referenceNeighborhoods(W,radius)
%REFERENCENEIGHBORHOODS Cache fixed neighborhoods across generations.

    persistent cachedW cachedRadius cachedNeighborhoods
    if isempty(cachedNeighborhoods) || ...
            ~isequal(W,cachedW) || radius ~= cachedRadius
        cachedW = W;
        cachedRadius = radius;
        cachedNeighborhoods = cell(size(W,1),1);
        for r = 1 : size(W,1)
            cachedNeighborhoods{r} = neighborRefs(W,r,radius);
        end
    end
    Neighborhoods = cachedNeighborhoods;
end

function D = pairDistance(A,B)
%PAIRDISTANCE Calculate Euclidean distances without an extra toolbox.

    if isempty(A) || isempty(B)
        D = zeros(size(A,1),size(B,1));
    else
        D = sqrt(max(0,sum(A.^2,2)+sum(B.^2,2)'-2*(A*B')));
    end
end

function Xn = normalizeRows(X)
%NORMALIZEROWS Min-max normalize each objective over the current rows.

    minimum = min(X,[],1);
    span = max(X,[],1)-minimum;
    span(span <= eps) = 1;
    Xn = (X-minimum)./span;
    Xn(~isfinite(Xn)) = 0;
end

function BMem = subsetMemory(BMem,keep)
%SUBSETMEMORY Apply a row mask consistently to every memory field.

    names = fieldnames(BMem);
    for i = 1 : numel(names)
        BMem.(names{i}) = BMem.(names{i})(keep,:);
    end
end

function BMem = ensureMemoryFields(BMem,D,M)
%ENSUREMEMORYFIELDS Complete or initialize the boundary-memory schema.

    if isempty(BMem) || ~isstruct(BMem)
        BMem = emptyBMem(0,D,M);
        return;
    end
    n = size(BMem.y_b,1);
    if ~isfield(BMem,'ref'); BMem.ref = zeros(n,1); end
    if ~isfield(BMem,'gap'); BMem.gap = nan(n,1); end
    if ~isfield(BMem,'x_b'); BMem.x_b = nan(n,D); end
    if ~isfield(BMem,'y_b'); BMem.y_b = nan(n,M); end
    if ~isfield(BMem,'x_i'); BMem.x_i = nan(n,D); end
end

function [D,M] = inferDimensions(Samples,PrevBMem)
%INFERDIMENSIONS Infer decision and objective dimensions.

    if ~isempty(Samples)
        D = size(Samples.decs,2);
        M = size(Samples.objs,2);
    elseif isstruct(PrevBMem) && isfield(PrevBMem,'x_b')
        D = size(PrevBMem.x_b,2);
        M = size(PrevBMem.y_b,2);
    else
        D = 0;
        M = 0;
    end
end

function BMem = emptyBMem(N,D,M)
%EMPTYBMEM Create an empty boundary-memory structure.

    BMem = struct( ...
        'ref',zeros(N,1), ...
        'gap',zeros(N,1), ...
        'x_b',nan(N,D), ...
        'y_b',zeros(N,M), ...
        'x_i',nan(N,D));
end

function value = optionInteger(Options,name,defaultValue,minimum)
%OPTIONINTEGER Read and normalize an integer option.

    value = defaultValue;
    if isstruct(Options) && isfield(Options,name) && ...
            ~isempty(Options.(name))
        value = double(Options.(name));
    end
    if isfinite(value)
        value = max(minimum,round(value));
    end
end
