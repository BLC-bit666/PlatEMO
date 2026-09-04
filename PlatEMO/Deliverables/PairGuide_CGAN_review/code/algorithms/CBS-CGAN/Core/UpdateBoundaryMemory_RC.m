function [BMem,RefScale,Trace] = UpdateBoundaryMemory_RC(PrevBMem, ...
        Population1,Offspring1,Population2,Offspring2,W,Options)
%UPDATEBOUNDARYMEMORY_RC Update the fixed boundary-anchor memory.
%   The memory pairs nondominated feasible anchors with nearby infeasible
%   solutions under reference-vector neighborhoods. Repeated rows are
%   intentionally retained because they define the training weights.
%   REFSCALE records the objective normalization used for those references.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Trace = emptyMemoryTrace();
    %% Collect and validate current evaluated solutions
    Samples = [Population1,Offspring1,Population2,Offspring2];
    [D,M] = inferDimensions(Samples,PrevBMem);
    RefScale = struct('minimum',zeros(1,M),'span',ones(1,M));
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
    [Ref,RefScale,Yn] = AssignReferenceVectors_CBS(Y,W);
    [BMem,Trace] = harvestCloud(X,Y,Yn,feasible,Ref,W,Options, ...
        pairableInfeasible,D,M);
    [BMem,Trace.madDropped] = filterGapCap(BMem,Options);
    Trace.paired = sum(all(isfinite(BMem.x_i),2));
    Trace.unpaired = size(BMem.x_b,1)-Trace.paired;
    Trace.retained = size(BMem.x_b,1);
    [Trace.previousUnpaired,Trace.previousUnpairedPaired] = ...
        previousUnpairedConversion(PrevBMem,BMem,D,M);
end

function [Cloud,Trace] = harvestCloud(X,Y,Yn,Feasible,Ref,W,Options, ...
        PairableInfeasible,D,M)
%HARVESTCLOUD Pair feasible anchors with nearby infeasible solutions.

    Trace = emptyMemoryTrace();
    Cloud = emptyBMem(0,D,M);
    if isempty(Y) || isempty(W)
        return;
    end
    frontDepth = optionInteger(Options,'frontDepth',2,1);
    pairRefCount = pairReferenceCount(Options);
    mainFeasible = false(size(Feasible));
    feasibleRows = find(Feasible);
    Trace.trueFeasible = numel(feasibleRows);
    if ~isempty(feasibleRows)
        if isfinite(frontDepth)
            rank = paretoRankLimited(Y(feasibleRows,:),frontDepth);
            mainFeasible(feasibleRows(rank <= frontDepth)) = true;
        else
            mainFeasible(feasibleRows) = true;
        end
    end
    Trace.afterFront = nnz(mainFeasible);
    Trace.frontDropped = Trace.trueFeasible-Trace.afterFront;
    allFeasibleRefs = unique(Ref(Feasible),'stable');
    frontRefs = unique(Ref(mainFeasible),'stable');
    Trace.frontOpportunityRefs = numel(setdiff(allFeasibleRefs,frontRefs));
    [mainFeasible,Trace.capDropped] = ...
        capAnchorsPerRef(mainFeasible,Y,Ref,Options);
    Trace.afterCap = nnz(mainFeasible);
    Cloud = emptyBMem(nnz(mainFeasible),D,M);
    row = 0;
    Neighborhoods = referenceNeighborhoods(W,pairRefCount);
    allInfeasible = find(~Feasible & PairableInfeasible);
    keepUnpaired = optionLogical(Options,'keepUnpairedAnchors',false);
    pairGaps = zeros(0,1);
    pairRanks = zeros(0,1);
    pairAngles = zeros(0,1);

    for r = 1 : size(W,1)
        neighborhood = Neighborhoods{r};
        feasibleIdx = find(mainFeasible & Ref == r);
        infeasibleIdx = find(~Feasible & PairableInfeasible & ...
            ismember(Ref,neighborhood));
        if isempty(feasibleIdx)
            continue;
        end
        distance = pairDistance(Yn(feasibleIdx,:),Yn(infeasibleIdx,:));
        [~,refOrder] = sort(sqrt(sum((W-W(r,:)).^2,2)),'ascend');
        refRank = zeros(size(W,1),1);
        refRank(refOrder) = 1:numel(refOrder);
        for a = 1 : numel(feasibleIdx)
            f = feasibleIdx(a);
            if ~isempty(allInfeasible)
                legalAll = ~feasibleDominatesInfeasible( ...
                    Y(f,:),Y(allInfeasible,:));
                legalRanks = refRank(Ref(allInfeasible(legalAll)));
                Trace.dominanceRejected = Trace.dominanceRejected+ ...
                    sum(~legalAll);
                Trace.legalAny = Trace.legalAny+any(legalRanks >= 1);
                Trace.legalWithin5 = Trace.legalWithin5+any(legalRanks <= 5);
                Trace.legalWithin10 = Trace.legalWithin10+any(legalRanks <= 10);
            end
            if keepUnpaired
                row = row+1;
                Cloud.ref(row,1) = r;
                Cloud.x_b(row,:) = X(f,:);
                Cloud.y_b(row,:) = Y(f,:);
            end
            if isempty(infeasibleIdx)
                continue;
            end
            legal = ~feasibleDominatesInfeasible( ...
                Y(f,:),Y(infeasibleIdx,:));
            if ~any(legal)
                continue;
            end
            [minGap,which] = min(distance(a,legal));
            legalIdx = infeasibleIdx(legal);
            if ~keepUnpaired
                row = row + 1;
                Cloud.ref(row,1) = r;
                Cloud.x_b(row,:) = X(f,:);
                Cloud.y_b(row,:) = Y(f,:);
            end
            partner = legalIdx(which);
            Cloud.ref(row,1) = r;
            Cloud.gap(row,1) = minGap;
            Cloud.x_i(row,:) = X(partner,:);
            pairGaps(end+1,1) = minGap; %#ok<AGROW>
            pairRanks(end+1,1) = refRank(Ref(partner)); %#ok<AGROW>
            pairAngles(end+1,1) = referenceAngle( ...
                W(r,:),W(Ref(partner),:)); %#ok<AGROW>
        end
    end
    Cloud = subsetMemory(Cloud,1:row);
    Trace.pairedBeforeMAD = numel(pairGaps);
    Trace.unpairedBeforeMAD = row-Trace.pairedBeforeMAD;
    Trace.pairRank1To5 = sum(pairRanks <= 5);
    Trace.pairRank6To10 = sum(pairRanks > 5 & pairRanks <= 10);
    Trace.pairRankOver10 = sum(pairRanks > 10);
    Trace.pairGapMedian = finiteMedian(pairGaps);
    Trace.pairGapP90 = finitePercentile(pairGaps,0.9);
    Trace.pairAngleMedian = finiteMedian(pairAngles);
    Trace.pairAngleP90 = finitePercentile(pairAngles,0.9);
end

function [keep,dropped] = capAnchorsPerRef(keep,Y,Ref,Options)
%CAPANCHORSPERREF Limit feasible anchors retained by each reference vector.

    before = nnz(keep);
    maxPerRef = optionInteger(Options,'maxAnchorsPerRef',Inf,1);
    if ~isfinite(maxPerRef)
        dropped = 0;
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
    dropped = before-nnz(keep);
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

function [Cloud,dropped] = filterGapCap(Cloud,Options)
%FILTERGAPCAP Remove anomalously distant boundary pairs using a MAD cap.

    dropped = 0;
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
    unpaired = ~isfinite(gaps);
    keepPaired = isfinite(gaps) & gaps <= cap + 1e-12;
    if sum(keepPaired) < min(minCount,numel(finiteGaps))
        pairedRows = find(isfinite(gaps));
        [~,order] = sort(gaps(pairedRows),'ascend');
        keepPaired = false(size(gaps));
        rescue = pairedRows(order(1:min(minCount,numel(order))));
        keepPaired(rescue) = true;
    end
    keep = unpaired | keepPaired;
    dropped = sum(isfinite(gaps) & ~keepPaired);
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

function refs = neighborRefs(W,r,count)
%NEIGHBORREFS Return the local reference-vector neighborhood.

    if isinf(count)
        refs = (1:size(W,1))';
        return;
    end
    distance = sqrt(sum((W-W(r,:)).^2,2));
    [~,order] = sort(distance,'ascend');
    refs = order(1:min(numel(order),count));
end

function Neighborhoods = referenceNeighborhoods(W,count)
%REFERENCENEIGHBORHOODS Cache fixed neighborhoods across generations.

    persistent cachedW cachedCount cachedNeighborhoods
    if isempty(cachedNeighborhoods) || ...
            ~isequal(W,cachedW) || count ~= cachedCount
        cachedW = W;
        cachedCount = count;
        cachedNeighborhoods = cell(size(W,1),1);
        for r = 1 : size(W,1)
            cachedNeighborhoods{r} = neighborRefs(W,r,count);
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
        'gap',nan(N,1), ...
        'x_b',nan(N,D), ...
        'y_b',zeros(N,M), ...
        'x_i',nan(N,D));
end

function count = pairReferenceCount(Options)
%PAIRREFERENCECOUNT Prefer the explicit count and retain old test support.

    if isstruct(Options) && isfield(Options,'pairNeighborRefCount') && ...
            ~isempty(Options.pairNeighborRefCount)
        count = optionInteger(Options,'pairNeighborRefCount',5,1);
    else
        radius = optionInteger(Options,'pairNeighborRefRadius',2,0);
        count = 1+2*radius;
    end
end

function value = optionLogical(Options,name,defaultValue)
    value = defaultValue;
    if isstruct(Options) && isfield(Options,name) && ...
            ~isempty(Options.(name))
        candidate = double(Options.(name));
        if ~isscalar(candidate) || ~isfinite(candidate) || ...
                ~ismember(candidate,[0 1])
            error('CBSRegionGAN:BadMemoryOption', ...
                '%s must be either 0 or 1.',name);
        end
        value = logical(candidate);
    end
end

function angle = referenceAngle(a,b)
    denominator = norm(a)*norm(b);
    if denominator <= eps
        angle = NaN;
    else
        cosine = max(-1,min(1,dot(a,b)/denominator));
        angle = acos(cosine)*180/pi;
    end
end

function value = finiteMedian(values)
    values = values(isfinite(values));
    if isempty(values)
        value = NaN;
    else
        value = median(values);
    end
end

function value = finitePercentile(values,fraction)
    values = sort(values(isfinite(values)));
    if isempty(values)
        value = NaN;
    else
        value = values(max(1,ceil(fraction*numel(values))));
    end
end

function [count,paired] = previousUnpairedConversion(Prev,Current,D,M)
    Prev = ensureMemoryFields(Prev,D,M);
    Current = ensureMemoryFields(Current,D,M);
    previousRows = all(isfinite(Prev.x_b),2) & ...
        ~all(isfinite(Prev.x_i),2);
    currentRows = all(isfinite(Current.x_b),2) & ...
        all(isfinite(Current.x_i),2);
    count = sum(previousRows);
    if count == 0 || ~any(currentRows)
        paired = 0;
    else
        paired = sum(ismember(Prev.x_b(previousRows,:), ...
            Current.x_b(currentRows,:),'rows'));
    end
end

function Trace = emptyMemoryTrace()
    Trace = struct('trueFeasible',0,'afterFront',0,'frontDropped',0, ...
        'frontOpportunityRefs',0,'afterCap',0,'capDropped',0, ...
        'retained',0,'pairedBeforeMAD',0,'unpairedBeforeMAD',0, ...
        'paired',0,'unpaired',0,'madDropped',0, ...
        'legalWithin5',0,'legalWithin10',0,'legalAny',0, ...
        'dominanceRejected',0,'pairRank1To5',0,'pairRank6To10',0, ...
        'pairRankOver10',0,'pairGapMedian',NaN,'pairGapP90',NaN, ...
        'pairAngleMedian',NaN,'pairAngleP90',NaN, ...
        'previousUnpaired',0,'previousUnpairedPaired',0);
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
