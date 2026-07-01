function [TrainX,TrainC,QueryC,BMem,Info] = BuildBoundaryDataset_CBS( ...
    BMem,Samples,W,Problem,Options)
%BUILDBOUNDARYDATASET_CBS Convert paired boundary memory into CGAN data.

    conditionMode = conditionModeFromOptions(Options);
    conditionDim = referenceConditionDim(W,Problem.M,conditionMode);
    if isempty(BMem) || ~isfield(BMem,'y_b') || isempty(BMem.y_b)
        TrainX = zeros(0,Problem.D);
        TrainC = zeros(0,conditionDim);
        QueryC = zeros(0,conditionDim);
        Info = emptyInfo(Problem.D,Problem.M,conditionMode);
        return;
    end

    BMem = ensureMemoryFields(BMem,Problem.D,Problem.M);
    Y = Samples.objs;
    valid = all(isfinite(Y),2);
    Y = Y(valid,:);

    [objMin,objSpan] = conditionScaleFromOptions(Options,[Y;BMem.y_b]);
    Info = emptyInfo(Problem.D,Problem.M,conditionMode);
    Info.objMin = objMin;
    Info.objSpan = objSpan;
    Info.bmem_input_count = size(BMem.y_b,1);

    [QueryY,QueryMeta] = buildExternalQueries( ...
        BMem,W,Options.queryConditionBudget);
    validTrain = all(isfinite(BMem.x_b),2) & ...
        all(isfinite(BMem.x_f),2) & all(isfinite(BMem.x_i),2) & ...
        all(isfinite(BMem.y_b),2) & all(isfinite(BMem.y_f),2) & ...
        all(isfinite(BMem.y_i),2) & isfinite(BMem.gap(:));
    TrainX = BMem.x_b(validTrain,:);
    TrainC = referenceConditionsFromRefs(W,BMem.ref(validTrain), ...
        Problem.M,conditionMode, ...
        BMem.y_b(validTrain,:),objMin,objSpan);
    Info.trainConditions = TrainC;
    Info.valid_train_count = sum(validTrain);
    Info.invalid_train_count = numel(validTrain) - Info.valid_train_count;
    Info.query_count = size(QueryY,1);

    Info.trainObjs = BMem.y_b(validTrain,:);
    Info.trainXf = BMem.x_f(validTrain,:);
    Info.trainXi = BMem.x_i(validTrain,:);
    Info.trainYf = BMem.y_f(validTrain,:);
    Info.trainYi = BMem.y_i(validTrain,:);
    Info.trainRef = BMem.ref(validTrain,:);
    Info.queryObjs = QueryY;
    Info.queryMeta = QueryMeta;

    if isempty(QueryY)
        QueryC = zeros(0,conditionDim);
    else
        QueryC = referenceConditionsFromRefs(W,QueryMeta.ref, ...
            Problem.M,conditionMode, ...
            QueryY,objMin,objSpan);
    end
    Info.queryConditions = QueryC;
end

function C = referenceConditionsFromRefs( ...
        W,refs,M,conditionMode,Y,objMin,objSpan)
    dim = referenceConditionDim(W,M,conditionMode);
    refDim = referenceTokenDim(W,M);
    refs = round(double(refs(:)));
    n = numel(refs);
    Ref = zeros(n,refDim);
    if ~isempty(W) && refDim > 0
        valid = isfinite(refs) & refs >= 1 & refs <= size(W,1);
        if any(valid)
            Ref(valid,1:refDim) = double(W(refs(valid),1:refDim));
        end
    end
    switch conditionMode
        case "ref_only"
            C = Ref;
        case "ref_y"
            C = [Ref,normalizeObjectiveCondition(Y,objMin,objSpan,M,n)];
        otherwise
            error('CBSCGAN:BadConditionMode', ...
                'Unsupported condition mode: %s.',conditionMode);
    end
    if size(C,2) ~= dim
        error('CBSCGAN:ConditionDimMismatch', ...
            'Condition mode %s produced %d columns, expected %d.', ...
            conditionMode,size(C,2),dim);
    end
    C(~isfinite(C)) = 0;
end

function YN = normalizeObjectiveCondition(Y,objMin,objSpan,M,n)
    if nargin < 5 || isempty(n)
        n = size(Y,1);
    end
    Y = double(Y);
    if isempty(Y)
        Y = zeros(n,M);
    end
    if size(Y,1) ~= n || size(Y,2) ~= M
        error('CBSCGAN:BadConditionObjectives', ...
            'Condition objective rows must be n-by-M.');
    end
    objMin = double(objMin(:)');
    objSpan = double(objSpan(:)');
    if numel(objMin) ~= M || numel(objSpan) ~= M
        objMin = zeros(1,M);
        objSpan = ones(1,M);
    end
    objSpan(objSpan <= eps) = 1;
    YN = (Y - objMin)./objSpan;
    YN(~isfinite(YN)) = 0;
end

function value = clippedUnitInterval(value)
    value(~isfinite(value)) = 0;
    value = max(0,min(1,value));
end

function dim = referenceConditionDim(W,M,conditionMode)
    refDim = referenceTokenDim(W,M);
    switch conditionMode
        case "ref_only"
            dim = refDim;
        case "ref_y"
            dim = refDim + M;
        otherwise
            error('CBSCGAN:BadConditionMode', ...
                'Unsupported condition mode: %s.',conditionMode);
    end
end

function dim = referenceTokenDim(W,M)
    if ~isempty(W) && isnumeric(W)
        dim = size(W,2);
    else
        dim = M;
    end
    dim = max(0,round(double(dim)));
end

function mode = conditionModeFromOptions(Options)
    mode = "ref_y";
    if isstruct(Options) && isfield(Options,'conditionMode') && ...
            ~isempty(Options.conditionMode)
        mode = string(Options.conditionMode);
    end
    mode = lower(strtrim(mode));
    switch mode
        case {"ref_only","ref","reference","reference_only"}
            mode = "ref_only";
        case {"ref_y","ref_obj","reference_objective"}
            mode = "ref_y";
        otherwise
            error('CBSCGAN:BadConditionMode', ...
                'Unsupported condition mode: %s.',mode);
    end
end

function [QueryY,QueryMeta] = buildExternalQueries(BMem,W,budget)
    QueryY = zeros(0,size(BMem.y_b,2));
    QueryMeta = emptyQueryMeta(size(BMem.x_b,2),size(BMem.y_b,2));
    if budget <= 0 || isempty(BMem.y_b) || size(BMem.y_b,1) < 2
        return;
    end

    usedRefs = unique(BMem.ref(:)');
    emptyRefs = setdiff(1:size(W,1),usedRefs);
    idx = sortedBoundaryNodes(BMem,W);
    for k = 1 : numel(idx)-1
        r1 = BMem.ref(idx(k));
        r2 = BMem.ref(idx(k+1));
        [missingRefs,tValues] = emptyRefsOnReferenceSegment( ...
            W,emptyRefs,r1,r2);
        for j = 1 : numel(missingRefs)
            t = clippedUnitInterval(tValues(j));
            y = (1-t)*BMem.y_b(idx(k),:) + t*BMem.y_b(idx(k+1),:);
            Support = interpolatePairSupport(BMem,idx(k),idx(k+1),t);
            [QueryY,QueryMeta] = appendQuery(QueryY,QueryMeta, ...
                y,missingRefs(j),[r1 r2],"missing_ref",t,Support);
            if size(QueryY,1) >= budget
                return;
            end
        end
    end

    segmentRows = adjacentBoundarySegments(BMem,W,idx);
    for row = 1 : size(segmentRows,1)
        if size(QueryY,1) >= budget
            return;
        end
        a = segmentRows(row,1);
        b = segmentRows(row,2);
        t = 0.5;
        y = 0.5*(BMem.y_b(a,:) + BMem.y_b(b,:));
        ref = nearestReference(W,0.5*(W(BMem.ref(a),:) + W(BMem.ref(b),:)));
        interval = sort([BMem.ref(a),BMem.ref(b)]);
        Support = interpolatePairSupport(BMem,a,b,t);
        [QueryY,QueryMeta] = appendQuery(QueryY,QueryMeta, ...
            y,ref,interval,"large_gap",t,Support);
    end
end

function rows = adjacentBoundarySegments(BMem,W,idx)
    if numel(idx) < 2
        rows = zeros(0,3);
        return;
    end
    rows = zeros(numel(idx)-1,3);
    for k = 1 : numel(idx)-1
        a = idx(k);
        b = idx(k+1);
        refDist = sqrt(sum((W(BMem.ref(a),:) - W(BMem.ref(b),:)).^2));
        rows(k,:) = [a,b,max([BMem.gap(a),BMem.gap(b),refDist])];
    end
    [~,ord] = sort(rows(:,3),'descend');
    rows = rows(ord,1:2);
end

function Support = interpolatePairSupport(BMem,a,b,t)
    Support = struct( ...
        'x_f',(1-t)*BMem.x_f(a,:) + t*BMem.x_f(b,:), ...
        'x_i',(1-t)*BMem.x_i(a,:) + t*BMem.x_i(b,:), ...
        'y_f',(1-t)*BMem.y_f(a,:) + t*BMem.y_f(b,:), ...
        'y_i',(1-t)*BMem.y_i(a,:) + t*BMem.y_i(b,:));
end

function ref = nearestReference(W,point)
    if isempty(W)
        ref = 1;
        return;
    end
    d = sqrt(sum((W - point).^2,2));
    [~,ref] = min(d);
end

function Meta = emptyQueryMeta(D,M)
    Meta = struct( ...
        'ref',zeros(0,1), ...
        'interp_t',zeros(0,1), ...
        'source_interval',zeros(0,2), ...
        'source_type',strings(0,1), ...
        'x_f',zeros(0,D), ...
        'x_i',zeros(0,D), ...
        'y_f',zeros(0,M), ...
        'y_i',zeros(0,M));
end

function [QueryY,Meta] = appendQuery( ...
    QueryY,Meta,y,ref,interval,sourceType,interpT,Support)
    QueryY(end+1,:) = y;
    Meta.ref(end+1,1) = ref;
    Meta.interp_t(end+1,1) = clippedUnitInterval(interpT);
    Meta.source_interval(end+1,:) = interval;
    Meta.source_type(end+1,1) = string(sourceType);
    Meta.x_f(end+1,:) = Support.x_f;
    Meta.x_i(end+1,:) = Support.x_i;
    Meta.y_f(end+1,:) = Support.y_f;
    Meta.y_i(end+1,:) = Support.y_i;
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

function [refs,t] = emptyRefsOnReferenceSegment(W,emptyRefs,r1,r2)
    refs = [];
    t = [];
    if isempty(emptyRefs)
        return;
    end
    A = W(r1,:);
    B = W(r2,:);
    AB = B - A;
    denom = sum(AB.^2);
    if denom <= eps
        return;
    end
    P = W(emptyRefs,:);
    tv = ((P - A)*AB')/denom;
    projection = A + tv.*AB;
    dist = sqrt(sum((P - projection).^2,2));
    span = sqrt(denom);
    onSegment = tv > 1e-9 & tv < 1-1e-9 & dist <= max(1e-12,0.25*span);
    refs = emptyRefs(onSegment);
    t = tv(onSegment);
    if ~isempty(refs)
        [~,ord] = sortrows([dist(onSegment),t]);
        refs = refs(ord);
        t = t(ord);
    end
end

function BMem = ensureMemoryFields(BMem,D,M)
    if ~isfield(BMem,'ref'); BMem.ref = zeros(size(BMem.y_b,1),1); end
    if ~isfield(BMem,'gap'); BMem.gap = nan(size(BMem.y_b,1),1); end
    if ~isfield(BMem,'x_b'); BMem.x_b = nan(size(BMem.y_b,1),D); end
    if ~isfield(BMem,'x_f'); BMem.x_f = nan(size(BMem.y_b,1),D); end
    if ~isfield(BMem,'y_f'); BMem.y_f = nan(size(BMem.y_b,1),M); end
    if ~isfield(BMem,'x_i'); BMem.x_i = nan(size(BMem.y_b,1),D); end
    if ~isfield(BMem,'y_i'); BMem.y_i = nan(size(BMem.y_b,1),M); end
    if size(BMem.y_b,2) ~= M
        error('CBSCGAN:BMemObjectiveDim', ...
            'Boundary memory objective dimension does not match Problem.M.');
    end
end

function Info = emptyInfo(D,M,conditionMode)
    if nargin < 3 || isempty(conditionMode)
        conditionMode = "ref_y";
    end
    Info = struct( ...
        'condition_mode',string(conditionMode), ...
        'objMin',zeros(1,M), ...
        'objSpan',ones(1,M), ...
        'bmem_input_count',0, ...
        'valid_train_count',0, ...
        'invalid_train_count',0, ...
        'query_count',0, ...
        'trainObjs',zeros(0,M), ...
        'trainXf',zeros(0,D), ...
        'trainXi',zeros(0,D), ...
        'trainYf',zeros(0,M), ...
        'trainYi',zeros(0,M), ...
        'trainRef',zeros(0,1), ...
        'trainConditions',zeros(0,0), ...
        'queryObjs',zeros(0,M), ...
        'queryConditions',zeros(0,0), ...
        'queryMeta',emptyQueryMeta(D,M));
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

function [MinV,SpanV] = conditionScaleFromOptions(Options,X)
    if isstruct(Options) && isfield(Options,'conditionScale') && ...
            isstruct(Options.conditionScale) && ...
            isfield(Options.conditionScale,'objMin') && ...
            isfield(Options.conditionScale,'objSpan')
        MinV = double(Options.conditionScale.objMin);
        SpanV = double(Options.conditionScale.objSpan);
        if size(MinV,1) ~= 1
            MinV = MinV(:)';
        end
        if size(SpanV,1) ~= 1
            SpanV = SpanV(:)';
        end
        if numel(MinV) == size(X,2) && numel(SpanV) == size(X,2)
            SpanV(SpanV <= eps) = 1;
            return;
        end
    end
    [~,MinV,SpanV] = normalizeRows(X);
end
