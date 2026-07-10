function [TrainX,TrainC,QueryC,BMem,Info] = BuildBoundaryDataset_RC( ...
    BMem,Samples,W,Problem,Options)
%BUILDBOUNDARYDATASET_RC Region-conditioned CGAN data from a boundary cloud.
%   Default condition = coarse reference direction W(region,:) only. Optional
%   modes append one scalar: a per-ref local boundary coordinate or a radial
%   objective-space projection.

    conditionMode = regionConditionModeFromOptions(Options);
    condDim = regionConditionDim(W,conditionMode);
    Info = emptyInfo(Problem.D,Problem.M,conditionMode,condDim);
    if isempty(BMem) || ~isfield(BMem,'x_b') || isempty(BMem.x_b)
        TrainX = zeros(0,Problem.D);
        TrainC = zeros(0,condDim);
        QueryC = zeros(0,condDim);
        return;
    end

    if isstruct(Samples) || ~isempty(Samples)
        Y = Samples.objs;
        Y = Y(all(isfinite(Y),2),:);
    else
        Y = zeros(0,Problem.M);
    end
    [objMin,objSpan] = conditionScale(Options,[Y;BMem.y_b]);
    Info.objMin = objMin;
    Info.objSpan = objSpan;

    validTrain = all(isfinite(BMem.x_b),2) & all(isfinite(BMem.y_b),2);
    TrainX = BMem.x_b(validTrain,:);
    TrainY = BMem.y_b(validTrain,:);
    refs = round(double(BMem.ref(validTrain)));
    [TrainC,TrainScalar] = regionConditionsWithScalar( ...
        W,refs,conditionMode,TrainY,objMin,objSpan);
    Info.trainConditions = TrainC;
    Info.trainConditionScalar = TrainScalar;
    Info.trainObjs = TrainY;
    Info.trainRef = refs;
    Info.valid_train_count = sum(validTrain);

    uRegions = unique(refs,'stable');
    [QueryC,QueryRegions,QueryScalar,AllQueryC,AllQueryRegions, ...
        AllQueryScalar] = buildRegionQueryConditions( ...
        W,uRegions,conditionMode,refs,TrainScalar);
    Info.queryConditions = QueryC;
    Info.queryConditionScalar = QueryScalar;
    Info.queryRegions = QueryRegions;
    Info.queryObjs = regionMeanObjs(TrainY,refs,QueryRegions);
    Info.query_count = size(QueryC,1);
    Info.allQueryConditions = AllQueryC;
    Info.allQueryRegions = AllQueryRegions;
    Info.allQueryConditionScalar = AllQueryScalar;

    if size(TrainC,2) ~= condDim || size(QueryC,2) ~= condDim
        error('CBSRegionCGAN:ConditionDimMismatch', ...
            'Region condition must have %d columns.',condDim);
    end
end

function mode = regionConditionModeFromOptions(Options)
    mode = "region";
    if isstruct(Options) && isfield(Options,'conditionMode') && ...
            ~isempty(Options.conditionMode)
        mode = string(Options.conditionMode);
    end
    mode = lower(strtrim(mode));
    switch mode
        case {"region","ref","ref_only","reference","reference_only"}
            mode = "region";
        case {"region_slocal","s_local","slocal","local","local_position"}
            mode = "region_slocal";
        case {"region_rho","rho","radial","region_radial"}
            mode = "region_rho";
        otherwise
            error('CBSRegionCGAN:BadConditionMode', ...
                'Unsupported region condition mode: %s.',mode);
    end
end

function dim = regionConditionDim(W,conditionMode)
    dim = size(W,2);
    switch conditionMode
        case "region"
            return;
        case {"region_slocal","region_rho"}
            dim = dim + 1;
        otherwise
            error('CBSRegionCGAN:BadConditionMode', ...
                'Unsupported region condition mode: %s.',conditionMode);
    end
end

function [C,Scalar] = regionConditionsWithScalar( ...
        W,refs,conditionMode,Y,objMin,objSpan)
    Base = regionConditions(W,refs);
    switch conditionMode
        case "region"
            Scalar = zeros(numel(refs),0);
            C = Base;
        case "region_slocal"
            Scalar = localBoundaryCoordinate(W,Y,refs,objMin,objSpan);
            C = [Base,Scalar];
        case "region_rho"
            Scalar = radialReferenceCoordinate(W,refs,Y,objMin,objSpan);
            C = [Base,Scalar];
        otherwise
            error('CBSRegionCGAN:BadConditionMode', ...
                'Unsupported region condition mode: %s.',conditionMode);
    end
    C(~isfinite(C)) = 0;
end

function C = regionConditions(W,refs)
    condDim = size(W,2);
    refs = round(double(refs(:)));
    n = numel(refs);
    C = zeros(n,condDim);
    if ~isempty(W) && condDim > 0
        ok = isfinite(refs) & refs >= 1 & refs <= size(W,1);
        if any(ok)
            C(ok,:) = double(W(refs(ok),:));
        end
    end
    C(~isfinite(C)) = 0;
end

function [QueryC,QueryRegions,QueryScalar,AllQueryC,AllQueryRegions, ...
        AllQueryScalar] = buildRegionQueryConditions( ...
        W,uRegions,conditionMode,trainRefs,trainScalar)
    if conditionMode == "region"
        QueryRegions = uRegions(:);
        QueryC = regionConditions(W,QueryRegions);
        QueryScalar = zeros(numel(QueryRegions),0);
        AllQueryRegions = (1:size(W,1))';
        AllQueryC = regionConditions(W,AllQueryRegions);
        AllQueryScalar = zeros(numel(AllQueryRegions),0);
        return;
    end

    [QueryRegions,QueryScalar] = scalarQueryGrid( ...
        uRegions,trainRefs,trainScalar,false);
    QueryC = [regionConditions(W,QueryRegions),QueryScalar];

    [AllQueryRegions,AllQueryScalar] = scalarQueryGrid( ...
        (1:size(W,1))',trainRefs,trainScalar,true);
    AllQueryC = [regionConditions(W,AllQueryRegions),AllQueryScalar];
end

function [Regions,Scalars] = scalarQueryGrid( ...
        candidateRegions,trainRefs,trainScalar,includeMissing)
    candidateRegions = round(double(candidateRegions(:)));
    trainRefs = round(double(trainRefs(:)));
    trainScalar = double(trainScalar(:));
    Regions = zeros(0,1);
    Scalars = zeros(0,1);
    for i = 1 : numel(candidateRegions)
        ref = candidateRegions(i);
        values = trainScalar(trainRefs == ref);
        values = clippedUnitInterval(values(isfinite(values)));
        if isempty(values)
            if includeMissing
                values = 0.5;
            else
                continue;
            end
        else
            values = scalarSupportValues(values);
        end
        Regions = [Regions;repmat(ref,numel(values),1)]; %#ok<AGROW>
        Scalars = [Scalars;values(:)]; %#ok<AGROW>
    end
end

function values = scalarSupportValues(values)
    values = clippedUnitInterval(values(:));
    if isempty(values)
        values = 0.5;
        return;
    end
    lo = min(values);
    hi = max(values);
    if hi - lo <= eps
        values = median(values);
    else
        values = [lo + 0.25*(hi - lo); lo + 0.75*(hi - lo)];
    end
    values = clippedUnitInterval(values);
end

function S = localBoundaryCoordinate(W,Y,refs,objMin,objSpan)
    n = numel(refs);
    S = 0.5*ones(n,1);
    if n == 0
        return;
    end
    YN = normalizeObjectiveCondition(Y,objMin,objSpan,size(Y,2),n);
    uRefs = unique(refs(isfinite(refs)),'stable');
    for i = 1 : numel(uRefs)
        rows = find(refs == uRefs(i));
        if numel(rows) <= 1
            S(rows) = 0.5;
            continue;
        end
        Scores = localProjectionScores(W,uRefs(i),YN(rows,:));
        lo = min(Scores);
        hi = max(Scores);
        if hi - lo <= eps
            S(rows) = 0.5;
        else
            S(rows) = (Scores - lo)./(hi - lo);
        end
    end
    S = clippedUnitInterval(S);
end

function Scores = localProjectionScores(W,ref,Y)
    Y = double(Y);
    Centered = Y - mean(Y,1);
    if isempty(Centered) || all(abs(Centered(:)) <= eps)
        Scores = zeros(size(Y,1),1);
        return;
    end
    radial = referenceWeights(W(ref,:),size(Y,2));
    radial = radial(:);
    radialNorm = norm(radial);
    if radialNorm > eps
        radial = radial./radialNorm;
        Centered = Centered - (Centered*radial)*radial';
    end
    if all(abs(Centered(:)) <= eps)
        Scores = zeros(size(Y,1),1);
        return;
    end
    [~,~,V] = svd(Centered,'econ');
    direction = V(:,1);
    direction = orientLocalDirection(direction,radial);
    Scores = Centered*direction;
end

function direction = orientLocalDirection(direction,radial)
    if numel(direction) == 2 && numel(radial) == 2 && norm(radial) > eps
        tangent = [-radial(2);radial(1)];
        if dot(direction,tangent) < 0
            direction = -direction;
        end
        return;
    end
    [~,idx] = max(abs(direction));
    if direction(idx) < 0
        direction = -direction;
    end
end

function Rho = radialReferenceCoordinate(W,refs,Y,objMin,objSpan)
    n = numel(refs);
    M = size(Y,2);
    YN = normalizeObjectiveCondition(Y,objMin,objSpan,M,n);
    Rho = zeros(n,1);
    for i = 1 : n
        ref = refs(i);
        if isfinite(ref) && ref >= 1 && ref <= size(W,1)
            weights = referenceWeights(W(ref,:),M);
            Rho(i) = sum(YN(i,:).*weights);
        else
            Rho(i) = mean(YN(i,:),'omitnan');
        end
    end
    Rho = clippedUnitInterval(Rho);
end

function weights = referenceWeights(w,M)
    weights = double(w(:)');
    if numel(weights) ~= M
        weights = weights(1:min(numel(weights),M));
        if numel(weights) < M
            weights(end+1:M) = 0;
        end
    end
    weights(~isfinite(weights) | weights < 0) = 0;
    total = sum(weights);
    if total <= eps
        weights = ones(1,M)./max(1,M);
    else
        weights = weights./total;
    end
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
        error('CBSRegionCGAN:BadConditionObjectives', ...
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
    value(~isfinite(value)) = 0.5;
    value = max(0,min(1,value));
end

function Q = regionMeanObjs(Yb,refs,uRegions)
    M = size(Yb,2);
    Q = zeros(numel(uRegions),M);
    for k = 1 : numel(uRegions)
        rows = refs == uRegions(k);
        if any(rows)
            Q(k,:) = mean(Yb(rows,:),1);
        end
    end
end

function [MinV,SpanV] = conditionScale(Options,X)
    if isstruct(Options) && isfield(Options,'conditionScale') && ...
            isstruct(Options.conditionScale) && ...
            isfield(Options.conditionScale,'objMin') && ...
            isfield(Options.conditionScale,'objSpan')
        MinV = double(Options.conditionScale.objMin(:)');
        SpanV = double(Options.conditionScale.objSpan(:)');
        if numel(MinV) == size(X,2) && numel(SpanV) == size(X,2)
            SpanV(SpanV <= eps) = 1;
            return;
        end
    end
    if isempty(X)
        MinV = zeros(1,size(X,2));
        SpanV = ones(1,size(X,2));
        return;
    end
    MinV = min(X,[],1);
    SpanV = max(X,[],1) - MinV;
    SpanV(SpanV <= eps) = 1;
end

function Info = emptyInfo(D,M,conditionMode,condDim)
    if nargin < 3 || isempty(conditionMode)
        conditionMode = "region";
    end
    if nargin < 4 || isempty(condDim)
        condDim = 0;
    end
    scalarDim = double(string(conditionMode) ~= "region");
    Info = struct( ...
        'condition_mode',string(conditionMode), ...
        'objMin',zeros(1,M), ...
        'objSpan',ones(1,M), ...
        'valid_train_count',0, ...
        'query_count',0, ...
        'trainConditions',zeros(0,condDim), ...
        'trainConditionScalar',zeros(0,scalarDim), ...
        'trainObjs',zeros(0,M), ...
        'trainRef',zeros(0,1), ...
        'queryConditions',zeros(0,condDim), ...
        'queryConditionScalar',zeros(0,scalarDim), ...
        'queryRegions',zeros(0,1), ...
        'queryObjs',zeros(0,M), ...
        'allQueryConditions',zeros(0,condDim), ...
        'allQueryConditionScalar',zeros(0,scalarDim), ...
        'allQueryRegions',zeros(0,1));
    Info.D = D;
end
