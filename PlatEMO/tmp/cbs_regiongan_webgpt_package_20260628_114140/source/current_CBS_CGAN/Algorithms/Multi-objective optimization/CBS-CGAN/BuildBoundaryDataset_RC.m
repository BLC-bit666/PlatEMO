function [TrainX,TrainC,QueryC,BMem,Info] = BuildBoundaryDataset_RC( ...
    BMem,Samples,W,Problem,Options)
%BUILDBOUNDARYDATASET_RC Region-conditioned CGAN data from a boundary cloud.
%   Condition = COARSE reference direction W(region,:) only (the objective y is
%   deliberately dropped, so z carries the within-region decision variation).
%   TrainX = every cloud decision; TrainC = its region direction. QueryC = one
%   condition per populated region, to be sampled with multiple random z.

    condDim = size(W,2);
    Info = emptyInfo(Problem.D,Problem.M);
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
    refs = round(double(BMem.ref(validTrain)));
    TrainC = regionConditions(W,refs);
    Info.trainConditions = TrainC;
    Info.trainObjs = BMem.y_b(validTrain,:);
    Info.trainRef = refs;
    Info.valid_train_count = sum(validTrain);

    uRegions = unique(refs,'stable');
    QueryC = regionConditions(W,uRegions);
    Info.queryConditions = QueryC;
    Info.queryRegions = uRegions;
    Info.queryObjs = regionMeanObjs(BMem.y_b(validTrain,:),refs,uRegions);
    Info.query_count = size(QueryC,1);

    if size(TrainC,2) ~= condDim || size(QueryC,2) ~= condDim
        error('CBSRegionCGAN:ConditionDimMismatch', ...
            'Region condition must have %d columns.',condDim);
    end
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

function Info = emptyInfo(D,M)
    Info = struct( ...
        'condition_mode',"region", ...
        'objMin',zeros(1,M), ...
        'objSpan',ones(1,M), ...
        'valid_train_count',0, ...
        'query_count',0, ...
        'trainConditions',zeros(0,0), ...
        'trainObjs',zeros(0,M), ...
        'trainRef',zeros(0,1), ...
        'queryConditions',zeros(0,0), ...
        'queryRegions',zeros(0,1), ...
        'queryObjs',zeros(0,M));
    Info.D = D;
end
