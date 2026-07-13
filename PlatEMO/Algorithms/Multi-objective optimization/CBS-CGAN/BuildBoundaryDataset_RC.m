function [TrainX,TrainC,QueryC,BMem,Info] = BuildBoundaryDataset_RC( ...
    BMem,Samples,W,Problem,Options)
%BUILDBOUNDARYDATASET_RC Build the reference-conditioned WGAN data set.
%   TrainX contains the feasible-side boundary decisions. TrainC contains
%   only their coarse reference vectors. QueryC contains the currently
%   populated reference vectors; all reference vectors are exposed through
%   Info for the fixed one-sixth frontier query.

    condDim = size(W,2);
    Info = emptyInfo(Problem.D,Problem.M,condDim);
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
    [Info.objMin,Info.objSpan] = conditionScale( ...
        Options,[Y;BMem.y_b]);

    validTrain = all(isfinite(BMem.x_b),2) & ...
        all(isfinite(BMem.y_b),2);
    TrainX = BMem.x_b(validTrain,:);
    TrainY = BMem.y_b(validTrain,:);
    refs = round(double(BMem.ref(validTrain)));
    TrainSource = memoryColumn(BMem,'source_f',validTrain,0);
    TrainAge = memoryColumn(BMem,'age_f',validTrain,0);
    TrainFrontRank = memoryColumn( ...
        BMem,'front_rank_f',validTrain,NaN);
    Info.raw_valid_train_count = size(TrainX,1);
    [uniqueKeep,exactDuplicateCount] = exactRefDecisionKeep(TrainX,refs);
    Info.exact_duplicate_train_count = exactDuplicateCount;
    Info.train_dedup_mode = trainDedupMode(Options);
    if Info.train_dedup_mode == "exact_ref_x"
        TrainX = TrainX(uniqueKeep,:);
        TrainY = TrainY(uniqueKeep,:);
        refs = refs(uniqueKeep,:);
        TrainSource = TrainSource(uniqueKeep,:);
        TrainAge = TrainAge(uniqueKeep,:);
        TrainFrontRank = TrainFrontRank(uniqueKeep,:);
    end
    Info.removed_duplicate_train_count = ...
        Info.raw_valid_train_count - size(TrainX,1);
    Info.train_dedup_enabled = double(Info.train_dedup_mode ~= "off");
    TrainC = referenceConditions(W,refs);

    Info.trainConditions = TrainC;
    Info.trainObjs = TrainY;
    Info.trainRef = refs;
    Info.trainSource = TrainSource;
    Info.trainAge = TrainAge;
    Info.trainFrontRank = TrainFrontRank;
    Info.valid_train_count = size(TrainX,1);

    queryRefs = unique(refs,'stable');
    QueryC = referenceConditions(W,queryRefs);
    Info.queryConditions = QueryC;
    Info.queryRegions = queryRefs(:);
    Info.query_count = size(QueryC,1);
    Info.allQueryRegions = (1:size(W,1))';
    Info.allQueryConditions = referenceConditions(W,Info.allQueryRegions);

    if size(TrainC,2) ~= condDim || size(QueryC,2) ~= condDim
        error('CBSRegionWGAN:ConditionDimMismatch', ...
            'Reference condition must have %d columns.',condDim);
    end
end

function [keep,duplicateCount] = exactRefDecisionKeep(X,refs)
%EXACTREFDECISIONKEEP Keep the first exact (ref,x) training row.
    n = size(X,1);
    keep = true(n,1);
    duplicateCount = 0;
    if n <= 1
        return;
    end
    [~,first] = unique([double(refs(:)),double(X)],'rows','stable');
    keep = false(n,1);
    keep(first) = true;
    duplicateCount = n - numel(first);
end

function mode = trainDedupMode(Options)
    mode = "off";
    if isstruct(Options) && isfield(Options,'trainDedupMode') && ...
            ~isempty(Options.trainDedupMode)
        mode = lower(strip(string(Options.trainDedupMode)));
    end
    if ~isscalar(mode) || ~ismember(mode,["off","exact_ref_x"])
        error('CBSRegionGAN:BadTrainDedupMode', ...
            'trainDedupMode must be "off" or "exact_ref_x".');
    end
end

function C = referenceConditions(W,refs)
    refs = round(double(refs(:)));
    C = zeros(numel(refs),size(W,2));
    if isempty(W)
        return;
    end
    valid = isfinite(refs) & refs >= 1 & refs <= size(W,1);
    if any(valid)
        C(valid,:) = double(W(refs(valid),:));
    end
    C(~isfinite(C)) = 0;
end

function value = memoryColumn(BMem,name,keep,defaultValue)
    n = numel(keep);
    value = repmat(double(defaultValue),n,1);
    if isstruct(BMem) && isfield(BMem,name) && ...
            numel(BMem.(name)) == n
        value = double(BMem.(name)(:));
    end
    value = value(keep,:);
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

function Info = emptyInfo(D,M,condDim)
    Info = struct( ...
        'objMin',zeros(1,M), ...
        'objSpan',ones(1,M), ...
        'raw_valid_train_count',0, ...
        'valid_train_count',0, ...
        'exact_duplicate_train_count',0, ...
        'removed_duplicate_train_count',0, ...
        'train_dedup_enabled',0, ...
        'train_dedup_mode',"off", ...
        'query_count',0, ...
        'trainConditions',zeros(0,condDim), ...
        'trainObjs',zeros(0,M), ...
        'trainRef',zeros(0,1), ...
        'trainSource',zeros(0,1), ...
        'trainAge',zeros(0,1), ...
        'trainFrontRank',zeros(0,1), ...
        'queryConditions',zeros(0,condDim), ...
        'queryRegions',zeros(0,1), ...
        'allQueryConditions',zeros(0,condDim), ...
        'allQueryRegions',zeros(0,1), ...
        'D',D);
end
