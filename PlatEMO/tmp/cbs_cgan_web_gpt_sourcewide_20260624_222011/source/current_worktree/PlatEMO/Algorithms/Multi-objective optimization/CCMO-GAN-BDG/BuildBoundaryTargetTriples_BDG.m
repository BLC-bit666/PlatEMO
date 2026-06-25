function [XBoundary,ConditionData,Diag] = BuildBoundaryTargetTriples_BDG( ...
        AF,AI,Problem,W,Options)
%BuildBoundaryTargetTriples_BDG Build target triples for boundary GAN.
%   Default endpoint mode preserves the original contract:
%   x_b is AFDec and condition is [AFNorm, AI-AF direction, optional ref].
%   fix.md modes add a scalar t and replace the endpoint target only when
%   explicitly requested by Options.targetMode.

    if nargin < 4
        W = [];
    end
    if nargin < 5 || isempty(Options)
        Options = struct();
    end

    D = ProblemDimension_BDG(Problem,AF);
    M = ProblemObjectiveCount_BDG(Problem,AF,AI);
    Options = NormalizeTargetTripleOptions_BDG(Options);
    refDim = TargetRefTokenDim_BDG(W,M,Options.conditionMode);
    hasT = ConditionHasT_BDG(Options.conditionMode);
    XBoundary = zeros(0,D);
    ConditionData = zeros(0,2*M + double(hasT) + refDim,'single');
    Diag = EmptyTargetTripleDiag_BDG(M,refDim,Options.conditionMode, ...
        Options.targetMode);

    [AFDec,AFObj] = ArchiveMatrices_BDG(AF,D,M);
    [AIDec,AIObj] = ArchiveMatrices_BDG(AI,D,M);
    nPair = min([size(AFDec,1),size(AIDec,1),size(AFObj,1),size(AIObj,1)]);
    Diag.target_pair_count = double(nPair);
    Diag.target_condition_dim = double(2*M + double(hasT) + refDim);
    if nPair < 2 || D <= 0 || M <= 0
        return;
    end

    AFDec = AFDec(1:nPair,:);
    AIDec = AIDec(1:nPair,:);
    AFObj = AFObj(1:nPair,:);
    AIObj = AIObj(1:nPair,:);
    finiteRows = all(isfinite(AFDec),2) & all(isfinite(AIDec),2) & ...
        all(isfinite(AFObj),2) & all(isfinite(AIObj),2);
    if ~any(finiteRows)
        return;
    end
    AFDec = AFDec(finiteRows,:);
    AIDec = AIDec(finiteRows,:);
    AFObj = AFObj(finiteRows,:);
    AIObj = AIObj(finiteRows,:);
    keepIndex = find(finiteRows);

    [AFNorm,AINorm,zmin,zmax] = NormalizePairedObjectives_BDG(AFObj,AIObj,M);
    Direction = AINorm - AFNorm;
    finiteTargets = all(isfinite(AFNorm),2) & all(isfinite(Direction),2);
    if ~any(finiteTargets) || sum(finiteTargets) < 2
        return;
    end

    AFDec = AFDec(finiteTargets,:);
    AIDec = AIDec(finiteTargets,:);
    AFNorm = AFNorm(finiteTargets,:);
    Direction = Direction(finiteTargets,:);
    keepIndex = keepIndex(finiteTargets);
    RefToken = TargetRefToken_BDG(AF,keepIndex,W,refDim);

    switch Options.targetMode
        case "endpoint"
            pairRows = (1:size(AFDec,1))';
            t = zeros(size(pairRows));
            XBoundary = AFDec;
            fallback = false(size(pairRows));
            nearSourceCount = 0;
            nearCandidateCount = 0;
            nearKeepCount = 0;
        case "decision_interp"
            [XBoundary,pairRows,t] = DecisionInterpolationTargets_BDG( ...
                AFDec,AIDec,Options.decisionInterpCount);
            fallback = false(size(pairRows));
            nearSourceCount = 0;
            nearCandidateCount = 0;
            nearKeepCount = 0;
        case "near_segment_feasible"
            [XBoundary,pairRows,t,fallback,nearSourceCount, ...
                nearCandidateCount,nearKeepCount] = ...
                NearSegmentFeasibleTargets_BDG(AFDec,AFNorm,Direction, ...
                Options,zmin,zmax,D,M);
        otherwise
            error('BuildBoundaryTargetTriples_BDG:BadTargetMode', ...
                'Unsupported targetMode: %s.',Options.targetMode);
    end

    if isempty(XBoundary) || numel(pairRows) < 2
        return;
    end
    ConditionData = ComposeConditionData_BDG(AFNorm,Direction,RefToken, ...
        pairRows,t,hasT,refDim);
    targetKeepIndex = keepIndex(pairRows);
    [XBoundary,ConditionData,pairRows,t,fallback,targetKeepIndex, ...
        FusedDiag] = ApplyFusedTargetFilter_BDG( ...
        XBoundary,ConditionData,pairRows,t,fallback,targetKeepIndex, ...
        AF,W,Options);
    if isempty(XBoundary) || numel(pairRows) < 2
        return;
    end
    Diag.target_keep_index = targetKeepIndex;
    Diag.target_triple_count = double(size(XBoundary,1));
    Diag.target_condition_dim = double(size(ConditionData,2));
    Diag.target_triple_ready = double(size(XBoundary,1) >= 2);
    Diag.target_y_min = min(AFNorm(pairRows,:),[],1);
    Diag.target_y_max = max(AFNorm(pairRows,:),[],1);
    dirNorm = sqrt(sum(Direction(pairRows,:).^2,2));
    Diag.target_direction_norm_mean = MeanFinite_BDG(dirNorm);
    Diag.target_t_min = MinFinite_BDG(t);
    Diag.target_t_max = MaxFinite_BDG(t);
    Diag.target_t_mean = MeanFinite_BDG(t);
    Diag.target_fallback_count = double(sum(fallback));
    Diag.target_near_segment_source_count = double(nearSourceCount);
    Diag.target_near_segment_candidate_count = double(nearCandidateCount);
    if Options.targetMode == "near_segment_feasible"
        Diag.target_near_segment_keep_count = double(sum(~fallback(:)));
    else
        Diag.target_near_segment_keep_count = double(nearKeepCount);
    end
    Diag = MergeStructFields_BDG(Diag,FusedDiag);
end

function Options = NormalizeTargetTripleOptions_BDG(Options)
    if ~isstruct(Options)
        Options = struct();
    end
    if ~isfield(Options,'conditionMode') || isempty(Options.conditionMode)
        Options.conditionMode = "yt_dt";
    end
    if ~isfield(Options,'targetMode') || isempty(Options.targetMode)
        Options.targetMode = "endpoint";
    end
    if ~isfield(Options,'nearSegmentSource')
        Options.nearSegmentSource = [];
    end
    if ~isfield(Options,'nearSegmentTau') || isempty(Options.nearSegmentTau)
        Options.nearSegmentTau = 0.20;
    end
    if ~isfield(Options,'nearSegmentMaxPerPair') || ...
            isempty(Options.nearSegmentMaxPerPair)
        Options.nearSegmentMaxPerPair = 5;
    end
    if ~isfield(Options,'decisionInterpCount') || ...
            isempty(Options.decisionInterpCount)
        Options.decisionInterpCount = 5;
    end
    if ~isfield(Options,'targetFusionMode') || isempty(Options.targetFusionMode)
        Options.targetFusionMode = "none";
    end
    if ~isfield(Options,'conditionKNNK') || isempty(Options.conditionKNNK)
        Options.conditionKNNK = 5;
    end
    if ~isfield(Options,'conditionKNNRetainRatio') || ...
            isempty(Options.conditionKNNRetainRatio)
        Options.conditionKNNRetainRatio = 0.70;
    end
    if ~isfield(Options,'conditionKNNMinCount') || ...
            isempty(Options.conditionKNNMinCount)
        Options.conditionKNNMinCount = 2;
    end
    if ~isfield(Options,'lower')
        Options.lower = [];
    end
    if ~isfield(Options,'upper')
        Options.upper = [];
    end
    Options.conditionMode = NormalizeConditionMode_BDG( ...
        Options.conditionMode);
    Options.targetMode = NormalizeTargetMode_BDG(Options.targetMode);
    Options.targetFusionMode = NormalizeTargetFusionMode_BDG( ...
        Options.targetFusionMode);
    Options.nearSegmentTau = max(0,double(Options.nearSegmentTau));
    Options.nearSegmentMaxPerPair = max(1,round(double( ...
        Options.nearSegmentMaxPerPair)));
    Options.decisionInterpCount = max(2,round(double( ...
        Options.decisionInterpCount)));
    Options.conditionKNNK = max(1,round(double(Options.conditionKNNK)));
    Options.conditionKNNRetainRatio = min(max( ...
        double(Options.conditionKNNRetainRatio),0),1);
    Options.conditionKNNMinCount = max(2,round(double( ...
        Options.conditionKNNMinCount)));
end

function mode = NormalizeConditionMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["yt_dt","yt_dt_ref","yt_dt_t","yt_dt_t_ref"];
    assert(ismember(mode,valid), ...
        'BuildBoundaryTargetTriples_BDG:BadConditionMode', ...
        'conditionMode must be one of: %s.',strjoin(valid,", "));
end

function mode = NormalizeTargetMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["endpoint","near_segment_feasible","decision_interp"];
    assert(ismember(mode,valid), ...
        'BuildBoundaryTargetTriples_BDG:BadTargetMode', ...
        'targetMode must be one of: %s.',strjoin(valid,", "));
end

function mode = NormalizeTargetFusionMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["none","condition_knn"];
    assert(ismember(mode,valid), ...
        'BuildBoundaryTargetTriples_BDG:BadTargetFusionMode', ...
        'targetFusionMode must be one of: %s.',strjoin(valid,", "));
end

function tf = ConditionHasT_BDG(mode)
    mode = NormalizeConditionMode_BDG(mode);
    tf = mode == "yt_dt_t" || mode == "yt_dt_t_ref";
end

function code = TargetModeCode_BDG(mode)
    mode = NormalizeTargetMode_BDG(mode);
    names = ["endpoint","near_segment_feasible","decision_interp"];
    code = find(names == mode,1,'first') - 1;
end

function dim = TargetRefTokenDim_BDG(W,M,mode)
    mode = NormalizeConditionMode_BDG(mode);
    if mode ~= "yt_dt_ref" && mode ~= "yt_dt_t_ref"
        dim = 0;
        return;
    end
    if ~isempty(W) && isnumeric(W)
        dim = size(W,2);
    else
        dim = M;
    end
    dim = max(0,round(double(dim)));
end

function C = ComposeConditionData_BDG(AFNorm,Direction,RefToken, ...
        pairRows,t,hasT,refDim)
    parts = {AFNorm(pairRows,:),Direction(pairRows,:)};
    if hasT
        parts{end+1} = t(:);
    end
    if refDim > 0
        parts{end+1} = RefToken(pairRows,:);
    end
    C = single([parts{:}]);
end

function RefToken = TargetRefToken_BDG(AF,idx,W,refDim)
    RefToken = zeros(numel(idx),refDim);
    if refDim <= 0 || isempty(idx) || isempty(W) || ...
            ~isstruct(AF) || ~isfield(AF,'ref') || isempty(AF.ref)
        return;
    end
    refs = double(AF.ref(:));
    for i = 1 : numel(idx)
        p = idx(i);
        if p < 1 || p > numel(refs)
            continue;
        end
        r = round(refs(p));
        if ~isfinite(r) || r < 1 || r > size(W,1)
            continue;
        end
        cols = min(refDim,size(W,2));
        RefToken(i,1:cols) = double(W(r,1:cols));
    end
end

function [X,C,pairRows,t,fallback,targetKeepIndex,Diag] = ...
        ApplyFusedTargetFilter_BDG(X,C,pairRows,t,fallback, ...
        targetKeepIndex,AF,W,Options)
    Diag = EmptyFusedTargetFilterDiag_BDG();
    nPre = size(X,1);
    if Options.targetFusionMode == "none"
        return;
    end

    [preRefCount,preRefCov] = TargetRefCoverage_BDG(AF,targetKeepIndex,W);
    switch Options.targetFusionMode
        case "condition_knn"
            [keep,spread,k] = BoundaryConditionKNNKeepMask_BDG( ...
                X,C,Options.conditionKNNK, ...
                Options.conditionKNNRetainRatio, ...
                Options.conditionKNNMinCount,Options.lower,Options.upper);
            Diag.target_filter_mode_code = 5;
            Diag.condition_knn_k = double(k);
            Diag.condition_knn_decision_spread_mean = MeanFinite_BDG(spread);
            Diag.condition_knn_decision_spread_median = ...
                MedianFinite_BDG(spread);
            Diag.condition_knn_decision_spread_p90 = ...
                PercentileFinite_BDG(spread,90);
        otherwise
            keep = true(nPre,1);
    end

    keep = logical(keep(:));
    X = X(keep,:);
    C = C(keep,:);
    pairRows = pairRows(keep);
    t = t(keep);
    fallback = fallback(keep);
    targetKeepIndex = targetKeepIndex(keep);
    nPost = size(X,1);
    [postRefCount,postRefCov] = TargetRefCoverage_BDG(AF,targetKeepIndex,W);

    Diag.target_filter_pre_count = double(nPre);
    Diag.target_filter_post_count = double(nPost);
    Diag.target_filter_retain_ratio = SafeRatio_BDG(nPost,nPre);
    Diag.target_filter_pre_ref_count = double(preRefCount);
    Diag.target_filter_pre_ref_cov = double(preRefCov);
    Diag.target_filter_post_ref_count = double(postRefCount);
    Diag.target_filter_post_ref_cov = double(postRefCov);
end

function Diag = EmptyFusedTargetFilterDiag_BDG()
    Diag = struct();
end

function [count,cov] = TargetRefCoverage_BDG(AF,targetKeepIndex,W)
    refs = zeros(0,1);
    if isstruct(AF) && isfield(AF,'ref') && ~isempty(AF.ref) && ...
            ~isempty(targetKeepIndex)
        allRefs = double(AF.ref(:));
        idx = round(double(targetKeepIndex(:)));
        valid = idx >= 1 & idx <= numel(allRefs);
        refs = allRefs(idx(valid));
    end
    refs = refs(isfinite(refs) & refs > 0);
    count = numel(unique(round(refs)));
    if isempty(W)
        cov = double(count > 0);
    else
        cov = double(count) / max(1,size(W,1));
    end
end

function S = MergeStructFields_BDG(S,Extra)
    if ~isstruct(Extra)
        return;
    end
    names = fieldnames(Extra);
    for i = 1 : numel(names)
        S.(names{i}) = Extra.(names{i});
    end
end

function [X,pairRows,t] = DecisionInterpolationTargets_BDG(AFDec,AIDec,count)
    nPair = size(AFDec,1);
    tGrid = linspace(0,1,count)';
    X = zeros(nPair*numel(tGrid),size(AFDec,2));
    pairRows = zeros(nPair*numel(tGrid),1);
    t = zeros(nPair*numel(tGrid),1);
    row = 0;
    for i = 1 : nPair
        for j = 1 : numel(tGrid)
            row = row + 1;
            pairRows(row) = i;
            t(row) = tGrid(j);
            X(row,:) = AFDec(i,:) + t(row) .* (AIDec(i,:) - AFDec(i,:));
        end
    end
end

function [X,pairRows,t,fallback,sourceCount,candidateCount,keepCount] = ...
        NearSegmentFeasibleTargets_BDG(AFDec,AFNorm,Direction,Options, ...
        zmin,zmax,D,M)
    [SourceDec,SourceObj] = FeasibleSourceMatrices_BDG( ...
        Options.nearSegmentSource,D,M);
    nPair = size(AFDec,1);
    maxPerPair = Options.nearSegmentMaxPerPair;
    tau = Options.nearSegmentTau;
    candidateCount = 0;
    keepCount = 0;

    if ~isempty(SourceDec)
        SourceNorm = NormalizeObjMatNoClip_BDG(SourceObj,zmin,zmax);
        finiteSource = all(isfinite(SourceNorm),2);
        SourceDec = SourceDec(finiteSource,:);
        SourceNorm = SourceNorm(finiteSource,:);
    else
        SourceNorm = zeros(0,M);
    end
    sourceCount = size(SourceDec,1);

    if sourceCount <= 0
        row = nPair;
        X = zeros(row,D);
        pairRows = zeros(row,1);
        t = zeros(row,1);
        fallback = false(row,1);
        X(1:row,:) = AFDec;
        pairRows(1:row) = (1:nPair)';
        t(1:row) = 0;
        fallback(1:row) = true;
    else
        candidateCount = double(nPair) * double(sourceCount);
        if candidateCount < NearSegmentBBoxThreshold_BDG()
            [X,pairRows,t,fallback,keepCount] = ...
                NearSegmentFullScanTargets_BDG(AFDec,AFNorm,Direction, ...
                SourceDec,SourceNorm,maxPerPair,tau,D);
        else
            [X,pairRows,t,fallback,keepCount] = ...
                NearSegmentBBoxTargets_BDG(AFNorm,Direction,SourceDec, ...
                SourceNorm,maxPerPair,tau,D,nPair);
        end
    end
end

function threshold = NearSegmentBBoxThreshold_BDG()
    threshold = 1e6;
end

function [X,pairRows,t,fallback,keepCount] = ...
        NearSegmentFullScanTargets_BDG(AFDec,AFNorm,Direction,SourceDec, ...
        SourceNorm,maxPerPair,tau,D)
    nPair = size(AFDec,1);
    X = zeros(0,D);
    pairRows = zeros(0,1);
    t = zeros(0,1);
    fallback = false(0,1);
    keepCount = 0;
    for i = 1 : nPair
        a = AFNorm(i,:);
        b = a + Direction(i,:);
        [dist,projT] = PointToSegmentDistances_BDG(SourceNorm,a,b);
        valid = isfinite(dist) & isfinite(projT);
        chosen = [];
        chosenT = [];
        if any(valid)
            idx = find(valid & dist <= tau);
            if isempty(idx)
                [~,ord] = sort(dist(valid),'ascend');
                validIdx = find(valid);
                idx = validIdx(ord(1:min(maxPerPair,numel(ord))));
            else
                [~,ord] = sort(dist(idx),'ascend');
                idx = idx(ord(1:min(maxPerPair,numel(ord))));
            end
            chosen = SourceDec(idx,:);
            chosenT = min(max(projT(idx),0),1);
        end
        if isempty(chosen)
            chosen = AFDec(i,:);
            chosenT = 0;
            fb = true;
        else
            fb = false(size(chosenT));
        end
        rowN = size(chosen,1);
        X = [X;chosen]; %#ok<AGROW>
        pairRows = [pairRows;repmat(i,rowN,1)]; %#ok<AGROW>
        t = [t;chosenT(:)]; %#ok<AGROW>
        fallback = [fallback;fb(:)]; %#ok<AGROW>
        keepCount = keepCount + rowN - sum(fb(:));
    end
end

function [X,pairRows,t,fallback,keepCount] = ...
        NearSegmentBBoxTargets_BDG(AFNorm,Direction,SourceDec,SourceNorm, ...
        maxPerPair,tau,D,nPair)
    rowCap = max(1,nPair * maxPerPair);
    X = zeros(rowCap,D);
    pairRows = zeros(rowCap,1);
    t = zeros(rowCap,1);
    fallback = false(rowCap,1);
    row = 0;
    keepCount = 0;
    tau2 = tau .* tau;
    for i = 1 : nPair
        a = AFNorm(i,:);
        v = Direction(i,:);
        b = a + v;
        lo = min(a,b) - tau;
        hi = max(a,b) + tau;
        box = all(SourceNorm >= lo & SourceNorm <= hi,2);
        idx = [];
        chosenT = [];
        if any(box)
            boxIdx = find(box);
            [boxDist2,boxT] = PointToSegmentDistancesSquared_BDG( ...
                SourceNorm(boxIdx,:),a,v);
            nearRows = find(boxDist2 <= tau2);
            if ~isempty(nearRows)
                nearOrd = SmallestKRows_BDG(boxDist2(nearRows), ...
                    maxPerPair);
                localRows = nearRows(nearOrd);
                idx = boxIdx(localRows);
                chosenT = boxT(localRows);
            end
        end
        if isempty(idx)
            [dist2,projT] = PointToSegmentDistancesSquared_BDG( ...
                SourceNorm,a,v);
            idx = SmallestKRows_BDG(dist2,maxPerPair);
            chosenT = projT(idx);
        end
        rowN = numel(idx);
        rows = row + (1:rowN);
        X(rows,:) = SourceDec(idx,:);
        pairRows(rows) = i;
        t(rows) = min(max(chosenT,0),1);
        fallback(rows) = false;
        row = row + rowN;
        keepCount = keepCount + rowN;
    end
    X = X(1:row,:);
    pairRows = pairRows(1:row);
    t = t(1:row);
    fallback = fallback(1:row);
end

function [dist,t] = PointToSegmentDistances_BDG(X,a,b)
    v = b - a;
    len2 = sum(v.^2);
    if len2 <= 1e-24
        t = zeros(size(X,1),1);
        proj = repmat(a,size(X,1),1);
    else
        t = sum((X - a).*v,2) ./ len2;
        proj = a + min(max(t,0),1).*v;
    end
    dist = sqrt(sum((X - proj).^2,2));
end

function [dist2,t] = PointToSegmentDistancesSquared_BDG(X,a,v)
    len2 = sum(v.^2);
    if len2 <= 1e-24
        t = zeros(size(X,1),1);
        proj = a;
    else
        t = sum((X - a).*v,2) ./ len2;
        proj = a + min(max(t,0),1).*v;
    end
    dist2 = sum((X - proj).^2,2);
end

function [Dec,Obj] = FeasibleSourceMatrices_BDG(Source,D,M)
    Dec = zeros(0,D);
    Obj = zeros(0,M);
    if ~isstruct(Source) || ~isfield(Source,'decs') || ...
            ~isfield(Source,'objs') || isempty(Source.decs) || ...
            isempty(Source.objs)
        return;
    end
    decs = double(Source.decs);
    objs = double(Source.objs);
    if size(decs,2) ~= D
        return;
    end
    if size(objs,2) > M
        objs = objs(:,1:M);
    elseif size(objs,2) < M
        objs = [objs,nan(size(objs,1),M-size(objs,2))];
    end
    n = min(size(decs,1),size(objs,1));
    decs = decs(1:n,:);
    objs = objs(1:n,:);
    feasible = true(n,1);
    if isfield(Source,'cons') && ~isempty(Source.cons)
        cons = double(Source.cons);
        cons = cons(1:min(n,size(cons,1)),:);
        feasible = false(n,1);
        feasible(1:size(cons,1)) = all(cons <= 0,2);
    elseif isfield(Source,'feasible') && ~isempty(Source.feasible)
        f = logical(Source.feasible(:));
        feasible = false(n,1);
        feasible(1:min(n,numel(f))) = f(1:min(n,numel(f)));
    end
    finite = all(isfinite(decs),2) & all(isfinite(objs),2);
    keep = feasible & finite;
    Dec = decs(keep,:);
    Obj = objs(keep,:);
end

function idx = SmallestKRows_BDG(values,k)
    values = values(:);
    k = min(k,numel(values));
    if k <= 0
        idx = zeros(0,1);
    elseif k == numel(values)
        [~,idx] = sort(values,'ascend');
    else
        try
            [~,idx] = mink(values,k);
        catch
            [~,idx] = sort(values,'ascend');
            idx = idx(1:k);
        end
    end
end

function Diag = EmptyTargetTripleDiag_BDG(M,refDim,conditionMode,targetMode)
    if nargin < 2
        refDim = 0;
    end
    if nargin < 3
        conditionMode = "yt_dt";
    end
    if nargin < 4
        targetMode = "endpoint";
    end
    conditionMode = NormalizeConditionMode_BDG(conditionMode);
    targetMode = NormalizeTargetMode_BDG(targetMode);
    Diag = struct( ...
        'target_triple_ready',0, ...
        'target_pair_count',0, ...
        'target_triple_count',0, ...
        'target_condition_dim',double(2*M + double(ConditionHasT_BDG(conditionMode)) + refDim), ...
        'target_condition_mode_code',ConditionModeCode_BDG(conditionMode), ...
        'target_ref_token_dim',double(refDim), ...
        'target_mode_code',double(TargetModeCode_BDG(targetMode)), ...
        'target_keep_index',zeros(0,1), ...
        'target_y_min',nan(1,M), ...
        'target_y_max',nan(1,M), ...
        'target_direction_norm_mean',NaN, ...
        'target_t_min',NaN, ...
        'target_t_max',NaN, ...
        'target_t_mean',NaN, ...
        'target_fallback_count',0, ...
        'target_near_segment_source_count',0, ...
        'target_near_segment_candidate_count',0, ...
        'target_near_segment_keep_count',0);
end

function code = ConditionModeCode_BDG(mode)
    mode = NormalizeConditionMode_BDG(mode);
    names = ["yt_dt","yt_dt_ref","yt_dt_t","yt_dt_t_ref"];
    code = double(find(names == mode,1,'first') - 1);
end

function D = ProblemDimension_BDG(Problem,AF)
    D = 0;
    if isfield_safe_BDG(Problem,'D')
        D = double(Problem.D);
    elseif isstruct(AF) && isfield(AF,'decs')
        D = size(AF.decs,2);
    end
    if isempty(D) || ~isscalar(D) || ~isfinite(D)
        D = 0;
    end
    D = max(0,round(D));
end

function M = ProblemObjectiveCount_BDG(Problem,AF,AI)
    M = 0;
    if isfield_safe_BDG(Problem,'M')
        M = double(Problem.M);
    elseif isstruct(AF) && isfield(AF,'objs')
        M = size(AF.objs,2);
    elseif isstruct(AI) && isfield(AI,'objs')
        M = size(AI.objs,2);
    end
    if isempty(M) || ~isscalar(M) || ~isfinite(M)
        M = 0;
    end
    M = max(0,round(M));
end

function flag = isfield_safe_BDG(S,name)
    flag = false;
    try
        flag = isfield(S,name);
    catch
    end
end

function [Dec,Obj] = ArchiveMatrices_BDG(A,D,M)
    Dec = zeros(0,D);
    Obj = zeros(0,M);
    if ~isstruct(A)
        return;
    end
    if isfield(A,'decs') && ~isempty(A.decs)
        Dec = double(A.decs);
        if size(Dec,2) ~= D
            Dec = zeros(0,D);
        end
    end
    if isfield(A,'objs') && ~isempty(A.objs)
        Obj = double(A.objs);
        if size(Obj,2) > M
            Obj = Obj(:,1:M);
        elseif size(Obj,2) < M
            Obj = [Obj,nan(size(Obj,1),M-size(Obj,2))];
        end
    end
end

function [AFNorm,AINorm,zmin,zmax] = NormalizePairedObjectives_BDG(AFObj,AIObj,M)
    allObj = [AFObj;AIObj];
    zmin = zeros(1,M);
    zmax = ones(1,M);
    for j = 1 : M
        values = allObj(:,j);
        values = values(isfinite(values));
        if isempty(values)
            continue;
        end
        zmin(j) = min(values);
        zmax(j) = max(values);
    end
    AFNorm = NormalizeObjMat_BDG(AFObj,zmin,zmax);
    AINorm = NormalizeObjMat_BDG(AIObj,zmin,zmax);
end

function X = NormalizeObjMat_BDG(X,zmin,zmax)
    X = NormalizeObjMatNoClip_BDG(X,zmin,zmax);
    X = min(max(X,0),1);
end

function X = NormalizeObjMatNoClip_BDG(X,zmin,zmax)
    X = (double(X) - double(zmin)) ./ ...
        (double(zmax) - double(zmin) + 1e-12);
end

function value = MeanFinite_BDG(X)
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
    else
        value = mean(X);
    end
end

function value = MedianFinite_BDG(X)
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
    else
        value = median(X);
    end
end

function value = PercentileFinite_BDG(X,p)
    X = sort(double(X(isfinite(X))));
    if isempty(X)
        value = NaN;
        return;
    end
    p = min(max(double(p),0),100);
    value = X(max(1,ceil((p/100)*numel(X))));
end

function value = SafeRatio_BDG(num,den)
    if den <= 0
        value = NaN;
    else
        value = double(num) / double(den);
    end
end

function value = MinFinite_BDG(X)
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
    else
        value = min(X);
    end
end

function value = MaxFinite_BDG(X)
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
    else
        value = max(X);
    end
end
