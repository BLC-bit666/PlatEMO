function [XOut,COut,AIOut,weights,Diag] = FilterBoundaryTargetTriples_BDG( ...
        X,C,AITrain,AF,AI,W,TripleDiag,Options)
%FilterBoundaryTargetTriples_BDG Apply CGAN-only target triple filters.
%   Filters and weights affect only the CGAN training set. The AF/AI
%   boundary archive itself is not modified.

    if nargin < 8 || isempty(Options)
        Options = struct();
    end
    Options = NormalizeFilterOptions_BDG(Options);
    XOut = X;
    COut = C;
    AIOut = AITrain;
    weights = [];
    n = size(X,1);
    pairIdx = TargetPairIndex_BDG(TripleDiag,n,AF);
    refs = ArchiveRefRows_BDG(AF,pairIdx);
    Diag = EmptyFilterDiag_BDG();
    Diag.target_filter_pre_count = double(n);
    Diag.target_filter_post_count = double(n);
    Diag.target_filter_retain_ratio = SafeRatio_BDG(n,n);
    Diag.target_triple_count = double(n);
    [Diag.target_filter_pre_ref_count,Diag.target_filter_pre_ref_cov] = ...
        RefCoverage_BDG(refs,W);
    [Diag.target_filter_post_ref_count,Diag.target_filter_post_ref_cov] = ...
        RefCoverage_BDG(refs,W);
    Diag.target_keep_index = pairIdx(:);
    Diag.train_direction_candidate_count = double(n);
    Diag.train_direction_keep_count = double(n);
    Diag.train_direction_retain_ratio = SafeRatio_BDG(n,n);

    [aiDom,mutualND,afDom] = PairDirectionClasses_BDG(AF,AI,pairIdx);
    Diag.pair_ai_dom_count = double(sum(aiDom));
    Diag.pair_mutual_nd_count = double(sum(mutualND));
    Diag.pair_af_dom_count = double(sum(afDom));
    Diag.pair_ai_dom_ratio = SafeRatio_BDG(sum(aiDom),n);
    Diag.pair_mutual_nd_ratio = SafeRatio_BDG(sum(mutualND),n);

    switch Options.trainFilterMode
        case "none"
            return;
        case "condition_knn"
            [keep,spread,k] = ConditionKNNKeepMask_BDG( ...
                X,C,Options,ProblemBounds_BDG(Options,n,size(X,2)));
            Diag.target_filter_mode_code = 1;
            Diag.condition_knn_k = double(k);
            Diag.condition_knn_decision_spread_mean = MeanFinite_BDG(spread);
            Diag.condition_knn_decision_spread_median = ...
                MedianFinite_BDG(spread);
            Diag.condition_knn_decision_spread_p90 = ...
                PercentileFinite_BDG(spread,90);
            [XOut,COut,AIOut,pairIdx,refs] = ApplyKeep_BDG( ...
                X,C,AITrain,pairIdx,refs,keep);
        case "local_mad_weight"
            [weights,spread,k] = LocalMADDecisionWeights_BDG( ...
                X,C,Options,ProblemBounds_BDG(Options,n,size(X,2)));
            Diag.target_filter_mode_code = 4;
            Diag.condition_knn_k = double(k);
            Diag.condition_knn_decision_spread_mean = MeanFinite_BDG(spread);
            Diag.condition_knn_decision_spread_median = ...
                MedianFinite_BDG(spread);
            Diag.condition_knn_decision_spread_p90 = ...
                PercentileFinite_BDG(spread,90);
        case "random_keep"
            keep = RandomKeepMask_BDG(n,Options.conditionKNNRetainRatio, ...
                Options.conditionKNNMinCount,Options.randomSeed);
            Diag.target_filter_mode_code = 3;
            [XOut,COut,AIOut,pairIdx,refs] = ApplyKeep_BDG( ...
                X,C,AITrain,pairIdx,refs,keep);
        case "ai_dom_only"
            Diag.target_filter_mode_code = 2;
            Diag.train_direction_candidate_count = double(n);
            if sum(aiDom) >= Options.aiDomMinCount
                keep = aiDom;
                Diag.train_direction_mode_code = 1;
                [XOut,COut,AIOut,pairIdx,refs] = ApplyKeep_BDG( ...
                    X,C,AITrain,pairIdx,refs,keep);
            else
                weights = ones(n,1);
                weights(mutualND) = Options.mutualNDWeight;
                weights(afDom) = 0;
                if sum(weights) <= 0
                    weights = [];
                end
                Diag.train_direction_mode_code = 2;
            end
        otherwise
            error('FilterBoundaryTargetTriples_BDG:BadMode', ...
                'Unsupported trainFilterMode: %s',Options.trainFilterMode);
    end

    Diag.target_filter_post_count = double(size(XOut,1));
    Diag.target_filter_retain_ratio = SafeRatio_BDG( ...
        Diag.target_filter_post_count,Diag.target_filter_pre_count);
    [Diag.target_filter_post_ref_count,Diag.target_filter_post_ref_cov] = ...
        RefCoverage_BDG(refs,W);
    Diag.target_triple_count = double(size(XOut,1));
    Diag.target_keep_index = pairIdx(:);
    Diag.train_direction_keep_count = double(size(XOut,1));
    Diag.train_direction_retain_ratio = SafeRatio_BDG(size(XOut,1),n);
    if ~isempty(weights)
        Diag.train_weight_count = double(numel(weights));
        Diag.train_weight_min = MinFinite_BDG(weights);
        Diag.train_weight_mean = MeanFinite_BDG(weights);
        Diag.train_weight_max = MaxFinite_BDG(weights);
    end
end

function Options = NormalizeFilterOptions_BDG(Options)
    if ~isstruct(Options)
        Options = struct();
    end
    Options = WithDefault_BDG(Options,'trainFilterMode',"none");
    Options = WithDefault_BDG(Options,'conditionKNNK',5);
    Options = WithDefault_BDG(Options,'conditionKNNRetainRatio',0.70);
    Options = WithDefault_BDG(Options,'conditionKNNMinCount',2);
    Options = WithDefault_BDG(Options,'randomSeed',0);
    Options = WithDefault_BDG(Options,'aiDomMinCount',2);
    Options = WithDefault_BDG(Options,'mutualNDWeight',0.10);
    Options = WithDefault_BDG(Options,'localMADAlpha',3.0);
    Options = WithDefault_BDG(Options,'localMADOutlierWeight',0.10);
    Options.trainFilterMode = NormalizeTrainFilterMode_BDG( ...
        Options.trainFilterMode);
    Options.conditionKNNK = max(1,round(double(Options.conditionKNNK)));
    Options.conditionKNNRetainRatio = min(max( ...
        double(Options.conditionKNNRetainRatio),0),1);
    Options.conditionKNNMinCount = max(2,round(double( ...
        Options.conditionKNNMinCount)));
    Options.randomSeed = NormalizeRandomSeed_BDG(Options.randomSeed);
    Options.aiDomMinCount = max(2,round(double(Options.aiDomMinCount)));
    Options.mutualNDWeight = min(max(double(Options.mutualNDWeight),0),1);
    Options.localMADAlpha = max(0,double(Options.localMADAlpha));
    Options.localMADOutlierWeight = min(max( ...
        double(Options.localMADOutlierWeight),0),1);
end

function mode = NormalizeTrainFilterMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["none","condition_knn","local_mad_weight", ...
        "random_keep","ai_dom_only"];
    assert(ismember(mode,valid), ...
        'FilterBoundaryTargetTriples_BDG:BadTrainFilterMode', ...
        'trainFilterMode must be one of: %s.',strjoin(valid,", "));
end

function seed = NormalizeRandomSeed_BDG(seed)
    seed = double(seed);
    if isempty(seed) || ~isscalar(seed) || ~isfinite(seed)
        seed = 0;
    end
    seed = mod(round(seed),2^32);
end

function S = WithDefault_BDG(S,fieldName,value)
    if ~isfield(S,fieldName) || isempty(S.(fieldName))
        S.(fieldName) = value;
    end
end

function Diag = EmptyFilterDiag_BDG()
    Diag = struct( ...
        'target_filter_mode_code',0, ...
        'target_filter_pre_count',0, ...
        'target_filter_post_count',0, ...
        'target_filter_retain_ratio',NaN, ...
        'target_filter_pre_ref_count',0, ...
        'target_filter_pre_ref_cov',0, ...
        'target_filter_post_ref_count',0, ...
        'target_filter_post_ref_cov',0, ...
        'condition_knn_k',0, ...
        'condition_knn_decision_spread_mean',NaN, ...
        'condition_knn_decision_spread_median',NaN, ...
        'condition_knn_decision_spread_p90',NaN, ...
        'pair_ai_dom_count',0, ...
        'pair_mutual_nd_count',0, ...
        'pair_af_dom_count',0, ...
        'pair_ai_dom_ratio',NaN, ...
        'pair_mutual_nd_ratio',NaN, ...
        'train_direction_candidate_count',0, ...
        'train_direction_keep_count',0, ...
        'train_direction_retain_ratio',NaN, ...
        'train_direction_mode_code',0, ...
        'train_weight_count',0, ...
        'train_weight_min',NaN, ...
        'train_weight_mean',NaN, ...
        'train_weight_max',NaN, ...
        'target_triple_count',0, ...
        'target_keep_index',zeros(0,1));
end

function pairIdx = TargetPairIndex_BDG(TripleDiag,n,AF)
    pairIdx = (1:n)';
    if isstruct(TripleDiag) && isfield(TripleDiag,'target_keep_index') && ...
            ~isempty(TripleDiag.target_keep_index)
        idx = round(double(TripleDiag.target_keep_index(:)));
        if numel(idx) >= n
            pairIdx = idx(1:n);
        end
    end
    maxN = ArchiveRowCount_BDG(AF);
    pairIdx = max(1,min(maxN,pairIdx));
end

function n = ArchiveRowCount_BDG(A)
    n = 0;
    if isstruct(A) && isfield(A,'decs')
        n = size(A.decs,1);
    elseif isstruct(A) && isfield(A,'objs')
        n = size(A.objs,1);
    end
    n = max(1,n);
end

function refs = ArchiveRefRows_BDG(AF,pairIdx)
    refs = zeros(numel(pairIdx),1);
    if ~isstruct(AF) || ~isfield(AF,'ref') || isempty(AF.ref)
        return;
    end
    allRefs = double(AF.ref(:));
    valid = pairIdx >= 1 & pairIdx <= numel(allRefs);
    refs(valid) = allRefs(pairIdx(valid));
end

function [count,cov] = RefCoverage_BDG(refs,W)
    refs = round(double(refs(:)));
    refs = refs(isfinite(refs) & refs > 0);
    count = numel(unique(refs));
    if isempty(W)
        cov = double(count > 0);
    else
        cov = double(count) / max(1,size(W,1));
    end
end

function [aiDom,mutualND,afDom] = PairDirectionClasses_BDG(AF,AI,pairIdx)
    n = numel(pairIdx);
    aiDom = false(n,1);
    afDom = false(n,1);
    AFObj = ArchiveObjRows_BDG(AF,pairIdx);
    AIObj = ArchiveObjRows_BDG(AI,pairIdx);
    if isempty(AFObj) || isempty(AIObj) || size(AFObj,2) ~= size(AIObj,2)
        mutualND = true(n,1);
        return;
    end
    epsTol = 1e-12;
    finite = all(isfinite(AFObj),2) & all(isfinite(AIObj),2);
    aiDom(finite) = all(AIObj(finite,:) <= AFObj(finite,:) + epsTol,2) & ...
        any(AIObj(finite,:) < AFObj(finite,:) - epsTol,2);
    afDom(finite) = all(AFObj(finite,:) <= AIObj(finite,:) + epsTol,2) & ...
        any(AFObj(finite,:) < AIObj(finite,:) - epsTol,2);
    mutualND = finite & ~aiDom & ~afDom;
    mutualND(~finite) = true;
end

function Obj = ArchiveObjRows_BDG(A,pairIdx)
    Obj = zeros(numel(pairIdx),0);
    if ~isstruct(A) || ~isfield(A,'objs') || isempty(A.objs)
        return;
    end
    allObj = double(A.objs);
    Obj = nan(numel(pairIdx),size(allObj,2));
    valid = pairIdx >= 1 & pairIdx <= size(allObj,1);
    Obj(valid,:) = allObj(pairIdx(valid),:);
end

function bounds = ProblemBounds_BDG(Options,n,D)
    bounds = struct('lower',zeros(1,D),'upper',ones(1,D));
    if isfield(Options,'lower') && isfield(Options,'upper') && ...
            numel(Options.lower) == D && numel(Options.upper) == D
        bounds.lower = double(Options.lower(:)');
        bounds.upper = double(Options.upper(:)');
    end
    if n <= 0
        bounds.lower = zeros(1,D);
        bounds.upper = ones(1,D);
    end
end

function [keep,spread,k] = ConditionKNNKeepMask_BDG(X,C,Options,bounds)
    n = size(X,1);
    keep = true(n,1);
    spread = nan(n,1);
    k = min(Options.conditionKNNK,max(0,n-1));
    if n <= Options.conditionKNNMinCount || isempty(C) || k <= 0
        return;
    end
    C = double(C);
    Xn = NormalizeDecision_BDG(double(X),bounds.lower,bounds.upper);
    dCond = pdist2(C,C);
    dCond(1:n+1:end) = Inf;
    for i = 1 : n
        [~,ord] = sort(dCond(i,:),'ascend');
        ord = ord(1:min(k,numel(ord)));
        d = sqrt(sum((Xn(ord,:) - Xn(i,:)).^2,2));
        spread(i) = MeanFinite_BDG(d);
    end
    if ~any(isfinite(spread))
        return;
    end
    nKeep = max(Options.conditionKNNMinCount, ...
        ceil(Options.conditionKNNRetainRatio * n));
    nKeep = min(n,max(2,nKeep));
    [~,ord] = sort(spread,'ascend','MissingPlacement','last');
    keep = false(n,1);
    keep(ord(1:nKeep)) = true;
end

function [weights,spread,k] = LocalMADDecisionWeights_BDG(X,C,Options,bounds)
    n = size(X,1);
    weights = ones(n,1);
    spread = nan(n,1);
    k = min(Options.conditionKNNK,max(0,n-1));
    if n <= Options.conditionKNNMinCount || isempty(C) || k <= 1
        weights = [];
        return;
    end
    C = double(C);
    Xn = NormalizeDecision_BDG(double(X),bounds.lower,bounds.upper);
    dCond = pdist2(C,C);
    dCond(1:n+1:end) = Inf;
    for i = 1 : n
        [~,ord] = sort(dCond(i,:),'ascend');
        ord = ord(1:min(k,numel(ord)));
        localX = Xn(ord,:);
        medoid = LocalDecisionMedoid_BDG(localX);
        localDist = sqrt(sum((localX - medoid).^2,2));
        dist = sqrt(sum((Xn(i,:) - medoid).^2,2));
        spread(i) = dist;
        center = MedianFinite_BDG(localDist);
        mad = MedianFinite_BDG(abs(localDist - center));
        scale = max(1.4826 * mad,1e-12);
        if isfinite(dist) && isfinite(center) && ...
                dist > center + Options.localMADAlpha * scale
            weights(i) = Options.localMADOutlierWeight;
        end
    end
    if all(abs(weights - 1) < 1e-12)
        weights = [];
    end
end

function medoid = LocalDecisionMedoid_BDG(X)
    if isempty(X)
        medoid = zeros(1,0);
        return;
    end
    if size(X,1) == 1
        medoid = X(1,:);
        return;
    end
    D = pdist2(X,X);
    score = sum(D,2,'omitnan');
    [~,idx] = min(score);
    medoid = X(idx,:);
end

function keep = RandomKeepMask_BDG(n,retainRatio,minCount,seed)
    keep = true(n,1);
    if n <= minCount
        return;
    end
    nKeep = max(minCount,ceil(double(retainRatio) * n));
    nKeep = min(n,max(2,nKeep));
    stream = RandStream('mt19937ar','Seed',seed);
    [~,ord] = sort(rand(stream,n,1),'ascend');
    keep = false(n,1);
    keep(ord(1:nKeep)) = true;
end

function Xn = NormalizeDecision_BDG(X,lower,upper)
    Xn = (double(X) - double(lower)) ./ ...
        (double(upper) - double(lower) + 1e-12);
    Xn = min(max(Xn,0),1);
end

function [XOut,COut,AIOut,pairIdx,refs] = ApplyKeep_BDG( ...
        X,C,AITrain,pairIdx,refs,keep)
    keep = logical(keep(:));
    XOut = X(keep,:);
    COut = C(keep,:);
    AIOut = AITrain(keep,:);
    pairIdx = pairIdx(keep);
    refs = refs(keep);
end

function value = SafeRatio_BDG(num,den)
    if den <= 0
        value = NaN;
    else
        value = double(num) / double(den);
    end
end

function value = MeanFinite_BDG(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
end

function value = MedianFinite_BDG(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = median(x);
    end
end

function value = PercentileFinite_BDG(x,p)
    x = sort(double(x(isfinite(x))));
    if isempty(x)
        value = NaN;
        return;
    end
    p = min(max(double(p),0),100);
    value = x(max(1,ceil((p/100)*numel(x))));
end

function value = MinFinite_BDG(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = min(x);
    end
end

function value = MaxFinite_BDG(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = max(x);
    end
end
