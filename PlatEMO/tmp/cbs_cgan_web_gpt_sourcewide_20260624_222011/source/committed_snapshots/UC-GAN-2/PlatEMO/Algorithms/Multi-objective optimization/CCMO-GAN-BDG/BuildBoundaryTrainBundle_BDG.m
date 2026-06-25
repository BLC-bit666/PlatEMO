function [Bundle,Diag] = BuildBoundaryTrainBundle_BDG( ...
        AF,AI,W,Problem,mode,Control,EvaluatedSource)
%BuildBoundaryTrainBundle_BDG Build refresh-local BDG GAN training bundle.

    if nargin < 6 || isempty(Control)
        Control = struct();
    end
    if nargin < 7
        EvaluatedSource = [];
    end
    mode = NormalizeGeneratorMode_BDG(mode);
    Control = NormalizeBundleControl_BDG(Control);
    filterMode = Control.trainFilterMode;
    targetFusionMode = "none";
    if filterMode == "condition_knn_fused"
        targetFusionMode = "condition_knn";
        filterMode = "none";
    end

    TripleOptions = struct( ...
        'conditionMode',Control.conditionMode, ...
        'targetMode',Control.targetMode, ...
        'nearSegmentSource',EvaluatedSource, ...
        'nearSegmentTau',Control.nearSegmentTau, ...
        'nearSegmentMaxPerPair',Control.nearSegmentMaxPerPair, ...
        'decisionInterpCount',Control.decisionInterpCount, ...
        'targetFusionMode',targetFusionMode, ...
        'conditionKNNK',5, ...
        'conditionKNNRetainRatio',Control.conditionKNNRetainRatio, ...
        'conditionKNNMinCount',2, ...
        'lower',Problem.lower, ...
        'upper',Problem.upper);

    tTargetBuild = tic;
    [TrainDecs,ConditionData,Diag] = BuildBoundaryTargetTriples_BDG( ...
        AF,AI,Problem,W,TripleOptions);
    Diag.t_target_build = toc(tTargetBuild);
    DiagAI = ArchiveDecisionRows_BDG(AI,Diag.target_keep_index,Problem.D);
    FilterOptions = struct( ...
        'trainFilterMode',filterMode, ...
        'conditionKNNK',5, ...
        'conditionKNNRetainRatio',Control.conditionKNNRetainRatio, ...
        'conditionKNNMinCount',2, ...
        'randomSeed',Control.trainFilterRandomSeed, ...
        'aiDomMinCount',2, ...
        'mutualNDWeight',0.10, ...
        'lower',Problem.lower, ...
        'upper',Problem.upper);
    tTargetFilter = tic;
    [TrainDecs,ConditionData,DiagAI,weights,FilterDiag] = ...
        FilterBoundaryTargetTriples_BDG(TrainDecs,ConditionData, ...
        DiagAI,AF,AI,W,Diag,FilterOptions);
    FilterDiag.t_target_filter = toc(tTargetFilter);
    if targetFusionMode == "condition_knn"
        FilterDiag = RestoreFusedTargetFilterDiag_BDG(FilterDiag,Diag);
    end
    Diag = MergeStructFields_BDG(Diag,FilterDiag);
    realLabels = [];
    if Control.targetRealLabelMode == "boundary_quality_eval"
        error('BuildBoundaryTrainBundle_BDG:UnsupportedRealLabelMode', ...
            'BoundaryTrainBundle currently supports binary real labels only.');
    end
    if mode == "objective_target_unconditioned"
        ConditionData = [];
    end

    Bundle = struct( ...
        'mode',mode, ...
        'decs',TrainDecs, ...
        'aiDecs',DiagAI, ...
        'conditionData',ConditionData, ...
        'sampleWeights',weights, ...
        'realLabels',realLabels, ...
        'pairIndex',double(Diag.target_keep_index(:)), ...
        'conditionDim',size(ConditionData,2));
    Diag = AddBundleEquivalenceDiag_BDG(Bundle,Diag,AI,W);
end

function Control = NormalizeBundleControl_BDG(Control)
    Control = WithDefault_BDG(Control,'conditionMode',"yt_dt_t_ref");
    Control = WithDefault_BDG(Control,'targetMode',"near_segment_feasible");
    Control = WithDefault_BDG(Control,'nearSegmentTau',0.20);
    Control = WithDefault_BDG(Control,'nearSegmentMaxPerPair',5);
    Control = WithDefault_BDG(Control,'decisionInterpCount',5);
    Control = WithDefault_BDG(Control,'trainFilterMode',"condition_knn");
    Control = WithDefault_BDG(Control,'conditionKNNRetainRatio',0.60);
    Control = WithDefault_BDG(Control,'trainFilterRandomSeed',0);
    Control = WithDefault_BDG(Control,'targetRealLabelMode',"binary");
    Control.conditionMode = string(Control.conditionMode);
    Control.targetMode = string(Control.targetMode);
    Control.nearSegmentTau = max(0,double(Control.nearSegmentTau));
    Control.nearSegmentMaxPerPair = max(1,round(double( ...
        Control.nearSegmentMaxPerPair)));
    Control.decisionInterpCount = max(2,round(double( ...
        Control.decisionInterpCount)));
    Control.trainFilterMode = lower(strtrim(string(Control.trainFilterMode)));
    Control.conditionKNNRetainRatio = min(max( ...
        double(Control.conditionKNNRetainRatio),0),1);
    Control.trainFilterRandomSeed = max(0,round(double( ...
        Control.trainFilterRandomSeed)));
    Control.targetRealLabelMode = lower(strtrim(string( ...
        Control.targetRealLabelMode)));
end

function S = WithDefault_BDG(S,fieldName,value)
    if ~isstruct(S)
        S = struct();
    end
    if ~isfield(S,fieldName) || isempty(S.(fieldName))
        S.(fieldName) = value;
    end
end

function mode = NormalizeGeneratorMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["objective_target_conditioned","objective_target_unconditioned"];
    assert(ismember(mode,valid), ...
        'BuildBoundaryTrainBundle_BDG:BadGeneratorMode', ...
        'mode must be one of: %s.',strjoin(valid,", "));
end

function Dec = ArchiveDecisionRows_BDG(A,idx,D)
    Dec = zeros(numel(idx),D);
    if ~isstruct(A) || ~isfield(A,'decs') || isempty(A.decs)
        return;
    end
    allDec = double(A.decs);
    idx = round(double(idx(:)));
    valid = idx >= 1 & idx <= size(allDec,1);
    Dec(valid,:) = allDec(idx(valid),:);
end

function Diag = AddBundleEquivalenceDiag_BDG(Bundle,Diag,AI,W)
    pairIndex = round(double(Bundle.pairIndex(:)));
    pairIndex = pairIndex(isfinite(pairIndex) & pairIndex > 0);
    Diag.train_bundle_mode_code = 1;
    Diag.bundle_condition_dim = double(Bundle.conditionDim);
    Diag.bundle_pair_count = double(numel(unique(pairIndex)));
    Diag.bundle_pair_cov = SafeRatio_BDG(Diag.bundle_pair_count, ...
        max(1,double(Diag.target_pair_count)));
    [Diag.bundle_ref_count,Diag.bundle_ref_cov] = ...
        BundleRefCoverage_BDG(AI,pairIndex,W);
    if isfield(Diag,'target_t_min')
        Diag.bundle_t_min = Diag.target_t_min;
        Diag.bundle_t_max = Diag.target_t_max;
        Diag.bundle_t_mean = Diag.target_t_mean;
    else
        Diag.bundle_t_min = NaN;
        Diag.bundle_t_max = NaN;
        Diag.bundle_t_mean = NaN;
    end
    ExpectedAI = ArchiveDecisionRows_BDG(AI,Bundle.pairIndex, ...
        size(Bundle.aiDecs,2));
    matches = all(abs(double(ExpectedAI) - double(Bundle.aiDecs)) < 1e-12,2);
    Diag.bundle_sample_pair_ai_match_count = double(sum(matches));
    Diag.bundle_sample_pair_ai_match_rate = SafeRatio_BDG( ...
        sum(matches),numel(matches));
end

function [count,cov] = BundleRefCoverage_BDG(AI,pairIndex,W)
    refs = zeros(0,1);
    if isstruct(AI) && isfield(AI,'ref') && ~isempty(AI.ref) && ...
            ~isempty(pairIndex)
        allRefs = double(AI.ref(:));
        valid = pairIndex >= 1 & pairIndex <= numel(allRefs);
        refs = allRefs(pairIndex(valid));
    end
    refs = refs(isfinite(refs) & refs > 0);
    count = numel(unique(round(refs)));
    if isempty(W)
        cov = double(count > 0);
    else
        cov = double(count) / max(1,size(W,1));
    end
end

function FilterDiag = RestoreFusedTargetFilterDiag_BDG(FilterDiag,BuildDiag)
    names = ["target_filter_mode_code", ...
        "target_filter_pre_count", ...
        "target_filter_post_count", ...
        "target_filter_retain_ratio", ...
        "target_filter_pre_ref_count", ...
        "target_filter_pre_ref_cov", ...
        "target_filter_post_ref_count", ...
        "target_filter_post_ref_cov", ...
        "condition_knn_k", ...
        "condition_knn_decision_spread_mean", ...
        "condition_knn_decision_spread_median", ...
        "condition_knn_decision_spread_p90"];
    for i = 1 : numel(names)
        name = char(names(i));
        if isfield(BuildDiag,name)
            FilterDiag.(name) = BuildDiag.(name);
        end
    end
end

function S = MergeStructFields_BDG(S,Extra)
    names = fieldnames(Extra);
    for i = 1 : numel(names)
        S.(names{i}) = Extra.(names{i});
    end
end

function value = SafeRatio_BDG(num,den)
    if den <= 0
        value = NaN;
    else
        value = double(num) / double(den);
    end
end
