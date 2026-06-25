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
    if Control.trainBundleMode == "pair_cell"
        [Bundle,Diag] = BuildPairCellBundle_BDG(AF,AI,W,Problem,mode, ...
            Control,EvaluatedSource);
        return;
    end
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
    Control = WithDefault_BDG(Control,'trainBundleMode',"refresh_local");
    Control = WithDefault_BDG(Control,'pairCellMaxPerPair',2);
    Control = WithDefault_BDG(Control,'pairCellSelectionMode',"geometry");
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
    Control.trainBundleMode = lower(strtrim(string(Control.trainBundleMode)));
    validBundleModes = ["refresh_local","pair_cell"];
    assert(ismember(Control.trainBundleMode,validBundleModes), ...
        'BuildBoundaryTrainBundle_BDG:BadTrainBundleMode', ...
        'trainBundleMode must be one of: %s.',strjoin(validBundleModes,", "));
    Control.pairCellMaxPerPair = max(1,round(double( ...
        Control.pairCellMaxPerPair)));
    Control.pairCellSelectionMode = NormalizePairCellSelectionMode_BDG( ...
        Control.pairCellSelectionMode);
end

function [Bundle,Diag] = BuildPairCellBundle_BDG(AF,AI,W,Problem,mode, ...
        Control,EvaluatedSource)
    if Control.targetRealLabelMode == "boundary_quality_eval"
        error('BuildBoundaryTrainBundle_BDG:UnsupportedRealLabelMode', ...
            'BoundaryTrainBundle currently supports binary real labels only.');
    end
    if Control.targetMode ~= "near_segment_feasible"
        error('BuildBoundaryTrainBundle_BDG:UnsupportedPairCellTargetMode', ...
            'pair_cell trainBundleMode supports near_segment_feasible only.');
    end

    tTargetBuild = tic;
    [TrainDecs,ConditionData,pairIndex,Diag] = ...
        BuildPairCellNearSegmentTargets_BDG(AF,AI,Problem,W,Control, ...
        EvaluatedSource);
    Diag.t_target_build = toc(tTargetBuild);
    Diag.t_target_filter = 0;

    DiagAI = ArchiveDecisionRows_BDG(AI,pairIndex,Problem.D);
    realLabels = [];
    weights = [];
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
        'pairIndex',double(pairIndex(:)), ...
        'conditionDim',size(ConditionData,2));
    Diag = AddBundleEquivalenceDiag_BDG(Bundle,Diag,AI,W);
    Diag.train_bundle_mode_code = 2;
end

function [XBoundary,ConditionData,targetKeepIndex,Diag] = ...
        BuildPairCellNearSegmentTargets_BDG(AF,AI,Problem,W,Control, ...
        EvaluatedSource)
    D = ProblemDimension_BDG(Problem,AF);
    M = ProblemObjectiveCount_BDG(Problem,AF,AI);
    refDim = TargetRefTokenDim_BDG(W,M,Control.conditionMode);
    hasT = ConditionHasT_BDG(Control.conditionMode);
    conditionDim = 2*M + double(hasT) + refDim;
    XBoundary = zeros(0,D);
    ConditionData = zeros(0,conditionDim,'single');
    targetKeepIndex = zeros(0,1);
    Diag = EmptyPairCellDiag_BDG(M,refDim,Control);
    Diag.target_condition_dim = double(conditionDim);

    [AFDec,AFObj] = ArchiveMatrices_BDG(AF,D,M);
    [AIDec,AIObj] = ArchiveMatrices_BDG(AI,D,M);
    nPair = min([size(AFDec,1),size(AIDec,1),size(AFObj,1),size(AIObj,1)]);
    Diag.target_pair_count = double(nPair);
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
    AFNorm = AFNorm(finiteTargets,:);
    Direction = Direction(finiteTargets,:);
    keepIndex = keepIndex(finiteTargets);
    RefToken = TargetRefToken_BDG(AF,keepIndex,W,refDim);

    [SourceDec,SourceObj,SourceMargin] = FeasibleSourceMatrices_BDG( ...
        EvaluatedSource,D,M);
    if ~isempty(SourceDec)
        SourceNorm = NormalizeObjMatNoClip_BDG(SourceObj,zmin,zmax);
        finiteSource = all(isfinite(SourceNorm),2);
        SourceDec = SourceDec(finiteSource,:);
        SourceNorm = SourceNorm(finiteSource,:);
        SourceMargin = SourceMargin(finiteSource,:);
    else
        SourceNorm = zeros(0,M);
        SourceMargin = zeros(0,1);
    end
    sourceCount = size(SourceDec,1);
    Diag.target_near_segment_source_count = double(sourceCount);
    Diag.target_near_segment_candidate_count = ...
        double(size(AFDec,1)) * double(sourceCount);

    maxCells = Control.pairCellMaxPerPair;
    rowCap = max(1,size(AFDec,1) * maxCells);
    X = zeros(rowCap,D);
    pairRows = zeros(rowCap,1);
    t = zeros(rowCap,1);
    fallback = false(rowCap,1);
    margin = nan(rowCap,1);
    row = 0;
    weakCandidateCount = 0;
    skippedPairCount = 0;
    for i = 1 : size(AFDec,1)
        [chosen,chosenT,chosenFallback,weakCount,chosenMargin] = ...
            SelectPairCells_BDG(AFDec(i,:),AFNorm(i,:),Direction(i,:), ...
            SourceDec,SourceNorm,SourceMargin,Control.nearSegmentTau, ...
            maxCells,D,Control.pairCellSelectionMode);
        rowN = size(chosen,1);
        weakCandidateCount = weakCandidateCount + weakCount;
        if rowN <= 0
            skippedPairCount = skippedPairCount + 1;
            continue;
        end
        rows = row + (1:rowN);
        if rows(end) > size(X,1)
            grow = max(rowN,size(X,1));
            X = [X;zeros(grow,D)]; %#ok<AGROW>
            pairRows = [pairRows;zeros(grow,1)]; %#ok<AGROW>
            t = [t;zeros(grow,1)]; %#ok<AGROW>
            fallback = [fallback;false(grow,1)]; %#ok<AGROW>
            margin = [margin;nan(grow,1)]; %#ok<AGROW>
        end
        X(rows,:) = chosen;
        pairRows(rows) = i;
        t(rows) = chosenT(:);
        fallback(rows) = chosenFallback(:);
        margin(rows) = chosenMargin(:);
        row = row + rowN;
    end

    XBoundary = X(1:row,:);
    pairRows = pairRows(1:row);
    t = t(1:row);
    fallback = fallback(1:row);
    margin = margin(1:row);
    if isempty(XBoundary) || numel(pairRows) < 2
        return;
    end
    ConditionData = ComposeConditionData_BDG(AFNorm,Direction,RefToken, ...
        pairRows,t,hasT,refDim);
    targetKeepIndex = keepIndex(pairRows);
    Diag.target_keep_index = targetKeepIndex(:);
    Diag.target_triple_count = double(size(XBoundary,1));
    Diag.target_triple_ready = double(size(XBoundary,1) >= 2);
    Diag.target_y_min = min(AFNorm(pairRows,:),[],1);
    Diag.target_y_max = max(AFNorm(pairRows,:),[],1);
    dirNorm = sqrt(sum(Direction(pairRows,:).^2,2));
    Diag.target_direction_norm_mean = MeanFinite_BDG(dirNorm);
    Diag.target_t_min = MinFinite_BDG(t);
    Diag.target_t_max = MaxFinite_BDG(t);
    Diag.target_t_mean = MeanFinite_BDG(t);
    Diag.target_fallback_count = double(sum(fallback));
    Diag.target_near_segment_keep_count = double(sum(~fallback(:)));
    Diag.target_weak_segment_candidate_count = double(weakCandidateCount);
    Diag.target_skipped_pair_count = double(skippedPairCount);
    Diag.target_compact_cell_max_per_pair = double(maxCells);
    Diag.target_pair_cell_selection_mode_code = ...
        PairCellSelectionModeCode_BDG(Control.pairCellSelectionMode);
    Diag.target_constraint_margin_count = double(sum(isfinite(margin)));
    Diag.target_constraint_margin_min = MinFinite_BDG(margin);
    Diag.target_constraint_margin_mean = MeanFinite_BDG(margin);
    Diag.target_constraint_margin_abs_mean = MeanFinite_BDG(abs(margin));
    Diag.target_constraint_margin_max = MaxFinite_BDG(margin);
    Diag.target_filter_pre_count = Diag.target_triple_count;
    Diag.target_filter_post_count = Diag.target_triple_count;
    Diag.target_filter_retain_ratio = 1;
    [refCount,refCov] = TargetRefCoverage_BDG(AF,targetKeepIndex,W);
    Diag.target_filter_pre_ref_count = double(refCount);
    Diag.target_filter_pre_ref_cov = double(refCov);
    Diag.target_filter_post_ref_count = double(refCount);
    Diag.target_filter_post_ref_cov = double(refCov);
    Diag.train_direction_candidate_count = Diag.target_triple_count;
    Diag.train_direction_keep_count = Diag.target_triple_count;
    Diag.train_direction_retain_ratio = 1;
end

function [chosen,chosenT,fallback,weakCount,chosenMargin] = ...
        SelectPairCells_BDG(~,afNorm,direction,SourceDec,SourceNorm, ...
        SourceMargin,tau,maxCells,D,selectionMode)
    chosen = zeros(0,D);
    chosenT = zeros(0,1);
    fallback = false(0,1);
    weakCount = 0;
    chosenMargin = zeros(0,1);
    if isempty(SourceDec)
        return;
    end
    margin = nan(size(SourceDec,1),1);
    if nargin >= 6 && ~isempty(SourceMargin)
        nMargin = min(size(SourceDec,1),numel(SourceMargin));
        margin(1:nMargin) = double(SourceMargin(1:nMargin));
    end
    if nargin < 10 || isempty(selectionMode)
        selectionMode = "geometry";
    end
    selectionMode = NormalizePairCellSelectionMode_BDG(selectionMode);

    [dist2,projT] = PointToSegmentDistancesSquared_BDG( ...
        SourceNorm,afNorm,direction);
    valid = isfinite(dist2) & isfinite(projT);
    if ~any(valid)
        return;
    end
    tau2 = tau .* tau;
    near = valid & dist2 <= tau2;
    weakCount = sum(valid & ~near);
    candidate = find(near);
    if isempty(candidate)
        return;
    end
    projT = min(max(projT,0),1);
    selected = SelectTBalancedRows_BDG(candidate,dist2,projT,maxCells, ...
        margin,selectionMode);
    chosen = SourceDec(selected,:);
    chosenT = projT(selected);
    fallback = false(numel(selected),1);
    chosenMargin = margin(selected);
end

function selected = SelectTBalancedRows_BDG(candidate,dist2,projT,maxCells, ...
        margin,selectionMode)
    candidate = candidate(:);
    if isempty(candidate)
        selected = zeros(0,1);
        return;
    end
    if nargin < 5 || isempty(margin)
        margin = nan(size(dist2));
    end
    if nargin < 6 || isempty(selectionMode)
        selectionMode = "geometry";
    end
    selectionMode = NormalizePairCellSelectionMode_BDG(selectionMode);
    maxCells = max(1,round(double(maxCells)));
    binId = min(max(floor(projT(candidate) .* maxCells) + 1,1),maxCells);
    selected = zeros(0,1);
    for b = 1 : maxCells
        rows = candidate(binId == b);
        if isempty(rows)
            continue;
        end
        selected(end+1,1) = SelectPairCellBinRow_BDG(rows,dist2, ...
            margin,selectionMode); %#ok<AGROW>
    end
    selected = selected(1:min(maxCells,numel(selected)));
    [~,ord] = sort(projT(selected),'ascend');
    selected = selected(ord);
end

function selected = SelectPairCellBinRow_BDG(rows,dist2,margin,selectionMode)
    rows = rows(:);
    if selectionMode == "constraint_margin"
        score = abs(double(margin(rows)));
        score(~isfinite(score)) = inf;
        if any(isfinite(score))
            keys = [score(:),double(dist2(rows(:)))];
            [~,ord] = sortrows(keys,[1 2]);
            selected = rows(ord(1));
            return;
        end
    end
    [~,bestLocal] = min(dist2(rows));
    selected = rows(bestLocal);
end

function Diag = EmptyPairCellDiag_BDG(M,refDim,Control)
    Diag = struct( ...
        'target_condition_mode_code',ConditionModeCode_BDG(Control.conditionMode), ...
        'target_mode_code',1, ...
        'target_pair_count',0, ...
        'target_triple_count',0, ...
        'target_triple_ready',0, ...
        'target_condition_dim',0, ...
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
        'target_near_segment_keep_count',0, ...
        'target_weak_segment_candidate_count',0, ...
        'target_skipped_pair_count',0, ...
        'target_compact_cell_max_per_pair',double(Control.pairCellMaxPerPair), ...
        'target_pair_cell_selection_mode_code', ...
            PairCellSelectionModeCode_BDG(Control.pairCellSelectionMode), ...
        'target_constraint_margin_count',0, ...
        'target_constraint_margin_min',NaN, ...
        'target_constraint_margin_mean',NaN, ...
        'target_constraint_margin_abs_mean',NaN, ...
        'target_constraint_margin_max',NaN, ...
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
        'train_direction_candidate_count',0, ...
        'train_direction_keep_count',0, ...
        'train_direction_retain_ratio',NaN, ...
        'train_direction_mode_code',0, ...
        'train_weight_count',0, ...
        'train_weight_min',NaN, ...
        'train_weight_mean',NaN, ...
        'train_weight_max',NaN, ...
        'target_ref_dim',double(refDim));
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

function D = ProblemDimension_BDG(Problem,AF)
    if isstruct(Problem) && isfield(Problem,'D') && ~isempty(Problem.D)
        D = round(double(Problem.D));
    elseif isprop(Problem,'D')
        D = round(double(Problem.D));
    elseif isstruct(AF) && isfield(AF,'decs') && ~isempty(AF.decs)
        D = size(AF.decs,2);
    else
        D = 0;
    end
end

function M = ProblemObjectiveCount_BDG(Problem,AF,AI)
    if isstruct(Problem) && isfield(Problem,'M') && ~isempty(Problem.M)
        M = round(double(Problem.M));
    elseif isprop(Problem,'M')
        M = round(double(Problem.M));
    elseif isstruct(AF) && isfield(AF,'objs') && ~isempty(AF.objs)
        M = size(AF.objs,2);
    elseif isstruct(AI) && isfield(AI,'objs') && ~isempty(AI.objs)
        M = size(AI.objs,2);
    else
        M = 0;
    end
end

function [Dec,Obj] = ArchiveMatrices_BDG(A,D,M)
    Dec = zeros(0,D);
    Obj = zeros(0,M);
    if ~isstruct(A) || ~isfield(A,'decs') || ~isfield(A,'objs') || ...
            isempty(A.decs) || isempty(A.objs)
        return;
    end
    decs = double(A.decs);
    objs = double(A.objs);
    if size(decs,2) ~= D
        return;
    end
    if size(objs,2) > M
        objs = objs(:,1:M);
    elseif size(objs,2) < M
        objs = [objs,nan(size(objs,1),M-size(objs,2))];
    end
    n = min(size(decs,1),size(objs,1));
    Dec = decs(1:n,:);
    Obj = objs(1:n,:);
end

function [AFNorm,AINorm,zmin,zmax] = NormalizePairedObjectives_BDG(AFObj,AIObj,M)
    Obj = [AFObj(:,1:M);AIObj(:,1:M)];
    zmin = min(Obj,[],1);
    zmax = max(Obj,[],1);
    range = zmax - zmin;
    range(range <= 1e-12) = 1;
    AFNorm = (AFObj(:,1:M) - zmin) ./ range;
    AINorm = (AIObj(:,1:M) - zmin) ./ range;
end

function XNorm = NormalizeObjMatNoClip_BDG(X,zmin,zmax)
    range = zmax - zmin;
    range(range <= 1e-12) = 1;
    XNorm = (double(X) - zmin) ./ range;
end

function [Dec,Obj,Margin] = FeasibleSourceMatrices_BDG(Source,D,M)
    Dec = zeros(0,D);
    Obj = zeros(0,M);
    Margin = zeros(0,1);
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
    margin = nan(n,1);
    if isfield(Source,'cons') && ~isempty(Source.cons)
        cons = double(Source.cons);
        if size(cons,2) > 0
            nCons = min(n,size(cons,1));
            cons = cons(1:nCons,:);
            feasible(1:nCons) = all(cons <= 0,2);
            margin(1:nCons) = max(cons,[],2);
        end
    end
    finite = all(isfinite(decs),2) & all(isfinite(objs),2) & feasible;
    Dec = decs(finite,:);
    Obj = objs(finite,:);
    Margin = margin(finite,:);
end

function mode = NormalizeConditionMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["yt_dt","yt_dt_ref","yt_dt_t","yt_dt_t_ref"];
    assert(ismember(mode,valid), ...
        'BuildBoundaryTrainBundle_BDG:BadConditionMode', ...
        'conditionMode must be one of: %s.',strjoin(valid,", "));
end

function tf = ConditionHasT_BDG(mode)
    mode = NormalizeConditionMode_BDG(mode);
    tf = mode == "yt_dt_t" || mode == "yt_dt_t_ref";
end

function code = ConditionModeCode_BDG(mode)
    mode = NormalizeConditionMode_BDG(mode);
    names = ["yt_dt","yt_dt_ref","yt_dt_t","yt_dt_t_ref"];
    code = find(names == mode,1,'first') - 1;
end

function mode = NormalizePairCellSelectionMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["geometry","constraint_margin"];
    assert(ismember(mode,valid), ...
        'BuildBoundaryTrainBundle_BDG:BadPairCellSelectionMode', ...
        'pairCellSelectionMode must be one of: %s.',strjoin(valid,", "));
end

function code = PairCellSelectionModeCode_BDG(mode)
    mode = NormalizePairCellSelectionMode_BDG(mode);
    names = ["geometry","constraint_margin"];
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

function [dist2,t] = PointToSegmentDistancesSquared_BDG(X,a,v)
    len2 = sum(v.^2);
    if len2 <= 1e-24
        t = zeros(size(X,1),1);
        proj = repmat(a,size(X,1),1);
    else
        t = sum((X - a).*v,2) ./ len2;
        proj = a + min(max(t,0),1).*v;
    end
    dist2 = sum((X - proj).^2,2);
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

function value = MeanFinite_BDG(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
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
