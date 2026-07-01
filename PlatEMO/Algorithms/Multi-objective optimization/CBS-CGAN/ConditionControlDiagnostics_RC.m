function Diag = ConditionControlDiagnostics_RC(ganKind,GAN,QueryC,QueryRefs, ...
    ~,SampleRefs,RawDec,BMem,W,Problem,GANOptions,Options)
%CONDITIONCONTROLDIAGNOSTICS_RC Post-hoc probes for region-conditioned GANs.
%   The probes do not feed back into training or evolutionary selection.

    Diag = emptyConditionDiag();
    if isempty(GAN) || isempty(QueryC) || isempty(W) || isempty(Problem)
        return;
    end
    if nargin < 12 || isempty(Options)
        Options = struct();
    end
    Options = fillDiagnosticOptions(Options);

    rngState = rng;
    cleanupRng = onCleanup(@()rng(rngState));
    originalFE = Problem.FE;
    cleanupFE = onCleanup(@()restoreProblemFE(Problem,originalFE));
    if ~isnan(Options.seed)
        rng(round(double(Options.seed)),'twister');
    end

    [SelectedC,SelectedRefs] = selectQueryConditions(QueryC,QueryRefs, ...
        Options.maxConditions);
    zCount = max(1,round(double(Options.zSamples)));
    if isempty(SelectedC) || isempty(SelectedRefs)
        return;
    end

    zDim = inferZDim(GAN,GANOptions);
    Z = finiteSigma(GANOptions)*randn(zCount,zDim);
    Diag.condition_diag_condition_count = size(SelectedC,1);
    Diag.condition_diag_z_count = zCount;

    Diag = diagnoseSameZDifferentC(Diag,ganKind,GAN,SelectedC, ...
        SelectedRefs,Z,BMem,W,Problem,GANOptions);
    Diag = diagnoseSameCDifferentZ(Diag,ganKind,GAN,SelectedC, ...
        SelectedRefs,Z,BMem,W,Problem,GANOptions);
    Diag = diagnoseQueryRegionMatch(Diag,RawDec,SampleRefs,BMem,W, ...
        Problem,Options.neighborRadius);
    Diag = diagnoseAllWRegionMatch(Diag,ganKind,GAN,BMem,W,Problem, ...
        GANOptions,Options);

    Diag.condition_effect_ratio_dec = safeRatio( ...
        Diag.same_z_diff_c_dec_median,Diag.same_c_diff_z_dec_median);
    Diag.condition_effect_ratio_obj = safeRatio( ...
        Diag.same_z_diff_c_obj_median,Diag.same_c_diff_z_obj_median);
end

function Diag = diagnoseSameZDifferentC(Diag,ganKind,GAN,C,Refs,Z, ...
    BMem,W,Problem,GANOptions)
    k = size(C,1);
    zCount = size(Z,1);
    ProbeC = repmat(C,zCount,1);
    ProbeZ = repelem(Z,k,1);
    [Dec,Obj] = sampleAndEvaluate(ganKind,GAN,ProbeC,ProbeZ,BMem, ...
        Problem,GANOptions);
    if isempty(Dec)
        return;
    end
    DecN = normalizeDecisions(Dec,Problem);
    ObjN = normalizeByBase(Obj,objectiveBase(BMem,Obj));
    Ref = assignReferencesLocal(Obj,W,objectiveBase(BMem,Obj));

    decDist = zeros(zCount,1);
    objDist = zeros(zCount,1);
    uniqueRate = zeros(zCount,1);
    for i = 1 : zCount
        rows = (i-1)*k + (1:k);
        decDist(i) = pairwiseRmsMedian(DecN(rows,:));
        objDist(i) = pairwiseRmsMedian(ObjN(rows,:));
        uniqueRate(i) = numel(unique(Ref(rows))) / max(1,numel(unique(Refs)));
    end
    Diag.same_z_diff_c_dec_median = median(decDist,'omitnan');
    Diag.same_z_diff_c_obj_median = median(objDist,'omitnan');
    Diag.same_z_diff_c_ref_unique_rate = mean(uniqueRate,'omitnan');
end

function Diag = diagnoseSameCDifferentZ(Diag,ganKind,GAN,C,Refs,Z, ...
    BMem,W,Problem,GANOptions)
    k = size(C,1);
    zCount = size(Z,1);
    ProbeC = repelem(C,zCount,1);
    ProbeZ = repmat(Z,k,1);
    [Dec,Obj] = sampleAndEvaluate(ganKind,GAN,ProbeC,ProbeZ,BMem, ...
        Problem,GANOptions);
    if isempty(Dec)
        return;
    end
    DecN = normalizeDecisions(Dec,Problem);
    ObjN = normalizeByBase(Obj,objectiveBase(BMem,Obj));
    Ref = assignReferencesLocal(Obj,W,objectiveBase(BMem,Obj));

    decDist = zeros(k,1);
    objDist = zeros(k,1);
    refLeak = zeros(k,1);
    for i = 1 : k
        rows = (i-1)*zCount + (1:zCount);
        decDist(i) = pairwiseRmsMedian(DecN(rows,:));
        objDist(i) = pairwiseRmsMedian(ObjN(rows,:));
        refLeak(i) = mean(Ref(rows) ~= Refs(i),'omitnan');
    end
    Diag.same_c_diff_z_dec_median = median(decDist,'omitnan');
    Diag.same_c_diff_z_obj_median = median(objDist,'omitnan');
    Diag.same_c_diff_z_ref_leak_rate = mean(refLeak,'omitnan');
    Diag.same_c_diff_z_collapse_rate = mean(decDist <= 1e-6,'omitnan');
end

function Diag = diagnoseQueryRegionMatch(Diag,RawDec,SampleRefs,BMem,W, ...
    Problem,neighborRadius)
    if isempty(RawDec) || isempty(SampleRefs)
        return;
    end
    n = min(size(RawDec,1),numel(SampleRefs));
    RawDec = RawDec(1:n,:);
    SampleRefs = round(double(SampleRefs(1:n)));
    valid = isfinite(SampleRefs) & SampleRefs >= 1 & SampleRefs <= size(W,1);
    if ~any(valid)
        return;
    end
    RawDec = RawDec(valid,:);
    SampleRefs = SampleRefs(valid);
    [Obj,Con] = EvaluateDecisions_CBS(Problem,RawDec);
    ObjBase = objectiveBase(BMem,Obj);
    AssignedRef = assignReferencesLocal(Obj,W,ObjBase);
    TargetRank = targetReferenceRank(Obj,W,ObjBase,SampleRefs);

    Diag.query_generated_count = size(RawDec,1);
    Diag.query_exact_ref_match_rate = mean(AssignedRef == SampleRefs,'omitnan');
    Diag.query_neighbor_ref_match_rate = mean(isNeighborReference( ...
        W,AssignedRef,SampleRefs,neighborRadius),'omitnan');
    Diag.query_target_ref_rank_median = median(TargetRank,'omitnan');
    Diag.query_target_ref_rank_mean = mean(TargetRank,'omitnan');
    if isempty(Con)
        Diag.query_feasible_rate_probe = 1;
    else
        Diag.query_feasible_rate_probe = mean(sum(max(0,Con),2) <= 0,'omitnan');
    end
    ShuffledRefs = circshift(SampleRefs,1);
    Diag.query_shuffled_exact_match_rate = mean( ...
        AssignedRef == ShuffledRefs,'omitnan');
end

function Diag = diagnoseAllWRegionMatch(Diag,ganKind,GAN,BMem,W,Problem, ...
    GANOptions,Options)
    if isempty(W)
        return;
    end
    nRef = size(W,1);
    zPerRef = max(1,round(double(Options.allWZPerRef)));
    zDim = inferZDim(GAN,GANOptions);
    Z = finiteSigma(GANOptions)*randn(zPerRef,zDim);
    ProbeC = repelem(double(W),zPerRef,1);
    ProbeZ = repmat(Z,nRef,1);
    TargetRefs = repelem((1:nRef)',zPerRef,1);
    [Dec,Obj] = sampleAndEvaluate(ganKind,GAN,ProbeC,ProbeZ,BMem, ...
        Problem,GANOptions);
    if isempty(Dec)
        return;
    end
    ObjBase = objectiveBase(BMem,Obj);
    AssignedRef = assignReferencesLocal(Obj,W,ObjBase);
    TargetRank = targetReferenceRank(Obj,W,ObjBase,TargetRefs);
    [~,Con] = EvaluateDecisions_CBS(Problem,Dec);

    Match = isIndexWindowReference(AssignedRef,TargetRefs,2);
    seenRefs = seenReferenceSet(BMem,nRef);
    seenRows = ismember(TargetRefs,seenRefs);
    unseenRows = ~seenRows;
    ShuffledRefs = shuffledReferenceTargets(nRef,zPerRef);
    ShuffledMatch = isIndexWindowReference(AssignedRef,ShuffledRefs,2);

    Diag.all_w_condition_count = nRef;
    Diag.all_w_z_per_ref = zPerRef;
    Diag.all_w_query_generated_count = size(Dec,1);
    Diag.all_w_exact_ref_match_rate = mean(AssignedRef == TargetRefs,'omitnan');
    Diag.all_w_pm2_ref_match_rate = mean(Match,'omitnan');
    Diag.all_w_shuffled_pm2_ref_match_rate = mean(ShuffledMatch,'omitnan');
    Diag.all_w_seen_count = sum(seenRows);
    Diag.all_w_unseen_count = sum(unseenRows);
    Diag.all_w_seen_pm2_ref_match_rate = meanOrNaN(Match(seenRows));
    Diag.all_w_unseen_pm2_ref_match_rate = meanOrNaN(Match(unseenRows));
    Diag.all_w_target_ref_rank_median = median(TargetRank,'omitnan');
    Diag.all_w_target_ref_rank_mean = mean(TargetRank,'omitnan');
    if isempty(Con)
        Diag.all_w_feasible_rate_probe = 1;
    else
        Diag.all_w_feasible_rate_probe = mean(sum(max(0,Con),2) <= 0, ...
            'omitnan');
    end
end

function [Dec,Obj] = sampleAndEvaluate(ganKind,GAN,ProbeC,ProbeZ,BMem, ...
    Problem,GANOptions)
    if isempty(ProbeC)
        Dec = zeros(0,Problem.D);
        Obj = zeros(0,Problem.M);
        return;
    end
    SampleOptions = GANOptions;
    SampleOptions.sampleZ = ProbeZ;
    switch lower(strtrim(string(ganKind)))
        case "cgan"
            Dec = BoundaryCGAN_CBS('samplebycondition',GAN,ProbeC,1, ...
                SampleOptions);
        case {"wgan-gp","wgangp","wgan"}
            Dec = BoundaryWGAN_RC('samplebycondition',GAN,ProbeC,1, ...
                SampleOptions);
        otherwise
            error('CBSRegionGAN:BadGANKind', ...
                'Unsupported region GAN kind: %s.',ganKind);
    end
    [Obj,~] = EvaluateDecisions_CBS(Problem,Dec);
    if isempty(Obj) && isfield(BMem,'y_b')
        Obj = zeros(size(Dec,1),size(BMem.y_b,2));
    end
end

function [SelectedC,SelectedRefs] = selectQueryConditions(QueryC,QueryRefs,maxK)
    QueryC = double(QueryC);
    QueryRefs = round(double(QueryRefs(:)));
    valid = all(isfinite(QueryC),2) & isfinite(QueryRefs);
    QueryC = QueryC(valid,:);
    QueryRefs = QueryRefs(valid);
    if isempty(QueryC)
        SelectedC = zeros(0,size(QueryC,2));
        SelectedRefs = zeros(0,1);
        return;
    end
    maxK = min(size(QueryC,1),max(1,round(double(maxK))));
    if size(QueryC,1) <= maxK
        SelectedC = QueryC;
        SelectedRefs = QueryRefs;
        return;
    end

    chosen = false(size(QueryC,1),1);
    [~,first] = max(sum((QueryC - mean(QueryC,1)).^2,2));
    chosen(first) = true;
    while sum(chosen) < maxK
        D = pairDistance(QueryC,QueryC(chosen,:));
        minD = min(D,[],2);
        minD(chosen) = -Inf;
        [~,next] = max(minD);
        chosen(next) = true;
    end
    SelectedC = QueryC(chosen,:);
    SelectedRefs = QueryRefs(chosen);
end

function Ref = assignReferencesLocal(Obj,W,ObjBase)
    if isempty(Obj)
        Ref = zeros(0,1);
        return;
    end
    Yn = normalizeByBase(Obj,ObjBase);
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    NormY = sqrt(sum(Yn.^2,2));
    Yu = Yn./max(NormY,eps);
    Score = Yu*Wn';
    [~,Ref] = max(Score,[],2);
    zeroRows = NormY <= eps;
    if any(zeroRows)
        Dz = pairDistance(Yn(zeroRows,:),W);
        [~,Ref(zeroRows)] = min(Dz,[],2);
    end
    Ref = reshape(Ref,size(Obj,1),1);
end

function Rank = targetReferenceRank(Obj,W,ObjBase,TargetRef)
    Yn = normalizeByBase(Obj,ObjBase);
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    NormY = sqrt(sum(Yn.^2,2));
    Yu = Yn./max(NormY,eps);
    Score = Yu*Wn';
    Rank = nan(size(Obj,1),1);
    for i = 1 : size(Obj,1)
        t = TargetRef(i);
        if ~isfinite(t) || t < 1 || t > size(W,1)
            continue;
        end
        Rank(i) = 1 + sum(Score(i,:) > Score(i,t));
    end
end

function Yes = isNeighborReference(W,AssignedRef,TargetRef,radius)
    radius = max(0,round(double(radius)));
    Yes = false(size(TargetRef));
    for i = 1 : numel(TargetRef)
        a = AssignedRef(i);
        t = TargetRef(i);
        if ~isfinite(a) || ~isfinite(t) || a < 1 || a > size(W,1) || ...
                t < 1 || t > size(W,1)
            continue;
        end
        d = sqrt(sum((W - W(t,:)).^2,2));
        [~,ord] = sort(d,'ascend');
        count = min(numel(ord),1 + 2*radius);
        Yes(i) = ismember(a,ord(1:count));
    end
end

function Yes = isIndexWindowReference(AssignedRef,TargetRef,radius)
    radius = max(0,round(double(radius)));
    AssignedRef = round(double(AssignedRef(:)));
    TargetRef = round(double(TargetRef(:)));
    Yes = isfinite(AssignedRef) & isfinite(TargetRef) & ...
        abs(AssignedRef - TargetRef) <= radius;
end

function Refs = seenReferenceSet(BMem,nRef)
    if isstruct(BMem) && isfield(BMem,'ref') && ~isempty(BMem.ref)
        Refs = unique(round(double(BMem.ref(:))));
        Refs = Refs(isfinite(Refs) & Refs >= 1 & Refs <= nRef);
    else
        Refs = zeros(0,1);
    end
end

function ShuffledRefs = shuffledReferenceTargets(nRef,zPerRef)
    RefMap = (1:nRef)';
    if nRef > 1
        RefMap = circshift(RefMap,max(1,round(nRef/3)));
    end
    ShuffledRefs = repelem(RefMap,zPerRef,1);
end

function DecN = normalizeDecisions(Dec,Problem)
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    span = upper - lower;
    span(span <= eps) = 1;
    DecN = (double(Dec) - lower)./span;
    DecN(~isfinite(DecN)) = 0;
end

function Base = objectiveBase(BMem,Obj)
    Base = Obj;
    if isstruct(BMem) && isfield(BMem,'y_b') && ~isempty(BMem.y_b)
        Base = [double(BMem.y_b);double(Obj)];
    end
    if isempty(Base)
        Base = zeros(0,size(Obj,2));
    end
end

function Xn = normalizeByBase(X,Base)
    if isempty(X)
        Xn = X;
        return;
    end
    if isempty(Base)
        Base = X;
    end
    MinV = min(Base,[],1);
    SpanV = max(Base,[],1) - MinV;
    SpanV(SpanV <= eps) = 1;
    Xn = (double(X) - MinV)./SpanV;
    Xn(~isfinite(Xn)) = 0;
end

function value = pairwiseRmsMedian(X)
    n = size(X,1);
    if n < 2
        value = 0;
        return;
    end
    D = pairDistance(X,X);
    mask = triu(true(n),1);
    value = median(D(mask),'omitnan');
end

function D = pairDistance(A,B)
    if isempty(A) || isempty(B)
        D = zeros(size(A,1),size(B,1));
        return;
    end
    D = sqrt(max(0,sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B')));
end

function value = safeRatio(a,b)
    if ~isfinite(a) || ~isfinite(b)
        value = NaN;
    else
        value = a / max(abs(b),eps);
    end
end

function value = meanOrNaN(X)
    if isempty(X)
        value = NaN;
    else
        value = mean(X,'omitnan');
    end
end

function zDim = inferZDim(GAN,GANOptions)
    if isstruct(GAN) && isfield(GAN,'zDim') && ~isempty(GAN.zDim)
        zDim = max(1,round(double(GAN.zDim)));
    elseif isstruct(GANOptions) && isfield(GANOptions,'zDim')
        zDim = max(1,round(double(GANOptions.zDim)));
    else
        zDim = 1;
    end
end

function sigma = finiteSigma(Options)
    sigma = 1.0;
    if isstruct(Options) && isfield(Options,'sigma') && ~isempty(Options.sigma)
        sigma = double(Options.sigma);
    end
    if ~isfinite(sigma)
        sigma = 1.0;
    end
end

function Options = fillDiagnosticOptions(Options)
    Options = ensureField(Options,'maxConditions',8);
    Options = ensureField(Options,'zSamples',8);
    Options = ensureField(Options,'neighborRadius',2);
    Options = ensureField(Options,'allWZPerRef',2);
    Options = ensureField(Options,'seed',NaN);
    Options.maxConditions = max(1,round(double(Options.maxConditions)));
    Options.zSamples = max(1,round(double(Options.zSamples)));
    Options.neighborRadius = max(0,round(double(Options.neighborRadius)));
    Options.allWZPerRef = max(1,round(double(Options.allWZPerRef)));
end

function S = ensureField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function restoreProblemFE(Problem,originalFE)
    try
        Problem.FE = originalFE;
    catch
    end
end

function Diag = emptyConditionDiag()
    Diag = struct( ...
        'condition_diag_condition_count',0, ...
        'condition_diag_z_count',0, ...
        'same_z_diff_c_dec_median',NaN, ...
        'same_z_diff_c_obj_median',NaN, ...
        'same_z_diff_c_ref_unique_rate',NaN, ...
        'same_c_diff_z_dec_median',NaN, ...
        'same_c_diff_z_obj_median',NaN, ...
        'same_c_diff_z_ref_leak_rate',NaN, ...
        'same_c_diff_z_collapse_rate',NaN, ...
        'condition_effect_ratio_dec',NaN, ...
        'condition_effect_ratio_obj',NaN, ...
        'query_generated_count',0, ...
        'query_exact_ref_match_rate',NaN, ...
        'query_neighbor_ref_match_rate',NaN, ...
        'query_target_ref_rank_median',NaN, ...
        'query_target_ref_rank_mean',NaN, ...
        'query_feasible_rate_probe',NaN, ...
        'query_shuffled_exact_match_rate',NaN, ...
        'all_w_condition_count',0, ...
        'all_w_z_per_ref',0, ...
        'all_w_query_generated_count',0, ...
        'all_w_exact_ref_match_rate',NaN, ...
        'all_w_pm2_ref_match_rate',NaN, ...
        'all_w_shuffled_pm2_ref_match_rate',NaN, ...
        'all_w_seen_count',0, ...
        'all_w_unseen_count',0, ...
        'all_w_seen_pm2_ref_match_rate',NaN, ...
        'all_w_unseen_pm2_ref_match_rate',NaN, ...
        'all_w_target_ref_rank_median',NaN, ...
        'all_w_target_ref_rank_mean',NaN, ...
        'all_w_feasible_rate_probe',NaN);
end
