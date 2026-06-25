function varargout = BoundaryQualityTarget_BDG(action,varargin)
%BoundaryQualityTarget_BDG - Shared boundary-quality targets for BDG.

    action = lower(strtrim(char(action)));
    switch action
        case 'builddata'
            varargout = {BuildBoundaryQualityData_BDG(varargin{:})};
        case 'evaluatedlabels'
            [Labels,pairIdx,Diag] = EvaluatedBoundaryLabels_BDG(varargin{:});
            varargout = {Labels,pairIdx,Diag};
        case 'generatedlabels'
            [Labels,outside,Obj,Cons,Diag] = GeneratedBoundaryLabels_BDG(varargin{:});
            varargout = {Labels,outside,Obj,Cons,Diag};
        otherwise
            error('BoundaryQualityTarget_BDG:BadAction', ...
                'Unsupported action: %s.',action);
    end
end

function Data = BuildBoundaryQualityData_BDG(AF,AI,Problem,tauSegment)
    if nargin < 4 || isempty(tauSegment)
        tauSegment = 0.20;
    end
    M = ProblemObjectiveCount_BDG(Problem,AF,AI);
    [lower,upper] = ProblemDecisionBounds_BDG(Problem);
    [AFObj,AIObj] = PairObjectiveMatrices_BDG(AF,AI,M);
    [zmin,zmax] = ObjectiveNormalizationBounds_BDG(AFObj,AIObj,M);
    [objLower,objUpper,hasObjBounds] = ObjectiveStandardBounds_BDG( ...
        Problem,M);

    Data = struct();
    Data.AFObj = AFObj;
    Data.AIObj = AIObj;
    Data.zmin = zmin;
    Data.zmax = zmax;
    Data.tauSegment = SanitizedTau_BDG(tauSegment,0.20);
    Data.tauOutside = 0.05;
    Data.objectiveStandardLower = objLower;
    Data.objectiveStandardUpper = objUpper;
    Data.hasObjectiveStandardBounds = hasObjBounds;
    Data.objectiveFcn = @(X)SafeProblemObjective_BDG(Problem,X,M);
    Data.constraintFcn = @(X)SafeProblemConstraint_BDG(Problem,X,M);
    Data.lower = lower;
    Data.upper = upper;
    Data.hasValidSegments = HasValidBoundaryQualitySegments_BDG( ...
        AFObj,AIObj,zmin,zmax,M);
    Data.ready = Data.hasValidSegments;
end

function [Labels,pairIdx,Diag] = EvaluatedBoundaryLabels_BDG(Objs,Cons,Data)
    if nargin < 2
        Cons = [];
    end
    [Labels,pairIdx,~,Diag] = BoundaryQualityLabels_BDG( ...
        double(Objs),Cons,Data,[]);
    Labels = single(Labels(:));
end

function [Labels,outside,Obj,Cons,Diag] = GeneratedBoundaryLabels_BDG( ...
        XScaled,pairIdx,Data)
    XRaw = ExtractNumericData_BDG(XScaled);
    n = size(XRaw,2);
    y = zeros(1,n,'single');
    outside = false(n,1);
    Obj = zeros(n,0);
    Cons = zeros(n,0);
    Diag = BoundaryQualityDiagnostics_BDG(y(:),outside,zeros(n,1),[]);
    if ~IsGeneratedDataReady_BDG(Data) || isempty(XRaw)
        Labels = dlarray(y,'CB');
        return;
    end

    lower = double(Data.lower(:)');
    upper = double(Data.upper(:)');
    if size(XRaw,1) ~= numel(lower) || numel(lower) ~= numel(upper)
        Labels = dlarray(y,'CB');
        return;
    end

    X = double(XRaw)';
    X = bsxfun(@plus,bsxfun(@times,(X + 1)./2,upper - lower),lower);
    X = min(max(X,lower),upper);
    try
        Obj = double(Data.objectiveFcn(X));
    catch
        Labels = dlarray(y,'CB');
        return;
    end
    if isempty(Obj) || size(Obj,1) ~= n
        Labels = dlarray(y,'CB');
        return;
    end
    try
        Cons = double(Data.constraintFcn(X));
    catch
        Cons = zeros(n,0);
    end
    Cons = ConformConstraintRows_BDG(Cons,n);

    [labelColumn,~,outside,Diag] = BoundaryQualityLabels_BDG( ...
        Obj,Cons,Data,pairIdx);
    Labels = dlarray(single(labelColumn(:)'),'CB');
end

function [Labels,pairIdx,outside,Diag] = BoundaryQualityLabels_BDG( ...
        Obj,Cons,Data,inputPairIdx)
    Obj = double(Obj);
    n = size(Obj,1);
    Labels = zeros(n,1,'single');
    pairIdx = zeros(n,1);
    outside = false(n,1);
    positiveCV = zeros(n,1);

    if n <= 0
        Diag = BoundaryQualityDiagnostics_BDG(Labels,outside,positiveCV,[]);
        return;
    end
    Cons = ConformConstraintRows_BDG(Cons,n);
    [positiveCV,invalidCons] = PositiveConstraintViolation_BDG(Cons,n);

    if ~IsLabelDataReady_BDG(Data) || size(Obj,2) ~= size(Data.AFObj,2)
        Diag = BoundaryQualityDiagnostics_BDG(Labels,outside,positiveCV,[]);
        return;
    end

    invalidObj = ~all(isfinite(Obj),2);
    ObjN = NormalizeByRangeNoClip_BDG(Obj,Data.zmin,Data.zmax);
    AFObj = NormalizeByRangeNoClip_BDG(Data.AFObj,Data.zmin,Data.zmax);
    AIObj = NormalizeByRangeNoClip_BDG(Data.AIObj,Data.zmin,Data.zmax);

    [dist,pairIdx] = NearestBoundarySegmentForPoints_BDG(ObjN,AFObj,AIObj);
    [dist,pairIdx] = ApplyInputPairIdx_BDG( ...
        ObjN,AFObj,AIObj,dist,pairIdx,inputPairIdx);

    tauSegment = TauForPairs_BDG(Data.tauSegment,pairIdx, ...
        size(AFObj,1),0.20);

    tauCV = ConstraintTau_BDG(positiveCV);

    [outsideSeverity,outside] = OutsideSeverity_BDG(Obj,Data);
    tauOutside = FieldTau_BDG(Data,'tauOutside',0.05);
    segmentScore = SoftBoundaryComponentScore_BDG(dist,tauSegment);
    feasibleScore = SoftBoundaryComponentScore_BDG(positiveCV,tauCV);
    outsideScore = SoftBoundaryComponentScore_BDG(outsideSeverity,tauOutside);

    quality = segmentScore .* feasibleScore .* outsideScore;
    invalid = invalidObj | invalidCons | ~isfinite(quality);
    quality(invalid) = 0;
    quality = min(max(quality,0),1);
    Labels = single(quality(:));
    Components = struct( ...
        'segmentScore',segmentScore, ...
        'feasibleScore',feasibleScore, ...
        'outsideScore',outsideScore, ...
        'outsideSeverity',outsideSeverity);
    Diag = BoundaryQualityDiagnostics_BDG(Labels,outside, ...
        positiveCV,Components);
end

function score = SoftBoundaryComponentScore_BDG(value,tau)
    value = max(double(value(:)),0);
    tau = double(tau);
    if isscalar(tau)
        tau = repmat(tau,numel(value),1);
    else
        tau = tau(:);
    end
    if numel(tau) ~= numel(value)
        tau = repmat(1,numel(value),1);
    end
    tau(~isfinite(tau) | tau <= 0) = 1;
    ratio = value ./ (tau + 1e-12);
    score = 1 ./ (1 + sqrt(ratio));
    score(~isfinite(value) | ~isfinite(score)) = 0;
end

function [AFObj,AIObj] = PairObjectiveMatrices_BDG(AF,AI,M)
    AFAll = PopulationObjectives_BDG(AF,M);
    AIAll = PopulationObjectives_BDG(AI,M);
    nPair = min(size(AFAll,1),size(AIAll,1));
    AFObj = AFAll(1:nPair,:);
    AIObj = AIAll(1:nPair,:);
end

function Obj = PopulationObjectives_BDG(P,M)
    Obj = zeros(0,M);
    try
        Obj = double(P.objs);
    catch
        return;
    end
    Obj = ConformObjectiveColumns_BDG(Obj,M);
end

function Obj = ConformObjectiveColumns_BDG(Obj,M)
    if M <= 0
        Obj = zeros(size(Obj,1),0);
    elseif size(Obj,2) > M
        Obj = Obj(:,1:M);
    elseif size(Obj,2) < M
        Obj = [Obj,nan(size(Obj,1),M-size(Obj,2))];
    end
end

function M = ProblemObjectiveCount_BDG(Problem,AF,AI)
    M = 0;
    try
        M = double(Problem.M);
    catch
    end
    if isempty(M) || ~isscalar(M) || ~isfinite(M) || M <= 0
        M = ObjectiveColumns_BDG(AF);
    end
    if isempty(M) || M <= 0
        M = ObjectiveColumns_BDG(AI);
    end
    M = max(0,round(double(M)));
end

function M = ObjectiveColumns_BDG(P)
    M = 0;
    try
        Obj = P.objs;
        M = size(Obj,2);
    catch
    end
end

function [lower,upper] = ProblemDecisionBounds_BDG(Problem)
    lower = [];
    upper = [];
    try
        lower = double(Problem.lower(:)');
        upper = double(Problem.upper(:)');
    catch
    end
    if isempty(lower) || isempty(upper) || numel(lower) ~= numel(upper)
        D = 0;
        try
            D = double(Problem.D);
        catch
        end
        D = max(0,round(D));
        lower = zeros(1,D);
        upper = ones(1,D);
    end
end

function [zmin,zmax] = ObjectiveNormalizationBounds_BDG(AFObj,AIObj,M)
    zmin = zeros(1,M);
    zmax = ones(1,M);
    allObj = [AFObj;AIObj];
    if isempty(allObj)
        return;
    end
    for j = 1 : M
        values = allObj(:,j);
        values = values(isfinite(values));
        if isempty(values)
            continue;
        end
        zmin(j) = min(values);
        zmax(j) = max(values);
    end
end

function flag = HasValidBoundaryQualitySegments_BDG(AFObj,AIObj,zmin,zmax,M)
    flag = false;
    if M <= 0 || isempty(AFObj) || isempty(AIObj)
        return;
    end
    if size(AFObj,2) ~= M || size(AIObj,2) ~= M || ...
            numel(zmin) ~= M || numel(zmax) ~= M
        return;
    end
    if ~all(isfinite(zmin(:))) || ~all(isfinite(zmax(:)))
        return;
    end
    nPair = min(size(AFObj,1),size(AIObj,1));
    if nPair <= 0
        return;
    end
    validSegments = all(isfinite(AFObj(1:nPair,:)),2) & ...
        all(isfinite(AIObj(1:nPair,:)),2);
    flag = any(validSegments);
end

function [lower,upper,ok] = ObjectiveStandardBounds_BDG(Problem,M)
    lower = nan(1,M);
    upper = nan(1,M);
    ok = false;
    PF = [];
    try
        PF = Problem.PF;
    catch
    end
    if isempty(PF)
        try
            PF = Problem.GetPF();
        catch
        end
    end
    if isnumeric(PF) && size(PF,2) >= M
        for j = 1 : M
            values = PF(:,j);
            values = values(isfinite(values));
            if isempty(values)
                return;
            end
            lower(j) = min(values);
            upper(j) = max(values);
        end
    elseif iscell(PF) && numel(PF) >= M
        for j = 1 : M
            values = PF{j}(:);
            values = values(isfinite(values));
            if isempty(values)
                return;
            end
            lower(j) = min(values);
            upper(j) = max(values);
        end
    else
        return;
    end
    ok = all(isfinite(lower)) && all(isfinite(upper));
end

function Obj = SafeProblemObjective_BDG(Problem,X,M)
    n = size(X,1);
    Obj = zeros(n,0);
    try
        Dec = Problem.CalDec(double(X));
        Obj = double(Problem.CalObj(Dec));
    catch
    end
    if ObjectiveRowsReady_BDG(Obj,n,M)
        Obj = ConformObjectiveColumns_BDG(Obj,M);
        return;
    end
    [Obj,~] = ProblemEvaluationNoFE_BDG(Problem,X,M);
end

function Cons = SafeProblemConstraint_BDG(Problem,X,M)
    n = size(X,1);
    useEvaluation = true;
    try
        Dec = Problem.CalDec(double(X));
        Obj = double(Problem.CalObj(Dec));
        useEvaluation = ~ObjectiveRowsReady_BDG(Obj,n,M);
    catch
    end
    if useEvaluation
        [~,Cons] = ProblemEvaluationNoFE_BDG(Problem,X,M);
    else
        try
            Cons = Problem.CalCon(Dec);
        catch
            Cons = zeros(n,0);
        end
    end
    if isempty(Cons)
        Cons = zeros(n,0);
        return;
    end
    Cons = ConformConstraintRows_BDG(double(Cons),n);
end

function flag = ObjectiveRowsReady_BDG(Obj,n,M)
    flag = ~isempty(Obj) && size(Obj,1) == n && size(Obj,2) >= M;
end

function [Obj,Cons] = ProblemEvaluationNoFE_BDG(Problem,X,M)
    n = size(X,1);
    Obj = zeros(n,0);
    Cons = zeros(n,0);
    oldFE = Problem.FE;
    cleanup = onCleanup(@()RestoreProblemFE_BDG(Problem,oldFE)); %#ok<NASGU>
    try
        Population = Problem.Evaluation(double(X));
        Obj = double(Population.objs);
        Cons = double(Population.cons);
    catch
        return;
    end
    Obj = ConformObjectiveColumns_BDG(Obj,M);
    Cons = ConformConstraintRows_BDG(Cons,n);
end

function RestoreProblemFE_BDG(Problem,oldFE)
    try
        Problem.FE = oldFE;
    catch
    end
end

function Cons = ConformConstraintRows_BDG(Cons,n)
    if isempty(Cons)
        Cons = zeros(n,0);
        return;
    end
    Cons = double(Cons);
    if size(Cons,1) == n
        return;
    elseif size(Cons,2) == n
        Cons = Cons';
    else
        Cons = nan(n,1);
    end
end

function [positiveCV,invalidCons] = PositiveConstraintViolation_BDG(Cons,n)
    positiveCV = zeros(n,1);
    invalidCons = false(n,1);
    if isempty(Cons)
        return;
    end
    invalidCons = any(~isfinite(Cons),2);
    positiveCV = max(max(Cons,[],2),0);
    positiveCV(invalidCons) = NaN;
end

function tauCV = ConstraintTau_BDG(positiveCV)
    positive = positiveCV(isfinite(positiveCV) & positiveCV > 0);
    if isempty(positive)
        tauCV = 1;
    else
        tauCV = median(positive);
        if ~isfinite(tauCV) || tauCV <= 0
            tauCV = 1;
        end
    end
end

function [dist,pairIdx] = NearestBoundarySegmentForPoints_BDG(P,A,B)
    n = size(P,1);
    nPair = min(size(A,1),size(B,1));
    dist = inf(n,1);
    pairIdx = zeros(n,1);
    for p = 1 : nPair
        d = DistancePointsToSegment_BDG(P,A(p,:),B(p,:));
        hit = d < dist;
        dist(hit) = d(hit);
        pairIdx(hit) = p;
    end
end

function [dist,pairIdx] = ApplyInputPairIdx_BDG( ...
        P,A,B,dist,pairIdx,inputPairIdx)
    n = size(P,1);
    nPair = min(size(A,1),size(B,1));
    if isempty(inputPairIdx) || nPair <= 0
        return;
    end
    p = round(double(inputPairIdx(:)));
    if isscalar(p) && n > 1
        p = repmat(p,n,1);
    end
    if numel(p) ~= n
        return;
    end
    finite = isfinite(p);
    if ~any(finite)
        return;
    end
    p(finite) = max(1,min(nPair,p(finite)));
    for j = unique(p(finite))'
        rows = finite & p == j;
        dist(rows) = DistancePointsToSegment_BDG(P(rows,:),A(j,:),B(j,:));
        pairIdx(rows) = j;
    end
end

function d = DistancePointsToSegment_BDG(P,a,b)
    n = size(P,1);
    d = inf(n,1);
    valid = all(isfinite(P),2) & all(isfinite(a)) & all(isfinite(b));
    if ~any(valid)
        return;
    end
    v = b - a;
    len2 = sum(v.^2);
    if len2 <= 1e-24
        R = bsxfun(@minus,P(valid,:),a);
    else
        R0 = bsxfun(@minus,P(valid,:),a);
        t = sum(bsxfun(@times,R0,v),2) ./ len2;
        t = min(max(t,0),1);
        projection = bsxfun(@plus,a,bsxfun(@times,t,v));
        R = P(valid,:) - projection;
    end
    d(valid) = sqrt(sum(R.^2,2));
end

function X = NormalizeByRangeNoClip_BDG(X,zmin,zmax)
    X = bsxfun(@rdivide,bsxfun(@minus,double(X),double(zmin)), ...
        double(zmax) - double(zmin) + 1e-12);
end

function [severity,outside] = OutsideSeverity_BDG(Obj,Data)
    n = size(Obj,1);
    severity = zeros(n,1);
    outside = false(n,1);
    if ~isfield(Data,'hasObjectiveStandardBounds') || ...
            ~logical(Data.hasObjectiveStandardBounds)
        return;
    end
    lower = double(Data.objectiveStandardLower(:)');
    upper = double(Data.objectiveStandardUpper(:)');
    if numel(lower) ~= size(Obj,2) || numel(upper) ~= size(Obj,2) || ...
            ~all(isfinite(lower)) || ~all(isfinite(upper))
        return;
    end
    width = upper - lower + 1e-12;
    below = max(bsxfun(@rdivide,bsxfun(@minus,lower,Obj),width),0);
    above = max(bsxfun(@rdivide,bsxfun(@minus,Obj,upper),width),0);
    overflow = below + above;
    finiteObj = all(isfinite(Obj),2);
    severity(finiteObj) = sqrt(sum(overflow(finiteObj,:).^2,2));
    outside = ~finiteObj | severity > 0;
end

function tau = TauForPairs_BDG(tauData,pairIdx,nPair,defaultTau)
    tauData = SanitizedTau_BDG(tauData,defaultTau);
    if isscalar(tauData)
        tau = repmat(tauData,numel(pairIdx),1);
        return;
    end
    tauVector = tauData(:);
    if numel(tauVector) ~= nPair
        tau = repmat(defaultTau,numel(pairIdx),1);
        return;
    end
    p = max(1,min(nPair,round(double(pairIdx(:)))));
    tau = tauVector(p);
    tau(~isfinite(tau) | tau <= 0) = defaultTau;
end

function tau = FieldTau_BDG(Data,fieldName,defaultTau)
    tau = defaultTau;
    if isfield(Data,fieldName)
        tau = SanitizedTau_BDG(Data.(fieldName),defaultTau);
        if ~isscalar(tau)
            tau = defaultTau;
        end
    end
end

function tau = SanitizedTau_BDG(tau,defaultTau)
    tau = double(tau);
    if isempty(tau)
        tau = defaultTau;
        return;
    end
    bad = ~isfinite(tau) | tau <= 0;
    tau(bad) = defaultTau;
end

function Diag = BoundaryQualityDiagnostics_BDG( ...
        Labels,outside,positiveCV,Components)
    if nargin < 4
        Components = [];
    end
    Labels = double(Labels(:));
    finiteLabels = Labels(isfinite(Labels));
    Diag = struct( ...
        'boundary_quality_count',double(numel(Labels)), ...
        'boundary_quality_mean',NaN, ...
        'boundary_quality_p10',NaN, ...
        'boundary_quality_p50',NaN, ...
        'boundary_quality_p90',NaN, ...
        'boundary_quality_zero_rate',NaN, ...
        'boundary_quality_high_rate',NaN, ...
        'boundary_quality_outside_rate',NaN, ...
        'boundary_quality_positive_cv_rate',NaN, ...
        'boundary_quality_segment_score_mean',NaN, ...
        'boundary_quality_feasible_score_mean',NaN, ...
        'boundary_quality_outside_score_mean',NaN, ...
        'boundary_quality_outside_severity_mean',NaN, ...
        'boundary_quality_positive_cv_mean',NaN);
    if ~isempty(finiteLabels)
        Diag.boundary_quality_mean = mean(finiteLabels);
        Diag.boundary_quality_p10 = PercentileFinite_BDG(finiteLabels,10);
        Diag.boundary_quality_p50 = PercentileFinite_BDG(finiteLabels,50);
        Diag.boundary_quality_p90 = PercentileFinite_BDG(finiteLabels,90);
        Diag.boundary_quality_zero_rate = ...
            mean(double(finiteLabels <= 1e-6));
        Diag.boundary_quality_high_rate = ...
            mean(double(finiteLabels >= 0.5));
    end
    if ~isempty(outside)
        Diag.boundary_quality_outside_rate = mean(double(outside(:)));
    end
    if ~isempty(positiveCV)
        Diag.boundary_quality_positive_cv_rate = ...
            mean(double(isfinite(positiveCV(:)) & positiveCV(:) > 0));
        Diag.boundary_quality_positive_cv_mean = ...
            MeanFinite_BDG(positiveCV);
    end
    if isstruct(Components)
        Diag.boundary_quality_segment_score_mean = ...
            MeanFinite_BDG(ComponentField_BDG(Components,'segmentScore'));
        Diag.boundary_quality_feasible_score_mean = ...
            MeanFinite_BDG(ComponentField_BDG(Components,'feasibleScore'));
        Diag.boundary_quality_outside_score_mean = ...
            MeanFinite_BDG(ComponentField_BDG(Components,'outsideScore'));
        Diag.boundary_quality_outside_severity_mean = ...
            MeanFinite_BDG(ComponentField_BDG(Components,'outsideSeverity'));
    end
end

function values = ComponentField_BDG(S,fieldName)
    values = [];
    if isfield(S,fieldName)
        values = S.(fieldName);
    end
end

function value = MeanFinite_BDG(X)
    X = double(X(:));
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
    else
        value = mean(X);
    end
end

function value = PercentileFinite_BDG(X,p)
    X = sort(X(isfinite(X)));
    if isempty(X)
        value = NaN;
        return;
    end
    p = min(max(double(p),0),100);
    idx = max(1,ceil((p/100) * numel(X)));
    value = X(idx);
end

function flag = IsLabelDataReady_BDG(Data)
    required = {'AFObj','AIObj','zmin','zmax','tauSegment', ...
        'tauOutside','objectiveStandardLower','objectiveStandardUpper', ...
        'hasObjectiveStandardBounds'};
    flag = isstruct(Data);
    for i = 1 : numel(required)
        flag = flag && isfield(Data,required{i});
    end
    flag = flag && IsTrueScalarField_BDG(Data,'ready') && ...
        IsTrueScalarField_BDG(Data,'hasValidSegments') && ...
        HasValidBoundaryQualitySegments_BDG(Data.AFObj,Data.AIObj, ...
        Data.zmin,Data.zmax,size(Data.AFObj,2));
end

function flag = IsGeneratedDataReady_BDG(Data)
    required = {'lower','upper','objectiveFcn','constraintFcn'};
    flag = IsLabelDataReady_BDG(Data);
    for i = 1 : numel(required)
        flag = flag && isfield(Data,required{i});
    end
end

function flag = IsTrueScalarField_BDG(Data,fieldName)
    flag = false;
    if ~isfield(Data,fieldName)
        return;
    end
    value = Data.(fieldName);
    if ~(isnumeric(value) || islogical(value)) || ~isscalar(value)
        return;
    end
    value = double(value);
    flag = isfinite(value) && value ~= 0;
end

function X = ExtractNumericData_BDG(X)
    try
        X = extractdata(X);
    catch
    end
    try
        X = gather(X);
    catch
    end
    X = double(X);
end
