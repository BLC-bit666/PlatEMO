function [XBoundary,ConditionData,Diag] = BuildBoundaryTargetTriples_BDG( ...
        AF,AI,Problem,W,Options)
%BuildBoundaryTargetTriples_BDG Build (x_b, y_t, d_t) target triples.
%   x_b is the AF-side feasible boundary decision. y_t is the normalized
%   AF objective vector. d_t is the normalized objective direction AI-AF.

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
    XBoundary = zeros(0,D);
    ConditionData = zeros(0,2*M + refDim,'single');
    Diag = EmptyTargetTripleDiag_BDG(M,refDim,Options.conditionMode);

    [AFDec,AFObj] = ArchiveMatrices_BDG(AF,D,M);
    [~,AIObj] = ArchiveMatrices_BDG(AI,D,M);
    nPair = min([size(AFDec,1),size(AFObj,1),size(AIObj,1)]);
    Diag.target_pair_count = double(nPair);
    Diag.target_condition_dim = double(2*M + refDim);
    if nPair < 2 || D <= 0 || M <= 0
        return;
    end

    AFDec = AFDec(1:nPair,:);
    AFObj = AFObj(1:nPair,:);
    AIObj = AIObj(1:nPair,:);
    finiteRows = all(isfinite(AFDec),2) & all(isfinite(AFObj),2) & ...
        all(isfinite(AIObj),2);
    if ~any(finiteRows)
        return;
    end
    AFDec = AFDec(finiteRows,:);
    AFObj = AFObj(finiteRows,:);
    AIObj = AIObj(finiteRows,:);
    keepIndex = find(finiteRows);

    [AFNorm,AINorm] = NormalizePairedObjectives_BDG(AFObj,AIObj,M);
    Direction = AINorm - AFNorm;
    finiteTargets = all(isfinite(AFNorm),2) & all(isfinite(Direction),2);
    if ~any(finiteTargets) || sum(finiteTargets) < 2
        return;
    end

    XBoundary = AFDec(finiteTargets,:);
    keepIndex = keepIndex(finiteTargets);
    RefToken = TargetRefToken_BDG(AF,keepIndex,W,refDim);
    ConditionData = single([AFNorm(finiteTargets,:), ...
        Direction(finiteTargets,:),RefToken]);
    Diag.target_keep_index = keepIndex;
    Diag.target_triple_count = double(size(XBoundary,1));
    Diag.target_condition_dim = double(size(ConditionData,2));
    Diag.target_triple_ready = double(size(XBoundary,1) >= 2);
    Diag.target_y_min = min(AFNorm(finiteTargets,:),[],1);
    Diag.target_y_max = max(AFNorm(finiteTargets,:),[],1);
    dirNorm = sqrt(sum(Direction(finiteTargets,:).^2,2));
    Diag.target_direction_norm_mean = MeanFinite_BDG(dirNorm);
end

function Options = NormalizeTargetTripleOptions_BDG(Options)
    if ~isstruct(Options)
        Options = struct();
    end
    if ~isfield(Options,'conditionMode') || isempty(Options.conditionMode)
        Options.conditionMode = "yt_dt";
    end
    Options.conditionMode = NormalizeConditionMode_BDG( ...
        Options.conditionMode);
end

function mode = NormalizeConditionMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["yt_dt","yt_dt_ref"];
    assert(ismember(mode,valid), ...
        'BuildBoundaryTargetTriples_BDG:BadConditionMode', ...
        'conditionMode must be one of: %s.',strjoin(valid,", "));
end

function dim = TargetRefTokenDim_BDG(W,M,mode)
    mode = NormalizeConditionMode_BDG(mode);
    if mode ~= "yt_dt_ref"
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

function Diag = EmptyTargetTripleDiag_BDG(M,refDim,conditionMode)
    if nargin < 2
        refDim = 0;
    end
    if nargin < 3
        conditionMode = "yt_dt";
    end
    conditionMode = NormalizeConditionMode_BDG(conditionMode);
    Diag = struct( ...
        'target_triple_ready',0, ...
        'target_pair_count',0, ...
        'target_triple_count',0, ...
        'target_condition_dim',double(2*M + refDim), ...
        'target_condition_mode_code',double(conditionMode == "yt_dt_ref"), ...
        'target_ref_token_dim',double(refDim), ...
        'target_keep_index',zeros(0,1), ...
        'target_y_min',nan(1,M), ...
        'target_y_max',nan(1,M), ...
        'target_direction_norm_mean',NaN);
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

function [AFNorm,AINorm] = NormalizePairedObjectives_BDG(AFObj,AIObj,M)
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
    X = (double(X) - double(zmin)) ./ ...
        (double(zmax) - double(zmin) + 1e-12);
    X = min(max(X,0),1);
end

function value = MeanFinite_BDG(X)
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
    else
        value = mean(X);
    end
end
