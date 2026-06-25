function [Summary,outFile] = run_CBS_CGAN_non_cgan_baseline_eval( ...
    inputPath,outFile,Options)
%RUN_CBS_CGAN_NON_CGAN_BASELINE_EVAL Compare simple non-CGAN holdout baselines.
%
% The input is a captured_dataset.mat file or a directory containing captured
% datasets.  The baselines use the same modulo-based holdout split as
% run_CBS_CGAN_holdout_ref_eval.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(inputPath)
        inputPath = fullfile(rootDir,'Data','CBS_CGAN');
    end
    if nargin < 2
        outFile = [];
    end
    if nargin < 3 || isempty(Options)
        Options = struct();
    end
    Options = normalizeOptions(Options);

    files = findCapturedDatasets(inputPath);
    Rows = repmat(emptySummaryRow(),0,1);
    for i = 1 : numel(files)
        Rows = [Rows; evaluateOneCapture(files(i),Options)]; %#ok<AGROW>
    end
    Summary = struct2table(Rows);
    if nargin < 2 || isempty(outFile)
        if isfolder(inputPath)
            outFile = fullfile(inputPath,'non_cgan_baseline_eval.csv');
        else
            outFile = fullfile(fileparts(inputPath), ...
                'non_cgan_baseline_eval.csv');
        end
    end
    if ~isempty(outFile)
        outDir = fileparts(outFile);
        if ~isempty(outDir) && ~isfolder(outDir)
            mkdir(outDir);
        end
        writetable(Summary,outFile);
    end
end

function Options = normalizeOptions(Options)
    Options = ensureField(Options,'holdoutModulo',3);
    Options = ensureField(Options,'holdoutRemainder',0);
    Options = ensureField(Options,'samplesPerCondition',1);
    Options = ensureField(Options,'mutationSigma',0.01);
    Options.holdoutModulo = max(2,round(double(Options.holdoutModulo)));
    Options.holdoutRemainder = mod(round(double(Options.holdoutRemainder)), ...
        Options.holdoutModulo);
    Options.samplesPerCondition = max(1,round(double( ...
        Options.samplesPerCondition)));
    Options.mutationSigma = max(0,double(Options.mutationSigma));
end

function S = ensureField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function files = findCapturedDatasets(inputPath)
    inputPath = string(inputPath);
    if isfile(inputPath)
        files = inputPath;
        return;
    end
    Listing = dir(fullfile(inputPath,'**','captured_dataset.mat'));
    files = strings(numel(Listing),1);
    for i = 1 : numel(Listing)
        files(i) = string(fullfile(Listing(i).folder,Listing(i).name));
    end
end

function Rows = evaluateOneCapture(file,Options)
    [problemName,runId] = parseCaptureName(file);
    Methods = ["replay";"interp";"mutate"];
    Rows = repmat(emptySummaryRow(),numel(Methods),1);
    for m = 1 : numel(Methods)
        Rows(m).file = string(file);
        Rows(m).problem = problemName;
        Rows(m).run = runId;
        Rows(m).method = Methods(m);
    end
    try
        Loaded = load(file,"Data");
        Data = Loaded.Data;
        BMem = Data.BMem;
        if isempty(BMem) || ~isfield(BMem,'y_b') || isempty(BMem.y_b) || ...
                ~isfield(BMem,'ref') || ~isfield(BMem,'x_b')
            Rows = markRows(Rows,"empty_bmem");
            return;
        end
        if ~isfield(BMem,'chain') || isempty(BMem.chain)
            BMem.chain = ones(size(BMem.y_b,1),1);
        end
        valid = all(isfinite(BMem.x_b),2);
        holdout = valid & mod(BMem.ref(:),Options.holdoutModulo) == ...
            Options.holdoutRemainder;
        train = valid & ~holdout;
        if sum(train) < 1 || sum(holdout) < 1
            Rows = markRows(Rows,"insufficient_refs");
            return;
        end

        Problem = makeProblem(problemName,size(BMem.x_b,2));
        W = referenceMatrixFromData(Data,BMem,Problem.M);
        TargetC = referenceConditionsFromBMem(BMem,W,holdout,Problem.M, ...
            Data.DatasetInfo);
        TrainC = referenceConditionsFromBMem(BMem,W,train,Problem.M, ...
            Data.DatasetInfo);
        TargetY = BMem.y_b(holdout,:);
        TrainX = BMem.x_b(train,:);
        ReplayDec = replayDecisions(TrainX,TrainC, ...
            TargetC,Options.samplesPerCondition);
        InterpDec = interpDecisions(BMem,train,holdout, ...
            Options.samplesPerCondition);
        MutateDec = mutateDecisions(ReplayDec,Problem,Options);

        Rows(1) = evaluateDecisions(Rows(1),Problem,Data, ...
            ReplayDec,TargetY,Options,sum(train), ...
            sum(holdout));
        Rows(2) = evaluateDecisions(Rows(2),Problem,Data, ...
            InterpDec,TargetY,Options,sum(train), ...
            sum(holdout));
        Rows(3) = evaluateDecisions(Rows(3),Problem,Data, ...
            MutateDec,TargetY,Options,sum(train), ...
            sum(holdout));
    catch err
        Rows = markRows(Rows,"failed");
        for i = 1 : numel(Rows)
            Rows(i).error_message = string(getReport(err,'extended', ...
                'hyperlinks','off'));
        end
    end
end

function Rows = markRows(Rows,status)
    for i = 1 : numel(Rows)
        Rows(i).status = string(status);
    end
end

function Dec = replayDecisions(TrainX,TrainC,TargetC,samplesPerCondition)
    D = pointDistance(TargetC,TrainC);
    [~,nearest] = min(D,[],2);
    nearest = repelem(nearest(:),samplesPerCondition,1);
    Dec = TrainX(nearest,:);
end

function Dec = interpDecisions(BMem,train,holdout,samplesPerCondition)
    H = find(holdout);
    Dec = zeros(numel(H)*samplesPerCondition,size(BMem.x_b,2));
    row = 0;
    for i = 1 : numel(H)
        h = H(i);
        cand = find(train & BMem.chain == BMem.chain(h));
        if numel(cand) < 2
            cand = find(train);
        end
        [~,ord] = sort(abs(BMem.ref(cand) - BMem.ref(h)) + ...
            1e-6*(1:numel(cand))');
        cand = cand(ord);
        if numel(cand) >= 2
            a = cand(1);
            b = cand(2);
            denom = BMem.ref(b) - BMem.ref(a);
            if abs(denom) <= eps
                t = 0.5;
            else
                t = (BMem.ref(h) - BMem.ref(a))/denom;
            end
            t = min(max(t,0),1);
            x = (1-t)*BMem.x_b(a,:) + t*BMem.x_b(b,:);
        else
            x = BMem.x_b(cand(1),:);
        end
        for s = 1 : samplesPerCondition
            row = row + 1;
            Dec(row,:) = x;
        end
    end
end

function Dec = mutateDecisions(BaseDec,Problem,Options)
    if isempty(BaseDec)
        Dec = BaseDec;
        return;
    end
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    span = upper - lower;
    span(span <= eps) = 1;
    Noise = randn(size(BaseDec)).*span.*Options.mutationSigma;
    Dec = min(max(BaseDec + Noise,lower),upper);
end

function Row = evaluateDecisions(Row,Problem,Data,Dec,TargetY,Options, ...
    trainCount,holdoutCount)
    Row.train_count = trainCount;
    Row.holdout_count = holdoutCount;
    Row.generated_count = size(Dec,1);
    if isempty(Dec)
        Row.status = "empty_generation";
        return;
    end
    Pop = Problem.Evaluation(Dec);
    TargetObj = repelem(TargetY,Options.samplesPerCondition,1);
    ObjDist = conditionObjectiveDistances(Pop.objs,TargetObj, ...
        Data.DatasetInfo);
    Row.holdout_obj_dist50 = percentileFinite(ObjDist,50);
    Row.holdout_obj_dist90 = percentileFinite(ObjDist,90);
    Row.feasible_rate = feasibleRate(Pop.cons);
    Row.status = "ok";
end

function Problem = makeProblem(problemName,D)
    Constructor = str2func(char(problemName));
    Problem = Constructor('N',100,'D',D,'maxFE',100000);
end

function [problemName,runId] = parseCaptureName(file)
    folder = string(fileparts(file));
    [~,leaf] = fileparts(folder);
    token = regexp(leaf,'^(.*)_run(\d+)$','tokens','once');
    if isempty(token)
        problemName = "";
        runId = NaN;
    else
        problemName = string(token{1});
        runId = str2double(token{2});
    end
end

function C = referenceConditionsFromBMem(BMem,W,rows,M,DatasetInfo)
    refs = round(double(BMem.ref(rows)));
    tau = tauFromBMem(BMem,rows,DatasetInfo);
    C = zeros(numel(refs),M+1);
    valid = isfinite(refs) & refs >= 1 & refs <= size(W,1);
    if any(valid)
        C(valid,1:M) = W(refs(valid),1:M);
    end
    C(:,end) = max(0,min(1,tau(:)));
end

function tau = tauFromBMem(BMem,rows,DatasetInfo)
    n = numel(BMem.ref(rows));
    if all(isfield(BMem,{'y_b','y_f','y_i'}))
        [objMin,objSpan] = conditionScaleFromDatasetInfo(DatasetInfo, ...
            size(BMem.y_b,2));
        tau = PairLocalTau_CBS(BMem.y_b(rows,:),BMem.y_f(rows,:), ...
            BMem.y_i(rows,:),objMin,objSpan);
    elseif isfield(BMem,'tau')
        tau = double(BMem.tau(rows));
    else
        tau = zeros(n,1);
    end
    tau(~isfinite(tau)) = 0;
end

function [objMin,objSpan] = conditionScaleFromDatasetInfo(DatasetInfo,M)
    if isstruct(DatasetInfo) && isfield(DatasetInfo,'objMin') && ...
            isfield(DatasetInfo,'objSpan') && ...
            numel(DatasetInfo.objMin) == M && ...
            numel(DatasetInfo.objSpan) == M
        objMin = double(DatasetInfo.objMin(:)');
        objSpan = double(DatasetInfo.objSpan(:)');
    else
        objMin = zeros(1,M);
        objSpan = ones(1,M);
    end
    objSpan(objSpan <= eps) = 1;
end

function W = referenceMatrixFromData(Data,BMem,M)
    if isfield(Data,'W') && ~isempty(Data.W) && size(Data.W,2) >= M
        W = double(Data.W(:,1:M));
        return;
    end
    nRef = max(1,max(round(double(BMem.ref(:)))));
    W = UniformPoint(nRef,M);
end

function Dist = conditionObjectiveDistances(Obj,TargetObj,DatasetInfo)
    n = min(size(Obj,1),size(TargetObj,1));
    ObjN = normalizeWithInfo(Obj(1:n,:),DatasetInfo.objMin, ...
        DatasetInfo.objSpan);
    TargetN = normalizeWithInfo(TargetObj(1:n,:),DatasetInfo.objMin, ...
        DatasetInfo.objSpan);
    Dist = sqrt(sum((ObjN - TargetN).^2,2));
end

function Rate = feasibleRate(Con)
    if isempty(Con)
        Rate = 1;
    else
        Rate = mean(double(sum(max(0,Con),2) <= 0));
    end
end

function Xn = normalizeWithInfo(X,MinV,SpanV)
    Xn = (X - MinV)./SpanV;
    Xn(~isfinite(Xn)) = 0;
end

function D = pointDistance(A,B)
    D2 = max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'),0);
    D = sqrt(D2);
end

function q = percentileFinite(X,p)
    X = X(isfinite(X));
    if isempty(X)
        q = NaN;
    else
        q = prctile(X,p);
    end
end

function Row = emptySummaryRow()
    Row = struct( ...
        'file',"", ...
        'problem',"", ...
        'run',NaN, ...
        'method',"", ...
        'train_count',0, ...
        'holdout_count',0, ...
        'generated_count',0, ...
        'holdout_obj_dist50',NaN, ...
        'holdout_obj_dist90',NaN, ...
        'feasible_rate',NaN, ...
        'status',"pending", ...
        'error_message',"");
end
