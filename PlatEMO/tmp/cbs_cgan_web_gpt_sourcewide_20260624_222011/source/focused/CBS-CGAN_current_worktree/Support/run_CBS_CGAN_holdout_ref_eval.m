function [Summary,outFile] = run_CBS_CGAN_holdout_ref_eval(inputPath,outFile,Options)
%RUN_CBS_CGAN_HOLDOUT_REF_EVAL Evaluate CGAN generation on held-out refs.
%
% inputPath is a captured_dataset.mat file or a directory containing captured
% datasets.  The diagnostic retrains the CGAN using non-holdout BMem refs and
% evaluates generated decisions for holdout refs through the real
% Problem.Evaluation path.

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
    Rows = repmat(emptySummaryRow(),numel(files),1);
    for i = 1 : numel(files)
        Rows(i) = evaluateOneCapture(files(i),Options);
    end
    Summary = struct2table(Rows);
    if nargin < 2 || isempty(outFile)
        if isfolder(inputPath)
            outFile = fullfile(inputPath,'holdout_ref_eval.csv');
        else
            outFile = fullfile(fileparts(inputPath),'holdout_ref_eval.csv');
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
    Options = ensureField(Options,'queryPerCondition',1);
    Options = ensureField(Options,'ganIter',50);
    Options.holdoutModulo = max(2,round(double(Options.holdoutModulo)));
    Options.holdoutRemainder = mod(round(double(Options.holdoutRemainder)), ...
        Options.holdoutModulo);
    Options.queryPerCondition = max(1,round(double(Options.queryPerCondition)));
    Options.ganIter = max(0,round(double(Options.ganIter)));
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

function Row = evaluateOneCapture(file,Options)
    Row = emptySummaryRow();
    Row.file = string(file);
    [problemName,runId] = parseCaptureName(file);
    Row.problem = problemName;
    Row.run = runId;
    try
        Loaded = load(file,"Data");
        Data = Loaded.Data;
        BMem = Data.BMem;
        if isempty(BMem) || ~isfield(BMem,'y_b') || isempty(BMem.y_b)
            Row.status = "empty_bmem";
            return;
        end
        valid = all(isfinite(BMem.x_b),2);
        if ~isfield(BMem,'ref')
            Row.status = "missing_ref";
            return;
        end
        holdout = valid & mod(BMem.ref(:),Options.holdoutModulo) == ...
            Options.holdoutRemainder;
        train = valid & ~holdout;
        Row.train_count = sum(train);
        Row.holdout_count = sum(holdout);
        if Row.train_count < 2 || Row.holdout_count < 1
            Row.status = "insufficient_refs";
            return;
        end

        Problem = makeProblem(problemName,size(BMem.x_b,2));
        W = referenceMatrixFromData(Data,BMem,Problem.M);
        TrainX = BMem.x_b(train,:);
        TrainC = referenceConditionsFromBMem(BMem,W,train,Problem.M, ...
            Data.DatasetInfo);
        HoldoutC = referenceConditionsFromBMem(BMem,W,holdout,Problem.M, ...
            Data.DatasetInfo);
        HoldoutY = BMem.y_b(holdout,:);
        GANOptions = Data.GANOptions;
        GANOptions.iter = Options.ganIter;
        if isfield(BMem,'x_f') && isfield(BMem,'x_i')
            GANOptions.trainXf = BMem.x_f(train,:);
            GANOptions.trainXi = BMem.x_i(train,:);
        end
        GAN = BoundaryCGAN_CBS('train',[],TrainX,TrainC,Problem,GANOptions);
        [RawDec,Info] = BoundaryCGAN_CBS('samplebycondition',GAN, ...
            HoldoutC,Options.queryPerCondition,GANOptions);
        if isempty(RawDec)
            Row.status = "empty_generation";
            return;
        end
        Pop = Problem.Evaluation(RawDec);
        ObjDist = conditionObjectiveDistances(Pop.objs,Info.query_index, ...
            HoldoutY, ...
            Data.DatasetInfo);
        Row.generated_count = size(RawDec,1);
        Row.holdout_obj_dist50 = percentileFinite(ObjDist,50);
        Row.holdout_obj_dist90 = percentileFinite(ObjDist,90);
        Row.feasible_rate = feasibleRate(Pop.cons);
        Row.status = "ok";
    catch err
        Row.status = "failed";
        Row.error_message = string(getReport(err,'extended', ...
            'hyperlinks','off'));
    end
end

function Problem = makeProblem(problemName,D)
    if strlength(problemName) == 0
        error('CBSCGAN:HoldoutProblemName', ...
            'Cannot infer problem name from captured dataset path.');
    end
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

function Dist = conditionObjectiveDistances(Obj,QueryIndex,TargetObj,DatasetInfo)
    n = min(size(Obj,1),numel(QueryIndex));
    ObjN = normalizeWithInfo(Obj(1:n,:),DatasetInfo.objMin, ...
        DatasetInfo.objSpan);
    TargetN = normalizeWithInfo(TargetObj,DatasetInfo.objMin, ...
        DatasetInfo.objSpan);
    QueryIndex = round(double(QueryIndex(1:n)));
    valid = isfinite(QueryIndex) & QueryIndex >= 1 & ...
        QueryIndex <= size(TargetN,1);
    Dist = NaN(n,1);
    if any(valid)
        Dist(valid) = sqrt(sum((ObjN(valid,:) - ...
            TargetN(QueryIndex(valid),:)).^2,2));
    end
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
        'train_count',0, ...
        'holdout_count',0, ...
        'generated_count',0, ...
        'holdout_obj_dist50',NaN, ...
        'holdout_obj_dist90',NaN, ...
        'feasible_rate',NaN, ...
        'status',"pending", ...
        'error_message',"");
end
