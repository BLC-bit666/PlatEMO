function varargout = RunRegionGAN_RC(action,varargin)
%RUNREGIONGAN_RC Pure helpers for region-conditioned GAN branch dispatch.
%   The evolutionary loop itself lives in CBS_RegionGAN_Base because PlatEMO
%   exposes ParameterSet, NotTerminated, and metric mutation through protected
%   ALGORITHM access.

    action = lower(strtrim(string(action)));
    switch action
        case "metricnames"
            [varargout{1:nargout}] = metricNames(varargin{:});
        case "allocatequery"
            [varargout{1:nargout}] = allocateQueryConditions(varargin{:});
        case "regionquerysamples"
            [varargout{1:nargout}] = regionQuerySamples(varargin{:});
        case "regionquerypoolcount"
            varargout{1} = regionQueryPoolCount(varargin{:});
        case "resolveganoptions"
            [varargout{1:nargout}] = resolveGANOptions(varargin{:});
        case "inittraintriggerstate"
            varargout{1} = initTrainTriggerState();
        case "traintrigger"
            [varargout{1:nargout}] = resolveTrainTrigger(varargin{:});
        case "updatetraintriggerstate"
            varargout{1} = updateTrainTriggerState(varargin{:});
        case "offspringgsurvivaldiagnostics"
            varargout{1} = offspringGSurvivalDiagnostics(varargin{:});
        case "trueboundarydiagnostics"
            varargout{1} = trueBoundaryDiagnostics(varargin{:});
        case "trainandsample"
            [varargout{1:nargout}] = trainAndSampleRegionGAN(varargin{:});
        otherwise
            error('CBSRegionGAN:BadRunnerAction', ...
                'Unsupported RunRegionGAN_RC action: %s.',action);
    end
end

function [SampleC,Counts,SampleRefs,DiagQueryC,DiagQueryRefs] = ...
        regionQuerySamples(QueryMode,QueryC,DatasetInfo,W, ...
        queryPerCondition,nGen)
    QueryMode = normalizeRegionQueryMode(QueryMode);
    switch QueryMode
        case "random_all_w"
            [PoolC,PoolRefs] = allRegionQueryPool(QueryC,DatasetInfo,W);
            totalBudget = max(0,round(double(nGen)));
            if isempty(PoolC) || totalBudget <= 0
                SampleC = zeros(0,size(PoolC,2));
                Counts = zeros(size(PoolC,1),1);
                SampleRefs = zeros(0,1);
                DiagQueryC = PoolC;
                DiagQueryRefs = PoolRefs;
                return;
            end
            idx = uniformRefConditionRows(PoolRefs,totalBudget);
            SampleC = PoolC(idx,:);
            Counts = accumarray(idx,1,[size(PoolC,1) 1],@sum,0);
            SampleRefs = queryRefsAt(PoolRefs,idx);
            DiagQueryC = PoolC;
            DiagQueryRefs = PoolRefs;
        case "frontier_anneal"
            [SampleC,Counts,SampleRefs,DiagQueryC,DiagQueryRefs] = ...
                frontierAnnealQuerySamples(QueryC,DatasetInfo,W,nGen);
        otherwise
            [SampleC,Counts] = allocateQueryConditions( ...
                QueryC,queryPerCondition,nGen);
            SampleRefs = expandRegionQueryRefs(DatasetInfo,Counts);
            DiagQueryC = QueryC;
            if isstruct(DatasetInfo) && isfield(DatasetInfo,'queryRegions')
                DiagQueryRefs = DatasetInfo.queryRegions(:);
            else
                DiagQueryRefs = zeros(size(QueryC,1),1);
            end
    end
end

function idx = uniformRefConditionRows(PoolRefs,totalBudget)
    PoolRefs = round(double(PoolRefs(:)));
    valid = isfinite(PoolRefs) & PoolRefs > 0;
    if ~any(valid)
        idx = randi(numel(PoolRefs),totalBudget,1);
        return;
    end
    uRefs = unique(PoolRefs(valid),'stable');
    idx = zeros(totalBudget,1);
    refDraws = randi(numel(uRefs),totalBudget,1);
    for i = 1 : totalBudget
        rows = find(PoolRefs == uRefs(refDraws(i)));
        idx(i) = rows(randi(numel(rows)));
    end
end

function count = regionQueryPoolCount(QueryMode,QueryC,DatasetInfo,W)
    QueryMode = normalizeRegionQueryMode(QueryMode);
    if QueryMode == "random_all_w"
        [PoolC,~] = allRegionQueryPool(QueryC,DatasetInfo,W);
        count = size(PoolC,1);
    elseif QueryMode == "frontier_anneal"
        [PoolC,~,~] = frontierAnnealQueryPool(QueryC,DatasetInfo,W);
        count = size(PoolC,1);
    else
        count = size(QueryC,1);
    end
end

function [SampleC,Counts,SampleRefs,DiagQueryC,DiagQueryRefs] = ...
        frontierAnnealQuerySamples(QueryC,DatasetInfo,W,nGen)
    [PoolC,PoolRefs,PoolGroup] = frontierAnnealQueryPool( ...
        QueryC,DatasetInfo,W);
    DiagQueryC = PoolC;
    DiagQueryRefs = PoolRefs;
    totalBudget = max(0,round(double(nGen)));
    if isempty(PoolC) || totalBudget <= 0
        SampleC = zeros(0,size(PoolC,2));
        Counts = zeros(size(PoolC,1),1);
        SampleRefs = zeros(0,1);
        return;
    end

    popRows = find(PoolGroup == 1);
    frontierRows = find(PoolGroup == 2);
    [popBudget,frontierBudget] = frontierAnnealBudgets( ...
        totalBudget,~isempty(popRows),~isempty(frontierRows));
    idx = zeros(popBudget + frontierBudget,1);
    row = 0;
    if popBudget > 0
        localIdx = uniformRefConditionRows(PoolRefs(popRows),popBudget);
        idx(row+1:row+popBudget) = popRows(localIdx);
        row = row + popBudget;
    end
    if frontierBudget > 0
        localIdx = uniformRefConditionRows( ...
            PoolRefs(frontierRows),frontierBudget);
        idx(row+1:row+frontierBudget) = frontierRows(localIdx);
        row = row + frontierBudget;
    end
    idx = idx(1:row);
    if ~isempty(idx)
        idx = idx(randperm(numel(idx)));
    end
    SampleC = PoolC(idx,:);
    Counts = accumarray(idx,1,[size(PoolC,1) 1],@sum,0);
    SampleRefs = queryRefsAt(PoolRefs,idx);
end

function [PoolC,PoolRefs,PoolGroup] = frontierAnnealQueryPool( ...
        QueryC,DatasetInfo,W)
    [PopC,PopRefs] = populatedRegionQueryPool(QueryC,DatasetInfo,W);
    [AllC,AllRefs] = allRegionQueryPool(QueryC,DatasetInfo,W);
    frontierRefs = oneHopFrontierRefs(W,PopRefs);
    [FrontierC,FrontierRefs] = queryRowsForRefs( ...
        AllC,AllRefs,frontierRefs,size(PopC,2));
    [PopC,FrontierC] = alignQueryPoolDims(PopC,FrontierC);
    PoolC = [PopC;FrontierC];
    PoolRefs = [PopRefs(:);FrontierRefs(:)];
    PoolGroup = [ones(size(PopC,1),1);2*ones(size(FrontierC,1),1)];
end

function [PopC,PopRefs] = populatedRegionQueryPool( ...
        QueryC,DatasetInfo,W)
    PopC = double(QueryC);
    if isempty(PopC) && isstruct(DatasetInfo) && ...
            isfield(DatasetInfo,'queryConditions') && ...
            ~isempty(DatasetInfo.queryConditions)
        PopC = double(DatasetInfo.queryConditions);
    end
    if isstruct(DatasetInfo) && isfield(DatasetInfo,'queryRegions') && ...
            numel(DatasetInfo.queryRegions) == size(PopC,1)
        PopRefs = round(double(DatasetInfo.queryRegions(:)));
    else
        PopRefs = inferConditionRefs(PopC,W);
    end
end

function [A,B] = alignQueryPoolDims(A,B)
    aDim = size(A,2);
    bDim = size(B,2);
    if aDim == bDim
        return;
    end
    if isempty(A) && aDim == 0
        A = zeros(0,bDim);
    elseif isempty(B) && bDim == 0
        B = zeros(0,aDim);
    end
end

function [RowsC,RowsRefs] = queryRowsForRefs(PoolC,PoolRefs,Refs,condDim)
    if nargin < 4 || isempty(condDim)
        condDim = size(PoolC,2);
    end
    Refs = round(double(Refs(:)));
    if isempty(PoolC) || isempty(Refs)
        RowsC = zeros(0,condDim);
        RowsRefs = zeros(0,1);
        return;
    end
    PoolRefs = round(double(PoolRefs(:)));
    keep = ismember(PoolRefs,Refs);
    RowsC = PoolC(keep,:);
    RowsRefs = PoolRefs(keep);
end

function Refs = oneHopFrontierRefs(W,PopRefs)
    if isempty(W)
        Refs = zeros(0,1);
        return;
    end
    nRef = size(W,1);
    PopRefs = unique(round(double(PopRefs(:))),'stable');
    PopRefs = PopRefs(isfinite(PopRefs) & PopRefs >= 1 & PopRefs <= nRef);
    if isempty(PopRefs)
        Refs = zeros(0,1);
        return;
    end
    CandidateRefs = zeros(0,1);
    for i = 1 : numel(PopRefs)
        CandidateRefs = [CandidateRefs; ...
            neighborRefsByW(W,PopRefs(i),1)]; %#ok<AGROW>
    end
    Refs = unique(CandidateRefs,'stable');
    Refs = Refs(~ismember(Refs,PopRefs));
end

function neigh = neighborRefsByW(W,r,radius)
    if radius <= 0
        neigh = r;
        return;
    end
    d = sqrt(sum((double(W) - double(W(r,:))).^2,2));
    [~,ord] = sort(d,'ascend');
    count = min(numel(ord),1 + 2*radius);
    neigh = ord(1:count);
end

function Refs = inferConditionRefs(C,W)
    C = double(C);
    if isempty(C) || isempty(W)
        Refs = zeros(size(C,1),1);
        return;
    end
    baseDim = min(size(C,2),size(W,2));
    if baseDim == 0
        Refs = zeros(size(C,1),1);
        return;
    end
    Dist = pairDistance(C(:,1:baseDim),double(W(:,1:baseDim)));
    [~,Refs] = min(Dist,[],2);
    Refs = Refs(:);
end

function D = pairDistance(A,B)
    if isempty(A) || isempty(B)
        D = zeros(size(A,1),size(B,1));
        return;
    end
    D = sqrt(max(0,sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B')));
end

function [popBudget,frontierBudget] = frontierAnnealBudgets( ...
        totalBudget,hasPopulated,hasFrontier)
    if totalBudget <= 0 || (~hasPopulated && ~hasFrontier)
        popBudget = 0;
        frontierBudget = 0;
    elseif hasPopulated && hasFrontier
        popBudget = round(0.60*totalBudget);
        popBudget = max(0,min(totalBudget,popBudget));
        frontierBudget = totalBudget - popBudget;
    elseif hasPopulated
        popBudget = totalBudget;
        frontierBudget = 0;
    else
        popBudget = 0;
        frontierBudget = totalBudget;
    end
end

function [PoolC,PoolRefs] = allRegionQueryPool(QueryC,DatasetInfo,W)
    if isstruct(DatasetInfo) && isfield(DatasetInfo,'allQueryConditions') && ...
            ~isempty(DatasetInfo.allQueryConditions)
        PoolC = double(DatasetInfo.allQueryConditions);
        if isfield(DatasetInfo,'allQueryRegions') && ...
                numel(DatasetInfo.allQueryRegions) == size(PoolC,1)
            PoolRefs = round(double(DatasetInfo.allQueryRegions(:)));
        else
            PoolRefs = zeros(size(PoolC,1),1);
        end
        return;
    end
    if ~isempty(W)
        PoolC = double(W);
        PoolRefs = (1:size(W,1))';
    else
        PoolC = zeros(0,size(QueryC,2));
        PoolRefs = zeros(0,1);
    end
end

function Refs = queryRefsAt(PoolRefs,idx)
    PoolRefs = round(double(PoolRefs(:)));
    if isempty(PoolRefs)
        Refs = zeros(numel(idx),1);
        return;
    end
    idx = max(1,min(numel(PoolRefs),round(double(idx(:)))));
    Refs = PoolRefs(idx);
end

function Refs = expandRegionQueryRefs(DatasetInfo,QueryAllocation)
    if ~isstruct(DatasetInfo) || ~isfield(DatasetInfo,'queryRegions') || ...
            isempty(DatasetInfo.queryRegions) || isempty(QueryAllocation)
        Refs = zeros(0,1);
        return;
    end
    queryRefs = round(double(DatasetInfo.queryRegions(:)));
    QueryAllocation = round(double(QueryAllocation(:)));
    n = min(numel(queryRefs),numel(QueryAllocation));
    Refs = zeros(sum(max(0,QueryAllocation(1:n))),1);
    row = 0;
    for i = 1 : n
        c = max(0,QueryAllocation(i));
        if c <= 0
            continue;
        end
        Refs(row+1:row+c) = queryRefs(i);
        row = row + c;
    end
    Refs = Refs(1:row);
end

function mode = normalizeRegionQueryMode(mode)
    mode = lower(strtrim(string(mode)));
    switch mode
        case {"boundary","boundary_populated","populated"}
            mode = "boundary_populated";
        case {"random","random_all_w","all_w_random"}
            mode = "random_all_w";
        case {"frontier_anneal","query_frontier_anneal", ...
                "support_frontier","populated_frontier"}
            mode = "frontier_anneal";
        otherwise
            error('CBSRegionGAN:BadQueryMode', ...
                'Unsupported region QueryC mode: %s.',mode);
    end
end

function [SampleC,Counts] = allocateQueryConditions(QueryC,queryPerCondition,nGen)
    K = size(QueryC,1);
    Counts = zeros(K,1);
    if isempty(QueryC) || K == 0
        SampleC = zeros(0,size(QueryC,2));
        return;
    end
    perRegionCap = max(1,round(double(queryPerCondition)));
    totalBudget = max(0,round(double(nGen)));
    if totalBudget <= 0
        SampleC = zeros(0,size(QueryC,2));
        return;
    end

    order = randperm(K);
    while sum(Counts) < totalBudget && any(Counts < perRegionCap)
        progressed = false;
        for p = 1 : K
            r = order(p);
            if Counts(r) >= perRegionCap
                continue;
            end
            Counts(r) = Counts(r) + 1;
            progressed = true;
            if sum(Counts) >= totalBudget
                break;
            end
        end
        if ~progressed
            break;
        end
    end

    SampleC = zeros(sum(Counts),size(QueryC,2));
    row = 0;
    for i = 1 : K
        c = Counts(i);
        if c <= 0
            continue;
        end
        SampleC(row+1:row+c,:) = repmat(QueryC(i,:),c,1);
        row = row + c;
    end
end

function [GAN,RawDec] = trainAndSampleRegionGAN(ganKind,GAN,TrainX,TrainC, ...
        QueryC,queryPerCondition,Problem,GANOptions)
    minTrainCount = 1;
    if isstruct(GANOptions) && isfield(GANOptions,'minTrainCount') && ...
            ~isempty(GANOptions.minTrainCount)
        minTrainCount = max(1,round(double(GANOptions.minTrainCount)));
    end
    if size(TrainX,1) < minTrainCount
        RawDec = zeros(0,Problem.D);
        return;
    end
    switch lower(strtrim(string(ganKind)))
        case "cgan"
            GAN = BoundaryCGAN_CBS('train',GAN,TrainX,TrainC, ...
                Problem,GANOptions);
            [RawDec,SampleInfo] = BoundaryCGAN_CBS('samplebycondition', ...
                GAN,QueryC,queryPerCondition,GANOptions);
            GAN.last_sample_info = SampleInfo;
        case "wgan-gp"
            GAN = BoundaryWGAN_RC('train',GAN,TrainX,TrainC, ...
                Problem,GANOptions);
            [RawDec,SampleInfo] = BoundaryWGAN_RC('samplebycondition', ...
                GAN,QueryC,queryPerCondition,GANOptions);
            GAN.last_sample_info = SampleInfo;
        otherwise
            error('CBSRegionGAN:BadGANKind', ...
                'Unsupported region GAN kind: %s.',ganKind);
    end
end

function Options = resolveGANOptions(Options,currentFE,maxFE,TriggerInfo)
    if nargin < 2 || isempty(currentFE)
        currentFE = 0;
    end
    if nargin < 3 || isempty(maxFE)
        maxFE = NaN;
    end
    if nargin < 4 || isempty(TriggerInfo)
        TriggerInfo = emptyTrainTriggerInfo();
    end
    if ~isstruct(Options)
        error('CBSRegionGAN:BadGANOptions', ...
            'GAN options must be a struct.');
    end
    if ~isfield(Options,'iter') || isempty(Options.iter)
        Options.iter = 0;
    end
    schedule = "";
    if isfield(Options,'ganIterSchedule') && ...
            ~isempty(Options.ganIterSchedule)
        schedule = lower(strtrim(string(Options.ganIterSchedule)));
    end
    Options.gan_iter_schedule = schedule;
    switch schedule
        case {"","fixed","none","off"}
            Options.iter = max(0,round(double(Options.iter)));
            Options.gan_iter_used = Options.iter;
        case {"linear","linear_fe","fe_linear"}
            startIter = optionScalar(Options,'ganIterStart',Options.iter);
            endIter = optionScalar(Options,'ganIterEnd',Options.iter);
            progress = finiteRatio(currentFE,maxFE);
            iter = round(startIter + (endIter - startIter)*progress);
            iter = min(max(iter,min(startIter,endIter)), ...
                max(startIter,endIter));
            Options.iter = max(0,round(double(iter)));
            Options.gan_iter_used = Options.iter;
        case {"two_level_data_change","data_change_two_level", ...
                "triggered_two_level"}
            boostIter = optionScalar(Options,'ganIterStart',Options.iter);
            baseIter = optionScalar(Options,'ganIterEnd',Options.iter);
            useBoost = triggerInfoFlag(TriggerInfo,'trigger_first_train') || ...
                triggerInfoFlag(TriggerInfo,'trigger_data_changed') || ...
                triggerInfoFlag(TriggerInfo,'trigger_ref_changed');
            if useBoost
                Options.iter = max(0,round(double(boostIter)));
            else
                Options.iter = max(0,round(double(baseIter)));
            end
            Options.gan_iter_used = Options.iter;
        otherwise
            error('CBSRegionGAN:BadGANIterSchedule', ...
                'Unsupported ganIterSchedule: %s.',schedule);
    end
end

function State = initTrainTriggerState()
    State = struct( ...
        'hasTrained',false, ...
        'lastTrainCount',0, ...
        'lastQueryUniqueRefCount',0);
end

function [TrainNow,Info] = resolveTrainTrigger(Options,State,gen, ...
        trainCount,queryUniqueRefCount,periodic)
    if nargin < 2 || isempty(State)
        State = initTrainTriggerState();
    end
    if nargin < 3 || isempty(gen)
        gen = 0;
    end
    if nargin < 4 || isempty(trainCount)
        trainCount = 0;
    end
    if nargin < 5 || isempty(queryUniqueRefCount)
        queryUniqueRefCount = 0;
    end
    if nargin < 6 || isempty(periodic)
        periodic = false;
    end
    periodic = logical(periodic);
    trainCount = max(0,round(double(trainCount)));
    queryUniqueRefCount = max(0,round(double(queryUniqueRefCount)));
    mode = optionString(Options,'trainTriggerMode',"off");
    deltaThreshold = optionScalar(Options,'trainTriggerDelta',0.20);
    deltaThreshold = max(0,double(deltaThreshold));

    Info = emptyTrainTriggerInfo();
    Info.trigger_mode = mode;
    Info.trigger_generation = double(gen);
    Info.trigger_periodic = double(periodic);
    Info.trigger_current_train_count = double(trainCount);
    Info.trigger_last_train_count = double(State.lastTrainCount);
    Info.trigger_current_query_unique_ref_count = ...
        double(queryUniqueRefCount);
    Info.trigger_last_query_unique_ref_count = ...
        double(State.lastQueryUniqueRefCount);

    if State.hasTrained
        denom = max(1,double(State.lastTrainCount));
        Info.trigger_train_count_delta = ...
            abs(double(trainCount) - double(State.lastTrainCount))/denom;
        Info.trigger_data_changed = double( ...
            Info.trigger_train_count_delta >= deltaThreshold);
        Info.trigger_ref_changed = double( ...
            queryUniqueRefCount ~= State.lastQueryUniqueRefCount);
    else
        Info.trigger_train_count_delta = NaN;
        Info.trigger_first_train = double(trainCount > 0);
    end

    switch mode
        case {"","off","none","periodic","fixed"}
            TrainNow = periodic;
        case {"deltatraincount20","delta_train_count20", ...
                "delta_train_count","data_change"}
            TrainNow = periodic || logical(Info.trigger_first_train) || ...
                logical(Info.trigger_data_changed) || ...
                logical(Info.trigger_ref_changed);
        otherwise
            error('CBSRegionGAN:BadTrainTriggerMode', ...
                'Unsupported trainTriggerMode: %s.',mode);
    end
    Info.trigger_train_now = double(TrainNow);
    Info.trigger_reason = trainTriggerReason(TrainNow,Info);
end

function State = updateTrainTriggerState(State,Info)
    if nargin < 1 || isempty(State)
        State = initTrainTriggerState();
    end
    if nargin < 2 || isempty(Info)
        return;
    end
    State.hasTrained = true;
    State.lastTrainCount = max(0,round(double( ...
        Info.trigger_current_train_count)));
    State.lastQueryUniqueRefCount = max(0,round(double( ...
        Info.trigger_current_query_unique_ref_count)));
end

function Info = emptyTrainTriggerInfo()
    Info = struct( ...
        'trigger_mode',"off", ...
        'trigger_reason',"", ...
        'trigger_generation',0, ...
        'trigger_train_now',0, ...
        'trigger_periodic',0, ...
        'trigger_first_train',0, ...
        'trigger_data_changed',0, ...
        'trigger_ref_changed',0, ...
        'trigger_train_count_delta',NaN, ...
        'trigger_last_train_count',0, ...
        'trigger_current_train_count',0, ...
        'trigger_last_query_unique_ref_count',0, ...
        'trigger_current_query_unique_ref_count',0);
end

function reason = trainTriggerReason(TrainNow,Info)
    if ~TrainNow
        reason = "waiting_period";
    elseif logical(Info.trigger_periodic)
        reason = "periodic";
    elseif logical(Info.trigger_first_train)
        reason = "first_train";
    elseif logical(Info.trigger_data_changed)
        reason = "train_count_delta";
    elseif logical(Info.trigger_ref_changed)
        reason = "ref_change";
    else
        reason = "triggered";
    end
end

function value = optionString(S,name,defaultValue)
    value = defaultValue;
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = S.(name);
    end
    value = lower(strtrim(string(value)));
end

function flag = triggerInfoFlag(Info,name)
    flag = isstruct(Info) && isfield(Info,name) && ...
        ~isempty(Info.(name)) && any(double(Info.(name)) ~= 0);
end

function Metrics = offspringGSurvivalDiagnostics(GDec,GCon,SelectedDec)
    if nargin < 1 || isempty(GDec)
        GDec = zeros(0,0);
    end
    if nargin < 2
        GCon = [];
    end
    if nargin < 3 || isempty(SelectedDec)
        SelectedDec = zeros(0,size(GDec,2));
    end
    GDec = double(GDec);
    SelectedDec = double(SelectedDec);
    Metrics = struct( ...
        'offspringG_count',0, ...
        'offspringG_survive_count',0, ...
        'offspringG_survival_rate',NaN, ...
        'offspringG_feasible_count',0, ...
        'offspringG_feasible_survive_count',0, ...
        'offspringG_feasible_survival_rate',NaN);
    if isempty(GDec)
        return;
    end
    survived = decisionRowsPresentMultiset(GDec,SelectedDec);
    Metrics.offspringG_count = size(GDec,1);
    Metrics.offspringG_survive_count = sum(survived);
    Metrics.offspringG_survival_rate = mean(double(survived));
    feasible = generatedFeasibleMask(GCon,size(GDec,1));
    Metrics.offspringG_feasible_count = sum(feasible);
    Metrics.offspringG_feasible_survive_count = sum(survived & feasible);
    if any(feasible)
        Metrics.offspringG_feasible_survival_rate = ...
            mean(double(survived(feasible)));
    end
end

function feasible = generatedFeasibleMask(GCon,n)
    if isempty(GCon)
        feasible = true(n,1);
        return;
    end
    GCon = double(GCon);
    if size(GCon,1) ~= n
        error('CBSRegionGAN:BadSurvivalConstraintRows', ...
            'Generated constraint rows must match generated decision rows.');
    end
    feasible = sum(max(0,GCon),2) <= 0;
end

function present = decisionRowsPresentMultiset(Query,Rows)
    present = false(size(Query,1),1);
    if isempty(Query) || isempty(Rows)
        return;
    end
    if size(Query,2) ~= size(Rows,2)
        error('CBSRegionGAN:BadSurvivalDecisionColumns', ...
            'Generated and selected decision columns must match.');
    end
    tol = 1e-12;
    used = false(size(Rows,1),1);
    for i = 1 : size(Query,1)
        diffToRows = max(abs(Rows - Query(i,:)),[],2);
        match = find(~used & diffToRows <= tol,1,'first');
        if ~isempty(match)
            present(i) = true;
            used(match) = true;
        end
    end
end

function Metrics = trueBoundaryDiagnostics(GeneratedObj,GeneratedCon,PF,Options)
    if nargin < 3
        PF = [];
    end
    if nargin < 4 || isempty(Options)
        Options = struct();
    end
    Metrics = emptyTrueBoundaryMetrics();
    if nargin < 1 || isempty(GeneratedObj)
        return;
    end
    GeneratedObj = double(GeneratedObj);
    if size(GeneratedObj,2) < 2
        return;
    end
    [Boundary,Scale,ArcInfo] = trueBoundaryPoints(PF);
    if isempty(Boundary)
        return;
    end

    Obj = normalizeTrueBoundaryPoints(GeneratedObj(:,1:2),Scale);
    BoundaryN = normalizeTrueBoundaryPoints(Boundary,Scale);
    D = pairwiseDistanceLocal(Obj,BoundaryN);
    if isempty(D)
        return;
    end
    [AbsDist,Nearest] = min(D,[],2);
    Feasible = generatedFeasibleMask(GeneratedCon,size(GeneratedObj,1));
    SignedDist = AbsDist;
    SignedDist(~Feasible) = -SignedDist(~Feasible);

    Metrics.bdist50_true = percentileFiniteLocal(abs(SignedDist),50);
    q90 = percentileFiniteLocal(SignedDist,90);
    q10 = percentileFiniteLocal(SignedDist,10);
    if isfinite(q90) && isfinite(q10)
        Metrics.bwidth90_10_true = q90 - q10;
    end

    epsDefault = 0.02*sqrt(2);
    epsTrue = max(0,optionScalar(Options,'coverEpsilon',epsDefault));
    binCount = max(1,round(optionScalar(Options,'coverBinCount',20)));
    nearFeasible = Feasible & isfinite(SignedDist) & ...
        SignedDist >= 0 & SignedDist <= epsTrue;
    Metrics.bcover_eps_true = boundaryCoverageRate( ...
        BoundaryN,ArcInfo,Nearest(nearFeasible),binCount);
end

function Metrics = emptyTrueBoundaryMetrics()
    Metrics = struct( ...
        'bdist50_true',NaN, ...
        'bwidth90_10_true',NaN, ...
        'bcover_eps_true',NaN);
end

function [Boundary,Scale,ArcInfo] = trueBoundaryPoints(PF)
    Boundary = zeros(0,2);
    Scale = struct('min',zeros(1,2),'span',ones(1,2));
    ArcInfo = boundaryArcInfo([],[]);
    if isnumeric(PF) && size(PF,2) >= 2
        Boundary = double(PF(:,1:2));
        Boundary = Boundary(all(isfinite(Boundary),2),:);
        Scale = trueBoundaryScale(Boundary);
        ArcInfo = boundaryArcInfo((1:size(Boundary,1))',1);
        return;
    end
    if ~iscell(PF) || numel(PF) < 3
        return;
    end
    X = double(PF{1});
    Y = double(PF{2});
    Z = double(PF{3});
    if isempty(X) || isempty(Y) || isempty(Z) || ...
            ~isequal(size(X),size(Y),size(Z))
        return;
    end
    Feasible = isfinite(Z);
    if ~any(Feasible(:)) || all(Feasible(:))
        Scale = trueBoundaryScale([X(:),Y(:)]);
        return;
    end

    Edge = false(size(Feasible));
    Edge(1:end-1,:) = Edge(1:end-1,:) | ...
        (Feasible(1:end-1,:) & ~Feasible(2:end,:));
    Edge(2:end,:) = Edge(2:end,:) | ...
        (Feasible(2:end,:) & ~Feasible(1:end-1,:));
    Edge(:,1:end-1) = Edge(:,1:end-1) | ...
        (Feasible(:,1:end-1) & ~Feasible(:,2:end));
    Edge(:,2:end) = Edge(:,2:end) | ...
        (Feasible(:,2:end) & ~Feasible(:,1:end-1));

    Edge = Edge & isfinite(X) & isfinite(Y);
    edgeLinear = find(Edge);
    Boundary = [X(edgeLinear),Y(edgeLinear)];
    Scale = trueBoundaryScale([X(:),Y(:)]);
    ArcInfo = boundaryArcInfoFromEdgeMask(Edge,edgeLinear,Boundary);
end

function Scale = trueBoundaryScale(Points)
    Points = double(Points);
    Points = Points(all(isfinite(Points),2),:);
    if isempty(Points)
        Scale = struct('min',zeros(1,2),'span',ones(1,2));
        return;
    end
    minV = min(Points(:,1:2),[],1);
    spanV = max(Points(:,1:2),[],1) - minV;
    spanV(spanV <= eps) = 1;
    Scale = struct('min',minV,'span',spanV);
end

function Xn = normalizeTrueBoundaryPoints(X,Scale)
    Xn = (double(X) - Scale.min)./Scale.span;
    Xn(~isfinite(Xn)) = 0;
end

function Info = boundaryArcInfo(Order,ComponentStart)
    if nargin < 1 || isempty(Order)
        Order = zeros(0,1);
    end
    if nargin < 2 || isempty(ComponentStart)
        if isempty(Order)
            ComponentStart = zeros(0,1);
        else
            ComponentStart = 1;
        end
    end
    Info = struct( ...
        'order',round(double(Order(:))), ...
        'component_start',round(double(ComponentStart(:))));
end

function ArcInfo = boundaryArcInfoFromEdgeMask(Edge,edgeLinear,Boundary)
    n = numel(edgeLinear);
    if n == 0
        ArcInfo = boundaryArcInfo([],[]);
        return;
    end

    IndexMap = zeros(size(Edge));
    IndexMap(edgeLinear) = 1:n;
    Visited = false(n,1);
    Order = zeros(n,1);
    ComponentStart = zeros(n,1);
    count = 0;
    componentCount = 0;
    for seed = 1 : n
        if Visited(seed)
            continue;
        end
        Component = collectBoundaryComponent(seed,IndexMap,edgeLinear);
        Visited(Component) = true;
        ComponentOrder = orderBoundaryComponent(Component,IndexMap, ...
            edgeLinear,Boundary);
        componentCount = componentCount + 1;
        ComponentStart(componentCount) = count + 1;
        nextCount = count + numel(ComponentOrder);
        Order(count+1:nextCount) = ComponentOrder(:);
        count = nextCount;
    end
    ArcInfo = boundaryArcInfo(Order(1:count),ComponentStart(1:componentCount));
end

function Component = collectBoundaryComponent(seed,IndexMap,edgeLinear)
    n = numel(edgeLinear);
    InComponent = false(n,1);
    Queue = zeros(n,1);
    head = 1;
    tail = 1;
    Queue(1) = seed;
    InComponent(seed) = true;
    mapSize = size(IndexMap);
    while head <= tail
        idx = Queue(head);
        head = head + 1;
        [r,c] = ind2sub(mapSize,edgeLinear(idx));
        for dr = -1 : 1
            rr = r + dr;
            if rr < 1 || rr > mapSize(1)
                continue;
            end
            for dc = -1 : 1
                if dr == 0 && dc == 0
                    continue;
                end
                cc = c + dc;
                if cc < 1 || cc > mapSize(2)
                    continue;
                end
                neighbor = IndexMap(rr,cc);
                if neighbor > 0 && ~InComponent(neighbor)
                    tail = tail + 1;
                    Queue(tail) = neighbor;
                    InComponent(neighbor) = true;
                end
            end
        end
    end
    Component = Queue(1:tail);
end

function OrderedIdx = orderBoundaryComponent(Component,IndexMap,edgeLinear,Boundary)
    Component = Component(:);
    n = numel(Component);
    if n <= 1
        OrderedIdx = Component;
        return;
    end

    InComponent = false(numel(edgeLinear),1);
    InComponent(Component) = true;
    neighborCount = zeros(n,1);
    for i = 1 : n
        neighborCount(i) = numel(boundaryNeighborIndices( ...
            Component(i),IndexMap,edgeLinear,InComponent));
    end
    endpoints = Component(neighborCount <= 1);
    if isempty(endpoints)
        current = chooseBoundaryStart(Component,Boundary);
    else
        current = chooseBoundaryStart(endpoints,Boundary);
    end

    OrderedIdx = zeros(n,1);
    Used = false(numel(edgeLinear),1);
    for pos = 1 : n
        OrderedIdx(pos) = current;
        Used(current) = true;
        Available = InComponent & ~Used;
        neighbors = boundaryNeighborIndices(current,IndexMap, ...
            edgeLinear,Available);
        if ~isempty(neighbors)
            current = chooseNearestBoundaryPoint(current,neighbors,Boundary);
            continue;
        end
        remaining = Component(~Used(Component));
        if isempty(remaining)
            continue;
        end
        current = chooseNearestBoundaryPoint(current,remaining,Boundary);
    end
end

function Neighbors = boundaryNeighborIndices(current,IndexMap,edgeLinear,Available)
    mapSize = size(IndexMap);
    [r,c] = ind2sub(mapSize,edgeLinear(current));
    Neighbors = zeros(8,1);
    count = 0;
    for dr = -1 : 1
        rr = r + dr;
        if rr < 1 || rr > mapSize(1)
            continue;
        end
        for dc = -1 : 1
            if dr == 0 && dc == 0
                continue;
            end
            cc = c + dc;
            if cc < 1 || cc > mapSize(2)
                continue;
            end
            neighbor = IndexMap(rr,cc);
            if neighbor > 0 && Available(neighbor)
                count = count + 1;
                Neighbors(count) = neighbor;
            end
        end
    end
    Neighbors = Neighbors(1:count);
end

function idx = chooseBoundaryStart(Candidates,Boundary)
    Candidates = Candidates(:);
    [~,ord] = sortrows(Boundary(Candidates,1:2),[1 2]);
    idx = Candidates(ord(1));
end

function idx = chooseNearestBoundaryPoint(current,Candidates,Boundary)
    Candidates = Candidates(:);
    D2 = sum((Boundary(Candidates,1:2) - Boundary(current,1:2)).^2,2);
    [~,ord] = sortrows([D2,double(Candidates)],[1 2]);
    idx = Candidates(ord(1));
end

function rate = boundaryCoverageRate(Boundary,ArcInfo,NearestIdx,binCount)
    if isempty(Boundary) || isempty(NearestIdx)
        rate = 0;
        return;
    end
    binCount = min(max(1,round(double(binCount))),size(Boundary,1));
    ord = sanitizeBoundaryOrder(ArcInfo,size(Boundary,1));
    componentStart = sanitizeComponentStart(ArcInfo,numel(ord));
    Ordered = Boundary(ord,:);
    step = sqrt(sum(diff(Ordered,1,1).^2,2));
    breakStep = componentStart(2:end) - 1;
    breakStep = breakStep(breakStep >= 1 & breakStep <= numel(step));
    step(breakStep) = 0;
    arc = [0;cumsum(step)];
    if arc(end) <= eps
        binsOrdered = ones(size(Boundary,1),1);
    else
        binsOrdered = min(binCount,max(1, ...
            ceil((arc./arc(end))*binCount)));
        binsOrdered(1) = 1;
    end
    bins = zeros(size(Boundary,1),1);
    bins(ord) = binsOrdered;
    NearestIdx = round(double(NearestIdx(:)));
    NearestIdx = NearestIdx(isfinite(NearestIdx) & NearestIdx >= 1 & ...
        NearestIdx <= size(Boundary,1));
    if isempty(NearestIdx)
        rate = 0;
    else
        rate = numel(unique(bins(NearestIdx)))/binCount;
    end
end

function Order = sanitizeBoundaryOrder(ArcInfo,n)
    Order = [];
    if isstruct(ArcInfo) && isfield(ArcInfo,'order')
        Order = round(double(ArcInfo.order(:)));
    end
    Order = Order(isfinite(Order) & Order >= 1 & Order <= n);
    Order = unique(Order,'stable');
    Missing = setdiff((1:n)',Order,'stable');
    Order = [Order;Missing];
    if isempty(Order)
        Order = (1:n)';
    end
end

function ComponentStart = sanitizeComponentStart(ArcInfo,n)
    ComponentStart = [];
    if isstruct(ArcInfo) && isfield(ArcInfo,'component_start')
        ComponentStart = round(double(ArcInfo.component_start(:)));
    end
    ComponentStart = ComponentStart(isfinite(ComponentStart) & ...
        ComponentStart >= 1 & ComponentStart <= n);
    ComponentStart = sort(unique([1;ComponentStart(:)]));
end

function D = pairwiseDistanceLocal(A,B)
    if isempty(A) || isempty(B)
        D = zeros(size(A,1),size(B,1));
        return;
    end
    D2 = max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'),0);
    D = sqrt(D2);
end

function q = percentileFiniteLocal(X,p)
    X = X(isfinite(X));
    if isempty(X)
        q = NaN;
    else
        q = prctile(X,p);
    end
end

function value = optionScalar(S,name,defaultValue)
    value = defaultValue;
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = S.(name);
    end
    value = double(value);
    if isempty(value) || ~isfinite(value(1))
        value = double(defaultValue);
    end
    value = value(1);
end

function progress = finiteRatio(currentFE,maxFE)
    currentFE = double(currentFE);
    maxFE = double(maxFE);
    if isempty(currentFE) || isempty(maxFE) || ~isfinite(currentFE(1)) || ...
            ~isfinite(maxFE(1)) || maxFE(1) <= 0
        progress = 0;
        return;
    end
    progress = min(max(currentFE(1)/maxFE(1),0),1);
end

function [lastMetric,historyMetric,cloudMetric] = metricNames(ganKind)
    switch lower(strtrim(string(ganKind)))
        case "cgan"
            lastMetric = 'region_cgan_last';
            historyMetric = 'region_cgan_history';
            cloudMetric = 'region_cgan_cloud';
        case "wgan-gp"
            lastMetric = 'region_wgan_gp_last';
            historyMetric = 'region_wgan_gp_history';
            cloudMetric = 'region_wgan_gp_cloud';
        otherwise
            error('CBSRegionGAN:BadGANKind', ...
                'Unsupported region GAN kind: %s.',ganKind);
    end
end
