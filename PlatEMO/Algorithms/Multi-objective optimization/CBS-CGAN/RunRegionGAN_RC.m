function varargout = RunRegionGAN_RC(action,varargin)
%RUNREGIONGAN_RC Pure helpers for region-conditioned GAN branch dispatch.
%   The evolutionary loop itself lives in CBS_RegionGAN_Base because PlatEMO
%   exposes ParameterSet, NotTerminated, and metric mutation through protected
%   ALGORITHM access.

    action = lower(strtrim(string(action)));
    switch action
        case "metricnames"
            [varargout{1:nargout}] = metricNames(varargin{:});
        case "regionquerysamples"
            [varargout{1:nargout}] = regionQuerySamples(varargin{:});
        case "regionquerypoolcount"
            varargout{1} = regionQueryPoolCount(varargin{:});
        case "offspringgsurvivaldiagnostics"
            varargout{1} = offspringGSurvivalDiagnostics(varargin{:});
        case "offspringgsurvivalmasks"
            [varargout{1:nargout}] = offspringGSurvivalMasks(varargin{:});
        case "offspringsurvivalhandlemasks"
            [varargout{1:nargout}] = offspringSurvivalHandleMasks(varargin{:});
        case "querygroupdiagnostics"
            varargout{1} = queryGroupDiagnostics(varargin{:});
        case "assignobjectivequeryrefs"
            varargout{1} = assignObjectiveQueryRefs(varargin{:});
        case "matchedfrontierdiagnostics"
            varargout{1} = matchedFrontierDiagnostics(varargin{:});
        case "trueboundarydiagnostics"
            varargout{1} = trueBoundaryDiagnostics(varargin{:});
        case "trueboundarysubsetdiagnostics"
            varargout{1} = trueBoundarySubsetDiagnostics(varargin{:});
        case "trainandsample"
            [varargout{1:nargout}] = trainAndSampleRegionGAN(varargin{:});
        otherwise
            error('CBSRegionGAN:BadRunnerAction', ...
                'Unsupported RunRegionGAN_RC action: %s.',action);
    end
end

function [SampleC,Counts,SampleRefs,DiagQueryC,DiagQueryRefs, ...
        SampleGroups] = regionQuerySamples(QueryC,DatasetInfo,W,nGen)
    [SampleC,Counts,SampleRefs,DiagQueryC,DiagQueryRefs] = ...
        oneSixthFrontierQuerySamples(QueryC,DatasetInfo,W,nGen);
    SampleGroups = queryGroupCodes(SampleRefs,DatasetInfo,W);
end

function Groups = queryGroupCodes(SampleRefs,DatasetInfo,W)
    SampleRefs = validateRegionQueryRefs(SampleRefs,W);
    if isempty(SampleRefs)
        Groups = zeros(0,1);
        return;
    end
    PopRefs = populatedRefsFromDatasetInfo(DatasetInfo,W);
    FrontierRefs = oneHopFrontierRefs(W,PopRefs);
    Groups = 3*ones(numel(SampleRefs),1);
    Groups(ismember(SampleRefs,FrontierRefs)) = 2;
    Groups(ismember(SampleRefs,PopRefs)) = 1;
end

function Refs = populatedRefsFromDatasetInfo(DatasetInfo,W)
    if isstruct(DatasetInfo) && isfield(DatasetInfo,'queryRegions') && ...
            ~isempty(DatasetInfo.queryRegions)
        Refs = double(DatasetInfo.queryRegions(:));
    elseif isstruct(DatasetInfo) && isfield(DatasetInfo,'trainRef') && ...
            ~isempty(DatasetInfo.trainRef)
        Refs = double(DatasetInfo.trainRef(:));
    else
        Refs = zeros(0,1);
    end
    Refs = validateRegionQueryRefs(Refs,W);
    Refs = unique(Refs,'stable');
end

function Refs = validateRegionQueryRefs(Refs,W)
    Refs = double(Refs(:));
    if isempty(Refs)
        return;
    end
    nRef = size(W,1);
    valid = isfinite(Refs) & Refs == fix(Refs) & ...
        Refs >= 1 & Refs <= nRef;
    if ~all(valid)
        error('CBSRegionGAN:BadSampleRef', ...
            'Sample refs must be finite integers in 1:size(W,1).');
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

function count = regionQueryPoolCount(QueryC,DatasetInfo,W)
    [PoolC,~,~] = oneSixthFrontierQueryPool(QueryC,DatasetInfo,W);
    count = size(PoolC,1);
end

function [SampleC,Counts,SampleRefs,DiagQueryC,DiagQueryRefs] = ...
        oneSixthFrontierQuerySamples(QueryC,DatasetInfo,W,nGen)
    [PoolC,PoolRefs,PoolGroup] = oneSixthFrontierQueryPool( ...
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
        totalBudget,~isempty(popRows),~isempty(frontierRows), ...
        1/6);
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
    SampleRefs = PoolRefs(idx);
end

function [PoolC,PoolRefs,PoolGroup] = oneSixthFrontierQueryPool( ...
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
        PopRefs = validateRegionQueryRefs(DatasetInfo.queryRegions,W);
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
        totalBudget,hasPopulated,hasFrontier,frontierRatio)
    if nargin < 4 || isempty(frontierRatio)
        frontierRatio = 0.40;
    end
    frontierRatio = max(0,min(1,double(frontierRatio)));
    if totalBudget <= 0 || (~hasPopulated && ~hasFrontier)
        popBudget = 0;
        frontierBudget = 0;
    elseif hasPopulated && hasFrontier
        frontierBudget = round(frontierRatio*totalBudget);
        if totalBudget >= 2 && frontierRatio > 0 && frontierRatio < 1
            frontierBudget = max(1,min(totalBudget - 1,frontierBudget));
        else
            frontierBudget = max(0,min(totalBudget,frontierBudget));
        end
        popBudget = totalBudget - frontierBudget;
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
            PoolRefs = validateRegionQueryRefs( ...
                DatasetInfo.allQueryRegions,W);
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

function [GAN,RawDec] = trainAndSampleRegionGAN(GAN,TrainX,TrainC, ...
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
    GAN = BoundaryWGAN_RC('train',GAN,TrainX,TrainC, ...
        Problem,GANOptions);
    [RawDec,SampleInfo] = BoundaryWGAN_RC('samplebycondition', ...
        GAN,QueryC,queryPerCondition,GANOptions);
    GAN.last_sample_info = SampleInfo;
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

function [InP1,InP2,InUnion] = offspringGSurvivalMasks(GDec,P1Dec,P2Dec)
    GDec = double(GDec);
    P1Dec = double(P1Dec);
    P2Dec = double(P2Dec);
    InP1 = decisionRowsPresentMultiset(GDec,P1Dec);
    InP2 = decisionRowsPresentMultiset(GDec,P2Dec);
    InUnion = decisionRowsPresentMultiset(GDec,[P1Dec;P2Dec]);
end

function [InP1,InP2,InUnion] = offspringSurvivalHandleMasks( ...
        Offspring,P1,P2)
    InP1 = solutionHandlesPresent(Offspring,P1);
    InP2 = solutionHandlesPresent(Offspring,P2);
    InUnion = solutionHandlesPresent(Offspring,[P1,P2]);
end

function present = solutionHandlesPresent(Query,Rows)
    present = false(numel(Query),1);
    if isempty(Query) || isempty(Rows)
        return;
    end
    try
        for i = 1 : numel(Query)
            present(i) = any(Query(i) == Rows);
        end
    catch Error
        throwAsCaller(MException('CBSRegionGAN:BadSurvivalHandles', ...
            'Offspring and selected rows must support handle identity: %s', ...
            Error.message));
    end
end

function Metrics = queryGroupDiagnostics(GeneratedObj,GeneratedCon,Groups, ...
        InP1,InP2,InUnion,PF,Options)
    if nargin < 7
        PF = [];
    end
    if nargin < 8 || isempty(Options)
        Options = struct();
    end
    GeneratedObj = double(GeneratedObj);
    n = size(GeneratedObj,1);
    if numel(Groups) ~= n
        error('CBSRegionGAN:BadQueryGroupRows', ...
            'Query group rows must match generated objective rows.');
    end
    Groups = double(Groups(:));
    if any(~isfinite(Groups) | Groups ~= fix(Groups) | ...
            Groups < 1 | Groups > 3)
        error('CBSRegionGAN:BadQueryGroupCode', ...
            'Query group codes must be finite integers in 1:3.');
    end
    if numel(InP1) ~= n || numel(InP2) ~= n || numel(InUnion) ~= n
        error('CBSRegionGAN:BadQueryMaskRows', ...
            'Query survival mask rows must match generated objective rows.');
    end
    InP1 = logical(InP1(:));
    InP2 = logical(InP2(:));
    InUnion = logical(InUnion(:));
    Feasible = generatedFeasibleMask(GeneratedCon,n);

    Metrics = struct();
    GroupNames = ["populated","frontier","remote"];
    for i = 1 : numel(GroupNames)
        InGroup = Groups == i;
        sampleCount = sum(InGroup);
        feasibleCount = sum(Feasible & InGroup);
        surviveP1Count = sum(InP1 & InGroup);
        surviveP2Count = sum(InP2 & InGroup);
        surviveUnionCount = sum(InUnion & InGroup);
        if sampleCount > 0
            feasibleRate = feasibleCount/sampleCount;
            surviveP1Rate = surviveP1Count/sampleCount;
            surviveP2Rate = surviveP2Count/sampleCount;
            surviveUnionRate = surviveUnionCount/sampleCount;
        else
            feasibleRate = NaN;
            surviveP1Rate = NaN;
            surviveP2Rate = NaN;
            surviveUnionRate = NaN;
        end
        if isempty(GeneratedCon)
            GroupCon = [];
        else
            GroupCon = GeneratedCon(InGroup,:);
        end
        Geometry = trueBoundaryDiagnostics( ...
            GeneratedObj(InGroup,:),GroupCon,PF,Options);
        Prefix = "query_" + GroupNames(i) + "_";
        Metrics.(char(Prefix + "sample_count")) = sampleCount;
        Metrics.(char(Prefix + "feasible_count")) = feasibleCount;
        Metrics.(char(Prefix + "feasible_rate")) = feasibleRate;
        Metrics.(char(Prefix + "survive_P1_count")) = surviveP1Count;
        Metrics.(char(Prefix + "survive_P1_rate")) = surviveP1Rate;
        Metrics.(char(Prefix + "survive_P2_count")) = surviveP2Count;
        Metrics.(char(Prefix + "survive_P2_rate")) = surviveP2Rate;
        Metrics.(char(Prefix + "survive_union_count")) = ...
            surviveUnionCount;
        Metrics.(char(Prefix + "survive_union_rate")) = ...
            surviveUnionRate;
        GeometryNames = fieldnames(Geometry);
        for j = 1 : numel(GeometryNames)
            Metrics.(char(Prefix + string(GeometryNames{j}))) = ...
                Geometry.(GeometryNames{j});
        end
    end
end

function Refs = assignObjectiveQueryRefs(Obj,W,objMin,objSpan)
    Obj = double(Obj);
    W = double(W);
    if ~ismatrix(Obj) || ~ismatrix(W) || ...
            size(Obj,2) ~= size(W,2) || isempty(W)
        error('CBSRegionGAN:BadObjectiveRefColumns', ...
            'Objectives and nonempty reference vectors must have equal columns.');
    end
    M = size(W,2);
    objMin = double(objMin(:)');
    objSpan = double(objSpan(:)');
    if numel(objMin) ~= M || numel(objSpan) ~= M
        error('CBSRegionGAN:BadObjectiveScale', ...
            'Objective scale must match reference-vector dimension.');
    end
    objSpan(~isfinite(objSpan) | objSpan <= eps) = 1;
    Yn = (Obj - objMin)./objSpan;
    Yn(~isfinite(Yn)) = 0;
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    NormY = sqrt(sum(Yn.^2,2));
    Yu = Yn./max(NormY,eps);
    [~,Refs] = max(Yu*Wn',[],2);
    zeroRows = NormY <= eps;
    if any(zeroRows)
        D = pairDistance(Yn(zeroRows,:),W);
        [~,Refs(zeroRows)] = min(D,[],2);
    end
    Refs = reshape(Refs,size(Obj,1),1);
end

function Metrics = matchedFrontierDiagnostics(GANData,DEData,PF,Options)
    if nargin < 3
        PF = [];
    end
    if nargin < 4 || isempty(Options)
        Options = struct();
    end
    GANData = validateMatchedData(GANData,true);
    DEData = validateMatchedData(DEData,false);
    frontierRefs = unique(GANData.ref(GANData.group == 2),'stable');
    deRefs = unique(DEData.ref,'stable');
    availableRefs = frontierRefs(ismember(frontierRefs,deRefs));
    unavailableRefs = frontierRefs(~ismember(frontierRefs,deRefs));

    Metrics = emptyMatchedFrontierMetrics();
    Metrics.frontier_query_ref_count = numel(frontierRefs);
    Metrics.frontier_matched_de_available = double(~isempty(availableRefs));
    Metrics.frontier_matched_de_available_ref_count = ...
        numel(availableRefs);
    Metrics.frontier_matched_de_unavailable_ref_count = ...
        numel(unavailableRefs);
    ganMask = GANData.group == 2 & ismember(GANData.ref,availableRefs);
    deMask = ismember(DEData.ref,availableRefs);
    Metrics = addMatchedSubsetMetrics(Metrics,"frontier_matched_gan_", ...
        GANData,ganMask,PF,Options);
    Metrics = addMatchedSubsetMetrics(Metrics,"frontier_matched_de_", ...
        DEData,deMask,PF,Options);
    Metrics = addRefEqualMatchedMetrics(Metrics, ...
        "frontier_matched_gan_",GANData,ganMask,availableRefs,PF,Options);
    Metrics = addRefEqualMatchedMetrics(Metrics, ...
        "frontier_matched_de_",DEData,deMask,availableRefs,PF,Options);
end

function Data = validateMatchedData(Data,requireGroup)
    required = {'obj','con','ref','survive_P1','survive_P2', ...
        'survive_union'};
    if requireGroup
        required{end+1} = 'group';
    end
    if ~isstruct(Data) || ~all(isfield(Data,required))
        error('CBSRegionGAN:BadMatchedData', ...
            'Matched diagnostic data is missing required fields.');
    end
    Data.obj = double(Data.obj);
    Data.con = double(Data.con);
    n = size(Data.obj,1);
    rowFields = {'ref','survive_P1','survive_P2','survive_union'};
    if requireGroup
        rowFields{end+1} = 'group';
    else
        Data.group = zeros(n,1);
    end
    for i = 1 : numel(rowFields)
        name = rowFields{i};
        if numel(Data.(name)) ~= n
            error('CBSRegionGAN:BadMatchedDataRows', ...
                'Matched diagnostic fields must have equal rows.');
        end
        Data.(name) = double(Data.(name)(:));
    end
    if size(Data.con,1) ~= n
        error('CBSRegionGAN:BadMatchedDataRows', ...
            'Matched constraint rows must equal objective rows.');
    end
    if any(~isfinite(Data.ref) | Data.ref ~= fix(Data.ref) | Data.ref < 1)
        error('CBSRegionGAN:BadMatchedRef', ...
            'Matched refs must be positive finite integers.');
    end
    if requireGroup && any(~isfinite(Data.group) | ...
            Data.group ~= fix(Data.group) | Data.group < 1 | Data.group > 3)
        error('CBSRegionGAN:BadMatchedGroup', ...
            'Matched query groups must be finite integers in 1:3.');
    end
    maskValues = [Data.survive_P1;Data.survive_P2;Data.survive_union];
    if any(~isfinite(maskValues) | ...
            (maskValues ~= 0 & maskValues ~= 1))
        error('CBSRegionGAN:BadMatchedSurvivalMask', ...
            'Matched survival masks must contain only finite zero/one values.');
    end
    Data.survive_P1 = logical(Data.survive_P1);
    Data.survive_P2 = logical(Data.survive_P2);
    Data.survive_union = logical(Data.survive_union);
end

function Metrics = emptyMatchedFrontierMetrics()
    Metrics = struct( ...
        'frontier_query_ref_count',0, ...
        'frontier_matched_de_available',0, ...
        'frontier_matched_de_available_ref_count',0, ...
        'frontier_matched_de_unavailable_ref_count',0);
    prefixes = ["frontier_matched_gan_","frontier_matched_de_"];
    for i = 1 : numel(prefixes)
        prefix = prefixes(i);
        countNames = ["sample_count","feasible_count", ...
            "survive_P1_count","survive_P2_count","survive_union_count"];
        rateNames = ["feasible_rate","survive_P1_rate", ...
            "survive_P2_rate","survive_union_rate"];
        geometryNames = ["bdist50_true","bwidth90_10_true", ...
            "bcover_eps_true"];
        for j = 1 : numel(countNames)
            Metrics.(char(prefix + countNames(j))) = 0;
        end
        for j = 1 : numel(rateNames)
            Metrics.(char(prefix + rateNames(j))) = NaN;
        end
        for j = 1 : numel(geometryNames)
            Metrics.(char(prefix + geometryNames(j))) = NaN;
        end
        refEqualNames = ["refeq_feasible_rate","refeq_survive_P1_rate", ...
            "refeq_survive_P2_rate","refeq_survive_union_rate", ...
            "refeq_bdist50_true","refeq_bwidth90_10_true", ...
            "refeq_bcover_eps_true"];
        for j = 1 : numel(refEqualNames)
            Metrics.(char(prefix + refEqualNames(j))) = NaN;
        end
    end
end

function Metrics = addMatchedSubsetMetrics(Metrics,prefix,Data,Mask, ...
        PF,Options)
    count = sum(Mask);
    feasible = generatedFeasibleMask(subsetConstraints(Data.con,Mask),count);
    Metrics.(char(prefix + "sample_count")) = count;
    Metrics.(char(prefix + "feasible_count")) = sum(feasible);
    Metrics.(char(prefix + "survive_P1_count")) = ...
        sum(Data.survive_P1(Mask));
    Metrics.(char(prefix + "survive_P2_count")) = ...
        sum(Data.survive_P2(Mask));
    Metrics.(char(prefix + "survive_union_count")) = ...
        sum(Data.survive_union(Mask));
    if count > 0
        Metrics.(char(prefix + "feasible_rate")) = mean(double(feasible));
        Metrics.(char(prefix + "survive_P1_rate")) = ...
            mean(double(Data.survive_P1(Mask)));
        Metrics.(char(prefix + "survive_P2_rate")) = ...
            mean(double(Data.survive_P2(Mask)));
        Metrics.(char(prefix + "survive_union_rate")) = ...
            mean(double(Data.survive_union(Mask)));
    end
    Geometry = trueBoundaryDiagnostics(Data.obj(Mask,:), ...
        subsetConstraints(Data.con,Mask),PF,Options);
    names = fieldnames(Geometry);
    for i = 1 : numel(names)
        Metrics.(char(prefix + string(names{i}))) = Geometry.(names{i});
    end
end

function Con = subsetConstraints(Con,Mask)
    if isempty(Con)
        Con = [];
    else
        Con = Con(Mask,:);
    end
end

function Metrics = addRefEqualMatchedMetrics(Metrics,prefix,Data,BaseMask, ...
        Refs,PF,Options)
    if isempty(Refs)
        return;
    end
    values = NaN(numel(Refs),7);
    for i = 1 : numel(Refs)
        mask = BaseMask & Data.ref == Refs(i);
        count = sum(mask);
        feasible = generatedFeasibleMask( ...
            subsetConstraints(Data.con,mask),count);
        if count > 0
            values(i,1) = mean(double(feasible));
            values(i,2) = mean(double(Data.survive_P1(mask)));
            values(i,3) = mean(double(Data.survive_P2(mask)));
            values(i,4) = mean(double(Data.survive_union(mask)));
        end
        Geometry = trueBoundaryDiagnostics(Data.obj(mask,:), ...
            subsetConstraints(Data.con,mask),PF,Options);
        values(i,5) = Geometry.bdist50_true;
        values(i,6) = Geometry.bwidth90_10_true;
        values(i,7) = Geometry.bcover_eps_true;
    end
    names = ["refeq_feasible_rate","refeq_survive_P1_rate", ...
        "refeq_survive_P2_rate","refeq_survive_union_rate", ...
        "refeq_bdist50_true","refeq_bwidth90_10_true", ...
        "refeq_bcover_eps_true"];
    for i = 1 : numel(names)
        Metrics.(char(prefix + names(i))) = meanFiniteMatched(values(:,i));
    end
end

function value = meanFiniteMatched(values)
    values = double(values(:));
    values = values(isfinite(values));
    if isempty(values)
        value = NaN;
    else
        value = mean(values);
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
    Metrics = summarizeTrueBoundaryMetrics(AbsDist,Nearest,Feasible, ...
        BoundaryN,ArcInfo,Options);
end

function Metrics = trueBoundarySubsetDiagnostics(GeneratedObj,GeneratedCon, ...
        PF,Subsets,Options)
%TRUEBOUNDARYSUBSETDIAGNOSTICS Reuse one boundary projection for subsets.
%   This read-only diagnostic is intended for offline archive audits. Each
%   column of Subsets selects rows from the already evaluated GeneratedObj.

    if nargin < 2
        GeneratedCon = [];
    end
    if nargin < 3
        PF = [];
    end
    if nargin < 4 || isempty(Subsets)
        Subsets = true(size(GeneratedObj,1),1);
    end
    if nargin < 5 || isempty(Options)
        Options = struct();
    end
    GeneratedObj = double(GeneratedObj);
    Subsets = logical(Subsets);
    if size(Subsets,1) ~= size(GeneratedObj,1)
        error('CBSRegionGAN:SubsetRowMismatch', ...
            'Subset masks must have one row per generated objective row.');
    end
    nSubset = size(Subsets,2);
    Metrics = repmat(emptyTrueBoundaryMetrics(),1,nSubset);
    if isempty(GeneratedObj) || size(GeneratedObj,2) < 2 || nSubset == 0
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
    for i = 1 : nSubset
        keep = Subsets(:,i);
        Metrics(i) = summarizeTrueBoundaryMetrics( ...
            AbsDist(keep),Nearest(keep),Feasible(keep), ...
            BoundaryN,ArcInfo,Options);
    end
end

function Metrics = summarizeTrueBoundaryMetrics(AbsDist,Nearest,Feasible, ...
        BoundaryN,ArcInfo,Options)
    Metrics = emptyTrueBoundaryMetrics();
    if isempty(AbsDist)
        return;
    end
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

function [lastMetric,historyMetric,cloudMetric] = metricNames()
    lastMetric = 'region_wgan_gp_last';
    historyMetric = 'region_wgan_gp_history';
    cloudMetric = 'region_wgan_gp_cloud';
end
