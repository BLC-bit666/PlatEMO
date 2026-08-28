function varargout = RunRegionGAN_RC(action,varargin)
%RUNREGIONGAN_RC Allocate mainline queries and dispatch WGAN training.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    switch lower(strtrim(string(action)))
        case "regionquerysamples"
            [varargout{1:nargout}] = regionQuerySamples(varargin{:});
        case "balancedquerysamples"
            [varargout{1:nargout}] = balancedQuerySamples(varargin{:});
        case "conditioncriticfilter"
            [varargout{1:nargout}] = conditionCriticFilter(varargin{:});
        case "trainandsample"
            [varargout{1:nargout}] = trainAndSample(varargin{:});
        otherwise
            error('CBSRegionGAN:BadRunnerAction', ...
                'Unsupported mainline action: %s.',action);
    end
end

function [SampleC,SampleRefs] = balancedQuerySamples(W,totalBudget)
%BALANCEDQUERYSAMPLES Allocate queries across every reference vector.

    W = double(W);
    totalBudget = max(0,round(double(totalBudget)));
    refCount = size(W,1);
    if totalBudget == 0 || refCount == 0
        SampleRefs = zeros(0,1);
        SampleC = zeros(0,size(W,2));
        return;
    end
    baseCount = floor(totalBudget/refCount);
    remainder = mod(totalBudget,refCount);
    SampleRefs = repelem((1:refCount)',baseCount,1);
    if remainder > 0
        extra = randperm(refCount,remainder)';
        SampleRefs = [SampleRefs;extra];
    end
    SampleRefs = SampleRefs(randperm(numel(SampleRefs)));
    SampleC = W(SampleRefs,:);
end

function [FilteredDec,FilteredRefs,KeepIdx,Percentile] = ...
        conditionCriticFilter(RawDec,RawRefs,RawScore,keepCount,W)
%CONDITIONCRITICFILTER Compare critic scores only within each condition.

    RawDec = double(RawDec);
    RawRefs = reshape(double(RawRefs),[],1);
    RawScore = reshape(double(RawScore),[],1);
    rowCount = size(RawDec,1);
    if numel(RawRefs) ~= rowCount || numel(RawScore) ~= rowCount
        error('CBSRegionGAN:CriticFilterShape', ...
            'Decisions, references, and critic scores need equal rows.');
    end
    keepCount = max(0,round(double(keepCount)));
    Percentile = nan(rowCount,1);
    valid = all(isfinite(RawDec),2) & isfinite(RawScore) & ...
        isfinite(RawRefs) & RawRefs == fix(RawRefs) & ...
        RawRefs >= 1 & RawRefs <= size(W,1);
    refs = unique(RawRefs(valid),'stable');
    for r = reshape(refs,1,[])
        rows = find(valid & RawRefs == r);
        [~,order] = sortrows([-RawScore(rows),rows],[1 2]);
        rank = zeros(numel(rows),1);
        rank(order) = (1:numel(rows))';
        Percentile(rows) = (numel(rows)-rank+0.5)/numel(rows);
    end
    rows = find(valid);
    if keepCount == 0 || isempty(rows)
        KeepIdx = zeros(0,1);
    else
        [~,order] = sortrows([-Percentile(rows),rows],[1 2]);
        KeepIdx = rows(order(1:min(keepCount,numel(order))));
        KeepIdx = sort(KeepIdx,'ascend');
    end
    FilteredDec = RawDec(KeepIdx,:);
    FilteredRefs = RawRefs(KeepIdx);
end

function [SampleC,SampleRefs] = regionQuerySamples(PopulatedRefs,W,nGen)
%REGIONQUERYSAMPLES Allocate 70%% of slots to memory-supported references.
%   The remaining 30%% target their one-hop empty neighbors. At nGen=20
%   this is the fixed 14+6 mainline allocation. If either pool is empty,
%   its slots deterministically reflow to the available pool.

    PopulatedRefs = validateRefs(PopulatedRefs,W);
    PopulatedRefs = unique(PopulatedRefs,'stable');
    FrontierRefs = oneHopFrontierRefs(W,PopulatedRefs);
    PoolRefs = [PopulatedRefs;FrontierRefs];
    totalBudget = max(0,round(double(nGen)));
    if isempty(PoolRefs) || totalBudget == 0
        SampleRefs = zeros(0,1);
        SampleC = zeros(0,size(W,2));
        return;
    end

    populatedRows = (1:numel(PopulatedRefs))';
    frontierRows = numel(PopulatedRefs)+(1:numel(FrontierRefs))';
    [populatedBudget,frontierBudget] = splitBudget( ...
        totalBudget,~isempty(populatedRows),~isempty(frontierRows));
    selected = zeros(populatedBudget+frontierBudget,1);
    next = 0;
    if populatedBudget > 0
        local = uniformRefRows(PoolRefs(populatedRows),populatedBudget);
        selected(1:populatedBudget) = populatedRows(local);
        next = populatedBudget;
    end
    if frontierBudget > 0
        local = uniformRefRows(PoolRefs(frontierRows),frontierBudget);
        selected(next+1:next+frontierBudget) = frontierRows(local);
        next = next+frontierBudget;
    end
    selected = selected(1:next);
    if ~isempty(selected)
        selected = selected(randperm(numel(selected)));
    end
    SampleRefs = PoolRefs(selected);
    SampleC = double(W(SampleRefs,:));
end

function idx = uniformRefRows(PoolRefs,totalBudget)
%UNIFORMREFROWS Sample reference rows while preserving the RNG sequence.

    PoolRefs = round(double(PoolRefs(:)));
    valid = isfinite(PoolRefs) & PoolRefs > 0;
    if ~any(valid)
        idx = randi(numel(PoolRefs),totalBudget,1);
        return;
    end
    refs = unique(PoolRefs(valid),'stable');
    idx = zeros(totalBudget,1);
    draws = randi(numel(refs),totalBudget,1);
    for i = 1 : totalBudget
        rows = find(PoolRefs == refs(draws(i)));
        idx(i) = rows(randi(numel(rows)));
    end
end

function [populatedBudget,frontierBudget] = ...
        splitBudget(totalBudget,hasPopulated,hasFrontier)
%SPLITBUDGET Assign 70%% populated and 30%% frontier slots.

    if ~hasPopulated && ~hasFrontier
        populatedBudget = 0;
        frontierBudget = 0;
    elseif hasPopulated && hasFrontier
        frontierBudget = round(0.3*totalBudget);
        if totalBudget >= 2
            frontierBudget = max(1,min(totalBudget-1,frontierBudget));
        else
            frontierBudget = max(0,min(totalBudget,frontierBudget));
        end
        populatedBudget = totalBudget-frontierBudget;
    elseif hasPopulated
        populatedBudget = totalBudget;
        frontierBudget = 0;
    else
        populatedBudget = 0;
        frontierBudget = totalBudget;
    end
end

function Refs = oneHopFrontierRefs(W,PopulatedRefs)
%ONEHOPFRONTIERREFS Return empty references adjacent to supported ones.

    if isempty(W) || isempty(PopulatedRefs)
        Refs = zeros(0,1);
        return;
    end
    candidates = cell(numel(PopulatedRefs),1);
    for i = 1 : numel(PopulatedRefs)
        candidates{i} = neighborRefs(W,PopulatedRefs(i),1);
    end
    Refs = unique(vertcat(candidates{:}),'stable');
    Refs = Refs(~ismember(Refs,PopulatedRefs));
end

function Refs = neighborRefs(W,r,radius)
%NEIGHBORREFS Return the closest reference-vector indices around r.

    distance = sqrt(sum((double(W)-double(W(r,:))).^2,2));
    [~,order] = sort(distance,'ascend');
    Refs = order(1:min(numel(order),1+2*radius));
end

function Refs = validateRefs(Refs,W)
%VALIDATEREFS Validate integer reference-vector indices.

    Refs = double(Refs(:));
    valid = isfinite(Refs) & Refs == fix(Refs) & ...
        Refs >= 1 & Refs <= size(W,1);
    if ~all(valid)
        error('CBSRegionGAN:BadSampleRef', ...
            'Reference indices must be valid integers.');
    end
end

function [GAN,RawDec,RawScore] = trainAndSample( ...
        GAN,TrainX,TrainC,SampleC,Problem,Options)
%TRAINANDSAMPLE Train on pairflag rows, then query feasible conditions.

    if size(TrainX,1) < Options.minTrainCount
        RawDec = zeros(0,Problem.D);
        RawScore = zeros(0,1);
        return;
    end
    GAN = BoundaryWGAN_RC('train',GAN,TrainX,TrainC,Problem,Options);
    if isempty(SampleC)
        RawDec = zeros(0,Problem.D);
        RawScore = zeros(0,1);
        return;
    end
    RawDec = BoundaryWGAN_RC('samplebycondition',GAN,SampleC,Options);
    RawScore = BoundaryWGAN_RC( ...
        'scorebycondition',GAN,RawDec,SampleC);
end
