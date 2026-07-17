function varargout = RunRegionGAN_RC(action,varargin)
%RUNREGIONGAN_RC Mainline query allocation and WGAN dispatch.
%   REGIONQUERYSAMPLES allocates a generation budget between populated
%   reference vectors and their one-hop frontier.
%   TRAINANDSAMPLE trains the WGAN-GP and samples full decision vectors.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Dispatch the requested operation
    switch lower(strtrim(string(action)))
        case "regionquerysamples"
            [varargout{1:nargout}] = regionQuerySamples(varargin{:});
        case "trainandsample"
            [varargout{1:nargout}] = trainAndSampleRegionGAN(varargin{:});
        otherwise
            error('CBSRegionGAN:BadRunnerAction', ...
                'Unsupported mainline action: %s.',action);
    end
end

function [SampleC,SampleRefs] = regionQuerySamples(PopulatedRefs,W,nGen)
%REGIONQUERYSAMPLES Allocate samples to populated and frontier references.

    PopulatedRefs = validateRefs(PopulatedRefs,W);
    PopulatedRefs = unique(PopulatedRefs,'stable');
    FrontierRefs = oneHopFrontierRefs(W,PopulatedRefs);
    PoolRefs = [PopulatedRefs;FrontierRefs];
    PoolGroup = [ones(numel(PopulatedRefs),1); ...
        2*ones(numel(FrontierRefs),1)];
    totalBudget = max(0,round(double(nGen)));
    if isempty(PoolRefs) || totalBudget == 0
        SampleRefs = zeros(0,1);
        SampleC = zeros(0,size(W,2));
        return;
    end

    populatedRows = find(PoolGroup == 1);
    frontierRows = find(PoolGroup == 2);
    [populatedBudget,frontierBudget] = queryBudgets( ...
        totalBudget,~isempty(populatedRows),~isempty(frontierRows));
    selected = zeros(populatedBudget + frontierBudget,1);
    next = 0;
    if populatedBudget > 0
        local = uniformRefRows(PoolRefs(populatedRows),populatedBudget);
        selected(1:populatedBudget) = populatedRows(local);
        next = populatedBudget;
    end
    if frontierBudget > 0
        local = uniformRefRows(PoolRefs(frontierRows),frontierBudget);
        selected(next+1:next+frontierBudget) = frontierRows(local);
        next = next + frontierBudget;
    end
    selected = selected(1:next);
    if ~isempty(selected)
        selected = selected(randperm(numel(selected)));
    end
    SampleRefs = PoolRefs(selected);
    SampleC = double(W(SampleRefs,:));
end

function idx = uniformRefRows(PoolRefs,totalBudget)
%UNIFORMREFROWS Draw reference rows with the validated RNG sequence.

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

function [populatedBudget,frontierBudget] = queryBudgets( ...
        totalBudget,hasPopulated,hasFrontier)
%QUERYBUDGETS Assign one sixth of the budget to an available frontier.

    if ~hasPopulated && ~hasFrontier
        populatedBudget = 0;
        frontierBudget = 0;
    elseif hasPopulated && hasFrontier
        frontierBudget = round(totalBudget/6);
        if totalBudget >= 2
            frontierBudget = max(1,min(totalBudget-1,frontierBudget));
        else
            frontierBudget = max(0,min(totalBudget,frontierBudget));
        end
        populatedBudget = totalBudget - frontierBudget;
    elseif hasPopulated
        populatedBudget = totalBudget;
        frontierBudget = 0;
    else
        populatedBudget = 0;
        frontierBudget = totalBudget;
    end
end

function Refs = oneHopFrontierRefs(W,PopulatedRefs)
%ONEHOPFRONTIERREFS Find neighboring references without anchor samples.

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

    if radius <= 0
        Refs = r;
        return;
    end
    distance = sqrt(sum((double(W) - double(W(r,:))).^2,2));
    [~,order] = sort(distance,'ascend');
    Refs = order(1:min(numel(order),1 + 2*radius));
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

function [GAN,RawDec] = trainAndSampleRegionGAN( ...
        GAN,TrainX,TrainC,SampleC,Problem,Options)
%TRAINANDSAMPLEREGIONGAN Train only when enough anchor rows are present.

    if size(TrainX,1) < Options.minTrainCount || isempty(SampleC)
        RawDec = zeros(0,Problem.D);
        return;
    end
    GAN = BoundaryWGAN_RC('train',GAN,TrainX,TrainC,Problem,Options);
    RawDec = BoundaryWGAN_RC('samplebycondition',GAN,SampleC,Options);
end
