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

function [SampleC,SampleRefs,SampleTypes] = regionQuerySamples( ...
        PopulatedRefs,W,nGen,allocMode)
%REGIONQUERYSAMPLES Allocate samples to populated and frontier references.
%   Modes: "legacy" (default) keeps the validated one-sixth one-hop
%   allocation and its exact RNG sequence; "scout" sends two thirds of the
%   budget to a two-hop frontier; "half" splits the budget evenly between
%   populated references and the one-hop frontier (jump-origin dose arm).
%   Logical values remain accepted for backward compatibility (true maps
%   to "scout"). SAMPLETYPES labels each row: 1 populated, 2 one-hop
%   frontier, 3 two-hop frontier.

    if nargin < 4 || isempty(allocMode)
        allocMode = "legacy";
    end
    if islogical(allocMode)
        if allocMode
            allocMode = "scout";
        else
            allocMode = "legacy";
        end
    end
    allocMode = lower(strtrim(string(allocMode)));
    if ~(isscalar(allocMode) && ...
            ismember(allocMode,["legacy","scout","half","wide"]))
        error('CBSRegionGAN:BadAllocMode', ...
            'Query allocation mode must be legacy, scout, half, or wide.');
    end
    PopulatedRefs = validateRefs(PopulatedRefs,W);
    PopulatedRefs = unique(PopulatedRefs,'stable');
    if allocMode == "scout"
        [SampleC,SampleRefs,SampleTypes] = scoutQuerySamples( ...
            PopulatedRefs,W,nGen,2/3);
        return;
    end
    if allocMode == "wide"
        % Guide-study allocation: 6:14 populated-to-empty at nGen=20.
        [SampleC,SampleRefs,SampleTypes] = scoutQuerySamples( ...
            PopulatedRefs,W,nGen,7/10);
        return;
    end
    FrontierRefs = oneHopFrontierRefs(W,PopulatedRefs);
    PoolRefs = [PopulatedRefs;FrontierRefs];
    PoolGroup = [ones(numel(PopulatedRefs),1); ...
        2*ones(numel(FrontierRefs),1)];
    totalBudget = max(0,round(double(nGen)));
    if isempty(PoolRefs) || totalBudget == 0
        SampleRefs = zeros(0,1);
        SampleC = zeros(0,size(W,2));
        SampleTypes = zeros(0,1);
        return;
    end

    populatedRows = find(PoolGroup == 1);
    frontierRows = find(PoolGroup == 2);
    if allocMode == "half"
        [populatedBudget,frontierBudget] = halfBudgets( ...
            totalBudget,~isempty(populatedRows),~isempty(frontierRows));
    else
        [populatedBudget,frontierBudget] = queryBudgets( ...
            totalBudget,~isempty(populatedRows),~isempty(frontierRows));
    end
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
    SampleTypes = PoolGroup(selected);
    SampleC = double(W(SampleRefs,:));
end

function [SampleC,SampleRefs,SampleTypes] = scoutQuerySamples( ...
        PopulatedRefs,W,nGen,frontierShare)
%SCOUTQUERYSAMPLES Frontier-majority allocation over a two-hop pool.
%   Two thirds of the budget target empty directions; within the frontier
%   share, one-hop directions receive two thirds and two-hop directions the
%   remainder. Missing pools reflow their budget (hop2 -> hop1 -> populated
%   and populated -> frontier) so the full budget is always spent whenever
%   any pool exists.

    if nargin < 4 || isempty(frontierShare)
        frontierShare = 2/3;
    end
    totalBudget = max(0,round(double(nGen)));
    Hop1 = oneHopFrontierRefs(W,PopulatedRefs);
    Hop2 = twoHopFrontierRefs(W,PopulatedRefs,Hop1);
    if (isempty(PopulatedRefs) && isempty(Hop1) && isempty(Hop2)) || ...
            totalBudget == 0
        SampleRefs = zeros(0,1);
        SampleC = zeros(0,size(W,2));
        SampleTypes = zeros(0,1);
        return;
    end

    frontierBudget = round(frontierShare*totalBudget);
    if isempty(Hop1) && isempty(Hop2)
        frontierBudget = 0;
    elseif isempty(PopulatedRefs)
        frontierBudget = totalBudget;
    end
    populatedBudget = totalBudget - frontierBudget;
    hop1Budget = ceil(2*frontierBudget/3);
    hop2Budget = frontierBudget - hop1Budget;
    if isempty(Hop2)
        hop1Budget = hop1Budget + hop2Budget;
        hop2Budget = 0;
    end
    if isempty(Hop1)
        hop2Budget = hop2Budget + hop1Budget;
        hop1Budget = 0;
    end

    pools = {PopulatedRefs,Hop1,Hop2};
    budgets = [populatedBudget,hop1Budget,hop2Budget];
    SampleRefs = zeros(totalBudget,1);
    SampleTypes = zeros(totalBudget,1);
    next = 0;
    for g = 1 : 3
        if budgets(g) <= 0
            continue;
        end
        local = uniformRefRows(pools{g},budgets(g));
        SampleRefs(next+1:next+budgets(g)) = pools{g}(local);
        SampleTypes(next+1:next+budgets(g)) = g;
        next = next + budgets(g);
    end
    SampleRefs = SampleRefs(1:next);
    SampleTypes = SampleTypes(1:next);
    if ~isempty(SampleRefs)
        order = randperm(numel(SampleRefs));
        SampleRefs = SampleRefs(order);
        SampleTypes = SampleTypes(order);
    end
    SampleC = double(W(SampleRefs,:));
end

function Refs = twoHopFrontierRefs(W,PopulatedRefs,Hop1)
%TWOHOPFRONTIERREFS Empty references adjacent to the one-hop frontier.

    if isempty(W) || isempty(Hop1)
        Refs = zeros(0,1);
        return;
    end
    candidates = cell(numel(Hop1),1);
    for i = 1 : numel(Hop1)
        candidates{i} = neighborRefs(W,Hop1(i),1);
    end
    Refs = unique(vertcat(candidates{:}),'stable');
    Refs = Refs(~ismember(Refs,[PopulatedRefs(:);Hop1(:)]));
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

function [populatedBudget,frontierBudget] = halfBudgets( ...
        totalBudget,hasPopulated,hasFrontier)
%HALFBUDGETS Split the budget evenly with the same reflow guards.

    if ~hasPopulated && ~hasFrontier
        populatedBudget = 0;
        frontierBudget = 0;
    elseif hasPopulated && hasFrontier
        frontierBudget = round(totalBudget/2);
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
