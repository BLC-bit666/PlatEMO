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
        case "resolveganoptions"
            [varargout{1:nargout}] = resolveGANOptions(varargin{:});
        case "trainandsample"
            [varargout{1:nargout}] = trainAndSampleRegionGAN(varargin{:});
        otherwise
            error('CBSRegionGAN:BadRunnerAction', ...
                'Unsupported RunRegionGAN_RC action: %s.',action);
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

function Options = resolveGANOptions(Options,currentFE,maxFE)
    if nargin < 2 || isempty(currentFE)
        currentFE = 0;
    end
    if nargin < 3 || isempty(maxFE)
        maxFE = NaN;
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
        otherwise
            error('CBSRegionGAN:BadGANIterSchedule', ...
                'Unsupported ganIterSchedule: %s.',schedule);
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
