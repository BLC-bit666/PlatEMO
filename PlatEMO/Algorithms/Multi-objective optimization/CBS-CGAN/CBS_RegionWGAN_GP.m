classdef CBS_RegionWGAN_GP < ALGORITHM
% <2026> <multi> <real> <constrained>
% Reference vector-conditioned boundary WGAN-GP
% nGen            ---  20 --- Number of generated solutions per eligible event
% zDim            ---   6 --- Dimension of the generator noise vector
% ganIter         --- 100 --- Generator updates per eligible training event
% ganMiniBatch    ---  32 --- Mini-batch size of WGAN-GP training
% nCritic         ---   4 --- Critic updates per generator update
% minGANTrainCount ---  32 --- Minimum feasible-anchor rows required for training
% sampleSigma     --- 0.3 --- Standard deviation of generator sampling noise

%------------------------------- Reference --------------------------------
% [1] Y. Tian, T. Zhang, J. Xiao, X. Zhang, and Y. Jin. A coevolutionary
% framework for constrained multi-objective optimization problems. IEEE
% Transactions on Evolutionary Computation, 2021, 25(1): 102-116.
% [2] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville.
% Improved training of Wasserstein GANs. Advances in Neural Information
% Processing Systems, 2017, 30.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties(Access = private,Transient)
        operatorMode = "ga_de_half";  % Mainline offspring operator (S2)
        boundarySearch = "on";        % Mainline post-stop line search (BLS)
        generatorMode = "wgan";       % Ablation switch: wgan | copynoise |
                                      % jump | jumptrivial
        scoutMode = "off";            % Mainline (2026-07-20 decision):
                                      % the guide mechanism replaces the
                                      % protected-injection mainline; the
                                      % protection study values stay
                                      % reachable. off | on | noprotect |
                                      % nofrontier | halffrontier
        metricsMode = "off";          % off | on (pure counting, RNG-free)
        guideMode = "on";             % off | on | mix. "on": unevaluated
                                      % CGAN guides steer part of P1's
                                      % offspring; "mix" reproduces the
                                      % same GA/DE split without guides
                                      % (ratio-confound control).
        guideShare = 0.2;             % guided fraction of P1 offspring
                                      % (GD20 mainline, 2026-07-20)
        guideCarve = "sym";           % sym: GA/DE shrink equally;
                                      % de: GA keeps 50%, DE donates
        guideWindow = "half";         % half | full: guide supply window
        blsWindow = "full";           % Mainline (2026-07-20 fusion):
                                      % boundary calibration runs all run.
                                      % late restores the pre-fusion path.
        blsFeed = "on";               % Calibration candidates join the
                                      % same-generation BMem harvest.
        initDecs = [];                % Optional N-by-D matrix seeding both
                                      % populations for branched-continuation
                                      % experiments; empty keeps the native
                                      % random initialization.
        scoutMetricsStore = [];       % Collected counters when metricsMode=on
    end

    methods
        function Algorithm = CBS_RegionWGAN_GP(varargin)
        %CBS_REGIONWGAN_GP Construct the algorithm and optional switches.
            Algorithm@ALGORITHM(varargin{:});
            Algorithm.operatorMode = readOperatorMode(varargin);
            Algorithm.boundarySearch = readBoundarySearch(varargin);
            Algorithm.generatorMode = readGeneratorMode(varargin);
            Algorithm.scoutMode = readScoutMode(varargin);
            Algorithm.metricsMode = readMetricsMode(varargin);
            Algorithm.guideMode = readGuideMode(varargin);
            Algorithm.guideShare = readGuideShare(varargin);
            Algorithm.guideCarve = readGuideCarve(varargin);
            Algorithm.guideWindow = readGuideWindow(varargin);
            Algorithm.blsWindow = readBlsWindow(varargin);
            Algorithm.blsFeed = readBlsFeed(varargin);
            Algorithm.initDecs = readInitDecs(varargin);
        end

        function mode = effectiveOperatorMode(Algorithm)
        %EFFECTIVEOPERATORMODE Return the live offspring-operator mode.
            mode = Algorithm.operatorMode;
        end

        function state = effectiveBoundarySearch(Algorithm)
        %EFFECTIVEBOUNDARYSEARCH Return the live boundary-search switch.
            state = Algorithm.boundarySearch;
        end

        function mode = effectiveGeneratorMode(Algorithm)
        %EFFECTIVEGENERATORMODE Return the live generator-ablation mode.
            mode = Algorithm.generatorMode;
        end

        function mode = effectiveScoutMode(Algorithm)
        %EFFECTIVESCOUTMODE Return the live scout-mode switch.
            mode = Algorithm.scoutMode;
        end

        function mode = effectiveMetricsMode(Algorithm)
        %EFFECTIVEMETRICSMODE Return the live metrics-recording switch.
            mode = Algorithm.metricsMode;
        end

        function mode = effectiveGuideMode(Algorithm)
        %EFFECTIVEGUIDEMODE Return the live guide-study switch.
            mode = Algorithm.guideMode;
        end

        function share = effectiveGuideShare(Algorithm)
        %EFFECTIVEGUIDESHARE Return the live guided-offspring share.
            share = Algorithm.guideShare;
        end

        function window = effectiveGuideWindow(Algorithm)
        %EFFECTIVEGUIDEWINDOW Return the live guide-supply window.
            window = Algorithm.guideWindow;
        end

        function carve = effectiveGuideCarve(Algorithm)
        %EFFECTIVEGUIDECARVE Return the live guided-slot carving rule.
            carve = Algorithm.guideCarve;
        end

        function window = effectiveBlsWindow(Algorithm)
        %EFFECTIVEBLSWINDOW Return the live boundary-search window.
            window = Algorithm.blsWindow;
        end

        function feed = effectiveBlsFeed(Algorithm)
        %EFFECTIVEBLSFEED Return the live BLS-to-BMem feed switch.
            feed = Algorithm.blsFeed;
        end

        function decs = effectiveInitDecs(Algorithm)
        %EFFECTIVEINITDECS Return the injected initial population, if any.
            decs = Algorithm.initDecs;
        end

        function Mx = collectedScoutMetrics(Algorithm)
        %COLLECTEDSCOUTMETRICS Return counters recorded when metricsMode=on.
            Mx = Algorithm.scoutMetricsStore;
        end

        function main(Algorithm,Problem)
            configurePlatEMOUtilityPath();

            %% Parameter setting
            Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
            [nGen,zDim,ganIter,ganMiniBatch,nCritic, ...
                minGANTrainCount,sampleSigma] = Algorithm.ParameterSet( ...
                Defaults.nGen,Defaults.zDim,Defaults.ganIter, ...
                Defaults.ganMiniBatch,Defaults.nCritic, ...
                Defaults.minGANTrainCount,Defaults.sampleSigma);
            Config = Defaults;
            Config.nGen = nGen;
            Config.zDim = zDim;
            Config.ganIter = ganIter;
            Config.ganMiniBatch = ganMiniBatch;
            Config.nCritic = nCritic;
            Config.minGANTrainCount = minGANTrainCount;
            Config.sampleSigma = sampleSigma;

            %% Optimization
            Algorithm.runRegionGAN(Problem,Config);
        end
    end

    methods(Static)
        function Defaults = mainlineDefaults()
        %MAINLINEDEFAULTS Return public defaults and fixed mainline constants.
            Defaults = struct( ...
                'nGen',20, ...
                'zDim',6, ...
                'ganIter',100, ...
                'ganMiniBatch',32, ...
                'ganLrD',1e-4, ...
                'ganLrG',1e-4, ...
                'frontDepth',2, ...
                'pairNeighborRefRadius',2, ...
                'refDivisor',2, ...
                'minBoundaryLength',2, ...
                'gpLambda',10, ...
                'nCritic',4, ...
                'maxAnchorsPerRef',5, ...
                'minGANTrainCount',32, ...
                'sampleSigma',0.3, ...
                'ganStopFraction',0.5, ...
                'generatorHidden',[32 32], ...
                'criticHidden',[32 32]);
        end
    end

    methods(Access = private)
        function runRegionGAN(Algorithm,Problem,Config)
        %RUNREGIONGAN Execute the two-population anchor-guided search.
            nGen = max(0,round(double(Config.nGen)));
            refDivisor = max(1,round(double(Config.refDivisor)));
            minBoundaryLength = max(1,round(double( ...
                Config.minBoundaryLength)));
            minGANTrainCount = max(1,round(double( ...
                Config.minGANTrainCount)));
            ganFELimit = Config.ganStopFraction*Problem.maxFE;

            %% Generate reference vectors and initialize state
            [W,~] = UniformPoint( ...
                max(2,round(Problem.N/refDivisor)),Problem.M);
            MemOptions = struct( ...
                'frontDepth',max(1,round(double(Config.frontDepth))), ...
                'pairNeighborRefRadius',max(0,round(double( ...
                    Config.pairNeighborRefRadius))), ...
                'minBoundaryLength',minBoundaryLength, ...
                'maxAnchorsPerRef',max(1,round(double( ...
                    Config.maxAnchorsPerRef))));
            GANOptions = regionGANOptions(Config,minGANTrainCount);

            if isempty(Algorithm.initDecs)
                Population1 = Problem.Initialization();
                Population2 = Problem.Initialization();
            else
                if ~isequal(size(Algorithm.initDecs), ...
                        [Problem.N,Problem.D])
                    error('CBSRegionGAN:BadInitDecsSize', ...
                        'initDecs must be a %d-by-%d matrix.', ...
                        Problem.N,Problem.D);
                end
                Population1 = Problem.Evaluation(Algorithm.initDecs);
                Population2 = Problem.Evaluation(Algorithm.initDecs);
            end
            Fitness1 = CalFitness_CBS( ...
                Population1.objs,Population1.cons);
            Fitness2 = CalFitness_CBS(Population2.objs);
            BMem = [];
            GAN = [];

            scoutQueriesOn = any(Algorithm.scoutMode == ...
                ["on","noprotect"]);
            if scoutQueriesOn
                queryAlloc = "scout";
            elseif Algorithm.scoutMode == "halffrontier"
                queryAlloc = "half";
            else
                queryAlloc = "legacy";
            end
            protectOn = any(Algorithm.scoutMode == ...
                ["on","nofrontier","halffrontier"]);
            datasetMode = "anchor";
            if any(Algorithm.generatorMode == ["jump","jumptrivial"])
                datasetMode = "landing";
            end
            recordOn = Algorithm.metricsMode == "on";
            useGuides = Algorithm.guideMode == "on";
            mixReproduction = any(Algorithm.guideMode == ["on","mix"]);
            if useGuides
                queryAlloc = "wide";
            end
            guideFELimit = ganFELimit;
            if useGuides && Algorithm.guideWindow == "full"
                guideFELimit = Problem.maxFE;
            end
            GuideBufDecs = zeros(0,Problem.D);
            GuideBufRefs = zeros(0,1);
            GuideBufTypes = zeros(0,1);
            halfWindow = ganFELimit/2;
            Mx = emptyScoutMetrics();
            Hood = [];
            if recordOn
                Hood = referenceHoods(W);
            end
            scoutMask1 = false(1,numel(Population1));

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                feStart = Problem.FE;
                bucket = 1 + double(feStart >= halfWindow);
                inWindow = feStart < ganFELimit;
                remainingFE = max(0,Problem.maxFE-Problem.FE);
                deBudget = min(2*Problem.N,remainingFE);
                count1 = min(Problem.N,ceil(deBudget/2));
                count2 = min(Problem.N,floor(deBudget/2));
                if mixReproduction
                    if Problem.FE >= guideFELimit
                        GuideBufDecs = zeros(0,Problem.D);
                        GuideBufRefs = zeros(0,1);
                        GuideBufTypes = zeros(0,1);
                    end
                    [Offspring1,Info1] = generateGuidedMixOffspring( ...
                        Problem,Population1,Fitness1,count1, ...
                        GuideBufDecs,GuideBufRefs,GuideBufTypes,W, ...
                        Algorithm.guideShare,Algorithm.guideCarve);
                else
                    [Offspring1,Info1] = generateRegionOffspring( ...
                        Problem,Population1,Fitness1,count1, ...
                        Algorithm.operatorMode);
                end
                Offspring2 = generateRegionOffspring(Problem, ...
                    Population2,Fitness2,count2,Algorithm.operatorMode);

                % Active boundary observation: the module's calibration
                % refiner contracts feasible-infeasible intervals on the
                % feasibility oracle; its evaluated candidates compete in
                % the ordinary selection and join the same-generation
                % boundary-memory harvest below.
                CalibrationCandidates = Offspring1([]);
                if Algorithm.boundarySearch == "on" && ...
                        (Algorithm.blsWindow == "full" || ...
                        Problem.FE >= ganFELimit)
                    remainingFE = max(0,Problem.maxFE-Problem.FE);
                    calibrationBudget = min(20,remainingFE);
                    if calibrationBudget > 0
                        CalibrationCandidates = ...
                            RefineBoundaryObservations_RC( ...
                            Problem,Population1,Population2, ...
                            calibrationBudget);
                    end
                end

                OffspringG = Offspring1([]);
                scoutTypes = zeros(0,1);
                scoutTargets = zeros(0,1);
                ganActiveLimit = ganFELimit;
                if useGuides
                    ganActiveLimit = guideFELimit;
                end
                if Problem.FE < ganActiveLimit
                    % Update the boundary memory before training so every
                    % evaluated offspring can contribute to the current
                    % event. After the configured stop point the memory,
                    % training, and sampling are all skipped and the saved
                    % budget flows back into the DE loop.
                    HarvestOffspring1 = Offspring1;
                    if Algorithm.blsFeed == "on" && ...
                            ~isempty(CalibrationCandidates)
                        HarvestOffspring1 = [Offspring1, ...
                            CalibrationCandidates];
                    end
                    BMem = UpdateBoundaryMemory_RC(BMem,Population1, ...
                        HarvestOffspring1,Population2,Offspring2,W, ...
                        MemOptions);
                    remainingFE = max(0,Problem.maxFE-Problem.FE);
                    eventBudget = min(nGen,remainingFE);
                    if ~isempty(BMem) && eventBudget > 0
                        [TrainX,TrainC,QueryRefs] = ...
                            BuildBoundaryDataset_RC(BMem,W,Problem, ...
                            datasetMode);
                        eligible = size(TrainX,1) >= ...
                            max(minBoundaryLength,minGANTrainCount) && ...
                            ~isempty(QueryRefs);
                        if eligible
                            [SampleC,SampleRefs,SampleTypes] = ...
                                RunRegionGAN_RC('regionquerysamples', ...
                                QueryRefs,W,eventBudget,queryAlloc);
                            if any(Algorithm.generatorMode == ...
                                    ["copynoise","jumptrivial"])
                                RawDec = copyNoiseSample(Problem, ...
                                    TrainX,TrainC,SampleC,W,SampleRefs);
                            else
                                [GAN,RawDec] = RunRegionGAN_RC( ...
                                    'trainandsample', ...
                                    GAN,TrainX,TrainC,SampleC,Problem, ...
                                    GANOptions);
                            end
                            remainingFE = max(0,Problem.maxFE-Problem.FE);
                            if size(RawDec,1) > remainingFE
                                RawDec = RawDec(1:remainingFE,:);
                            end
                            if ~isempty(RawDec) && useGuides
                                % Guides are never evaluated: they only
                                % steer the next generations' guided DE.
                                GuideBufDecs = RawDec;
                                GuideBufRefs = reshape(SampleRefs( ...
                                    1:size(RawDec,1)),[],1);
                                GuideBufTypes = reshape(SampleTypes( ...
                                    1:size(RawDec,1)),[],1);
                                if recordOn
                                    b = max(1,min(2,bucket));
                                    Mx.guideEvents(b) = ...
                                        Mx.guideEvents(b) + 1;
                                end
                            elseif ~isempty(RawDec)
                                OffspringG = Problem.Evaluation(RawDec);
                                scoutTypes = SampleTypes( ...
                                    1:numel(OffspringG));
                                scoutTargets = SampleRefs( ...
                                    1:numel(OffspringG));
                            end
                        end
                    end
                end

                ParentObjs1 = [];
                ParentCons1 = [];
                if recordOn && inWindow
                    ParentObjs1 = Population1.objs;
                    ParentCons1 = Population1.cons;
                end

                Union1 = [Population1,Population2,Offspring1, ...
                    Offspring2,OffspringG,CalibrationCandidates];
                Union2 = Union1;
                o1Start = numel(Population1) + numel(Population2);
                o1Idx = o1Start + (1:numel(Offspring1));
                gStart = o1Start + numel(Offspring1) + numel(Offspring2);
                gIdx = gStart + (1:numel(OffspringG));
                protectIdx = [];
                if protectOn && ~isempty(gIdx)
                    protectIdx = gIdx(1:min(numel(gIdx),20));
                end
                [NewPop1,NewFit1,sel1,uFit1] = ...
                    EnvironmentalSelection_CBS( ...
                    Union1,Problem.N,true,protectIdx);
                [Population2,Fitness2] = EnvironmentalSelection_CBS( ...
                    Union2,Problem.N,false);

                if recordOn && inWindow
                    Mx = tallyScoutMetrics(Mx,bucket,W,Hood, ...
                        ParentObjs1,ParentCons1,Offspring1,Info1, ...
                        scoutMask1,OffspringG,scoutTypes,scoutTargets, ...
                        o1Idx,gIdx,protectIdx,sel1,uFit1);
                end
                scoutMask1 = ismember(sel1,gIdx);
                Population1 = NewPop1;
                Fitness1 = NewFit1;
                if recordOn
                    feas1 = sum(max(0,Population1.cons),2) <= 0;
                    coverage = 0;
                    if any(feas1)
                        coverage = numel(unique(assignRefsLocal( ...
                            Population1(feas1).objs,W)));
                    end
                    Mx.covFE(end+1,1) = Problem.FE;
                    Mx.covCount(end+1,1) = coverage;
                    Algorithm.scoutMetricsStore = Mx;
                end
            end
        end
    end
end

function state = readBoundarySearch(argumentList)
%READBOUNDARYSEARCH Read the optional post-stop line-search switch.

    state = "on";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'boundarySearch'),names),1,'last');
    if isempty(index)
        return;
    end
    state = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(state) && ismember(state,["off","on"]))
        error('CBSRegionGAN:BadBoundarySearch', ...
            'boundarySearch must be off or on.');
    end
end

function mode = readGeneratorMode(argumentList)
%READGENERATORMODE Read the optional generator-ablation switch.

    mode = "wgan";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'generatorMode'),names),1,'last');
    if isempty(index)
        return;
    end
    mode = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(mode) && ismember(mode, ...
            ["wgan","copynoise","jump","jumptrivial"]))
        error('CBSRegionGAN:BadGeneratorMode', ...
            'generatorMode must be wgan, copynoise, jump, or jumptrivial.');
    end
end

function RawDec = copyNoiseSample(Problem,TrainX,TrainC,SampleC,W, ...
        SampleRefs)
%COPYNOISESAMPLE Ablation generator: anchor row plus Gaussian noise.
%   Keeps the exact evaluation budget, eligibility gate, and one-sixth
%   query allocation of the mainline CGAN, but replaces adversarial
%   learning by drawing a uniform training row of the queried reference
%   (duplicate rows keep their weight) and perturbing it with sigma=0.05
%   Gaussian noise in the [-1,1]-scaled decision space. Frontier
%   references without training rows fall back to the full training set.

    lower = double(Problem.lower);
    upper = double(Problem.upper);
    span = upper - lower;
    span(span <= eps) = 1;
    slotCount = size(SampleC,1);
    D = size(TrainX,2);
    RawDec = zeros(slotCount,D);
    for slot = 1 : slotCount
        rows = find(ismember(TrainC,W(SampleRefs(slot),:),'rows'));
        if isempty(rows)
            rows = (1:size(TrainX,1))';
        end
        pick = rows(randi(numel(rows)));
        scaled = 2*(double(TrainX(pick,:))-lower)./span - 1;
        scaled = scaled + 0.05*randn(1,D);
        scaled = max(-1,min(1,scaled));
        RawDec(slot,:) = lower + (scaled+1).*span/2;
    end
end

function mode = readScoutMode(argumentList)
%READSCOUTMODE Read the optional scout-study switch.
%   The mainline default is "nofrontier" (2026-07-19 decision): CGAN
%   offspring receive guaranteed P1 slots instead of competing in the
%   environmental selection, combined with the validated legacy query
%   allocation. "on" adds frontier-majority two-hop queries, "noprotect"
%   keeps only the queries, "halffrontier" uses an even one-hop origin
%   split, and "off" (the 2026-07-20 mainline default) leaves the
%   selection untouched for the guide mechanism.

    mode = "off";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'scoutMode'),names),1,'last');
    if isempty(index)
        return;
    end
    mode = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(mode) && ismember(mode, ...
            ["off","on","noprotect","nofrontier","halffrontier"]))
        error('CBSRegionGAN:BadScoutMode', ...
            ['scoutMode must be off, on, noprotect, nofrontier, ' ...
            'or halffrontier.']);
    end
end

function mode = readGuideMode(argumentList)
%READGUIDEMODE Read the optional guide-study switch.
%   "on" turns the CGAN into an unevaluated guide supplier: every event
%   caches nGen guide decision vectors (6:14 populated-to-empty queries)
%   and 30 of P1's offspring step from the nearest feasible parent toward
%   a guide. Guides never consume evaluations, never enter the union, and
%   are never protected.

    mode = "on";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'guideMode'),names),1,'last');
    if isempty(index)
        return;
    end
    mode = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(mode) && ismember(mode,["off","on","mix"]))
        error('CBSRegionGAN:BadGuideMode', ...
            'guideMode must be off, on, or mix.');
    end
end

function share = readGuideShare(argumentList)
%READGUIDESHARE Read the optional guided-offspring share.

    share = 0.2;
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'guideShare'),names),1,'last');
    if isempty(index)
        return;
    end
    share = double(argumentList{2*index});
    if ~(isscalar(share) && isfinite(share) && share > 0 && share <= 0.5)
        error('CBSRegionGAN:BadGuideShare', ...
            'guideShare must lie in (0, 0.5].');
    end
end

function carve = readGuideCarve(argumentList)
%READGUIDECARVE Read the optional guided-slot carving rule.

    carve = "sym";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'guideCarve'),names),1,'last');
    if isempty(index)
        return;
    end
    carve = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(carve) && ismember(carve,["sym","de"]))
        error('CBSRegionGAN:BadGuideCarve', ...
            'guideCarve must be sym or de.');
    end
end

function window = readBlsWindow(argumentList)
%READBLSWINDOW Read the optional boundary-search window switch.
%   "late" is the validated mainline (second half only); "full" runs the
%   search throughout, letting its boundary iterates serve the module in
%   the CGAN-active window as well.

    window = "full";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'blsWindow'),names),1,'last');
    if isempty(index)
        return;
    end
    window = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(window) && ismember(window,["late","full"]))
        error('CBSRegionGAN:BadBlsWindow', ...
            'blsWindow must be late or full.');
    end
end

function feed = readBlsFeed(argumentList)
%READBLSFEED Read the optional BLS-to-BMem feed switch.

    feed = "on";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'blsFeed'),names),1,'last');
    if isempty(index)
        return;
    end
    feed = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(feed) && ismember(feed,["off","on"]))
        error('CBSRegionGAN:BadBlsFeed', ...
            'blsFeed must be off or on.');
    end
end

function decs = readInitDecs(argumentList)
%READINITDECS Read the optional initial-population injection matrix.
%   Empty (default) keeps the native random initialization. A finite
%   numeric matrix seeds BOTH populations with the given decision
%   vectors, so branched-continuation experiments can restart from a
%   saved population snapshot. The matrix must match Problem.N-by-D,
%   which is checked at initialization time.

    decs = [];
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'initDecs'),names),1,'last');
    if isempty(index)
        return;
    end
    decs = argumentList{2*index};
    if ~(isnumeric(decs) && ismatrix(decs) && ~isempty(decs) && ...
            all(isfinite(decs(:))))
        error('CBSRegionGAN:BadInitDecs', ...
            'initDecs must be a nonempty finite numeric matrix.');
    end
    decs = double(decs);
end

function window = readGuideWindow(argumentList)
%READGUIDEWINDOW Read the optional guide-supply window.

    window = "half";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'guideWindow'),names),1,'last');
    if isempty(index)
        return;
    end
    window = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(window) && ismember(window,["half","full"]))
        error('CBSRegionGAN:BadGuideWindow', ...
            'guideWindow must be half or full.');
    end
end

function [Offspring,Info] = generateGuidedMixOffspring( ...
        Problem,Population,Fitness,count,GuideDecs,GuideRefs, ...
        GuideTypes,W,guidedShare,guideCarve)
%GENERATEGUIDEDMIXOFFSPRING (1-s)/2 GA + (1-s)/2 plain DE + s guided DE.
%   Guided children step from the nearest feasible parent toward an
%   unevaluated CGAN guide: child = a + F*(g - a), F cycling over
%   {0.4, 0.65, 0.85}, built through the platform OperatorDE (CR=1) so
%   crossover and polynomial mutation follow the validated conventions.
%   Without guides or feasible parents the guided slots fall back to the
%   plain DE path, so the mechanism degrades to the S2 mainline.

    count = max(0,min(numel(Population),round(double(count))));
    if count == 0
        Offspring = Population([]);
        Info = struct('deBase',nan(0,1),'guided',false(0,1), ...
            'targetRef',nan(0,1),'guideF',nan(0,1),'guideType',nan(0,1));
        return;
    end
    if nargin < 10 || isempty(guideCarve)
        guideCarve = "sym";
    end
    if string(guideCarve) == "de"
        % GA keeps its validated half; guided slots come from DE only.
        gaCount = min(count,round(0.5*count));
        guidedCount = min(count-gaCount, ...
            round(double(guidedShare)*count));
        plainCount = count - gaCount - guidedCount;
    else
        plainShare = (1-double(guidedShare))/2;
        gaCount = min(count,round(plainShare*count));
        plainCount = min(count-gaCount,round(plainShare*count));
        guidedCount = count - gaCount - plainCount;
    end

    if gaCount > 0
        Offspring = gaHalfOffspring(Problem,Population,Fitness,gaCount);
    else
        Offspring = Population([]);
    end
    deBase = nan(gaCount,1);
    guidedFlag = false(gaCount,1);
    targetRef = nan(gaCount,1);
    guideF = nan(gaCount,1);
    guideType = nan(gaCount,1);
    if plainCount > 0
        [PlainOffspring,plainIdx] = classicDEOffspring( ...
            Problem,Population,Fitness,plainCount);
        Offspring = [Offspring,PlainOffspring];
        deBase = [deBase;reshape(plainIdx,[],1)];
        guidedFlag = [guidedFlag;false(plainCount,1)];
        targetRef = [targetRef;nan(plainCount,1)];
        guideF = [guideF;nan(plainCount,1)];
        guideType = [guideType;nan(plainCount,1)];
    end
    feasible = sum(max(0,Population.cons),2) <= 0;
    if guidedCount > 0 && (isempty(GuideDecs) || ~any(feasible))
        [FallbackOffspring,fallbackIdx] = classicDEOffspring( ...
            Problem,Population,Fitness,guidedCount);
        Offspring = [Offspring,FallbackOffspring];
        deBase = [deBase;reshape(fallbackIdx,[],1)];
        guidedFlag = [guidedFlag;false(guidedCount,1)];
        targetRef = [targetRef;nan(guidedCount,1)];
        guideF = [guideF;nan(guidedCount,1)];
        guideType = [guideType;nan(guidedCount,1)];
        guidedCount = 0;
    end
    if guidedCount > 0
        FeasiblePop = Population(feasible);
        feasFitness = reshape(Fitness(feasible),[],1);
        feasRefs = assignRefsLocal(FeasiblePop.objs,W);
        FeasDecs = FeasiblePop.decs;
        nGuides = size(GuideDecs,1);
        parentRow = zeros(nGuides,1);
        for gi = 1 : nGuides
            distance = sqrt(sum((W(feasRefs,:) - ...
                W(GuideRefs(gi),:)).^2,2));
            ties = find(distance <= min(distance) + 1e-12);
            [~,k] = min(feasFitness(ties));
            parentRow(gi) = ties(k);
        end
        ladder = [0.4,0.65,0.85];
        childGuide = mod((0:guidedCount-1)',nGuides) + 1;
        childF = reshape(ladder(mod(0:guidedCount-1, ...
            numel(ladder))+1),[],1);
        ChildDecs = zeros(guidedCount,Problem.D);
        for b = 1 : numel(ladder)
            rows = find(childF == ladder(b));
            if isempty(rows)
                continue;
            end
            A = FeasDecs(parentRow(childGuide(rows)),:);
            G = GuideDecs(childGuide(rows),:);
            ChildDecs(rows,:) = OperatorDE(Problem,A,G,A, ...
                {1,ladder(b),1,20});
        end
        GuidedOffspring = Problem.Evaluation(ChildDecs);
        Offspring = [Offspring,GuidedOffspring];
        deBase = [deBase;nan(guidedCount,1)];
        guidedFlag = [guidedFlag;true(guidedCount,1)];
        targetRef = [targetRef; ...
            reshape(GuideRefs(childGuide),[],1)];
        guideF = [guideF;childF];
        if isempty(GuideTypes)
            guideType = [guideType;nan(guidedCount,1)];
        else
            guideType = [guideType; ...
                reshape(GuideTypes(childGuide),[],1)];
        end
    end
    Info = struct('deBase',deBase,'guided',guidedFlag, ...
        'targetRef',targetRef,'guideF',guideF,'guideType',guideType);
end

function mode = readMetricsMode(argumentList)
%READMETRICSMODE Read the optional metrics-recording switch.

    mode = "off";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'metricsMode'),names),1,'last');
    if isempty(index)
        return;
    end
    mode = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(mode) && ismember(mode,["off","on"]))
        error('CBSRegionGAN:BadMetricsMode', ...
            'metricsMode must be off or on.');
    end
end

function Mx = emptyScoutMetrics()
%EMPTYSCOUTMETRICS Initialize the two-window scout study counters.
%   Row 1 covers the first half of the CGAN-active window and row 2 the
%   second half (for maxFE=200k these are 0-50k and 50k-100k FE).

    Z = zeros(2,1);
    Mx = struct( ...
        'ganEvents',Z, ...
        'ganPopGen',Z,'ganPopFeas',Z,'ganPopND',Z,'ganPopP1',Z, ...
        'ganFrontGen',Z,'ganFrontFeas',Z,'ganFrontND',Z, ...
        'ganFrontP1',Z,'ganFrontEmpty',Z,'ganHop2Gen',Z, ...
        'ctrlHit',Z,'ctrlTot',Z, ...
        'protUsed',Z, ...
        'guideEvents',Z, ...
        'gdGen',Z,'gdFeas',Z,'gdND',Z,'gdP1',Z,'gdEmpty',Z, ...
        'gdFGen',zeros(2,3),'gdFFeas',zeros(2,3),'gdFP1',zeros(2,3), ...
        'gdTGen',zeros(2,3),'gdTFeas',zeros(2,3),'gdTP1',zeros(2,3), ...
        'scGen',Z,'scFeas',Z,'scP1',Z,'scEmpty',Z, ...
        'o1Gen',Z,'o1Feas',Z,'o1P1',Z, ...
        'covFE',zeros(0,1),'covCount',zeros(0,1));
end

function Hood = referenceHoods(W)
%REFERENCEHOODS One-hop neighborhoods (self plus two nearest references).

    n = size(W,1);
    width = min(3,n);
    Hood = zeros(n,width);
    for r = 1 : n
        distance = sqrt(sum((W-W(r,:)).^2,2));
        [~,order] = sort(distance,'ascend');
        Hood(r,:) = reshape(order(1:width),1,[]);
    end
end

function Ref = assignRefsLocal(Y,W)
%ASSIGNREFSLOCAL Batch min-max normalization plus reference assignment.
%   Mirrors the UpdateBoundaryMemory_RC assignment semantics for metrics
%   only; deterministic and RNG-free.

    n = size(Y,1);
    minimum = min(Y,[],1);
    span = max(Y,[],1)-minimum;
    span(span <= eps) = 1;
    Yn = (Y-minimum)./span;
    Yn(~isfinite(Yn)) = 0;
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    NormY = sqrt(sum(Yn.^2,2));
    Yu = Yn./max(NormY,eps);
    [~,Ref] = max(Yu*Wn',[],2);
    zeroRows = NormY <= eps;
    if any(zeroRows)
        distance2 = max(0,sum(Yn(zeroRows,:).^2,2) + ...
            sum(W.^2,2)' - 2*(Yn(zeroRows,:)*W'));
        [~,Ref(zeroRows)] = min(distance2,[],2);
    end
    Ref = reshape(Ref,n,1);
end

function Mx = tallyScoutMetrics(Mx,bucket,W,Hood,ParentObjs1, ...
        ParentCons1,Offspring1,Info1,scoutMask1,OffspringG,scoutTypes, ...
        scoutTargets,o1Idx,gIdx,protectIdx,sel1,uFit1)
%TALLYSCOUTMETRICS Accumulate the per-window scout study counters.
%   Pure arithmetic over already-evaluated solutions; consumes no random
%   numbers, so recording cannot perturb any trajectory.

    b = max(1,min(2,bucket));
    O1Objs = Offspring1.objs;
    O1Cons = Offspring1.cons;
    nP = size(ParentObjs1,1);
    nO = size(O1Objs,1);
    if isempty(ParentCons1)
        ParentCons1 = zeros(nP,1);
    end
    if isempty(O1Cons)
        O1Cons = zeros(nO,1);
    end
    GObjs = zeros(0,size(ParentObjs1,2));
    GCons = zeros(0,1);
    if ~isempty(gIdx)
        GObjs = OffspringG.objs;
        GCons = OffspringG.cons;
        if isempty(GCons)
            GCons = zeros(size(GObjs,1),1);
        end
    end
    refsAll = assignRefsLocal([ParentObjs1;O1Objs;GObjs],W);
    parentRefs = refsAll(1:nP);
    o1Refs = refsAll(nP+1:nP+nO);
    gRefs = refsAll(nP+nO+1:end);
    parentFeasible = sum(max(0,ParentCons1),2) <= 0;
    populatedNow = unique(parentRefs(parentFeasible));

    Mx.protUsed(b) = Mx.protUsed(b) + numel(protectIdx);

    o1Feas = sum(max(0,O1Cons),2) <= 0;
    o1In = ismember(reshape(o1Idx,[],1),sel1(:));
    Mx.o1Gen(b) = Mx.o1Gen(b) + nO;
    Mx.o1Feas(b) = Mx.o1Feas(b) + sum(o1Feas);
    Mx.o1P1(b) = Mx.o1P1(b) + sum(o1In);

    db = Info1.deBase;
    scRows = false(nO,1);
    hasBase = ~isnan(db);
    if any(hasBase) && any(scoutMask1)
        scRows(hasBase) = reshape(scoutMask1(db(hasBase)),[],1);
    end
    if any(scRows)
        Mx.scGen(b) = Mx.scGen(b) + sum(scRows);
        Mx.scFeas(b) = Mx.scFeas(b) + sum(scRows & o1Feas);
        Mx.scP1(b) = Mx.scP1(b) + sum(scRows & o1In);
        Mx.scEmpty(b) = Mx.scEmpty(b) + ...
            sum(scRows & ~ismember(o1Refs,populatedNow));
    end

    if isfield(Info1,'guided') && any(Info1.guided)
        gd = reshape(Info1.guided,[],1);
        Mx.gdGen(b) = Mx.gdGen(b) + sum(gd);
        Mx.gdFeas(b) = Mx.gdFeas(b) + sum(gd & o1Feas);
        Mx.gdND(b) = Mx.gdND(b) + ...
            sum(reshape(uFit1(o1Idx(gd)) < 1,[],1));
        Mx.gdP1(b) = Mx.gdP1(b) + sum(gd & o1In);
        Mx.gdEmpty(b) = Mx.gdEmpty(b) + ...
            sum(gd & ~ismember(o1Refs,populatedNow));
        ladder = [0.4,0.65,0.85];
        gF = reshape(Info1.guideF,[],1);
        gT = reshape(Info1.guideType,[],1);
        for slot = 1 : 3
            fRows = gd & abs(gF-ladder(slot)) < 1e-9;
            Mx.gdFGen(b,slot) = Mx.gdFGen(b,slot) + sum(fRows);
            Mx.gdFFeas(b,slot) = Mx.gdFFeas(b,slot) + ...
                sum(fRows & o1Feas);
            Mx.gdFP1(b,slot) = Mx.gdFP1(b,slot) + sum(fRows & o1In);
            tRows = gd & gT == slot;
            Mx.gdTGen(b,slot) = Mx.gdTGen(b,slot) + sum(tRows);
            Mx.gdTFeas(b,slot) = Mx.gdTFeas(b,slot) + ...
                sum(tRows & o1Feas);
            Mx.gdTP1(b,slot) = Mx.gdTP1(b,slot) + sum(tRows & o1In);
        end
    end

    if ~isempty(gIdx)
        Mx.ganEvents(b) = Mx.ganEvents(b) + 1;
        gFeas = sum(max(0,GCons),2) <= 0;
        gIn = ismember(reshape(gIdx,[],1),sel1(:));
        gND = reshape(uFit1(gIdx) < 1,[],1);
        isFront = reshape(scoutTypes >= 2,[],1);
        landEmpty = ~ismember(gRefs,populatedNow);
        Mx.ganPopGen(b) = Mx.ganPopGen(b) + sum(~isFront);
        Mx.ganPopFeas(b) = Mx.ganPopFeas(b) + sum(~isFront & gFeas);
        Mx.ganPopND(b) = Mx.ganPopND(b) + sum(~isFront & gND);
        Mx.ganPopP1(b) = Mx.ganPopP1(b) + sum(~isFront & gIn);
        Mx.ganFrontGen(b) = Mx.ganFrontGen(b) + sum(isFront);
        Mx.ganFrontFeas(b) = Mx.ganFrontFeas(b) + sum(isFront & gFeas);
        Mx.ganFrontND(b) = Mx.ganFrontND(b) + sum(isFront & gND);
        Mx.ganFrontP1(b) = Mx.ganFrontP1(b) + sum(isFront & gIn);
        Mx.ganFrontEmpty(b) = Mx.ganFrontEmpty(b) + ...
            sum(isFront & landEmpty);
        Mx.ganHop2Gen(b) = Mx.ganHop2Gen(b) + sum(scoutTypes == 3);
        popRows = reshape(find(~isFront),1,[]);
        for t = popRows
            Mx.ctrlTot(b) = Mx.ctrlTot(b) + 1;
            if any(Hood(scoutTargets(t),:) == gRefs(t))
                Mx.ctrlHit(b) = Mx.ctrlHit(b) + 1;
            end
        end
    end
end

function mode = readOperatorMode(argumentList)
%READOPERATORMODE Read the optional engineering offspring-operator switch.

    mode = "ga_de_half";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'operatorMode'),names),1,'last');
    if isempty(index)
        return;
    end
    mode = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(mode) && ismember(mode,["de","imtcmo_de","ga_de_half"]))
        error('CBSRegionGAN:BadOperatorMode', ...
            'operatorMode must be de, imtcmo_de, or ga_de_half.');
    end
end

function [Offspring,Info] = generateRegionOffspring( ...
        Problem,Population,Fitness,count,mode)
%GENERATEREGIONOFFSPRING Budget-limited offspring under the selected mode.
%   The default "de" branch is the validated mainline path, byte-identical
%   to the previous generateRegionDEOffspring. The two engineering
%   alternatives keep exactly the same evaluation budget: "imtcmo_de" uses
%   half DE/rand/1 plus half DE/pbest/1 with per-solution random F and CR,
%   and "ga_de_half" replaces half of the offspring with platform SBX+PM.
%   INFO.deBase records, per offspring row, the population index of its DE
%   base vector (NaN for GA/pbest rows); it captures existing RNG draws
%   without adding any, so trajectories are unchanged.

    count = max(0,min(numel(Population),round(double(count))));
    if count == 0
        Offspring = Population([]);
        Info = struct('deBase',nan(0,1));
        return;
    end
    if mode == "imtcmo_de" && numel(Population) >= 5
        Offspring = randPBestDEOffspring(Problem,Population,Fitness,count);
        Info = struct('deBase',nan(numel(Offspring),1));
    elseif mode == "ga_de_half"
        gaCount = ceil(count/2);
        Offspring = gaHalfOffspring(Problem,Population,Fitness,gaCount);
        deBase = nan(numel(Offspring),1);
        if count > gaCount
            [DeOffspring,deIdx] = classicDEOffspring( ...
                Problem,Population,Fitness,count-gaCount);
            Offspring = [Offspring,DeOffspring];
            deBase = [deBase;reshape(deIdx,[],1)];
        end
        Info = struct('deBase',deBase);
    else
        [Offspring,deIdx] = classicDEOffspring( ...
            Problem,Population,Fitness,count);
        Info = struct('deBase',reshape(deIdx,[],1));
    end
end

function [Offspring,baseIdx] = classicDEOffspring( ...
        Problem,Population,Fitness,count)
%CLASSICDEOFFSPRING The validated mainline DE path (unchanged wiring).

    matingPool = platformTournamentSelection(2,2*count,Fitness);
    if count == numel(Population)
        base = Population;
        baseIdx = 1:count;
    else
        baseIdx = randperm(numel(Population),count);
        base = Population(baseIdx);
    end
    Offspring = OperatorDE(Problem,base, ...
        Population(matingPool(1:count)), ...
        Population(matingPool(count+1:end)));
end

function Offspring = gaHalfOffspring(Problem,Population,Fitness,count)
%GAHALFOFFSPRING SBX+PM offspring through the platform half operator.

    matingPool = platformTournamentSelection(2,2*count,Fitness);
    Offspring = OperatorGAhalf(Problem,Population(matingPool));
end

function Offspring = randPBestDEOffspring(Problem,Population,Fitness,count)
%RANDPBESTDEOFFSPRING Canonical DE/rand/1 plus DE/pbest/1 offspring.
%   The operator recipe (random F in {0.6,0.8,1.0} and CR in {0.1,0.2,1.0}
%   per solution, rand/1 plus pbest/1 halves with p = 0.1) is ported from
%   the bundled IMTCMO implementation by Kangjia Qiao. The rand/1 mates are
%   plain mutually distinct random triples; the IMTCMO-specific neighbor
%   pairing strategy is intentionally not adopted.

    n = numel(Population);
    randCount = ceil(count/2);
    bestCount = count - randCount;
    Decs = Population.decs;

    order = randperm(n);
    [r1,r2,~] = distinctRandomTriples(n,order);
    Rand1 = deVariation(Problem, ...
        Decs(order(1:randCount),:), ...
        Decs(r1(1:randCount),:) - Decs(r2(1:randCount),:));

    if bestCount > 0
        perm = randperm(n);
        [q1,q2,~] = distinctRandomTriples(n,perm);
        base = Decs(perm(1:bestCount),:);
        pNP = max(round(0.1*n),2);
        [~,indBest] = sort(reshape(Fitness,[],1),'ascend');
        randindex = max(1,ceil(rand(1,bestCount)*pNP));
        pbest = Decs(indBest(randindex),:);
        PBest1 = deVariation(Problem,base, ...
            pbest - base + Decs(q1(1:bestCount),:) - Decs(q2(1:bestCount),:));
    else
        PBest1 = zeros(0,size(Decs,2));
    end
    Offspring = Problem.Evaluation([Rand1;PBest1]);
end

function Offspring = deVariation(Problem,Base,Difference)
%DEVARIATION Random-parameter DE variation with polynomial mutation.
%   Per-row F and CR are drawn from {0.6,0.8,1.0} and {0.1,0.2,1.0} as in
%   the bundled IMTCMO operators; polynomial mutation follows the platform
%   convention with proM = 1 and disM = 20.

    [N,D] = size(Base);
    [proM,disM] = deal(1,20);
    Fm = [0.6,0.8,1.0];
    CRm = [0.1,0.2,1.0];
    F = Fm(randi(3,N,1))';
    F = F(:,ones(1,D));
    CR = CRm(randi(3,N,1))';
    Site = rand(N,D) < CR;
    Offspring = Base;
    Offspring(Site) = Offspring(Site) + F(Site).*Difference(Site);

    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site = rand(N,D) < proM/D;
    mu = rand(N,D);
    temp = Site & mu <= 0.5;
    Offspring = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).* ...
        ((2.*mu(temp)+(1-2.*mu(temp)).* ...
        (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))) ...
        .^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu > 0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).* ...
        (1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).* ...
        (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))) ...
        .^(disM+1)).^(1/(disM+1)));
end

function [r1,r2,r3] = distinctRandomTriples(n,r0)
%DISTINCTRANDOMTRIPLES Random indices distinct from r0 and one another.
%   Ported from the bundled IMTCMO gnR1R2R3 helper.

    count = numel(r0);
    r1 = floor(rand(1,count)*n)+1;
    for i = 1 : 999
        pos = r1 == r0;
        if ~any(pos); break; end
        r1(pos) = floor(rand(1,sum(pos))*n)+1;
    end
    r2 = floor(rand(1,count)*n)+1;
    for i = 1 : 999
        pos = (r2 == r1) | (r2 == r0);
        if ~any(pos); break; end
        r2(pos) = floor(rand(1,sum(pos))*n)+1;
    end
    r3 = floor(rand(1,count)*n)+1;
    for i = 1 : 999
        pos = (r3 == r1) | (r3 == r2) | (r3 == r0);
        if ~any(pos); break; end
        r3(pos) = floor(rand(1,sum(pos))*n)+1;
    end
end

function index = platformTournamentSelection(K,N,Fitness)
%PLATFORMTOURNAMENTSELECTION Use PlatEMO selection with a stable tie break.
%   The rank adapter preserves the established row-order tie behavior while
%   the stochastic tournament itself is delegated to TournamentSelection.

    [~,rank] = sortrows(reshape(Fitness,[],1));
    [~,rank] = sort(rank);
    index = TournamentSelection(K,N,rank);
end

function configurePlatEMOUtilityPath()
%CONFIGUREPLATEMOUTILITYPATH Give official utility functions precedence.
%   Another bundled algorithm contains a same-named local selection helper;
%   placing the platform utility folder first avoids accidental shadowing.

    algorithmRoot = fileparts(mfilename('fullpath'));
    algorithmsRoot = fileparts(fileparts(algorithmRoot));
    utilityRoot = fullfile(algorithmsRoot,'Utility functions');
    currentSelection = string(which('TournamentSelection'));
    if ~startsWith(currentSelection,string(utilityRoot) + filesep)
        addpath(utilityRoot,'-begin');
    end
end

function Options = regionGANOptions(Config,minTrainCount)
%REGIONGANOPTIONS Convert the validated mainline configuration to WGAN options.

    Options = struct( ...
        'zDim',max(1,round(double(Config.zDim))), ...
        'iter',max(0,round(double(Config.ganIter))), ...
        'miniBatch',max(1,round(double(Config.ganMiniBatch))), ...
        'lrD',double(Config.ganLrD), ...
        'lrG',double(Config.ganLrG), ...
        'sampleSigma',double(Config.sampleSigma), ...
        'gpLambda',max(0,double(Config.gpLambda)), ...
        'nCritic',max(1,round(double(Config.nCritic))), ...
        'generatorHidden',double(Config.generatorHidden(:)'), ...
        'criticHidden',double(Config.criticHidden(:)'), ...
        'minTrainCount',double(minTrainCount));
end
