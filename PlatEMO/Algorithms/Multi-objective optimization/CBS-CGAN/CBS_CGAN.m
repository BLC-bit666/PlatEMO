classdef CBS_CGAN < ALGORITHM
% <multi> <real> <constrained>
% Current Boundary Skeleton CGAN.
% Mainline: thin-boundary memory with adversarial CGAN, ganIter=50.
%
% trainGap --- 1 --- CGAN retraining interval in generations
% archiveGap --- 1 --- Boundary memory refresh interval in generations
% nGen --- 20 --- Maximum raw CGAN offspring evaluated per refresh
% zDim --- 2 --- Latent dimension for local boundary perturbation
% ganIter --- 50 --- CGAN training iterations per refresh
% ganMiniBatch --- 32 --- CGAN mini-batch size
% ganLrD --- 0.0001 --- Discriminator learning rate
% ganLrG --- 0.0002 --- Generator learning rate
% ganDPretrainIter --- 0 --- D-only pretraining iterations
% ganDSteps --- 1 --- D updates per CGAN iteration
% ganGSteps --- 1 --- G updates per CGAN iteration
% pairNeighborRefRadius --- 4 --- Reference-neighborhood radius for F/I pairing
% maxCandidatePairsPerRef --- 1 --- Candidate boundary pairs retained per reference
% maxCanonicalPairsPerRef --- follows maxCandidatePairsPerRef --- Canonical pairs retained per reference
% minBoundaryLength --- 2 --- Minimum support points for the current boundary
% queryPerCondition --- 1 --- Generated decisions per external query condition

    methods
        function main(Algorithm,Problem)
            [trainGap,archiveGap,nGen,zDim,ganIter,ganMiniBatch, ...
                ganLrD,ganLrG,ganDPretrainIter,ganDSteps,ganGSteps, ...
                pairNeighborRefRadius,maxCandidatePairsPerRef, ...
                minBoundaryLength,queryPerCondition] = ...
                Algorithm.ParameterSet( ...
                1,1,20,2,50,32,1e-4,2e-4,0,1,1,4,1,2,1);

            trainGap = max(1,round(double(trainGap)));
            archiveGap = max(1,round(double(archiveGap)));
            nGen = max(0,round(double(nGen)));
            zDim = max(0,round(double(zDim)));
            ganIter = max(0,round(double(ganIter)));
            ganMiniBatch = max(1,round(double(ganMiniBatch)));
            ganDPretrainIter = max(0,round(double(ganDPretrainIter)));
            ganDSteps = max(1,round(double(ganDSteps)));
            ganGSteps = max(1,round(double(ganGSteps)));
            pairNeighborRefRadius = max(0,round(double(pairNeighborRefRadius)));
            maxCandidatePairsPerRef = max(1,round(double(maxCandidatePairsPerRef)));
            minBoundaryLength = max(1,round(double(minBoundaryLength)));
            queryPerCondition = max(1,round(double(queryPerCondition)));
            minTrainCount = minBoundaryLength;

            [W,~] = UniformPoint(Problem.N,Problem.M);
            BoundaryOptions = struct( ...
                'pairNeighborRefRadius',pairNeighborRefRadius, ...
                'maxCandidatePairsPerRef',maxCandidatePairsPerRef, ...
                'maxCanonicalPairsPerRef',maxCandidatePairsPerRef, ...
                'minBoundaryLength',minBoundaryLength);
            BoundaryOptions = applyCBSBoundaryControlOptions(BoundaryOptions);
            DatasetOptions = struct( ...
                'pairNeighborRefRadius',pairNeighborRefRadius, ...
                'queryConditionBudget', ...
                    max(1,ceil(max(nGen,1)/queryPerCondition)));
            DatasetOptions.conditionMode = cbsConditionModeFromControl( ...
                "ref_y");
            [BoundaryOptions,DatasetOptions] = Algorithm.configureBoundaryBranch( ...
                BoundaryOptions,DatasetOptions);
            GANOptions = struct( ...
                'zDim',zDim, ...
                'iter',ganIter, ...
                'miniBatch',ganMiniBatch, ...
                'lrD',double(ganLrD), ...
                'lrG',double(ganLrG), ...
                'dPretrainIter',ganDPretrainIter, ...
                'dSteps',ganDSteps, ...
                'gSteps',ganGSteps, ...
                'sigma',0.05);
            GANOptions = applyCBSGANControlOptions(GANOptions);

            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            ConditionScale = initializeCBSConditionScale( ...
                [Population1,Population2],Problem);
            DatasetOptions.conditionScale = ConditionScale;
            Fitness1 = CalFitness_CBS(Population1.objs,Population1.cons);
            Fitness2 = CalFitness_CBS(Population2.objs);

            BMem = [];
            GAN = [];
            gen = 0;
            LastDiag = emptyCBSMainDiag();
            Algorithm.metric.cbs_cgan_last = LastDiag;
            Algorithm.metric.cbs_cgan_history = LastDiag([]);
            StageObserver = initCBSStageObserver(Algorithm,Problem);
            Algorithm.metric.cbs_cgan_stage_snapshots = ...
                StageObserver.snapshots;

            while Algorithm.NotTerminated(Population1)
                gen = gen + 1;

                MatingPool1 = TournamentSelection(2,2*Problem.N,Fitness1);
                MatingPool2 = TournamentSelection(2,2*Problem.N,Fitness2);
                Offspring1 = OperatorDE(Problem,Population1, ...
                    Population1(MatingPool1(1:end/2)), ...
                    Population1(MatingPool1(end/2+1:end)));
                Offspring2 = OperatorDE(Problem,Population2, ...
                    Population2(MatingPool2(1:end/2)), ...
                    Population2(MatingPool2(end/2+1:end)));

                if mod(gen,archiveGap) == 0
                    [BMem,BoundaryDiag] = Algorithm.updateBoundaryMemory( ...
                        BMem, ...
                        Population1,Offspring1,Population2,Offspring2, ...
                        W,BoundaryOptions);
                else
                    BoundaryDiag = boundaryDiagFromMemory(BMem);
                end

                OffspringG = Offspring1([]);
                LastDiag = emptyCBSMainDiag();
                LastDiag.generation = gen;
                LastDiag.condition_mode = string(DatasetOptions.conditionMode);
                LastDiag.bmem_count = BoundaryDiag.bmem_count;
                LastDiag.boundary_count = BoundaryDiag.boundary_count;
                LastDiag.bmem_ref_coverage = bmemRefCoverage(BMem,W);
                LastDiag.finite_gap_count = getStructFieldOrDefault( ...
                    BoundaryDiag,'finite_gap_count',0);
                LastDiag.inf_gap_count = getStructFieldOrDefault( ...
                    BoundaryDiag,'inf_gap_count',0);
                LastDiag.median_gap = getStructFieldOrDefault( ...
                    BoundaryDiag,'median_gap',NaN);
                LastDiag.max_gap = getStructFieldOrDefault( ...
                    BoundaryDiag,'max_gap',NaN);
                LastDiag = copyMetricFields(LastDiag,BoundaryDiag, ...
                    boundaryPipelineFieldNames(),0);
                DatasetInfo = emptyCBSDatasetInfo(Problem.M);
                QueryInfo = emptyCBSQueryInfo(Problem.M);

                if mod(gen,trainGap) == 0 && ~isempty(BMem) && nGen > 0
                    [TrainX,TrainC,QueryC,BMem,DatasetInfo] = ...
                        Algorithm.buildBoundaryDataset(BMem, ...
                        [Population1,Offspring1,Population2,Offspring2], ...
                        W,Problem,DatasetOptions);
                    LastDiag.train_count = size(TrainX,1);
                    LastDiag.query_count = size(QueryC,1);
                    LastDiag.condition_dim = size(TrainC,2);
                    LastDiag.condition_mode = string( ...
                        getStructFieldOrDefault(DatasetInfo, ...
                        'condition_mode',DatasetOptions.conditionMode));
                    LastDiag.dataset_bmem_input_count = ...
                        getStructFieldOrDefault(DatasetInfo, ...
                        'bmem_input_count',0);
                    LastDiag.dataset_valid_train_count = ...
                        getStructFieldOrDefault(DatasetInfo, ...
                        'valid_train_count',0);
                    LastDiag.dataset_invalid_train_count = ...
                        getStructFieldOrDefault(DatasetInfo, ...
                        'invalid_train_count',0);
                    LastDiag.dataset_query_count = ...
                        getStructFieldOrDefault(DatasetInfo,'query_count',0);
                    LastDiag.train_param_ratio = trainParameterRatio( ...
                        LastDiag.train_count,Problem.D,LastDiag.condition_dim, ...
                        zDim,GANOptions);
                    LastDiag.gan_sample_reuse = ganSampleReuse( ...
                        LastDiag.train_count,ganIter,ganMiniBatch, ...
                        ganDPretrainIter,ganDSteps,ganGSteps);
                    PairTrainStats = trainPairDistanceStats(DatasetInfo);
                    LastDiag.train_pair_dec_dist50 = PairTrainStats(1);
                    LastDiag.train_pair_dec_dist90 = PairTrainStats(2);
                    QuerySourceStats = querySourceCounts(DatasetInfo);
                    LastDiag.missing_ref_query_count = QuerySourceStats(1);
                    LastDiag.large_gap_query_count = QuerySourceStats(2);

                    if size(TrainX,1) >= minTrainCount && ...
                            ~isempty(TrainC) && ~isempty(QueryC)
                        TrainGANOptions = GANOptions;
                        GAN = BoundaryCGAN_CBS('train',GAN,TrainX,TrainC, ...
                            Problem,TrainGANOptions);
                        TrainSampleOptions = applyCBSSampleControlOptions( ...
                            GANOptions,size(TrainC,1));
                        [TrainRecDec,~] = BoundaryCGAN_CBS( ...
                            'samplebycondition',GAN,TrainC,1, ...
                            TrainSampleOptions);
                        TrainRecStats = trainReconstructionStats( ...
                            Problem,TrainRecDec,TrainX,DatasetInfo);
                        LastDiag.train_x_rec90 = TrainRecStats(1);
                        LastDiag.train_y_rec90 = TrainRecStats(2);
                        SampleGANOptions = applyCBSSampleControlOptions( ...
                            GANOptions,size(QueryC,1)*queryPerCondition);
                        LastDiag.sample_z_mode = sampleZModeFromOptions( ...
                            SampleGANOptions);
                        [RawDec,QueryInfo] = BoundaryCGAN_CBS( ...
                            'samplebycondition',GAN, ...
                            QueryC,queryPerCondition,SampleGANOptions);
                        if size(RawDec,1) > nGen
                            RawDec = RawDec(1:nGen,:);
                            QueryInfo = trimCBSQueryInfo(QueryInfo,nGen);
                        end
                        LastDiag.raw_generated_count = size(RawDec,1);
                        LastDiag.generated_per_train = safeRatio( ...
                            LastDiag.raw_generated_count,LastDiag.train_count);
                        if ~isempty(RawDec)
                            OffspringG = Problem.Evaluation(RawDec);
                            CVG = sum(max(0,OffspringG.cons),2);
                            LastDiag.feasible_generated_count = sum(CVG <= 0);
                            LastDiag.feasible_rate = mean(double(CVG <= 0));
                            Dist = generatedBoundaryDistances( ...
                                OffspringG.objs,BMem,DatasetInfo);
                            DistStats = summarizeFiniteDistances(Dist);
                            LastDiag.segment_distance_min = DistStats(1);
                            LastDiag.segment_distance_mean = DistStats(2);
                            LastDiag.segment_distance_max = DistStats(3);
                            LastDiag.boundary_dist50 = percentileFinite(Dist,50);
                            LastDiag.boundary_dist90 = percentileFinite(Dist,90);
                            LastDiag.query_width90 = generatedQueryWidth90( ...
                                OffspringG.objs,QueryInfo.query_index, ...
                                DatasetInfo);
                            LastDiag.segment_width90 = generatedSegmentWidth90( ...
                                OffspringG.objs,QueryInfo.query_index, ...
                                DatasetInfo,BMem);
                            LastDiag.segment_width90_ratio = ...
                                segmentWidthRatio( ...
                                LastDiag.segment_width90,BMem,DatasetInfo);
                            SideStats = generatedPairSideStats( ...
                                RawDec,QueryInfo.query_index,DatasetInfo);
                            LastDiag.side_rate = SideStats(1);
                            LastDiag.pair_margin50 = SideStats(2);
                            LastDiag.ref_cover = generatedRefCover( ...
                                CVG <= 0,QueryInfo.query_index, ...
                                DatasetInfo,BMem);
                            QueryObjDist = queryObjectiveDistances( ...
                                OffspringG.objs,QueryInfo.query_index, ...
                                DatasetInfo);
                            LastDiag.query_obj_dist50 = ...
                                percentileFinite(QueryObjDist,50);
                            LastDiag.query_obj_dist90 = ...
                                percentileFinite(QueryObjDist,90);
                            LastDiag.missing_ref_query_obj_dist90 = ...
                                percentileFinite(querySourceDistances( ...
                                QueryObjDist,QueryInfo.query_index, ...
                                DatasetInfo,"missing_ref"),90);
                            LastDiag.large_gap_query_obj_dist90 = ...
                                percentileFinite(querySourceDistances( ...
                                QueryObjDist,QueryInfo.query_index, ...
                                DatasetInfo,"large_gap"),90);
                        end
                        if cbsVisualDiagnosticsFromControl()
                            DatasetInfo.visualDiagnostics = makeCBSVisualDiagnostics( ...
                                GAN,TrainC,QueryC,Problem,GANOptions, ...
                                queryPerCondition,nGen);
                        end
                    end
                end

                LastDiag.bmem_count = countBMemNodes(BMem);
                LastDiag.boundary_count = countBMemBoundaries(BMem);
                LastDiag.bmem_ref_coverage = bmemRefCoverage(BMem,W);
                Algorithm.metric.cbs_cgan_last = LastDiag;
                Algorithm.metric.cbs_cgan_history(end+1) = LastDiag;
                [StageObserver,captured] = maybeCaptureCBSStages( ...
                    StageObserver,Algorithm,Problem,gen,BMem,OffspringG, ...
                    QueryInfo,DatasetInfo,LastDiag);
                if captured
                    Algorithm.metric.cbs_cgan_stage_snapshots = ...
                        StageObserver.snapshots;
                end

                UnionPopulation = [Population1,Population2,Offspring1,Offspring2,OffspringG];
                [Population1,Fitness1] = EnvironmentalSelection_CBS( ...
                    UnionPopulation,Problem.N,true);
                [Population2,Fitness2] = EnvironmentalSelection_CBS( ...
                    UnionPopulation,Problem.N,false);
            end
        end

        function [BoundaryOptions,DatasetOptions] = configureBoundaryBranch( ...
                Algorithm,BoundaryOptions,DatasetOptions) %#ok<INUSD>
        end

        function [BMem,Diag] = updateBoundaryMemory(Algorithm, ...
                PrevBMem,Population1,Offspring1,Population2,Offspring2, ...
                W,Options) %#ok<INUSL>
            [BMem,Diag] = UpdateBoundaryMemory_CBS(PrevBMem, ...
                Population1,Offspring1,Population2,Offspring2,W,Options);
        end

        function [TrainX,TrainC,QueryC,BMem,Info] = buildBoundaryDataset( ...
                Algorithm,BMem,Samples,W,Problem,Options) %#ok<INUSL>
            [TrainX,TrainC,QueryC,BMem,Info] = BuildBoundaryDataset_CBS( ...
                BMem,Samples,W,Problem,Options);
        end
    end
end

function Diag = emptyCBSMainDiag()
    Diag = struct( ...
        'generation',0, ...
        'condition_mode',"ref_y", ...
        'bmem_count',0, ...
        'boundary_count',0, ...
        'bmem_ref_coverage',NaN, ...
        'finite_gap_count',0, ...
        'inf_gap_count',0, ...
        'median_gap',NaN, ...
        'max_gap',NaN, ...
        'train_count',0, ...
        'query_count',0, ...
        'condition_dim',NaN, ...
        'train_param_ratio',NaN, ...
        'gan_sample_reuse',NaN, ...
        'train_pair_dec_dist50',NaN, ...
        'train_pair_dec_dist90',NaN, ...
        'train_x_rec90',NaN, ...
        'train_y_rec90',NaN, ...
        'sample_z_mode',"zero", ...
        'missing_ref_query_count',0, ...
        'large_gap_query_count',0, ...
        'raw_generated_count',0, ...
        'feasible_generated_count',0, ...
        'generated_per_train',NaN, ...
        'feasible_rate',NaN, ...
        'boundary_dist50',NaN, ...
        'boundary_dist90',NaN, ...
        'query_width90',NaN, ...
        'segment_width90',NaN, ...
        'segment_width90_ratio',NaN, ...
        'side_rate',NaN, ...
        'pair_margin50',NaN, ...
        'ref_cover',NaN, ...
        'query_obj_dist50',NaN, ...
        'query_obj_dist90',NaN, ...
        'missing_ref_query_obj_dist90',NaN, ...
        'large_gap_query_obj_dist90',NaN, ...
        'segment_distance_min',NaN, ...
        'segment_distance_mean',NaN, ...
        'segment_distance_max',NaN);
    Diag = addDefaultMetricFields(Diag,boundaryPipelineFieldNames(),0);
    Diag = addDefaultMetricFields(Diag,datasetPipelineFieldNames(),0);
end

function mode = cbsConditionModeFromControl(defaultMode)
    mode = string(defaultMode);
    if isappdata(0,'CBS_CGAN_ExperimentControl')
        Control = getappdata(0,'CBS_CGAN_ExperimentControl');
        if isstruct(Control) && isfield(Control,'conditionMode') && ...
                ~isempty(Control.conditionMode)
            mode = string(Control.conditionMode);
        end
    end
    mode = lower(strtrim(mode));
end

function Options = applyCBSGANControlOptions(Options)
    Control = cbsExperimentControl();
    if isempty(Control)
        return;
    end
    names = {'advWeight','trainZMode','trainMode', ...
        'generatorHidden','discriminatorHidden'};
    for i = 1 : numel(names)
        if isfield(Control,names{i}) && ~isempty(Control.(names{i}))
            Options.(names{i}) = Control.(names{i});
        end
    end
end

function Options = applyCBSBoundaryControlOptions(Options)
    Control = cbsExperimentControl();
    if isempty(Control)
        return;
    end
    names = {'boundaryTargetMode'};
    for i = 1 : numel(names)
        if isfield(Control,names{i}) && ~isempty(Control.(names{i}))
            Options.(names{i}) = Control.(names{i});
        end
    end
end

function Options = applyCBSSampleControlOptions(Options,n)
    Control = cbsExperimentControl();
    mode = "zero";
    if ~isempty(Control) && isfield(Control,'sampleZMode') && ...
            ~isempty(Control.sampleZMode)
        mode = string(Control.sampleZMode);
    end
    mode = lower(strtrim(mode));
    Options.sampleZMode = mode;
    switch mode
        case "zero"
            zDim = max(0,round(double(Options.zDim)));
            Options.sampleZ = zeros(max(0,round(double(n))),zDim);
        case "random"
            if isfield(Options,'sampleZ')
                Options = rmfield(Options,'sampleZ');
            end
        otherwise
            error('CBSCGAN:BadSampleZMode', ...
                'Unsupported sampleZMode: %s.',mode);
    end
end

function mode = sampleZModeFromOptions(Options)
    mode = "zero";
    if isstruct(Options) && isfield(Options,'sampleZMode') && ...
            ~isempty(Options.sampleZMode)
        mode = lower(strtrim(string(Options.sampleZMode)));
    elseif isstruct(Options) && isfield(Options,'sampleZ') && ...
            ~isempty(Options.sampleZ)
        mode = "fixed";
    end
end

function Control = cbsExperimentControl()
    if isappdata(0,'CBS_CGAN_ExperimentControl')
        Control = getappdata(0,'CBS_CGAN_ExperimentControl');
        if ~isstruct(Control)
            Control = [];
        end
    else
        Control = [];
    end
end

function value = getStructFieldOrDefault(S,name,defaultValue)
    if isstruct(S) && isfield(S,name)
        value = S.(name);
    else
        value = defaultValue;
    end
end

function S = copyMetricFields(S,Source,names,defaultValue)
    for i = 1 : numel(names)
        S.(names{i}) = getStructFieldOrDefault(Source,names{i},defaultValue);
    end
end

function S = addDefaultMetricFields(S,names,defaultValue)
    for i = 1 : numel(names)
        if ~isfield(S,names{i})
            S.(names{i}) = defaultValue;
        end
    end
end

function names = boundaryPipelineFieldNames()
    names = {'boundary_sample_current_count', ...
        'boundary_sample_prev_pair_count', ...
        'boundary_sample_merged_count', ...
        'boundary_sample_finite_count', ...
        'boundary_feasible_count', ...
        'boundary_infeasible_count', ...
        'boundary_main_feasible_count', ...
        'boundary_ref_with_main_feasible_count', ...
        'boundary_ref_with_neighbor_infeasible_count', ...
        'boundary_ref_pairable_count', ...
        'boundary_pair_distance_count', ...
        'boundary_pair_dominated_skip_count', ...
        'boundary_pair_appended_count', ...
        'boundary_candidate_raw_count', ...
        'boundary_candidate_finite_count', ...
        'boundary_candidate_pareto_count', ...
        'boundary_candidate_gap_count', ...
        'boundary_candidate_dedup_count', ...
        'boundary_candidate_final_count'};
end

function names = datasetPipelineFieldNames()
    names = {'dataset_bmem_input_count', ...
        'dataset_valid_train_count', ...
        'dataset_invalid_train_count', ...
        'dataset_query_count'};
end

function Diag = boundaryDiagFromMemory(BMem)
    Diag = struct( ...
        'bmem_count',countBMemNodes(BMem), ...
        'boundary_count',countBMemBoundaries(BMem), ...
        'finite_gap_count',countFiniteGaps(BMem), ...
        'inf_gap_count',countInfGaps(BMem), ...
        'median_gap',medianFiniteGap(BMem), ...
        'max_gap',maxFiniteGap(BMem));
end

function N = countBMemNodes(BMem)
    if isempty(BMem) || ~isfield(BMem,'y_b')
        N = 0;
    else
        N = size(BMem.y_b,1);
    end
end

function N = countBMemBoundaries(BMem)
    if isempty(BMem) || ~isfield(BMem,'y_b') || isempty(BMem.y_b)
        N = 0;
    elseif isfield(BMem,'chain') && ~isempty(BMem.chain)
        N = numel(unique(BMem.chain(:)'));
    else
        N = 1;
    end
end

function N = countFiniteGaps(BMem)
    if isempty(BMem) || ~isfield(BMem,'gap')
        N = 0;
    else
        N = sum(isfinite(BMem.gap));
    end
end

function N = countInfGaps(BMem)
    if isempty(BMem) || ~isfield(BMem,'gap')
        N = 0;
    else
        N = sum(isinf(BMem.gap));
    end
end

function value = medianFiniteGap(BMem)
    gaps = finiteGaps(BMem);
    if isempty(gaps)
        value = NaN;
    else
        value = median(gaps);
    end
end

function value = maxFiniteGap(BMem)
    gaps = finiteGaps(BMem);
    if isempty(gaps)
        value = NaN;
    else
        value = max(gaps);
    end
end

function gaps = finiteGaps(BMem)
    if isempty(BMem) || ~isfield(BMem,'gap')
        gaps = zeros(0,1);
    else
        gaps = BMem.gap(isfinite(BMem.gap));
    end
end

function Info = emptyCBSDatasetInfo(M)
    Info = struct( ...
        'objMin',zeros(1,M), ...
        'objSpan',ones(1,M), ...
        'trainObjs',zeros(0,M), ...
        'queryObjs',zeros(0,M), ...
        'trainConditions',zeros(0,0), ...
        'queryConditions',zeros(0,0), ...
        'trainXf',zeros(0,0), ...
        'trainXi',zeros(0,0), ...
        'queryMeta',emptyCBSQueryMeta(), ...
        'visualDiagnostics',emptyCBSVisualDiagnostics(M));
end

function Meta = emptyCBSQueryMeta()
    Meta = struct( ...
        'ref',zeros(0,1), ...
        'chain',zeros(0,1), ...
        'source_interval',zeros(0,2), ...
        'source_type',strings(0,1));
end

function Info = emptyCBSQueryInfo(M)
    Info = struct( ...
        'query_index',zeros(0,1), ...
        'condition',zeros(0,M), ...
        'z',zeros(0,0));
end

function Info = trimCBSQueryInfo(Info,N)
    N = max(0,min(round(double(N)),numel(Info.query_index)));
    Info.query_index = Info.query_index(1:N,:);
    if isfield(Info,'condition')
        Info.condition = Info.condition(1:N,:);
    end
    if isfield(Info,'z')
        Info.z = Info.z(1:N,:);
    end
end

function enabled = cbsVisualDiagnosticsFromControl()
    enabled = false;
    if isappdata(0,'CBS_CGAN_ExperimentControl')
        Control = getappdata(0,'CBS_CGAN_ExperimentControl');
        if isstruct(Control) && isfield(Control,'visualDiagnostics') && ...
                ~isempty(Control.visualDiagnostics)
            enabled = logical(Control.visualDiagnostics);
        end
    end
end

function Visual = emptyCBSVisualDiagnostics(M)
    Visual = struct( ...
        'query_zero_objs',zeros(0,M), ...
        'train_zero_objs',zeros(0,M));
end

function Visual = makeCBSVisualDiagnostics( ...
    GAN,TrainC,QueryC,Problem,GANOptions,queryPerCondition,nGen)
    Visual = emptyCBSVisualDiagnostics(Problem.M);
    if isempty(GAN) || ~isfield(GAN,'netG')
        return;
    end
    Visual.query_zero_objs = visualSampleObjs(GAN,QueryC,queryPerCondition, ...
        Problem,GANOptions,nGen,"zero");
    Visual.train_zero_objs = visualSampleObjs(GAN,TrainC,1,Problem, ...
        GANOptions,inf,"zero");
end

function Obj = visualSampleObjs(GAN,C,perCondition,Problem,GANOptions,maxRows,mode)
    Obj = zeros(0,Problem.M);
    if isempty(C)
        return;
    end
    perCondition = max(1,round(double(perCondition)));
    n = size(C,1)*perCondition;
    SampleOptions = GANOptions;
    switch string(mode)
        case "zero"
            SampleOptions.sampleZ = zeros(n,GAN.zDim);
        otherwise
            error('CBSCGAN:BadVisualSampleMode', ...
                'Unsupported visual sample mode: %s.',mode);
    end
    [Dec,~] = BoundaryCGAN_CBS('samplebycondition',GAN,C, ...
        perCondition,SampleOptions);
    if isfinite(maxRows) && size(Dec,1) > maxRows
        Dec = Dec(1:maxRows,:);
    end
    if ~isempty(Dec)
        Obj = EvaluateDecisions_CBS(Problem,Dec);
    end
end

function Observer = initCBSStageObserver(Algorithm,Problem)
    Observer = struct( ...
        'enabled',false, ...
        'targets',zeros(0,1), ...
        'next',1, ...
        'snapshots',emptyCBSStageSnapshot(Problem.M), ...
        'run',doubleOrNaN(Algorithm.run), ...
        'problem',string(class(Problem)));
    Observer.snapshots = Observer.snapshots([]);
    if isappdata(0,'CBS_CGAN_ExperimentControl')
        Control = getappdata(0,'CBS_CGAN_ExperimentControl');
        if isstruct(Control) && isfield(Control,'stageTargets')
            targets = double(Control.stageTargets(:));
            targets = unique(targets(isfinite(targets) & targets > 0), ...
                'stable');
            Observer.enabled = ~isempty(targets);
            Observer.targets = targets(:);
        end
    end
end

function [Observer,captured] = maybeCaptureCBSStages(Observer,Algorithm, ...
    Problem,gen,BMem,OffspringG,QueryInfo,DatasetInfo,Diag)
    captured = false;
    if ~Observer.enabled
        return;
    end
    while Observer.next <= numel(Observer.targets) && ...
            Problem.FE >= Observer.targets(Observer.next)
        targetFE = Observer.targets(Observer.next);
        Snapshot = makeCBSStageSnapshot(Observer,Algorithm,Problem,gen, ...
            targetFE,BMem,OffspringG,QueryInfo,DatasetInfo,Diag);
        Observer.snapshots(end+1) = Snapshot;
        Observer.next = Observer.next + 1;
        captured = true;
    end
end

function Snapshot = makeCBSStageSnapshot(Observer,Algorithm,Problem,gen, ...
    targetFE,BMem,OffspringG,QueryInfo,DatasetInfo,Diag)
    Snapshot = emptyCBSStageSnapshot(Problem.M);
    Snapshot.problem = string(class(Problem));
    Snapshot.run = doubleOrNaN(Algorithm.run);
    if ~isfinite(Snapshot.run)
        Snapshot.run = Observer.run;
    end
    Snapshot.target_FE = double(targetFE);
    Snapshot.actual_FE = double(Problem.FE);
    Snapshot.gen = double(gen);
    Snapshot.condition_mode = string(Diag.condition_mode);
    Snapshot.bmem_count = double(Diag.bmem_count);
    Snapshot.boundary_count = double(Diag.boundary_count);
    Snapshot.bmem_ref_coverage = double(Diag.bmem_ref_coverage);
    Snapshot.finite_gap_count = double(Diag.finite_gap_count);
    Snapshot.inf_gap_count = double(Diag.inf_gap_count);
    Snapshot.median_gap = double(Diag.median_gap);
    Snapshot.max_gap = double(Diag.max_gap);
    Snapshot = copyMetricFields(Snapshot,Diag,boundaryPipelineFieldNames(),0);
    Snapshot.train_count = double(Diag.train_count);
    Snapshot.query_count = double(Diag.query_count);
    Snapshot.condition_dim = double(Diag.condition_dim);
    Snapshot = copyMetricFields(Snapshot,Diag,datasetPipelineFieldNames(),0);
    Snapshot.train_param_ratio = double(Diag.train_param_ratio);
    Snapshot.gan_sample_reuse = double(Diag.gan_sample_reuse);
    Snapshot.train_pair_dec_dist50 = double(Diag.train_pair_dec_dist50);
    Snapshot.train_pair_dec_dist90 = double(Diag.train_pair_dec_dist90);
    Snapshot.train_x_rec90 = double(Diag.train_x_rec90);
    Snapshot.train_y_rec90 = double(Diag.train_y_rec90);
    Snapshot.sample_z_mode = string(Diag.sample_z_mode);
    Snapshot.missing_ref_query_count = double(Diag.missing_ref_query_count);
    Snapshot.large_gap_query_count = double(Diag.large_gap_query_count);
    Snapshot.raw_generated_count = double(Diag.raw_generated_count);
    Snapshot.feasible_generated_count = double(Diag.feasible_generated_count);
    Snapshot.generated_per_train = double(Diag.generated_per_train);
    Snapshot.feasible_rate = double(Diag.feasible_rate);
    Snapshot.boundary_dist50 = double(Diag.boundary_dist50);
    Snapshot.boundary_dist90 = double(Diag.boundary_dist90);
    Snapshot.query_width90 = double(Diag.query_width90);
    Snapshot.segment_width90 = double(Diag.segment_width90);
    Snapshot.segment_width90_ratio = double(Diag.segment_width90_ratio);
    Snapshot.side_rate = double(Diag.side_rate);
    Snapshot.pair_margin50 = double(Diag.pair_margin50);
    Snapshot.ref_cover = double(Diag.ref_cover);
    Snapshot.query_obj_dist50 = double(Diag.query_obj_dist50);
    Snapshot.query_obj_dist90 = double(Diag.query_obj_dist90);
    Snapshot.missing_ref_query_obj_dist90 = ...
        double(Diag.missing_ref_query_obj_dist90);
    Snapshot.large_gap_query_obj_dist90 = ...
        double(Diag.large_gap_query_obj_dist90);
    if isfield(DatasetInfo,'trainObjs') && ~isempty(DatasetInfo.trainObjs)
        Snapshot.train_objs = DatasetInfo.trainObjs;
    elseif ~isempty(BMem) && isfield(BMem,'y_b')
        Snapshot.train_objs = BMem.y_b;
    end
    if isfield(DatasetInfo,'trainConditions')
        Snapshot.train_condition = DatasetInfo.trainConditions;
    end
    if isfield(DatasetInfo,'trainRef')
        Snapshot.train_ref = DatasetInfo.trainRef;
    end
    if ~isempty(OffspringG)
        Snapshot.generated_objs = OffspringG.objs;
        Snapshot.generated_cons = OffspringG.cons;
        if isempty(OffspringG.cons)
            Snapshot.generated_feasible = true(numel(OffspringG),1);
        else
            Snapshot.generated_feasible = sum(max(0,OffspringG.cons),2) <= 0;
        end
    end
    Snapshot.query_index = QueryInfo.query_index(:);
    if isfield(QueryInfo,'condition')
        Snapshot.query_condition = QueryInfo.condition;
    end
    if isfield(DatasetInfo,'queryObjs')
        Snapshot.query_objs = DatasetInfo.queryObjs;
    end
    if isfield(DatasetInfo,'queryMeta') && ...
            isstruct(DatasetInfo.queryMeta)
        QueryMeta = DatasetInfo.queryMeta;
        if isfield(QueryMeta,'ref')
            Snapshot.query_ref = QueryMeta.ref;
        end
        if isfield(QueryMeta,'chain')
            Snapshot.query_chain = QueryMeta.chain;
        end
        if isfield(QueryMeta,'source_interval')
            Snapshot.query_source_interval = QueryMeta.source_interval;
        end
        if isfield(QueryMeta,'source_type')
            Snapshot.query_source_type = QueryMeta.source_type;
        end
    end
    Snapshot.obj_min = DatasetInfo.objMin;
    Snapshot.obj_span = DatasetInfo.objSpan;
    if isfield(DatasetInfo,'visualDiagnostics')
        Visual = DatasetInfo.visualDiagnostics;
        Snapshot.visual_query_zero_objs = Visual.query_zero_objs;
        Snapshot.visual_train_zero_objs = Visual.train_zero_objs;
    end
end

function Snapshot = emptyCBSStageSnapshot(M)
    Snapshot = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'target_FE',NaN, ...
        'actual_FE',NaN, ...
        'gen',NaN, ...
        'condition_mode',"ref_y", ...
        'bmem_count',0, ...
        'boundary_count',0, ...
        'bmem_ref_coverage',NaN, ...
        'finite_gap_count',0, ...
        'inf_gap_count',0, ...
        'median_gap',NaN, ...
        'max_gap',NaN, ...
        'train_count',0, ...
        'query_count',0, ...
        'condition_dim',NaN, ...
        'train_param_ratio',NaN, ...
        'gan_sample_reuse',NaN, ...
        'train_pair_dec_dist50',NaN, ...
        'train_pair_dec_dist90',NaN, ...
        'train_x_rec90',NaN, ...
        'train_y_rec90',NaN, ...
        'sample_z_mode',"zero", ...
        'missing_ref_query_count',0, ...
        'large_gap_query_count',0, ...
        'raw_generated_count',0, ...
        'feasible_generated_count',0, ...
        'generated_per_train',NaN, ...
        'feasible_rate',NaN, ...
        'boundary_dist50',NaN, ...
        'boundary_dist90',NaN, ...
        'query_width90',NaN, ...
        'segment_width90',NaN, ...
        'segment_width90_ratio',NaN, ...
        'side_rate',NaN, ...
        'pair_margin50',NaN, ...
        'ref_cover',NaN, ...
        'query_obj_dist50',NaN, ...
        'query_obj_dist90',NaN, ...
        'missing_ref_query_obj_dist90',NaN, ...
        'large_gap_query_obj_dist90',NaN, ...
        'train_objs',zeros(0,M), ...
        'train_condition',zeros(0,M), ...
        'train_ref',zeros(0,1), ...
        'generated_objs',zeros(0,M), ...
        'generated_cons',zeros(0,0), ...
        'generated_feasible',false(0,1), ...
        'query_index',zeros(0,1), ...
        'query_condition',zeros(0,M), ...
        'query_objs',zeros(0,M), ...
        'query_ref',zeros(0,1), ...
        'query_chain',zeros(0,1), ...
        'query_source_interval',zeros(0,2), ...
        'query_source_type',strings(0,1), ...
        'visual_query_zero_objs',zeros(0,M), ...
        'visual_train_zero_objs',zeros(0,M), ...
        'obj_min',zeros(1,M), ...
        'obj_span',ones(1,M));
    Snapshot = addDefaultMetricFields(Snapshot,boundaryPipelineFieldNames(),0);
    Snapshot = addDefaultMetricFields(Snapshot,datasetPipelineFieldNames(),0);
end

function value = doubleOrNaN(value)
    if isempty(value)
        value = NaN;
    else
        value = double(value);
    end
end

function Scale = initializeCBSConditionScale(Population,Problem)
    if isempty(Population)
        Obj = zeros(0,Problem.M);
    else
        Obj = Population.objs;
    end
    Obj = double(Obj);
    valid = all(isfinite(Obj),2);
    Obj = Obj(valid,:);
    if isempty(Obj)
        objMin = zeros(1,Problem.M);
        objSpan = ones(1,Problem.M);
    else
        objMin = min(Obj,[],1);
        objMax = max(Obj,[],1);
        objSpan = objMax - objMin;
        objSpan(objSpan <= eps) = 1;
    end
    Scale = struct('objMin',objMin,'objSpan',objSpan);
end

function Dist = generatedBoundaryDistances(GeneratedObj,BMem,DatasetInfo)
    if isempty(GeneratedObj) || isempty(BMem) || isempty(BMem.y_b)
        Dist = NaN(0,1);
        return;
    end
    Y = normalizeWithInfo(GeneratedObj,DatasetInfo.objMin,DatasetInfo.objSpan);
    BY = normalizeWithInfo(BMem.y_b,DatasetInfo.objMin,DatasetInfo.objSpan);
    Dist = inf(size(Y,1),1);
    if isfield(BMem,'chain') && ~isempty(BMem.chain)
        groups = unique(BMem.chain(:)');
    else
        groups = 1;
    end
    for g = groups
        if isfield(BMem,'chain') && ~isempty(BMem.chain)
            idx = find(BMem.chain(:)' == g);
        else
            idx = (1:size(BMem.y_b,1))';
        end
        [~,ord] = sort(BMem.ref(idx));
        idx = idx(ord);
        if numel(idx) < 2
            Dist = min(Dist,min(pointDistance(Y,BY(idx,:)),[],2));
        else
            for k = 1 : numel(idx)-1
                A = BY(idx(k),:);
                B = BY(idx(k+1),:);
                Dist = min(Dist,pointSegmentDistance(Y,A,B));
            end
        end
    end
end

function Stats = summarizeFiniteDistances(Dist)
    finite = Dist(isfinite(Dist));
    if isempty(finite)
        Stats = [NaN,NaN,NaN];
    else
        Stats = [min(finite),mean(finite),max(finite)];
    end
end

function Stats = trainReconstructionStats(Problem,Dec,TrainX,DatasetInfo)
    Stats = [NaN,NaN];
    if isempty(Dec) || isempty(TrainX)
        return;
    end
    n = min(size(Dec,1),size(TrainX,1));
    if n <= 0 || size(Dec,2) ~= size(TrainX,2)
        return;
    end
    Dec = Dec(1:n,:);
    TrainX = TrainX(1:n,:);
    xDist = sqrt(sum((Dec - TrainX).^2,2));
    Stats(1) = percentileFinite(xDist,90);
    if ~isstruct(DatasetInfo) || ~isfield(DatasetInfo,'trainObjs') || ...
            isempty(DatasetInfo.trainObjs)
        return;
    end
    targetObj = DatasetInfo.trainObjs;
    nObj = min(n,size(targetObj,1));
    if nObj <= 0
        return;
    end
    Obj = EvaluateDecisions_CBS(Problem,Dec(1:nObj,:));
    if size(Obj,2) ~= size(targetObj,2)
        return;
    end
    yDist = sqrt(sum((Obj - targetObj(1:nObj,:)).^2,2));
    Stats(2) = percentileFinite(yDist,90);
end

function q = percentileFinite(X,p)
    X = X(isfinite(X));
    if isempty(X)
        q = NaN;
    else
        q = prctile(X,p);
    end
end

function Dist = queryObjectiveDistances(Obj,QueryIndex,DatasetInfo)
    if isempty(Obj) || isempty(QueryIndex) || ...
            ~isfield(DatasetInfo,'queryObjs') || isempty(DatasetInfo.queryObjs)
        Dist = NaN(0,1);
        return;
    end
    n = min(size(Obj,1),numel(QueryIndex));
    ObjN = normalizeWithInfo(Obj(1:n,:),DatasetInfo.objMin, ...
        DatasetInfo.objSpan);
    QueryObjN = normalizeWithInfo(DatasetInfo.queryObjs, ...
        DatasetInfo.objMin,DatasetInfo.objSpan);
    QueryIndex = round(double(QueryIndex(1:n)));
    valid = isfinite(QueryIndex) & QueryIndex >= 1 & ...
        QueryIndex <= size(QueryObjN,1);
    Dist = NaN(n,1);
    if any(valid)
        target = QueryObjN(QueryIndex(valid),:);
        Dist(valid) = sqrt(sum((ObjN(valid,:) - target).^2,2));
    end
end

function Dist = querySourceDistances(DistAll,QueryIndex,DatasetInfo,sourceName)
    Dist = NaN(0,1);
    if isempty(DistAll) || isempty(QueryIndex) || ...
            ~isstruct(DatasetInfo) || ~isfield(DatasetInfo,'queryMeta') || ...
            ~isstruct(DatasetInfo.queryMeta) || ...
            ~isfield(DatasetInfo.queryMeta,'source_type')
        return;
    end
    n = min(numel(DistAll),numel(QueryIndex));
    DistAll = DistAll(1:n);
    QueryIndex = round(double(QueryIndex(1:n)));
    sourceType = string(DatasetInfo.queryMeta.source_type(:));
    valid = isfinite(QueryIndex) & QueryIndex >= 1 & ...
        QueryIndex <= numel(sourceType);
    keep = false(n,1);
    keep(valid) = sourceType(QueryIndex(valid)) == string(sourceName);
    Dist = DistAll(keep);
end

function Width = generatedQueryWidth90(GeneratedObj,QueryIndex,DatasetInfo)
    Width = NaN;
    if isempty(GeneratedObj) || isempty(QueryIndex)
        return;
    end
    Y = normalizeWithInfo(GeneratedObj,DatasetInfo.objMin,DatasetInfo.objSpan);
    QueryIndex = QueryIndex(:);
    n = min(size(Y,1),numel(QueryIndex));
    Y = Y(1:n,:);
    QueryIndex = QueryIndex(1:n);
    groups = unique(QueryIndex(isfinite(QueryIndex) & QueryIndex > 0));
    widths = NaN(numel(groups),1);
    for i = 1 : numel(groups)
        rows = QueryIndex == groups(i);
        if ~any(rows)
            continue;
        end
        center = mean(Y(rows,:),1);
        dist = sqrt(sum((Y(rows,:) - center).^2,2));
        widths(i) = percentileFinite(dist,90);
    end
    widths = widths(isfinite(widths));
    if ~isempty(widths)
        Width = median(widths);
    end
end

function Width = generatedSegmentWidth90( ...
    GeneratedObj,QueryIndex,DatasetInfo,BMem)
    Width = NaN;
    if isempty(GeneratedObj) || isempty(QueryIndex) || ...
            isempty(BMem) || ~isfield(BMem,'ref') || ...
            ~isfield(BMem,'y_b') || size(BMem.y_b,1) < 2 || ...
            ~isfield(DatasetInfo,'queryMeta') || ...
            ~isfield(DatasetInfo.queryMeta,'source_interval')
        return;
    end
    n = min(size(GeneratedObj,1),numel(QueryIndex));
    QueryIndex = round(double(QueryIndex(1:n)));
    valid = isfinite(QueryIndex) & QueryIndex >= 1 & ...
        QueryIndex <= size(DatasetInfo.queryMeta.source_interval,1);
    if ~any(valid)
        return;
    end
    ObjN = normalizeWithInfo(GeneratedObj(1:n,:), ...
        DatasetInfo.objMin,DatasetInfo.objSpan);
    BObjN = normalizeWithInfo(BMem.y_b, ...
        DatasetInfo.objMin,DatasetInfo.objSpan);
    dist = NaN(n,1);
    rows = find(valid(:))';
    for row = rows
        interval = DatasetInfo.queryMeta.source_interval(QueryIndex(row),:);
        [a,b] = sourceSegmentRows(BMem,interval);
        if isempty(a) || isempty(b)
            continue;
        end
        dist(row) = pointSegmentDistance(ObjN(row,:),BObjN(a,:),BObjN(b,:));
    end
    Width = percentileFinite(dist,90);
end

function [a,b] = sourceSegmentRows(BMem,interval)
    a = [];
    b = [];
    if isempty(interval) || numel(interval) < 2 || any(~isfinite(interval))
        return;
    end
    r1 = round(double(interval(1)));
    r2 = round(double(interval(2)));
    a = find(BMem.ref(:) == r1,1,'first');
    b = find(BMem.ref(:) == r2,1,'first');
    if isempty(a) || isempty(b)
        a = [];
        b = [];
    end
end

function Ratio = segmentWidthRatio(width90,BMem,DatasetInfo)
    Ratio = NaN;
    if ~isfinite(width90)
        return;
    end
    medLen = medianBoundarySegmentLength(BMem,DatasetInfo);
    if isfinite(medLen) && medLen > eps
        Ratio = width90/medLen;
    end
end

function medLen = medianBoundarySegmentLength(BMem,DatasetInfo)
    medLen = NaN;
    if isempty(BMem) || ~isfield(BMem,'y_b') || size(BMem.y_b,1) < 2
        return;
    end
    Y = normalizeWithInfo(BMem.y_b,DatasetInfo.objMin,DatasetInfo.objSpan);
    [~,ord] = sort(BMem.ref(:));
    Y = Y(ord,:);
    len = sqrt(sum(diff(Y,1,1).^2,2));
    len = len(isfinite(len));
    if ~isempty(len)
        medLen = median(len);
    end
end

function Stats = generatedPairSideStats(GeneratedDec,QueryIndex,DatasetInfo)
    Stats = [NaN,NaN];
    if isempty(GeneratedDec) || isempty(QueryIndex) || ...
            ~isfield(DatasetInfo,'queryMeta') || ...
            ~isfield(DatasetInfo.queryMeta,'x_f') || ...
            isempty(DatasetInfo.queryMeta.x_f)
        return;
    end
    n = min(size(GeneratedDec,1),numel(QueryIndex));
    QueryIndex = round(double(QueryIndex(1:n)));
    valid = isfinite(QueryIndex) & QueryIndex >= 1 & ...
        QueryIndex <= size(DatasetInfo.queryMeta.x_f,1);
    if ~any(valid)
        return;
    end
    X = GeneratedDec(1:n,:);
    XF = DatasetInfo.queryMeta.x_f(QueryIndex(valid),:);
    XI = DatasetInfo.queryMeta.x_i(QueryIndex(valid),:);
    distF = sqrt(sum((X(valid,:) - XF).^2,2));
    distI = sqrt(sum((X(valid,:) - XI).^2,2));
    side = distF < distI;
    Stats = [mean(double(side)),percentileFinite(distI - distF,50)];
end

function Cover = generatedRefCover(GeneratedFeasible,QueryIndex,DatasetInfo,BMem)
    Cover = NaN;
    if isempty(BMem) || ~isfield(BMem,'ref') || isempty(BMem.ref) || ...
            isempty(GeneratedFeasible) || isempty(QueryIndex) || ...
            ~isfield(DatasetInfo,'queryMeta') || ...
            ~isfield(DatasetInfo.queryMeta,'source_interval')
        return;
    end
    pairedRefs = unique(BMem.ref(:));
    pairedRefs = pairedRefs(isfinite(pairedRefs) & pairedRefs > 0);
    if isempty(pairedRefs)
        return;
    end
    n = min(numel(GeneratedFeasible),numel(QueryIndex));
    QueryIndex = round(double(QueryIndex(1:n)));
    valid = isfinite(QueryIndex) & QueryIndex >= 1 & ...
        QueryIndex <= size(DatasetInfo.queryMeta.source_interval,1);
    if ~any(valid)
        Cover = 0;
        return;
    end
    successQuery = valid & logical(GeneratedFeasible(1:n));
    if ~any(successQuery)
        Cover = 0;
        return;
    end
    intervals = DatasetInfo.queryMeta.source_interval( ...
        QueryIndex(successQuery),:);
    successRefs = unique(intervals(:));
    successRefs = successRefs(isfinite(successRefs) & successRefs > 0);
    Cover = numel(intersect(successRefs,pairedRefs))/numel(pairedRefs);
end

function Cover = bmemRefCoverage(BMem,W)
    Cover = NaN;
    if isempty(W) || isempty(BMem) || ~isfield(BMem,'ref') || ...
            isempty(BMem.ref)
        return;
    end
    refs = unique(round(double(BMem.ref(:))));
    refs = refs(isfinite(refs) & refs >= 1 & refs <= size(W,1));
    Cover = numel(refs)/size(W,1);
end

function Ratio = trainParameterRatio(trainCount,D,conditionDim,zDim,GANOptions)
    if nargin < 5
        GANOptions = struct();
    end
    paramCount = estimateCGANParameterCount(D,conditionDim,zDim,GANOptions);
    Ratio = safeRatio(trainCount,paramCount);
end

function Count = estimateCGANParameterCount(D,conditionDim,zDim,GANOptions)
    D = max(0,round(double(D)));
    conditionDim = max(0,round(double(conditionDim)));
    zDim = max(0,round(double(zDim)));
    generatorHidden = hiddenVectorFromOptions(GANOptions, ...
        'generatorHidden',[64 64 64]);
    discriminatorHidden = hiddenVectorFromOptions(GANOptions, ...
        'discriminatorHidden',[64 64 32]);
    genInput = conditionDim + zDim;
    genCount = denseNetworkParameterCount(genInput,generatorHidden,D);
    discInput = D + conditionDim;
    discCount = denseNetworkParameterCount(discInput,discriminatorHidden,1);
    Count = genCount + discCount;
end

function Count = denseNetworkParameterCount(inputDim,hidden,outDim)
    dims = [max(0,round(double(inputDim))),hidden(:)', ...
        max(0,round(double(outDim)))];
    Count = 0;
    for i = 1 : numel(dims)-1
        Count = Count + dims(i)*dims(i+1) + dims(i+1);
    end
end

function hidden = hiddenVectorFromOptions(Options,name,defaultValue)
    hidden = defaultValue;
    if isstruct(Options) && isfield(Options,name) && ~isempty(Options.(name))
        hidden = double(Options.(name)(:)');
        hidden = hidden(isfinite(hidden) & hidden > 0);
        if isempty(hidden)
            hidden = defaultValue;
        else
            hidden = max(1,round(hidden));
        end
    end
end

function Reuse = ganSampleReuse(trainCount,ganIter,miniBatch,dPretrain,dSteps,gSteps)
    trainCount = max(0,round(double(trainCount)));
    if trainCount <= 0
        Reuse = NaN;
        return;
    end
    miniBatch = min(max(1,round(double(miniBatch))),trainCount);
    updateCount = max(0,round(double(dPretrain))) + ...
        max(0,round(double(ganIter))) * ...
        (max(1,round(double(dSteps))) + max(1,round(double(gSteps))));
    Reuse = updateCount*miniBatch/trainCount;
end

function Stats = trainPairDistanceStats(DatasetInfo)
    Stats = [NaN,NaN];
    if ~isstruct(DatasetInfo) || ~isfield(DatasetInfo,'trainXf') || ...
            ~isfield(DatasetInfo,'trainXi') || isempty(DatasetInfo.trainXf) || ...
            isempty(DatasetInfo.trainXi)
        return;
    end
    n = min(size(DatasetInfo.trainXf,1),size(DatasetInfo.trainXi,1));
    if n <= 0
        return;
    end
    diff = DatasetInfo.trainXf(1:n,:) - DatasetInfo.trainXi(1:n,:);
    dist = sqrt(sum(diff.^2,2));
    Stats = [percentileFinite(dist,50),percentileFinite(dist,90)];
end

function Stats = querySourceCounts(DatasetInfo)
    Stats = [0,0];
    if ~isstruct(DatasetInfo) || ~isfield(DatasetInfo,'queryMeta') || ...
            ~isstruct(DatasetInfo.queryMeta) || ...
            ~isfield(DatasetInfo.queryMeta,'source_type')
        return;
    end
    sourceType = string(DatasetInfo.queryMeta.source_type(:));
    Stats = [sum(sourceType == "missing_ref"), ...
        sum(sourceType == "large_gap")];
end

function Ratio = safeRatio(num,den)
    num = double(num);
    den = double(den);
    if isempty(num) || isempty(den) || ~isfinite(num) || ...
            ~isfinite(den) || den <= 0
        Ratio = NaN;
    else
        Ratio = num/den;
    end
end

function Xn = normalizeWithInfo(X,MinV,SpanV)
    Xn = (X - MinV)./SpanV;
    Xn(~isfinite(Xn)) = 0;
end

function D = pointDistance(A,B)
    AA = sum(A.^2,2);
    BB = sum(B.^2,2)';
    D2 = max(AA + BB - 2*(A*B'),0);
    D = sqrt(D2);
end

function D = pointSegmentDistance(P,A,B)
    AB = B - A;
    denom = sum(AB.^2);
    if denom <= eps
        D = sqrt(sum((P - A).^2,2));
        return;
    end
    T = ((P - A)*AB')/denom;
    T = max(0,min(1,T));
    Projection = A + T.*AB;
    D = sqrt(sum((P - Projection).^2,2));
end
