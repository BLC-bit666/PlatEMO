classdef CCMO_GAN_BDG < ALGORITHM
% <multi> <real> <constrained>
% CCMO + direct boundary generation GAN for binary/unknown constraints
%
% trainGap --- 20 --- GAN retraining interval in generations
% nGen     --- 20 --- Number of GAN offspring injected each refresh
% minTrain --- 20 --- Minimum AF size for GAN training
% bisectK  --- 4  --- Bisection steps for boundary refinement
% repairK  --- 2  --- Bisection steps for repairing infeasible GAN offspring
% zDim     --- 8  --- Latent dimension of GAN
% perRef   --- 3  --- Max AF samples kept per reference vector
% nearTau  --- 0.20 --- Max normalized objective distance for cross-pop pairing
% xTau     --- 0.15 --- Decision-space clustering threshold
% ganIter  --- 80 --- GAN training iterations per refresh
% jitterN  --- 0  --- AF jitter diagnostic samples per refresh, no injection
% probeRawN --- 0 --- Extra raw GAN diagnostic samples per refresh, no repair/injection
% jitterSigmaFactor --- 0.25 --- AF jitter sigma multiplier on Gap_dec_med
% trainMode --- 0 --- 0: train on full AF, 1: train on AF_core
% corePerRef --- 2 --- Max AF_core training samples kept per reference vector
% coreCap --- 100 --- Max AF_core training samples
% coreDecWeight --- 0.45 --- Decision-gap weight in AF_core boundary score
% oracleN --- 0 --- AF_core oracle diagnostic samples per refresh, no injection
% oracleSigmaFactor --- 0.25 --- AF_core oracle sigma multiplier on Gap_dec_med
% bankMode --- 0 --- 0: train on current AF_core, 1: train on FIFO AF_core bank
% bankCap --- 200 --- Max AF_core samples kept in FIFO training bank
%
% This is the minimum validating version:
% 1) keep CCMO-DE skeleton,
% 2) build AF/AI from local boundary pairs,
% 3) train a standard GAN only on AF,
% 4) inject OffspringG only into Population1.

    methods
        function main(Algorithm,Problem)
            nGenDefault = max(10,ceil(0.2*Problem.N));
            [trainGap,nGen,minTrain,bisectK,repairK,zDim,perRef,nearTau,xTau,ganIter, ...
                jitterN,probeRawN,jitterSigmaFactor,trainMode,corePerRef,coreCap, ...
                coreDecWeight,oracleN,oracleSigmaFactor,bankMode,bankCap] = ...
                Algorithm.ParameterSet(20,nGenDefault,20,4,2,8,3,0.20,0.15,80, ...
                0,0,0.25,0,2,100,0.45,0,0.25,0,200);
            Params = struct( ...
                'trainGap',trainGap, ...
                'nGen',nGen, ...
                'minTrain',minTrain, ...
                'bisectK',bisectK, ...
                'repairK',repairK, ...
                'zDim',zDim, ...
                'perRef',perRef, ...
                'nearTau',nearTau, ...
                'xTau',xTau, ...
                'ganIter',ganIter, ...
                'jitterN',jitterN, ...
                'probeRawN',probeRawN, ...
                'jitterSigmaFactor',jitterSigmaFactor, ...
                'trainMode',trainMode, ...
                'corePerRef',corePerRef, ...
                'coreCap',coreCap, ...
                'coreDecWeight',coreDecWeight, ...
                'oracleN',oracleN, ...
                'oracleSigmaFactor',oracleSigmaFactor, ...
                'bankMode',bankMode, ...
                'bankCap',bankCap);

            [W,~] = UniformPoint(Problem.N,Problem.M);
            Observer = InitMetricsObserver_BDG(Algorithm,Problem,Params);
            Algorithm.metric.analysis_folder   = Observer.folder;
            Algorithm.metric.analysis_meta_csv = Observer.meta_file;
            Algorithm.metric.analysis_core_csv = Observer.core_file;

            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            Fitness1    = CalFitness_BDG(Population1.objs,Population1.cons);
            Fitness2    = CalFitness_BDG(Population2.objs);

            AF = InitArchive_BDG(Problem.D,Problem.M);
            AI = InitArchive_BDG(Problem.D,Problem.M);
            AFBank = InitArchive_BDG(Problem.D,Problem.M);
            AIBank = InitArchive_BDG(Problem.D,Problem.M);
            GAN = [];
            gen = 0;

            while Algorithm.NotTerminated(Population1)
                gen = gen + 1;

                % ===== Original CCMO-DE reproduction =====
                MatingPool1 = TournamentSelection(2,2*Problem.N,Fitness1);
                MatingPool2 = TournamentSelection(2,2*Problem.N,Fitness2);

                Offspring1  = OperatorDE(Problem,Population1,...
                    Population1(MatingPool1(1:end/2)),...
                    Population1(MatingPool1(end/2+1:end)));

                Offspring2  = OperatorDE(Problem,Population2,...
                    Population2(MatingPool2(1:end/2)),...
                    Population2(MatingPool2(end/2+1:end)));

                % ===== Boundary archive update =====
                [AF,AI,ArchiveDiag] = UpdateBoundaryArchive_BDG(Problem,AF,AI,...
                    Population1,Offspring1,Population2,Offspring2,...
                    W,bisectK,perRef,nearTau,xTau);

                % ===== GAN training and sampling =====
                OffspringG = Offspring1([]);
                if mod(gen,trainGap) == 0
                    GanDiag = EmptyGanDiag_BDG(Problem.M,Problem.D);
                    [AFCore,AICore] = SelectAFCore_BDG(Problem,AF,AI,W, ...
                        trainMode,minTrain,corePerRef,coreCap,coreDecWeight);
                    [AFBank,AIBank] = UpdateAFCoreBank_BDG(AFBank,AIBank, ...
                        AFCore,AICore,bankMode,bankCap,minTrain);
                    CoreDiag = BuildAFCoreDiag_BDG(Problem,AFCore,AICore,W, ...
                        trainMode,coreDecWeight);
                    BankDiag = BuildAFBankDiag_BDG(Problem,AFBank,AIBank,W, ...
                        bankMode,bankCap);
                    AFTrain = SelectTrainingArchive_BDG( ...
                        AFCore,AICore,AFBank,AIBank,bankMode);
                    refreshFlag = false;
                    skipReason = "AF_below_minTrain";
                    if size(AF.decs,1) >= minTrain
                        if size(AFCore.decs,1) >= minTrain
                            if size(AFTrain.decs,1) >= minTrain
                                GAN = BoundaryGAN_BDG('train',AFTrain.decs, ...
                                    Problem.lower,Problem.upper,zDim,ganIter,GAN);
                                [OffspringG,GanDiag] = BoundaryGAN_BDG('sample',Problem,GAN,AF,AI,W,nGen,repairK);
                                GanDiag = EnsureGanDiag_BDG(GanDiag,Problem.M,Problem.D);
                                if oracleN > 0
                                    GanDiag = MergeDiagFields_BDG(GanDiag, ...
                                        EvaluateAFOracle_BDG(Problem,AFCore,AICore, ...
                                        oracleN,oracleSigmaFactor));
                                end
                                if jitterN > 0
                                    GanDiag = MergeDiagFields_BDG(GanDiag, ...
                                        EvaluateAFJitter_BDG(Problem,AF,AI,jitterN,jitterSigmaFactor));
                                end
                                if probeRawN > 0
                                    GanDiag = MergeDiagFields_BDG(GanDiag, ...
                                        BoundaryGAN_BDG('probe',Problem,GAN,probeRawN));
                                end
                                refreshFlag = true;
                                skipReason = "trained";
                            else
                                skipReason = "AF_bank_below_minTrain";
                            end
                        else
                            skipReason = "AF_core_below_minTrain";
                        end
                    end
                    AppendMetricsRow_BDG(Observer.core_file, ...
                        BuildMetricsRow_BDG(gen,Problem,AF,AI,W,GanDiag, ...
                        ArchiveDiag,CoreDiag,BankDiag,refreshFlag,skipReason));
                end

                % ===== Keep CCMO logic, inject only into Population1 =====
                [Population1,Fitness1] = EnvironmentalSelection_BDG(...
                    [Population1,Offspring1,Offspring2,OffspringG],Problem.N,true);

                [Population2,Fitness2] = EnvironmentalSelection_BDG(...
                    [Population2,Offspring1,Offspring2],Problem.N,false);
            end
        end
    end
end

function A = InitArchive_BDG(D,M)
    A.decs  = zeros(0,D);
    A.objs  = zeros(0,M);
    A.ref   = zeros(0,1);
    A.score = zeros(0,1);
end

function Diag = EmptyGanDiag_BDG(M,D)
    if nargin < 2
        D = 0;
    end
    Diag = struct( ...
        'raw_count',0, ...
        'raw_decs',zeros(0,D), ...
        'raw_objs',zeros(0,M), ...
        'raw_feasible',false(0,1), ...
        'repair_count',0, ...
        'af_jitter_count',0, ...
        'af_jitter_decs',zeros(0,D), ...
        'af_jitter_objs',zeros(0,M), ...
        'af_jitter_feasible',false(0,1), ...
        'af_jitter_FE',0, ...
        'probe_raw_count',0, ...
        'probe_raw_decs',zeros(0,D), ...
        'probe_raw_objs',zeros(0,M), ...
        'probe_raw_feasible',false(0,1), ...
        'probe_raw_FE',0, ...
        'af_oracle_count',0, ...
        'af_oracle_decs',zeros(0,D), ...
        'af_oracle_objs',zeros(0,M), ...
        'af_oracle_feasible',false(0,1), ...
        'af_oracle_FE',0);
end

function Observer = InitMetricsObserver_BDG(Algorithm,Problem,Params)
    rootDir = fileparts(which('platemo'));
    baseDir = fullfile(rootDir,'Data','CCMO_GAN_BDG');
    [~,~] = mkdir(baseDir);
    [~,token] = fileparts(tempname(baseDir));
    runId = ResolveRunId_BDG(Algorithm);
    runFolder = fullfile(baseDir,sprintf('%s_%s_run%d_%s', ...
        class(Algorithm),class(Problem),runId,token));
    [~,~] = mkdir(runFolder);

    Observer = struct( ...
        'folder',runFolder, ...
        'meta_file',fullfile(runFolder,'run_meta.csv'), ...
        'core_file',fullfile(runFolder,'core_metrics.csv'));

    WriteCsvHeader_BDG(Observer.meta_file,{ ...
        'algorithm','problem','family','run','M','D','N','maxFE', ...
        'trainGap','nGen','minTrain','bisectK','repairK','zDim', ...
        'perRef','nearTau','xTau','ganIter','jitterN','probeRawN', ...
        'jitterSigmaFactor','trainMode','corePerRef','coreCap', ...
        'coreDecWeight','oracleN','oracleSigmaFactor','bankMode', ...
        'bankCap','output_folder'});
    AppendCsvRow_BDG(Observer.meta_file,{ ...
        class(Algorithm),class(Problem),ProblemFamily_BDG(class(Problem)),runId, ...
        Problem.M,Problem.D,Problem.N,Problem.maxFE, ...
        Params.trainGap,Params.nGen,Params.minTrain,Params.bisectK, ...
        Params.repairK,Params.zDim,Params.perRef,Params.nearTau, ...
        Params.xTau,Params.ganIter,Params.jitterN,Params.probeRawN, ...
        Params.jitterSigmaFactor,Params.trainMode,Params.corePerRef, ...
        Params.coreCap,Params.coreDecWeight,Params.oracleN, ...
        Params.oracleSigmaFactor,Params.bankMode,Params.bankCap,runFolder});

    WriteCsvHeader_BDG(Observer.core_file,CoreMetricColumns_BDG());
end

function Row = BuildMetricsRow_BDG(gen,Problem,AF,AI,W,GanDiag, ...
    ArchiveDiag,CoreDiag,BankDiag,refreshFlag,skipReason)
    [gapObj,zmin,zmax] = PairObjectiveGap_BDG(AF,AI);
    gapDec = PairDecisionGap_BDG(Problem,AF,AI);
    refCov = ReferenceCoverage_BDG(AF,W);
    rawFR = NaN;
    rawBHR = NaN;
    rawBHRCond = NaN;
    repairRate = NaN;
    rawCount = 0;
    rawFeasibleCount = 0;
    rawHitCount = 0;
    rawMinAIDistMed = NaN;
    rawMinAIDistP10 = NaN;
    rawMinAFDistMed = NaN;
    replay = BoundaryHitMetrics_BDG(Problem,AF.decs,AF.objs, ...
        true(size(AF.decs,1),1),AF,AI,gapObj,zmin,zmax);
    jitter = BoundaryHitMetrics_BDG(Problem,GanDiag.af_jitter_decs, ...
        GanDiag.af_jitter_objs,GanDiag.af_jitter_feasible,AF,AI,gapObj,zmin,zmax);
    probe = BoundaryHitMetrics_BDG(Problem,GanDiag.probe_raw_decs, ...
        GanDiag.probe_raw_objs,GanDiag.probe_raw_feasible,AF,AI,gapObj,zmin,zmax);
    oracle = BoundaryHitMetrics_BDG(Problem,GanDiag.af_oracle_decs, ...
        GanDiag.af_oracle_objs,GanDiag.af_oracle_feasible,AF,AI,gapObj,zmin,zmax);

    if GanDiag.raw_count > 0
        rawCount = double(GanDiag.raw_count);
        rawFeasible = GanDiag.raw_feasible(:);
        rawFeasibleCount = sum(rawFeasible);
        rawFR = mean(double(rawFeasible));
        repairRate = double(GanDiag.repair_count) / double(GanDiag.raw_count);
        hit = false(GanDiag.raw_count,1);
        if ~isempty(AI.objs) && isfinite(gapObj) && ~isempty(GanDiag.raw_objs)
            rawObjN = NormalizeObjMatMetrics_BDG(GanDiag.raw_objs,zmin,zmax);
            aiObjN  = NormalizeObjMatMetrics_BDG(AI.objs,zmin,zmax);
            feasibleIdx = find(rawFeasible);
            if ~isempty(feasibleIdx) && ~isempty(aiObjN)
                dist = pdist2(rawObjN(feasibleIdx,:),aiObjN);
                minAIDist = min(dist,[],2);
                hit(feasibleIdx) = minAIDist <= gapObj;
                rawMinAIDistMed = MedianFinite_BDG(minAIDist);
                rawMinAIDistP10 = PercentileFinite_BDG(minAIDist,10);
            end
            rawBHR = mean(double(hit));
        end
        rawHitCount = sum(hit);
        if rawFeasibleCount > 0 && ~isnan(rawBHR)
            rawBHRCond = rawHitCount / rawFeasibleCount;
        end
        rawMinAFDistMed = RawMinAFDistance_BDG(GanDiag.raw_objs,AF);
    end

    Row = { ...
        gen,Problem.FE,size(AF.decs,1),size(AI.decs,1), ...
        gapObj,gapDec,refCov,double(refreshFlag),skipReason, ...
        ArchiveDiag.AF_candidate_count,ArchiveDiag.AF_retention_ratio, ...
        ArchiveDiag.AF_front1_count,ArchiveDiag.AF_front1_ratio, ...
        ArchiveDiag.AF_kept_front1_ratio,ArchiveDiag.score_gap_obj_med, ...
        ArchiveDiag.score_gap_dec_med,ArchiveDiag.score_conv_med, ...
        ArchiveDiag.score_rank_med, ...
        ArchiveDiag.pair_count_flip,ArchiveDiag.pair_count_cross, ...
        ArchiveDiag.pair_count_self,ArchiveDiag.pair_eval_count, ...
        ArchiveDiag.pair_eval_FE,rawCount,rawFeasibleCount,rawHitCount, ...
        rawMinAIDistMed,rawMinAIDistP10,rawMinAFDistMed, ...
        rawFR,rawBHR,rawBHRCond,repairRate, ...
        CoreDiag.train_mode,CoreDiag.size,CoreDiag.ref_cov, ...
        CoreDiag.gap_obj_med,CoreDiag.gap_dec_med,CoreDiag.score_med, ...
        CoreDiag.front1_ratio,CoreDiag.dec_weight, ...
        BankDiag.mode,BankDiag.size,BankDiag.ref_cov, ...
        BankDiag.gap_obj_med,BankDiag.gap_dec_med,BankDiag.score_med, ...
        BankDiag.front1_ratio,BankDiag.cap, ...
        replay.count,replay.hit(1),replay.hit(2),replay.hit(3), ...
        replay.bhr(1),replay.bhr(2),replay.bhr(3),replay.minAIObjMed, ...
        GanDiag.af_oracle_count,GanDiag.af_oracle_FE,oracle.fr, ...
        oracle.hit(1),oracle.hit(2),oracle.hit(3), ...
        oracle.bhr(1),oracle.bhr(2),oracle.bhr(3), ...
        oracle.bhrCond(1),oracle.bhrCond(2),oracle.bhrCond(3), ...
        oracle.minAIObjMed,oracle.minAIDecMed,oracle.minAFDecMed, ...
        GanDiag.af_jitter_count,GanDiag.af_jitter_FE,jitter.fr, ...
        jitter.hit(1),jitter.hit(2),jitter.hit(3), ...
        jitter.bhrCond(1),jitter.bhrCond(2),jitter.bhrCond(3), ...
        jitter.minAIObjMed, ...
        GanDiag.probe_raw_count,GanDiag.probe_raw_FE,probe.fr, ...
        probe.hit(1),probe.hit(2),probe.hit(3), ...
        probe.bhrCond(1),probe.bhrCond(2),probe.bhrCond(3), ...
        probe.minAIObjMed,probe.minAIDecMed,probe.minAFDecMed};
end

function Columns = CoreMetricColumns_BDG()
    Columns = {'gen','FE','AF_size','AI_size','Gap_obj_med','Gap_dec_med', ...
        'Ref_cov','refresh_flag','skip_reason', ...
        'AF_candidate_count','AF_retention_ratio', ...
        'AF_front1_count','AF_front1_ratio','AF_kept_front1_ratio', ...
        'score_gap_obj_med','score_gap_dec_med','score_conv_med','score_rank_med', ...
        'pair_count_flip','pair_count_cross','pair_count_self', ...
        'pair_eval_count','pair_eval_FE', ...
        'Raw_count','Raw_feasible_count','Raw_hit_count', ...
        'Raw_minAI_dist_med','Raw_minAI_dist_p10', ...
        'Raw_minAF_dist_med','Raw_FR','Raw_BHR','Raw_BHR_cond','Repair_rate', ...
        'AF_core_train_mode','AF_core_size','AF_core_ref_cov', ...
        'AF_core_gap_obj_med','AF_core_gap_dec_med','AF_core_score_med', ...
        'AF_core_front1_ratio','AF_core_dec_weight', ...
        'AF_bank_mode','AF_bank_size','AF_bank_ref_cov', ...
        'AF_bank_gap_obj_med','AF_bank_gap_dec_med','AF_bank_score_med', ...
        'AF_bank_front1_ratio','AF_bank_cap', ...
        'AF_replay_count','AF_replay_hit1','AF_replay_hit2','AF_replay_hit3', ...
        'AF_replay_BHR1','AF_replay_BHR2','AF_replay_BHR3', ...
        'AF_replay_minAI_obj_med', ...
        'AF_oracle_count','AF_oracle_FE','AF_oracle_FR', ...
        'AF_oracle_hit1','AF_oracle_hit2','AF_oracle_hit3', ...
        'AF_oracle_BHR1','AF_oracle_BHR2','AF_oracle_BHR3', ...
        'AF_oracle_BHR_cond1','AF_oracle_BHR_cond2','AF_oracle_BHR_cond3', ...
        'AF_oracle_minAI_obj_med','AF_oracle_minAI_dec_med', ...
        'AF_oracle_minAF_dec_med', ...
        'AF_jitter_count','AF_jitter_FE','AF_jitter_FR', ...
        'AF_jitter_hit1','AF_jitter_hit2','AF_jitter_hit3', ...
        'AF_jitter_BHR_cond1','AF_jitter_BHR_cond2','AF_jitter_BHR_cond3', ...
        'AF_jitter_minAI_obj_med', ...
        'Probe_Raw_count','Probe_Raw_FE','Probe_Raw_FR', ...
        'Probe_Raw_hit1','Probe_Raw_hit2','Probe_Raw_hit3', ...
        'Probe_Raw_BHR_cond1','Probe_Raw_BHR_cond2','Probe_Raw_BHR_cond3', ...
        'Probe_Raw_minAI_dist_obj_med','Probe_Raw_minAI_dist_dec_med', ...
        'Probe_Raw_minAF_dist_dec_med'};
end

function Diag = EnsureGanDiag_BDG(Diag,M,D)
    Empty = EmptyGanDiag_BDG(M,D);
    names = fieldnames(Empty);
    for i = 1 : numel(names)
        if ~isfield(Diag,names{i})
            Diag.(names{i}) = Empty.(names{i});
        end
    end
end

function A = MergeDiagFields_BDG(A,B)
    names = fieldnames(B);
    for i = 1 : numel(names)
        A.(names{i}) = B.(names{i});
    end
end

function [AFCore,AICore] = SelectAFCore_BDG(Problem,AF,AI,W,trainMode, ...
    minTrain,corePerRef,coreCap,coreDecWeight)
    AFCore = AF;
    AICore = AI;
    if trainMode < 1 || isempty(AF.decs) || isempty(AI.decs)
        return;
    end

    N = size(AF.decs,1);
    if N ~= size(AI.decs,1)
        return;
    end

    corePerRef = max(1,round(double(corePerRef)));
    if coreCap <= 0
        cap = N;
    else
        cap = min(N,max(round(double(coreCap)),round(double(minTrain))));
    end
    if cap <= 0
        AFCore = InitArchive_BDG(Problem.D,Problem.M);
        AICore = InitArchive_BDG(Problem.D,Problem.M);
        return;
    end

    ref = AF.ref(:);
    if numel(ref) ~= N || all(ref <= 0)
        ref = AssignRefFromObjMetrics_BDG(AF.objs,W);
    end

    score = AFCoreScore_BDG(Problem,AF,AI,coreDecWeight);
    keep = false(N,1);
    refs = unique(ref(ref > 0));
    for i = 1 : numel(refs)
        idx = find(ref == refs(i));
        [~,ord] = sort(score(idx),'ascend');
        idx = idx(ord(1:min(corePerRef,numel(ord))));
        keep(idx) = true;
    end

    [~,globalOrd] = sort(score,'ascend');
    need = min(cap,N);
    if sum(keep) < need
        fill = globalOrd(~keep(globalOrd));
        keep(fill(1:min(need-sum(keep),numel(fill)))) = true;
    elseif sum(keep) > cap
        chosen = find(keep);
        [~,ord] = sort(score(chosen),'ascend');
        keep(:) = false;
        keep(chosen(ord(1:cap))) = true;
    end

    AFCore.decs = AF.decs(keep,:);
    AFCore.objs = AF.objs(keep,:);
    AFCore.ref = ref(keep);
    AFCore.score = score(keep);
    AICore.decs = AI.decs(keep,:);
    AICore.objs = AI.objs(keep,:);
    AICore.ref = ref(keep);
    AICore.score = score(keep);
end

function score = AFCoreScore_BDG(Problem,AF,AI,coreDecWeight)
    if isempty(AF.decs) || isempty(AI.decs)
        score = zeros(size(AF.decs,1),1);
        return;
    end
    allObj = [AF.objs;AI.objs];
    zmin = min(allObj,[],1);
    zmax = max(allObj,[],1);
    fObj = NormalizeObjMatMetrics_BDG(AF.objs,zmin,zmax);
    iObj = NormalizeObjMatMetrics_BDG(AI.objs,zmin,zmax);
    fDec = NormalizeDecMatMetrics_BDG(AF.decs,Problem.lower,Problem.upper);
    iDec = NormalizeDecMatMetrics_BDG(AI.decs,Problem.lower,Problem.upper);
    gapObj = sqrt(sum((fObj - iObj).^2,2));
    gapDec = sqrt(sum((fDec - iDec).^2,2)) / sqrt(size(fDec,2));
    conv = AF.score(:);
    if numel(conv) ~= size(AF.decs,1)
        conv = zeros(size(AF.decs,1),1);
    end

    decW = min(max(double(coreDecWeight),0),0.90);
    objW = max(0,0.90 - decW);
    score = objW*NormalizeVector_BDG(gapObj) + ...
        decW*NormalizeVector_BDG(gapDec) + 0.10*NormalizeVector_BDG(conv);
end

function Diag = BuildAFCoreDiag_BDG(Problem,AFCore,AICore,W,trainMode,coreDecWeight)
    Diag = struct( ...
        'train_mode',double(trainMode >= 1), ...
        'size',size(AFCore.decs,1), ...
        'ref_cov',ReferenceCoverage_BDG(AFCore,W), ...
        'gap_obj_med',NaN, ...
        'gap_dec_med',NaN, ...
        'score_med',MedianFinite_BDG(AFCore.score(:)), ...
        'front1_ratio',NaN, ...
        'dec_weight',double(coreDecWeight));

    Diag.gap_obj_med = PairObjectiveGap_BDG(AFCore,AICore);
    Diag.gap_dec_med = PairDecisionGap_BDG(Problem,AFCore,AICore);
    if ~isempty(AFCore.objs)
        rank = ParetoRankMetrics_BDG(AFCore.objs);
        Diag.front1_ratio = mean(double(rank == 1));
    end
end

function Diag = BuildAFBankDiag_BDG(Problem,AFBank,AIBank,W,bankMode,bankCap)
    Diag = struct( ...
        'mode',double(bankMode >= 1), ...
        'size',size(AFBank.decs,1), ...
        'ref_cov',ReferenceCoverage_BDG(AFBank,W), ...
        'gap_obj_med',NaN, ...
        'gap_dec_med',NaN, ...
        'score_med',MedianFinite_BDG(AFBank.score(:)), ...
        'front1_ratio',NaN, ...
        'cap',double(bankCap));

    Diag.gap_obj_med = PairObjectiveGap_BDG(AFBank,AIBank);
    Diag.gap_dec_med = PairDecisionGap_BDG(Problem,AFBank,AIBank);
    if ~isempty(AFBank.objs)
        rank = ParetoRankMetrics_BDG(AFBank.objs);
        Diag.front1_ratio = mean(double(rank == 1));
    end
end

function AFTrain = SelectTrainingArchive_BDG(AFCore,~,AFBank,~,bankMode)
    if bankMode >= 1
        AFTrain = AFBank;
    else
        AFTrain = AFCore;
    end
end

function [AFBank,AIBank] = UpdateAFCoreBank_BDG(AFBank,AIBank,AFCore,AICore, ...
    bankMode,bankCap,minTrain)
    if bankMode < 1 || isempty(AFCore.decs) || isempty(AICore.decs)
        return;
    end
    if size(AFCore.decs,1) ~= size(AICore.decs,1)
        return;
    end

    AFBank.decs = [AFBank.decs;AFCore.decs];
    AFBank.objs = [AFBank.objs;AFCore.objs];
    AFBank.ref = [AFBank.ref;AFCore.ref(:)];
    AFBank.score = [AFBank.score;AFCore.score(:)];

    AIBank.decs = [AIBank.decs;AICore.decs];
    AIBank.objs = [AIBank.objs;AICore.objs];
    AIBank.ref = [AIBank.ref;AICore.ref(:)];
    AIBank.score = [AIBank.score;AICore.score(:)];

    cap = round(double(bankCap));
    if cap > 0
        cap = max(cap,round(double(minTrain)));
        if size(AFBank.decs,1) > cap
            keep = (size(AFBank.decs,1)-cap+1) : size(AFBank.decs,1);
            AFBank = SliceArchive_BDG(AFBank,keep);
            AIBank = SliceArchive_BDG(AIBank,keep);
        end
    end
end

function A = SliceArchive_BDG(A,keep)
    keep = keep(:);
    A.decs = A.decs(keep,:);
    A.objs = A.objs(keep,:);
    A.ref = A.ref(keep,:);
    A.score = A.score(keep,:);
end

function y = NormalizeVector_BDG(x)
    x = double(x(:));
    finite = isfinite(x);
    y = ones(size(x));
    if ~any(finite)
        y(:) = 1;
        return;
    end
    xmin = min(x(finite));
    xmax = max(x(finite));
    y(finite) = (x(finite) - xmin) ./ (xmax - xmin + 1e-12);
    y(~finite) = 1;
end

function ref = AssignRefFromObjMetrics_BDG(PopObj,W)
    if isempty(PopObj)
        ref = zeros(0,1);
        return;
    end
    if isempty(W)
        ref = ones(size(PopObj,1),1);
        return;
    end
    Y = NormalizeObjMatMetrics_BDG(PopObj,min(PopObj,[],1),max(PopObj,[],1));
    Wn = W ./ sqrt(sum(W.^2,2) + 1e-12);
    Yn = Y ./ sqrt(sum(Y.^2,2) + 1e-12);
    [~,ref] = max(Yn*Wn',[],2);
end

function rank = ParetoRankMetrics_BDG(PopObj)
    N = size(PopObj,1);
    dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if all(PopObj(i,:) <= PopObj(j,:)) && any(PopObj(i,:) < PopObj(j,:))
                dominate(i,j) = true;
            elseif all(PopObj(j,:) <= PopObj(i,:)) && any(PopObj(j,:) < PopObj(i,:))
                dominate(j,i) = true;
            end
        end
    end
    rank = inf(N,1);
    current = find(sum(dominate,1)' == 0);
    r = 1;
    while ~isempty(current)
        rank(current) = r;
        dominated = any(dominate(current,:),1)';
        dominate(current,:) = false;
        dominate(:,current) = false;
        current = find(dominated & isinf(rank) & sum(dominate,1)' == 0);
        r = r + 1;
    end
    rank(isinf(rank)) = r;
end

function Diag = EvaluateAFJitter_BDG(Problem,AF,AI,jitterN,jitterSigmaFactor)
    Diag = struct( ...
        'af_jitter_count',0, ...
        'af_jitter_decs',zeros(0,Problem.D), ...
        'af_jitter_objs',zeros(0,Problem.M), ...
        'af_jitter_feasible',false(0,1), ...
        'af_jitter_FE',0);
    if jitterN <= 0 || isempty(AF.decs) || isempty(AI.decs)
        return;
    end
    gapDec = PairDecisionGap_BDG(Problem,AF,AI);
    if ~isfinite(gapDec)
        return;
    end
    nJitter = max(0,round(double(jitterN)));
    if nJitter == 0
        return;
    end
    idx = randi(size(AF.decs,1),nJitter,1);
    Xn = NormalizeDecMatMetrics_BDG(AF.decs(idx,:),Problem.lower,Problem.upper);
    sigma = max(0,double(jitterSigmaFactor)) * gapDec;
    Xn = min(max(Xn + sigma*randn(size(Xn)),0),1);
    X = Xn .* (Problem.upper - Problem.lower) + Problem.lower;
    S = Problem.Evaluation(X);
    Diag.af_jitter_count = numel(S);
    Diag.af_jitter_decs = S.decs;
    Diag.af_jitter_objs = S.objs;
    Diag.af_jitter_feasible = IsFeasiblePopulation_BDG(S);
    Diag.af_jitter_FE = numel(S);
end

function Diag = EvaluateAFOracle_BDG(Problem,AFCore,AICore,oracleN,oracleSigmaFactor)
    Diag = struct( ...
        'af_oracle_count',0, ...
        'af_oracle_decs',zeros(0,Problem.D), ...
        'af_oracle_objs',zeros(0,Problem.M), ...
        'af_oracle_feasible',false(0,1), ...
        'af_oracle_FE',0);
    if oracleN <= 0 || isempty(AFCore.decs) || isempty(AICore.decs)
        return;
    end
    gapDec = PairDecisionGap_BDG(Problem,AFCore,AICore);
    if ~isfinite(gapDec)
        return;
    end
    nOracle = max(0,round(double(oracleN)));
    if nOracle == 0
        return;
    end
    idx = randi(size(AFCore.decs,1),nOracle,1);
    Xn = NormalizeDecMatMetrics_BDG(AFCore.decs(idx,:),Problem.lower,Problem.upper);
    sigma = max(0,double(oracleSigmaFactor)) * gapDec;
    Xn = min(max(Xn + sigma*randn(size(Xn)),0),1);
    X = Xn .* (Problem.upper - Problem.lower) + Problem.lower;
    S = Problem.Evaluation(X);
    Diag.af_oracle_count = numel(S);
    Diag.af_oracle_decs = S.decs;
    Diag.af_oracle_objs = S.objs;
    Diag.af_oracle_feasible = IsFeasiblePopulation_BDG(S);
    Diag.af_oracle_FE = numel(S);
end

function Metrics = BoundaryHitMetrics_BDG(Problem,decs,objs,feasible,AF,AI,gapObj,zmin,zmax)
    Metrics = struct( ...
        'count',0, ...
        'fr',NaN, ...
        'hit',zeros(1,3), ...
        'bhr',NaN(1,3), ...
        'bhrCond',NaN(1,3), ...
        'minAIObjMed',NaN, ...
        'minAIDecMed',NaN, ...
        'minAFDecMed',NaN);
    if isempty(decs)
        return;
    end
    Metrics.count = size(decs,1);
    feasible = feasible(:);
    if numel(feasible) ~= Metrics.count
        feasible = false(Metrics.count,1);
    end
    Metrics.fr = mean(double(feasible));
    feasibleCount = sum(feasible);

    if ~isempty(objs) && ~isempty(AI.objs)
        sampleObjN = NormalizeObjMatMetrics_BDG(objs,zmin,zmax);
        aiObjN = NormalizeObjMatMetrics_BDG(AI.objs,zmin,zmax);
        distObj = pdist2(sampleObjN,aiObjN);
        minObj = min(distObj,[],2);
        Metrics.minAIObjMed = MedianFinite_BDG(minObj);
        if isfinite(gapObj)
            for k = 1 : 3
                hit = feasible & minObj <= k*gapObj;
                Metrics.hit(k) = sum(hit);
                Metrics.bhr(k) = mean(double(hit));
                if feasibleCount > 0
                    Metrics.bhrCond(k) = Metrics.hit(k) / feasibleCount;
                end
            end
        end
    end

    sampleDecN = NormalizeDecMatMetrics_BDG(decs,Problem.lower,Problem.upper);
    if ~isempty(AI.decs)
        aiDecN = NormalizeDecMatMetrics_BDG(AI.decs,Problem.lower,Problem.upper);
        distDec = pdist2(sampleDecN,aiDecN) / sqrt(size(sampleDecN,2));
        Metrics.minAIDecMed = MedianFinite_BDG(min(distDec,[],2));
    end
    if ~isempty(AF.decs)
        afDecN = NormalizeDecMatMetrics_BDG(AF.decs,Problem.lower,Problem.upper);
        distDec = pdist2(sampleDecN,afDecN) / sqrt(size(sampleDecN,2));
        Metrics.minAFDecMed = MedianFinite_BDG(min(distDec,[],2));
    end
end

function flag = IsFeasiblePopulation_BDG(Population)
    if isempty(Population)
        flag = false(0,1);
    elseif isempty(Population.cons)
        flag = true(numel(Population),1);
    else
        flag = all(Population.cons <= 0,2);
    end
end

function [gap,zmin,zmax] = PairObjectiveGap_BDG(AF,AI)
    gap = NaN;
    zmin = zeros(1,size(AF.objs,2));
    zmax = ones(1,size(AF.objs,2));
    if isempty(AF.decs) || isempty(AI.decs)
        return;
    end
    allObj = [AF.objs;AI.objs];
    zmin = min(allObj,[],1);
    zmax = max(allObj,[],1);
    fObj = NormalizeObjMatMetrics_BDG(AF.objs,zmin,zmax);
    iObj = NormalizeObjMatMetrics_BDG(AI.objs,zmin,zmax);
    gap = MedianFinite_BDG(sqrt(sum((fObj - iObj).^2,2)));
end

function gap = PairDecisionGap_BDG(Problem,AF,AI)
    gap = NaN;
    if isempty(AF.decs) || isempty(AI.decs)
        return;
    end
    fDec = NormalizeDecMatMetrics_BDG(AF.decs,Problem.lower,Problem.upper);
    iDec = NormalizeDecMatMetrics_BDG(AI.decs,Problem.lower,Problem.upper);
    gap = MedianFinite_BDG(sqrt(sum((fDec - iDec).^2,2)) / sqrt(size(fDec,2)));
end

function refCov = ReferenceCoverage_BDG(AF,W)
    if isempty(AF.decs) || isempty(W)
        refCov = 0;
    else
        ref = AF.ref(:);
        ref = ref(ref > 0);
        refCov = numel(unique(ref)) / size(W,1);
    end
end

function X = NormalizeObjMatMetrics_BDG(X,zmin,zmax)
    if isempty(X)
        return;
    end
    X = (X - zmin) ./ (zmax - zmin + 1e-12);
    X = min(max(X,0),1);
end

function X = NormalizeDecMatMetrics_BDG(X,lower,upper)
    X = (X - lower) ./ (upper - lower + 1e-12);
    X = min(max(X,0),1);
end

function Value = MedianFinite_BDG(X)
    X = X(isfinite(X));
    if isempty(X)
        Value = NaN;
    else
        Value = median(X);
    end
end

function Value = PercentileFinite_BDG(X,p)
    X = sort(X(isfinite(X)));
    if isempty(X)
        Value = NaN;
    else
        pos = 1 + (numel(X)-1) * p / 100;
        lo = floor(pos);
        hi = ceil(pos);
        if lo == hi
            Value = X(lo);
        else
            Value = X(lo) + (pos-lo) * (X(hi)-X(lo));
        end
    end
end

function distMed = RawMinAFDistance_BDG(rawObjs,AF)
    distMed = NaN;
    if isempty(rawObjs) || isempty(AF.objs)
        return;
    end
    allObj = [rawObjs;AF.objs];
    zmin = min(allObj,[],1);
    zmax = max(allObj,[],1);
    rawObjN = NormalizeObjMatMetrics_BDG(rawObjs,zmin,zmax);
    afObjN  = NormalizeObjMatMetrics_BDG(AF.objs,zmin,zmax);
    dist = pdist2(rawObjN,afObjN);
    distMed = MedianFinite_BDG(min(dist,[],2));
end

function RunId = ResolveRunId_BDG(Algorithm)
    if isempty(Algorithm.run)
        RunId = 1;
    else
        RunId = Algorithm.run;
    end
end

function Family = ProblemFamily_BDG(problemName)
    if startsWith(string(problemName),"DASCMOP")
        Family = "DASCMOP_BC";
    elseif startsWith(string(problemName),"LIRCMOP")
        Family = "LIRCMOP_BC";
    else
        Family = "Other";
    end
end

function WriteCsvHeader_BDG(filePath,names)
    fid = fopen(filePath,'w');
    cleaner = onCleanup(@()fclose(fid));
    fprintf(fid,'%s\n',strjoin(names,','));
end

function AppendMetricsRow_BDG(filePath,row)
    AppendCsvRow_BDG(filePath,row);
end

function AppendCsvRow_BDG(filePath,row)
    fid = fopen(filePath,'a');
    cleaner = onCleanup(@()fclose(fid));
    text = cellfun(@FormatCsvValue_BDG,row,'UniformOutput',false);
    fprintf(fid,'%s\n',strjoin(text,','));
end

function text = FormatCsvValue_BDG(value)
    if isstring(value) || ischar(value)
        text = char(string(value));
        text = ['"',strrep(text,'"','""'),'"'];
    elseif isnumeric(value) || islogical(value)
        if isempty(value)
            text = 'NaN';
        else
            text = sprintf('%.17g',double(value(1)));
        end
    else
        text = char(string(value));
        text = ['"',strrep(text,'"','""'),'"'];
    end
end

function [Population,Fitness] = EnvironmentalSelection_BDG(Population,N,isOrigin)
% Same logic as current CCMO EnvironmentalSelection
    if isOrigin
        Fitness = CalFitness_BDG(Population.objs,Population.cons);
    else
        Fitness = CalFitness_BDG(Population.objs);
    end

    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del = Truncation_BDG(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end

    Population = Population(Next);
    Fitness    = Fitness(Next);
    [Fitness,rank] = sort(Fitness);
    Population = Population(rank);
end

function Fitness = CalFitness_BDG(PopObj,PopCon)
% Same logic as current CCMO CalFitness
    N = size(PopObj,1);
    if nargin == 1
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end

    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end

    S = sum(Dominate,2);
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end

    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    Fitness = R + D';
end

function Del = Truncation_BDG(PopObj,K)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain = find(~Del);
        Temp   = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
