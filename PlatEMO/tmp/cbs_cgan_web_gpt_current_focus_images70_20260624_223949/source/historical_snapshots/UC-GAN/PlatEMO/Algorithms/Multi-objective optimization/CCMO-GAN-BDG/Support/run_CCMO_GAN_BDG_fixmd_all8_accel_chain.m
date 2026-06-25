function [RunSummary,VariantSummary,outRoot] = ...
        run_CCMO_GAN_BDG_fixmd_all8_accel_chain( ...
        workerCount,problemNames,N,D,maxFE,runIds,selectedStages)
%run_CCMO_GAN_BDG_fixmd_all8_accel_chain Run FixMD all8 acceleration chain.

    rootDir = fileparts(which('platemo'));
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(workerCount)
        workerCount = 8;
    end
    if nargin < 2 || isempty(problemNames)
        problemNames = all8ProblemList_BDG();
    end
    if nargin < 3 || isempty(N)
        N = 100;
    end
    if nargin < 4
        D = [];
    end
    if nargin < 5 || isempty(maxFE)
        maxFE = 100000;
    end
    if nargin < 6 || isempty(runIds)
        runIds = 1:3;
    end
    if nargin < 7 || isempty(selectedStages)
        selectedStages = ["E0","E1","E2","E3","E4","E5","E6"];
    end

    workerCount = max(1,round(double(workerCount)));
    problemNames = cellstr(string(problemNames(:)));
    runIds = double(runIds(:)');
    assert(workerCount == 8 || maxFE < 100000, ...
        'run_CCMO_GAN_BDG_fixmd_all8_accel_chain:BadWorkerCount', ...
        'Formal FixMD all8 runs require workerCount=8.');
    assertNoForbiddenFormalProblems_BDG(problemNames);

    stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
    outRoot = fullfile(rootDir,'Data','CCMO_GAN_BDG', ...
        ['fixmd_all8_accel_chain_',stamp]);
    mkdir(outRoot);
    manifestFile = fullfile(outRoot,'chain_manifest.csv');
    Specs = filterSpecsByStage_BDG(buildExperimentSpecs_BDG(),selectedStages);
    Manifest = repmat(emptyManifestRow_BDG(),numel(Specs),1);

    for i = 1 : numel(Specs)
        Spec = Specs(i);
        fprintf('=== %s %s (%d/%d) ===\n',Spec.stage,Spec.label,i,numel(Specs));
        Manifest(i) = runOneSpec_BDG(Spec,outRoot,workerCount, ...
            problemNames,N,D,maxFE,runIds);
        writetable(struct2table(Manifest(1:i)),manifestFile);
    end

    [RunSummary,VariantSummary] = summarizeAll8Chain_BDG(outRoot, ...
        struct2table(Manifest));
end

function Row = runOneSpec_BDG(Spec,outRoot,workerCount,problemNames,N,D,maxFE,runIds)
    Row = emptyManifestRow_BDG();
    Row.stage = Spec.stage;
    Row.label = Spec.label;
    Row.variant = Spec.variant;
    Row.ganIter = Spec.ganIter;
    Row.ganTrainMode = Spec.ganTrainMode;
    Row.trainGap = Spec.trainGap;
    Row.targets_text = joinString_BDG(string(Spec.targets));
    Row.status = "not_started";
    Row.outDir = string(fullfile(outRoot,char(Spec.stage),char(Spec.label)));
    mkdir(char(Row.outDir));
    try
        diagOptions = struct( ...
            'variant',Spec.variant, ...
            'control',Spec.control, ...
            'algorithmParams',{algorithmParams_BDG( ...
                Spec.trainGap,Spec.ganIter,Spec.ganTrainMode, ...
                Spec.stageProbeN)}, ...
            'conditionCount',Spec.conditionCount, ...
            'zPerCondition',Spec.zPerCondition, ...
            'normalN',Spec.normalN);
        [Summary,actualOutDir] = run_CCMO_GAN_BDG_boundary_diagnostics( ...
            char(Row.outDir),workerCount,problemNames,N,D,maxFE,runIds, ...
            Spec.targets,diagOptions);
        Row.outDir = string(actualOutDir);
        if all(string(Summary.status) == "ok")
            Row.status = "ok";
        else
            Row.status = "has_error";
        end
    catch err
        Row.status = "failed";
        Row.error_message = string(getReport(err,'extended', ...
            'hyperlinks','off'));
    end
end

function Specs = buildExperimentSpecs_BDG()
    Specs = repmat(emptySpec_BDG(),0,1);
    Specs(end+1) = spec_BDG("E0","E0_nearseg_baseline", ...
        "FixMD_GNDk60_nearseg_huber_z0",50,"epoch",1, ...
        nearsegControl_BDG(),[100000],30,10,200,200);
    Specs(end+1) = spec_BDG("E0","E0_endpoint_baseline", ...
        "FixMD_GNDk60_endpoint_huber_z0",50,"epoch",1, ...
        endpointControl_BDG(),[100000],30,10,200,200);

    Specs(end+1) = spec_BDG("E1","E1_nearseg_diag_on", ...
        "FixMD_GNDk60_nearseg_huber_z0",50,"epoch",1, ...
        nearsegControl_BDG(),[30000 70000 100000],30,10,200,200);
    Specs(end+1) = spec_BDG("E1","E1_nearseg_diag_off", ...
        "FixMD_GNDk60_nearseg_huber_z0",50,"epoch",1, ...
        nearsegControl_BDG(),0,0,0,0,200);

    Specs(end+1) = spec_BDG("E2","E2_iter50", ...
        "FixMD_GNDk60_nearseg_huber_z0",50,"iter",1, ...
        nearsegControl_BDG(),[100000],30,10,200,200);
    Specs(end+1) = spec_BDG("E2","E2_iter100", ...
        "FixMD_GNDk60_nearseg_huber_z0",100,"iter",1, ...
        nearsegControl_BDG(),[100000],30,10,200,200);

    Specs(end+1) = spec_BDG("E3","E3_fused", ...
        "FixMD_GNDk60_nearseg_fused_huber_z0",50,"epoch",1, ...
        fusedControl_BDG(0.60),[100000],30,10,200,200);
    Specs(end+1) = spec_BDG("E3","E3_fused80_conservative", ...
        "FixMD_GNDk60_nearseg_fused80_huber_z0",50,"epoch",1, ...
        fusedControl_BDG(0.80),[100000],30,10,200,200);

    Specs(end+1) = spec_BDG("E4","E4_iter50_fused", ...
        "FixMD_GNDk60_nearseg_fused_huber_z0",50,"iter",1, ...
        fusedControl_BDG(0.60),[100000],30,10,200,200);
    Specs(end+1) = spec_BDG("E4","E4_iter100_fused", ...
        "FixMD_GNDk60_nearseg_fused_huber_z0",100,"iter",1, ...
        fusedControl_BDG(0.60),[100000],30,10,200,200);
    Specs(end+1) = spec_BDG("E4","E4_iter50_fused80", ...
        "FixMD_GNDk60_nearseg_fused80_huber_z0",50,"iter",1, ...
        fusedControl_BDG(0.80),[100000],30,10,200,200);

    Specs(end+1) = spec_BDG("E5","E5_iter50_bundle", ...
        "FixMD_GNDk60_nearseg_bundle_huber_z0",50,"iter",1, ...
        bundleControl_BDG(0.60),[100000],30,10,200,200);

    Specs(end+1) = spec_BDG("E6","E6_iter30_fused", ...
        "FixMD_GNDk60_nearseg_fused_huber_z0",30,"iter",1, ...
        fusedControl_BDG(0.60),[100000],30,10,200,200);
    Specs(end+1) = spec_BDG("E6","E6_iter50_fused", ...
        "FixMD_GNDk60_nearseg_fused_huber_z0",50,"iter",1, ...
        fusedControl_BDG(0.60),[100000],30,10,200,200);
    Specs(end+1) = spec_BDG("E6","E6_iter100_fused", ...
        "FixMD_GNDk60_nearseg_fused_huber_z0",100,"iter",1, ...
        fusedControl_BDG(0.60),[100000],30,10,200,200);
    Specs(end+1) = spec_BDG("E6","E6_iter50_fused_trainGap2", ...
        "FixMD_GNDk60_nearseg_fused_huber_z0",50,"iter",2, ...
        fusedControl_BDG(0.60),[100000],30,10,200,200);
end

function Spec = spec_BDG(stage,label,variant,ganIter,ganTrainMode, ...
        trainGap,control,targets,conditionCount,zPerCondition,normalN, ...
        stageProbeN)
    Spec = emptySpec_BDG();
    Spec.stage = string(stage);
    Spec.label = string(label);
    Spec.variant = string(variant);
    Spec.ganIter = double(ganIter);
    Spec.ganTrainMode = string(ganTrainMode);
    Spec.trainGap = double(trainGap);
    Spec.control = control;
    Spec.targets = double(targets);
    Spec.conditionCount = double(conditionCount);
    Spec.zPerCondition = double(zPerCondition);
    Spec.normalN = double(normalN);
    Spec.stageProbeN = double(stageProbeN);
end

function Spec = emptySpec_BDG()
    Spec = struct('stage',"",'label',"",'variant',"", ...
        'ganIter',NaN,'ganTrainMode',"",'trainGap',NaN, ...
        'control',struct(),'targets',[],'conditionCount',0, ...
        'zPerCondition',0,'normalN',0,'stageProbeN',0);
end

function Control = nearsegControl_BDG()
    Control = struct( ...
        'archiveParetoFilterMode',"global_af_nd", ...
        'archivePairDirectionMode',"af_not_dominates_ai", ...
        'archiveSourceCapMode',"none", ...
        'archivePairRefMode',"neighbor4", ...
        'trainFilterMode',"condition_knn", ...
        'conditionKNNRetainRatio',0.60, ...
        'cganTrainMinRefCov',0.00, ...
        'cganTrainMinTargetTriples',0, ...
        'conditionMode',"yt_dt_t_ref", ...
        'targetMode',"near_segment_feasible", ...
        'nearSegmentTau',0.20, ...
        'nearSegmentMaxPerPair',5, ...
        'decisionInterpCount',5, ...
        'targetRealLabelMode',"binary", ...
        'generatorMode',"objective_target_conditioned", ...
        'generatorLossMode',"conditional_adversarial_huber", ...
        'reconstructionWeight',250, ...
        'reconstructionHuberDelta',0.10, ...
        'ganCriticMode',"target_conditioned", ...
        'trainBundleMode',"none");
end

function Control = endpointControl_BDG()
    Control = nearsegControl_BDG();
    Control.targetMode = "endpoint";
end

function Control = fusedControl_BDG(retainRatio)
    Control = nearsegControl_BDG();
    Control.trainFilterMode = "condition_knn_fused";
    Control.conditionKNNRetainRatio = double(retainRatio);
end

function Control = bundleControl_BDG(retainRatio)
    Control = fusedControl_BDG(retainRatio);
    Control.trainBundleMode = "refresh_local";
end

function Params = algorithmParams_BDG(trainGap,ganIter,ganTrainMode,stageProbeN)
    Params = {trainGap,20,0,5,0.20,ganIter,0,stageProbeN,0,1,1,1, ...
        "g2sl",64,0.9,1e-4,1e-4,ganTrainMode};
end

function Specs = filterSpecsByStage_BDG(Specs,selectedStages)
    selectedStages = string(selectedStages(:));
    if any(selectedStages == "all")
        return;
    end
    keep = false(size(Specs));
    for i = 1 : numel(Specs)
        keep(i) = any(string(Specs(i).stage) == selectedStages) || ...
            any(string(Specs(i).label) == selectedStages);
    end
    Specs = Specs(keep);
end

function names = all8ProblemList_BDG()
    names = { ...
        'DASCMOP1_BC'; ...
        'DASCMOP2_BC'; ...
        'LIRCMOP5_BC'; ...
        'LIRCMOP6_BC'; ...
        'LIRCMOP7_BC'; ...
        'LIRCMOP8_BC'; ...
        'LIRCMOP9_BC'; ...
        'LIRCMOP10_BC'};
end

function assertNoForbiddenFormalProblems_BDG(problemNames)
    names = string(problemNames(:));
    forbidden = ["DASCMOP4_BC","DASCMOP5_BC"];
    assert(~any(ismember(names,forbidden)), ...
        'run_CCMO_GAN_BDG_fixmd_all8_accel_chain:ForbiddenProblem', ...
        'DASCMOP4_BC and DASCMOP5_BC are forbidden in formal all8 runs.');
end

function Row = emptyManifestRow_BDG()
    Row = struct('stage',"",'label',"",'variant',"",'ganIter',NaN, ...
        'ganTrainMode',"",'trainGap',NaN,'targets_text',"", ...
        'outDir',"",'status',"",'error_message',"");
end

function text = joinString_BDG(values)
    values = values(:);
    values = values(strlength(values) > 0);
    text = strjoin(cellstr(values),';');
end

function [RunSummary,VariantSummary] = summarizeAll8Chain_BDG( ...
        outRoot,Manifest)
    Rows = repmat(emptyChainRunRow_BDG(),0,1);
    rowCount = 0;
    for i = 1 : height(Manifest)
        outDir = char(string(Manifest.outDir(i)));
        summaryFile = fullfile(outDir,'boundary_diagnostic_run_summary.csv');
        metricsFile = fullfile(outDir,'gan_diagnostic_metrics_all.csv');
        Diag = readOptionalTable_BDG(metricsFile);
        if ~isfile(summaryFile)
            rowCount = rowCount + 1;
            Rows(rowCount,1) = buildMissingChainRow_BDG(Manifest,i, ...
                summaryFile,metricsFile); %#ok<AGROW>
            continue;
        end
        Summary = readOptionalTable_BDG(summaryFile);
        if isempty(Summary)
            rowCount = rowCount + 1;
            Rows(rowCount,1) = buildMissingChainRow_BDG(Manifest,i, ...
                summaryFile,metricsFile); %#ok<AGROW>
            continue;
        end
        for r = 1 : height(Summary)
            rowCount = rowCount + 1;
            Rows(rowCount,1) = buildChainRunRow_BDG(Manifest,i, ...
                Summary,r,Diag,summaryFile,metricsFile); %#ok<AGROW>
        end
    end

    RunSummary = struct2table(Rows);
    if isempty(RunSummary)
        VariantSummary = table();
    else
        RunSummary = applyChainStatusLabels_BDG(RunSummary);
        VariantSummary = summarizeVariants_BDG(RunSummary);
    end
    writetable(RunSummary,fullfile(outRoot,'all8_run_summary.csv'));
    writetable(VariantSummary,fullfile(outRoot,'all8_variant_summary.csv'));
    writeChainAnalysis_BDG(outRoot,RunSummary,VariantSummary,Manifest);
end

function Row = buildMissingChainRow_BDG(Manifest,i,summaryFile,metricsFile)
    Row = emptyChainRunRow_BDG();
    Row = copyManifestFields_BDG(Row,Manifest,i);
    Row.manifest_status = tableString_BDG(Manifest,'status',i);
    Row.run_status = "missing_summary";
    Row.status_label = "FAIL";
    Row.status_reason = "missing boundary_diagnostic_run_summary.csv";
    Row.summary_file = string(summaryFile);
    Row.diagnostic_metrics_file = string(metricsFile);
end

function Row = buildChainRunRow_BDG(Manifest,i,Summary,r,Diag, ...
        summaryFile,metricsFile)
    Row = emptyChainRunRow_BDG();
    Row = copyManifestFields_BDG(Row,Manifest,i);
    Row.manifest_status = tableString_BDG(Manifest,'status',i);
    Row.problem = tableString_BDG(Summary,'problem',r);
    Row.family = tableString_BDG(Summary,'family',r);
    Row.run = tableNumber_BDG(Summary,'run',r);
    Row.seed = tableNumber_BDG(Summary,'seed',r);
    Row.N = tableNumber_BDG(Summary,'N',r);
    Row.D = tableNumber_BDG(Summary,'D',r);
    Row.maxFE = tableNumber_BDG(Summary,'maxFE',r);
    Row.finalFE = tableNumber_BDG(Summary,'finalFE',r);
    Row.runtime_total = tableNumber_BDG(Summary,'runtime',r);
    Row.IGD = tableNumber_BDG(Summary,'IGD',r);
    Row.HV = tableNumber_BDG(Summary,'HV',r);
    Row.final_feasible_rate = tableNumber_BDG(Summary,'Feasible_rate',r);
    Row.run_status = tableString_BDG(Summary,'status',r);
    Row.summary_file = string(summaryFile);
    Row.core_metrics_file = tableString_BDG(Summary,'core_metrics_file',r);
    Row.stage_metrics_file = tableString_BDG(Summary,'stage_metrics_file',r);
    Row.diagnostic_metrics_file = string(metricsFile);
    Row.snapshot_count = tableNumber_BDG(Summary,'snapshot_count',r);
    Row.diagnostic_snapshot_count = tableNumber_BDG( ...
        Summary,'diagnostic_snapshot_count',r);

    coreFile = char(Row.core_metrics_file);
    if strlength(Row.core_metrics_file) > 0 && isfile(coreFile)
        Core = readOptionalTable_BDG(coreFile);
        if ~isempty(Core)
            Row = copyNumericVars_BDG(Row,Core,height(Core), ...
                chainCoreMetricVars_BDG());
        else
            Row.status_reason = "empty core metrics";
        end
    else
        Row.status_reason = "missing core metrics";
    end

    DiagRow = latestDiagnosticRow_BDG(Diag,Row.problem,Row.run);
    if ~isempty(DiagRow)
        Row.diagnostic_target_FE = tableNumber_BDG(DiagRow,'target_FE',1);
        Row.diagnostic_actual_FE = tableNumber_BDG(DiagRow,'actual_FE',1);
        Row = copyNumericVars_BDG(Row,DiagRow,1, ...
            chainDiagnosticMetricVars_BDG());
    end
end

function Row = copyManifestFields_BDG(Row,Manifest,i)
    Row.stage = tableString_BDG(Manifest,'stage',i);
    Row.label = tableString_BDG(Manifest,'label',i);
    Row.variant = tableString_BDG(Manifest,'variant',i);
    Row.ganIter = tableNumber_BDG(Manifest,'ganIter',i);
    Row.ganTrainMode = tableString_BDG(Manifest,'ganTrainMode',i);
    Row.trainGap = tableNumber_BDG(Manifest,'trainGap',i);
    Row.targets_text = tableString_BDG(Manifest,'targets_text',i);
    Row.spec_outDir = tableString_BDG(Manifest,'outDir',i);
end

function RunSummary = applyChainStatusLabels_BDG(RunSummary)
    for i = 1 : height(RunSummary)
        [label,reason] = classifyChainRun_BDG(RunSummary,i);
        RunSummary.status_label(i) = label;
        RunSummary.status_reason(i) = reason;
    end
end

function [label,reason] = classifyChainRun_BDG(T,i)
    label = "PASS";
    reason = "ok";
    if string(T.manifest_status(i)) ~= "ok" || ...
            string(T.run_status(i)) ~= "ok"
        label = "FAIL";
        reason = "run failed or missing summary";
        return;
    end
    if strlength(string(T.core_metrics_file(i))) == 0 || ...
            ~isfinite(double(T.runtime_core_est(i)))
        label = "FAIL";
        reason = "missing final core metrics";
        return;
    end
    currentLabel = string(T.label(i));
    if currentLabel == "E0_nearseg_baseline" || ...
            currentLabel == "E0_endpoint_baseline"
        return;
    end
    base = find(string(T.label) == "E0_nearseg_baseline" & ...
        string(T.problem) == string(T.problem(i)) & ...
        double(T.run) == double(T.run(i)) & ...
        string(T.run_status) == "ok",1);
    if isempty(base)
        reason = "ok; no E0 nearseg baseline row for quality comparison";
        return;
    end

    warnings = strings(0,1);
    warnings = appendQualityWarning_BDG(warnings,T.BoundaryHit_all(i), ...
        T.BoundaryHit_all(base),-0.10,"BoundaryHit_all");
    warnings = appendQualityWarning_BDG(warnings,T.RawGANFeasibleRate(i), ...
        T.RawGANFeasibleRate(base),-0.10,"RawGANFeasibleRate");
    warnings = appendDistanceWarning_BDG(warnings, ...
        T.GAN_to_Segment_Dist90(i),T.GAN_to_Segment_Dist90(base), ...
        "GAN_to_Segment_Dist90");
    warnings = appendDistanceWarning_BDG(warnings, ...
        T.normal_obj_seg_dist90(i),T.normal_obj_seg_dist90(base), ...
        "normal_obj_seg_dist90");
    if ~isempty(warnings)
        label = "QUALITY_WARN";
        reason = strjoin(cellstr(warnings),'; ');
    end
end

function warnings = appendQualityWarning_BDG(warnings,value,base, ...
        dropLimit,name)
    value = double(value);
    base = double(base);
    if isfinite(value) && isfinite(base) && value < base + dropLimit
        warnings(end+1,1) = string(name) + " drop";
    end
end

function warnings = appendDistanceWarning_BDG(warnings,value,base,name)
    value = double(value);
    base = double(base);
    if ~isfinite(value) || ~isfinite(base)
        return;
    end
    if base > 0 && value > 1.25 * base
        warnings(end+1,1) = string(name) + " increase";
    elseif base <= 0 && value > 0.05
        warnings(end+1,1) = string(name) + " increase";
    end
end

function VariantSummary = summarizeVariants_BDG(RunSummary)
    groupVars = {'stage','label','variant','ganTrainMode','ganIter','trainGap'};
    metricVars = chainSummaryMetricVars_BDG();
    names = string(RunSummary.Properties.VariableNames);
    numericMask = varfun(@isnumeric,RunSummary,'OutputFormat','uniform');
    numericNames = string(RunSummary.Properties.VariableNames(numericMask));
    metricVars = intersect(metricVars,intersect(names,numericNames),'stable');
    if isempty(RunSummary) || isempty(metricVars)
        VariantSummary = table();
        return;
    end
    VariantSummary = groupsummary(RunSummary,groupVars,'mean', ...
        cellstr(metricVars));
    VariantSummary.pass_count = statusCountByVariant_BDG( ...
        RunSummary,VariantSummary,"PASS");
    VariantSummary.quality_warn_count = statusCountByVariant_BDG( ...
        RunSummary,VariantSummary,"QUALITY_WARN");
    VariantSummary.fail_count = statusCountByVariant_BDG( ...
        RunSummary,VariantSummary,"FAIL");
end

function counts = statusCountByVariant_BDG(RunSummary,VariantSummary,status)
    counts = zeros(height(VariantSummary),1);
    for i = 1 : height(VariantSummary)
        mask = string(RunSummary.stage) == string(VariantSummary.stage(i)) & ...
            string(RunSummary.label) == string(VariantSummary.label(i)) & ...
            string(RunSummary.variant) == string(VariantSummary.variant(i)) & ...
            string(RunSummary.ganTrainMode) == ...
                string(VariantSummary.ganTrainMode(i)) & ...
            double(RunSummary.ganIter) == double(VariantSummary.ganIter(i)) & ...
            double(RunSummary.trainGap) == double(VariantSummary.trainGap(i)) & ...
            string(RunSummary.status_label) == string(status);
        counts(i) = sum(mask);
    end
end

function writeChainAnalysis_BDG(outRoot,RunSummary,VariantSummary,Manifest)
    analysisFile = fullfile(outRoot,'all8_final_analysis.md');
    lines = strings(0,1);
    lines(end+1,1) = "# CCMO_GAN_BDG FixMD all8 acceleration chain";
    lines(end+1,1) = "";
    lines(end+1,1) = "Generated by run_CCMO_GAN_BDG_fixmd_all8_accel_chain.";
    lines(end+1,1) = "Formal problem set excludes DASCMOP4_BC and DASCMOP5_BC.";
    lines(end+1,1) = "";
    lines(end+1,1) = "## Files";
    lines(end+1,1) = "- all8_run_summary.csv";
    lines(end+1,1) = "- all8_variant_summary.csv";
    lines(end+1,1) = "- chain_manifest.csv";
    lines(end+1,1) = "";
    lines(end+1,1) = "## Coverage";
    lines(end+1,1) = sprintf("- specs: %d",height(Manifest));
    lines(end+1,1) = sprintf("- run rows: %d",height(RunSummary));
    lines(end+1,1) = sprintf("- PASS: %d", ...
        sum(string(RunSummary.status_label) == "PASS"));
    lines(end+1,1) = sprintf("- QUALITY_WARN: %d", ...
        sum(string(RunSummary.status_label) == "QUALITY_WARN"));
    lines(end+1,1) = sprintf("- FAIL: %d", ...
        sum(string(RunSummary.status_label) == "FAIL"));
    lines(end+1,1) = "";
    lines(end+1,1) = "## Six Questions";
    lines(end+1,1) = "- Q1 epoch amplification: " + ...
        epochAnswer_BDG(VariantSummary);
    lines(end+1,1) = "- Q2 nearseg build-then-delete: " + ...
        nearsegFusionAnswer_BDG(VariantSummary);
    lines(end+1,1) = "- Q3 diagnostics/export share: " + ...
        diagnosticsShareAnswer_BDG(RunSummary);
    lines(end+1,1) = "- Q4 fused builder quality: " + ...
        fusedQualityAnswer_BDG(VariantSummary);
    lines(end+1,1) = "- Q5 BoundaryTrainBundle mainline value: " + ...
        bundleAnswer_BDG(VariantSummary);
    lines(end+1,1) = "- Q6 recommended all8 mainline: " + ...
        recommendMainline_BDG(VariantSummary);
    writelines_BDG(lines,analysisFile);
end

function text = epochAnswer_BDG(V)
    base = variantMean_BDG(V,"E0_nearseg_baseline","runtime_core_est");
    iter50 = variantMean_BDG(V,"E2_iter50","runtime_core_est");
    trainBase = variantMean_BDG(V,"E0_nearseg_baseline","t_gan_train");
    trainIter = variantMean_BDG(V,"E2_iter50","t_gan_train");
    if isfinite(base) && isfinite(iter50)
        text = sprintf("iter50/core ratio %.3g, gan_train ratio %.3g.", ...
            ratio_BDG(iter50,base),ratio_BDG(trainIter,trainBase));
    else
        text = "insufficient completed E0/E2 rows.";
    end
end

function text = nearsegFusionAnswer_BDG(V)
    baseBuild = variantMean_BDG(V,"E0_nearseg_baseline","t_target_build");
    fusedBuild = variantMean_BDG(V,"E3_fused","t_target_build");
    baseFilter = variantMean_BDG(V,"E0_nearseg_baseline","t_target_filter");
    fusedFilter = variantMean_BDG(V,"E3_fused","t_target_filter");
    if isfinite(baseBuild) && isfinite(fusedBuild)
        text = sprintf("target_build ratio %.3g, target_filter ratio %.3g.", ...
            ratio_BDG(fusedBuild,baseBuild), ...
            ratio_BDG(fusedFilter,baseFilter));
    else
        text = "insufficient completed E0/E3 rows.";
    end
end

function text = diagnosticsShareAnswer_BDG(T)
    core = finiteMean_BDG(T.runtime_core_est);
    diag = finiteMean_BDG(T.t_gan_diagnose);
    probe = finiteMean_BDG(T.t_stage_probe);
    snap = finiteMean_BDG(T.t_snapshot_export);
    if isfinite(core) && core > 0
        text = sprintf("diagnose %.2f%%, stage_probe %.2f%%, snapshot_export %.2f%% of runtime_core_est.", ...
            100*diag/core,100*probe/core,100*snap/core);
    else
        text = "insufficient timer rows.";
    end
end

function text = fusedQualityAnswer_BDG(V)
    hit = variantMean_BDG(V,"E3_fused","BoundaryHit_all");
    baseHit = variantMean_BDG(V,"E0_nearseg_baseline","BoundaryHit_all");
    dist = variantMean_BDG(V,"E3_fused","GAN_to_Segment_Dist90");
    baseDist = variantMean_BDG(V,"E0_nearseg_baseline", ...
        "GAN_to_Segment_Dist90");
    if isfinite(hit) && isfinite(baseHit)
        text = sprintf("BoundaryHit_all delta %.3g, GAN_to_Segment_Dist90 ratio %.3g.", ...
            hit - baseHit,ratio_BDG(dist,baseDist));
    else
        text = "insufficient completed E0/E3 quality rows.";
    end
end

function text = bundleAnswer_BDG(V)
    bundle = V(string(V.label) == "E5_iter50_bundle",:);
    if isempty(bundle)
        text = "insufficient completed E5 rows.";
        return;
    end
    match = tableMeanVar_BDG(bundle,"mean_bundle_sample_pair_ai_match_rate");
    core = tableMeanVar_BDG(bundle,"mean_runtime_core_est");
    text = sprintf("sample pair/AI match %.3g, mean runtime_core_est %.3g; treat as refresh-local equivalence evidence before deeper mainline adoption.", ...
        match,core);
end

function text = recommendMainline_BDG(V)
    if isempty(V)
        text = "no completed variant summary.";
        return;
    end
    candidateStages = ["E4","E5","E6"];
    mask = ismember(string(V.stage),candidateStages);
    if ismember("fail_count",string(V.Properties.VariableNames))
        mask = mask & double(V.fail_count) == 0;
    end
    C = V(mask,:);
    if isempty(C)
        text = "no completed E4/E5/E6 candidate without FAIL rows.";
        return;
    end
    warn = zeros(height(C),1);
    if ismember("quality_warn_count",string(C.Properties.VariableNames))
        warn = double(C.quality_warn_count);
    end
    runtime = tableMeanColumn_BDG(C,"mean_runtime_core_est");
    runtime(~isfinite(runtime)) = inf;
    [~,idx] = sortrows([warn,runtime],[1 2]);
    best = idx(1);
    text = sprintf("%s (%s, ganIter=%g, trainGap=%g), mean_runtime_core_est %.3g, warnings %g.", ...
        string(C.label(best)),string(C.ganTrainMode(best)), ...
        double(C.ganIter(best)),double(C.trainGap(best)), ...
        runtime(best),warn(best));
end

function value = variantMean_BDG(V,label,metric)
    if isempty(V)
        value = NaN;
        return;
    end
    varName = "mean_" + string(metric);
    mask = string(V.label) == string(label);
    value = tableMeanVar_BDG(V(mask,:),varName);
end

function value = tableMeanVar_BDG(T,varName)
    if isempty(T) || ~ismember(string(varName),string(T.Properties.VariableNames))
        value = NaN;
        return;
    end
    value = finiteMean_BDG(T.(char(varName)));
end

function values = tableMeanColumn_BDG(T,varName)
    values = NaN(height(T),1);
    if isempty(T) || ~ismember(string(varName),string(T.Properties.VariableNames))
        return;
    end
    values = double(T.(char(varName)));
end

function r = ratio_BDG(value,base)
    if isfinite(value) && isfinite(base) && base ~= 0
        r = value / base;
    else
        r = NaN;
    end
end

function m = finiteMean_BDG(values)
    values = double(values(:));
    values = values(isfinite(values));
    if isempty(values)
        m = NaN;
    else
        m = mean(values);
    end
end

function DiagRow = latestDiagnosticRow_BDG(Diag,problem,run)
    DiagRow = table();
    if isempty(Diag) || ~ismember("problem_name", ...
            string(Diag.Properties.VariableNames)) || ...
            ~ismember("run_id",string(Diag.Properties.VariableNames))
        return;
    end
    mask = string(Diag.problem_name) == string(problem) & ...
        double(Diag.run_id) == double(run);
    idx = find(mask);
    if isempty(idx)
        return;
    end
    if ismember("target_FE",string(Diag.Properties.VariableNames))
        targetFE = double(Diag.target_FE(idx));
        finite = isfinite(targetFE);
        if any(finite)
            maxFE = max(targetFE(finite));
            idx = idx(targetFE == maxFE);
        end
    end
    DiagRow = Diag(idx(end),:);
end

function T = readOptionalTable_BDG(fileName)
    if strlength(string(fileName)) == 0 || ~isfile(char(fileName))
        T = table();
        return;
    end
    try
        T = readtable(char(fileName),'TextType','string', ...
            'Delimiter',',','ReadVariableNames',true, ...
            'VariableNamingRule','preserve');
    catch
        T = table();
    end
end

function Row = copyNumericVars_BDG(Row,T,index,vars)
    names = string(T.Properties.VariableNames);
    for i = 1 : numel(vars)
        varName = string(vars(i));
        if ismember(varName,names)
            Row.(char(varName)) = tableNumber_BDG(T,varName,index);
        end
    end
end

function value = tableString_BDG(T,name,index)
    names = string(T.Properties.VariableNames);
    if ~ismember(string(name),names) || index > height(T)
        value = "";
        return;
    end
    raw = T.(char(name))(index);
    if isempty(raw)
        value = "";
    else
        value = string(raw);
        if ismissing(value)
            value = "";
        end
    end
end

function value = tableNumber_BDG(T,name,index)
    names = string(T.Properties.VariableNames);
    if ~ismember(string(name),names) || index > height(T)
        value = NaN;
        return;
    end
    raw = T.(char(name))(index);
    if isnumeric(raw) || islogical(raw)
        value = double(raw);
    else
        value = str2double(string(raw));
    end
    if isempty(value)
        value = NaN;
    end
end

function writelines_BDG(lines,fileName)
    fid = fopen(fileName,'w');
    if fid < 0
        error('run_CCMO_GAN_BDG_fixmd_all8_accel_chain:WriteFailed', ...
            'Unable to write %s.',fileName);
    end
    cleanup = onCleanup(@()fclose(fid)); %#ok<NASGU>
    for i = 1 : numel(lines)
        fprintf(fid,'%s\n',char(lines(i)));
    end
end

function vars = chainSummaryMetricVars_BDG()
    vars = [ ...
        "runtime_total", ...
        chainCoreMetricVars_BDG(), ...
        "diagnostic_target_FE", ...
        "diagnostic_actual_FE", ...
        chainDiagnosticMetricVars_BDG()];
end

function vars = chainCoreMetricVars_BDG()
    vars = [ ...
        "t_archive_update", ...
        "t_source_build", ...
        "t_target_build", ...
        "t_target_filter", ...
        "t_gan_train", ...
        "t_gan_diagnose", ...
        "t_stage_probe", ...
        "t_snapshot_export", ...
        "runtime_core_est", ...
        "target_pair_count", ...
        "target_near_segment_keep_count", ...
        "target_triple_count", ...
        "target_filter_pre_count", ...
        "target_filter_post_count", ...
        "target_filter_retain_ratio", ...
        "target_filter_pre_ref_cov", ...
        "target_filter_post_ref_cov", ...
        "target_t_min", ...
        "target_t_max", ...
        "target_t_mean", ...
        "train_ref_cov", ...
        "bundle_condition_dim", ...
        "bundle_pair_count", ...
        "bundle_pair_cov", ...
        "bundle_ref_count", ...
        "bundle_ref_cov", ...
        "bundle_t_min", ...
        "bundle_t_max", ...
        "bundle_t_mean", ...
        "bundle_sample_pair_ai_match_rate", ...
        "gan_g_adv_loss_mean", ...
        "gan_g_rec_loss_mean", ...
        "gan_g_loss_count", ...
        "BoundaryHit_all", ...
        "GAN_to_Segment_Dist90", ...
        "RawGANFeasibleRate"];
end

function vars = chainDiagnosticMetricVars_BDG()
    vars = [ ...
        "normal_feasible_rate", ...
        "normal_obj_seg_dist90", ...
        "normal_dec_seg_dist90", ...
        "normal_own_pair_nearest_rate", ...
        "fixedz_feasible_rate", ...
        "fixedz_obj_seg_dist90", ...
        "zsweep_feasible_rate", ...
        "zsweep_obj_seg_dist90"];
end

function Row = emptyChainRunRow_BDG()
    Row = struct( ...
        'stage',"", ...
        'label',"", ...
        'variant',"", ...
        'ganIter',NaN, ...
        'ganTrainMode',"", ...
        'trainGap',NaN, ...
        'targets_text',"", ...
        'spec_outDir',"", ...
        'manifest_status',"", ...
        'problem',"", ...
        'family',"", ...
        'run',NaN, ...
        'seed',NaN, ...
        'N',NaN, ...
        'D',NaN, ...
        'maxFE',NaN, ...
        'finalFE',NaN, ...
        'runtime_total',NaN, ...
        'IGD',NaN, ...
        'HV',NaN, ...
        'final_feasible_rate',NaN, ...
        'snapshot_count',NaN, ...
        'diagnostic_snapshot_count',NaN, ...
        'diagnostic_target_FE',NaN, ...
        'diagnostic_actual_FE',NaN, ...
        'run_status',"", ...
        'status_label',"", ...
        'status_reason',"", ...
        'summary_file',"", ...
        'core_metrics_file',"", ...
        'stage_metrics_file',"", ...
        'diagnostic_metrics_file',"");
    vars = unique([chainCoreMetricVars_BDG(), ...
        chainDiagnosticMetricVars_BDG()],'stable');
    for i = 1 : numel(vars)
        Row.(char(vars(i))) = NaN;
    end
end
