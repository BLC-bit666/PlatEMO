function Manifest = run_CCMO_GAN_BDG_fixmd_pairt_diagnostic_suite( ...
        outRoot,suiteName,workerCount)
% Run the no-plot fix.md pair+t boundary diagnostics.

    rootDir = fileparts(which('platemo'));
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outRoot)
        outRoot = fullfile(rootDir,'Data','CCMO_GAN_BDG', ...
            ['fixmd_pairt_diag_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(suiteName)
        suiteName = "matched_global_goal10";
    end
    if nargin < 3 || isempty(workerCount)
        workerCount = 6;
    end
    suiteName = lower(strtrim(string(suiteName)));
    workerCount = max(1,round(double(workerCount)));
    if ~isfolder(outRoot)
        mkdir(outRoot);
    end

    Suites = resolveSuiteList_BDG(suiteName);
    Rows = readExistingManifestRows_BDG(fullfile(outRoot, ...
        'suite_manifest.csv'));
    for s = 1 : numel(Suites)
        suite = Suites(s);
        problems = suiteProblems_BDG(suite);
        variants = suiteVariants_BDG(suite);
        for v = 1 : numel(variants)
            Spec = variantSpec_BDG(variants(v));
            variantOut = fullfile(outRoot,char(suite),char(Spec.name));
            summaryFile = fullfile(variantOut, ...
                'boundary_diagnostic_run_summary.csv');
            metricsFile = fullfile(variantOut, ...
                'gan_diagnostic_metrics_all.csv');
            if isfile(summaryFile)
                runStatus = summaryFileStatus_BDG(summaryFile);
                fprintf('skip existing %s/%s\n',suite,Spec.name);
            else
                diagOptions = buildDiagOptions_BDG(Spec);
                Summary = run_CCMO_GAN_BDG_boundary_diagnostics( ...
                    variantOut,workerCount,cellstr(problems),100,[], ...
                    100000,1:3,100000,diagOptions);
                runStatus = manifestStatus_BDG(Summary);
            end
            NewRow = emptyManifestRow_BDG();
            NewRow.suite = suite;
            NewRow.variant = Spec.name;
            NewRow.problems = strjoin(problems,';');
            NewRow.problem_count = numel(problems);
            NewRow.run_count = 3;
            NewRow.worker_count = workerCount;
            NewRow.outDir = string(variantOut);
            NewRow.summary_file = string(summaryFile);
            NewRow.metrics_file = string(metricsFile);
            NewRow.status = runStatus;
            Rows = upsertManifestRow_BDG(Rows,NewRow);
        end
    end

    Manifest = struct2table(Rows);
    writetable(Manifest,fullfile(outRoot,'suite_manifest.csv'));
    if exist('summarize_CCMO_GAN_BDG_fixmd_pairt_results','file') == 2
        summarize_CCMO_GAN_BDG_fixmd_pairt_results(outRoot);
    end
end

function Suites = resolveSuiteList_BDG(suiteName)
    switch suiteName
        case "all"
            Suites = "matched_global_goal10";
        case {"four_recommended","ten_recommended", ...
                "six_recommended_extra","goal_supplemental_six", ...
                "matched_global_goal10","matched_global_goal8", ...
                "matched_global_endpoint_goal10"}
            Suites = suiteName;
        otherwise
            error('run_CCMO_GAN_BDG_fixmd_pairt_diagnostic_suite:BadSuite', ...
                'Unknown suiteName: %s.',suiteName);
    end
end

function problems = suiteProblems_BDG(suite)
    switch suite
        case "four_recommended"
            problems = ["DASCMOP1_BC","DASCMOP2_BC", ...
                "LIRCMOP7_BC","LIRCMOP8_BC"];
        case "ten_recommended"
            problems = ["DASCMOP1_BC","DASCMOP2_BC", ...
                "DASCMOP3_BC","DASCMOP4_BC", ...
                "LIRCMOP1_BC","LIRCMOP2_BC", ...
                "LIRCMOP3_BC","LIRCMOP4_BC", ...
                "LIRCMOP7_BC","LIRCMOP8_BC"];
        case "six_recommended_extra"
            problems = ["DASCMOP3_BC","DASCMOP4_BC", ...
                "LIRCMOP1_BC","LIRCMOP2_BC", ...
                "LIRCMOP3_BC","LIRCMOP4_BC"];
        case "goal_supplemental_six"
            problems = ["DASCMOP4_BC","DASCMOP5_BC", ...
                "LIRCMOP5_BC","LIRCMOP6_BC", ...
                "LIRCMOP9_BC","LIRCMOP10_BC"];
        case {"matched_global_goal10","matched_global_endpoint_goal10"}
            problems = ["DASCMOP1_BC","DASCMOP2_BC", ...
                "DASCMOP4_BC","DASCMOP5_BC", ...
                "LIRCMOP5_BC","LIRCMOP6_BC", ...
                "LIRCMOP7_BC","LIRCMOP8_BC", ...
                "LIRCMOP9_BC","LIRCMOP10_BC"];
        case "matched_global_goal8"
            problems = ["DASCMOP1_BC","DASCMOP2_BC", ...
                "LIRCMOP5_BC","LIRCMOP6_BC", ...
                "LIRCMOP7_BC","LIRCMOP8_BC", ...
                "LIRCMOP9_BC","LIRCMOP10_BC"];
        otherwise
            error('run_CCMO_GAN_BDG_fixmd_pairt_diagnostic_suite:BadSuite', ...
                'Unknown suite: %s.',suite);
    end
end

function variants = suiteVariants_BDG(suite)
    switch suite
        case "four_recommended"
            variants = "FixMD_GNDk60_nearseg_huber_z0";
        case {"ten_recommended","six_recommended_extra", ...
                "goal_supplemental_six"}
            variants = "FixMD_GNDk60_nearseg_huber_z0";
        case {"matched_global_goal10","matched_global_goal8"}
            variants = "FixMD_GNDk60_nearseg_huber_z0";
        case "matched_global_endpoint_goal10"
            variants = "FixMD_GNDk60_endpoint_huber_z0";
        otherwise
            error('run_CCMO_GAN_BDG_fixmd_pairt_diagnostic_suite:BadSuite', ...
                'Unknown suite: %s.',suite);
    end
end

function Spec = variantSpec_BDG(name)
    name = string(name);
    Spec = struct('name',name,'zDim',4,'control',struct('variant',name));
    switch name
        case "FixMD_GNDk60_nearseg_huber_z0"
            Spec.zDim = 0;
            Spec.control = baseControl_BDG(name, ...
                "condition_knn",0.60,"yt_dt_t_ref", ...
                "near_segment_feasible","conditional_adversarial_huber", ...
                250,0.10);
        case "FixMD_GNDk60_endpoint_huber_z0"
            Spec.zDim = 0;
            Spec.control = baseControl_BDG(name, ...
                "condition_knn",0.60,"yt_dt_t_ref", ...
                "endpoint","conditional_adversarial_huber", ...
                250,0.10);
        otherwise
            error('run_CCMO_GAN_BDG_fixmd_pairt_diagnostic_suite:BadVariant', ...
                'Unknown variant: %s.',name);
    end
end

function Control = baseControl_BDG(name,filterMode,filterRatio, ...
        conditionMode,targetMode,lossMode,reconstructionWeight,huberDelta)
    Control = struct( ...
        'variant',string(name), ...
        'archiveParetoFilterMode',"global_af_nd", ...
        'archivePairDirectionMode',"af_not_dominates_ai", ...
        'archiveSourceCapMode',"none", ...
        'archivePairRefMode',"neighbor4", ...
        'trainFilterMode',string(filterMode), ...
        'conditionKNNRetainRatio',double(filterRatio), ...
        'cganTrainMinRefCov',0, ...
        'cganTrainMinTargetTriples',0, ...
        'conditionMode',string(conditionMode), ...
        'targetMode',string(targetMode), ...
        'nearSegmentTau',0.20, ...
        'nearSegmentMaxPerPair',5, ...
        'decisionInterpCount',5, ...
        'targetRealLabelMode',"binary", ...
        'generatorMode',"objective_target_conditioned", ...
        'ganCriticMode',"target_conditioned", ...
        'generatorLossMode',string(lossMode), ...
        'reconstructionWeight',double(reconstructionWeight), ...
        'reconstructionHuberDelta',double(huberDelta));
end

function diagOptions = buildDiagOptions_BDG(Spec)
    params = {1,20,Spec.zDim,5,0.20,50,0,200,0,1,1,1, ...
        "g2sl",64,0.9,1e-4,1e-4,"epoch"};
    diagOptions = struct( ...
        'conditionCount',30, ...
        'zPerCondition',10, ...
        'normalN',200, ...
        'seed',20260614, ...
        'variant',Spec.name, ...
        'control',Spec.control, ...
        'algorithmParams',{params});
end

function status = manifestStatus_BDG(Summary)
    if isempty(Summary)
        status = "empty";
    elseif ~ismember("status",string(Summary.Properties.VariableNames))
        status = "unknown";
    elseif all(string(Summary.status) == "ok")
        status = "ok";
    else
        status = "has_error";
    end
end

function status = summaryFileStatus_BDG(summaryFile)
    C = readcell(summaryFile,'Delimiter',',');
    if size(C,1) < 2
        status = "empty";
        return;
    end
    Header = string(C(1,:));
    idx = find(Header == "status",1);
    if isempty(idx)
        status = "unknown";
        return;
    end
    values = string(C(2:end,idx));
    if all(values == "ok")
        status = "ok";
    else
        status = "has_error";
    end
end

function Rows = readExistingManifestRows_BDG(manifestFile)
    Rows = repmat(emptyManifestRow_BDG(),0,1);
    if ~isfile(manifestFile)
        return;
    end
    C = readcell(manifestFile,'Delimiter',',');
    if size(C,1) < 2
        return;
    end
    Header = string(C(1,:));
    Data = C(2:end,:);
    for i = 1 : size(Data,1)
        Row = emptyManifestRow_BDG();
        Row.suite = readManifestString_BDG(Header,Data(i,:),"suite");
        Row.variant = readManifestString_BDG(Header,Data(i,:),"variant");
        Row.problems = readManifestString_BDG(Header,Data(i,:),"problems");
        Row.problem_count = readManifestDouble_BDG(Header,Data(i,:), ...
            "problem_count");
        Row.run_count = readManifestDouble_BDG(Header,Data(i,:),"run_count");
        Row.worker_count = readManifestDouble_BDG(Header,Data(i,:), ...
            "worker_count");
        Row.outDir = readManifestString_BDG(Header,Data(i,:),"outDir");
        Row.summary_file = readManifestString_BDG(Header,Data(i,:), ...
            "summary_file");
        Row.metrics_file = readManifestString_BDG(Header,Data(i,:), ...
            "metrics_file");
        Row.status = readManifestString_BDG(Header,Data(i,:),"status");
        Rows = upsertManifestRow_BDG(Rows,Row);
    end
end

function Rows = upsertManifestRow_BDG(Rows,Row)
    hit = [];
    for i = 1 : numel(Rows)
        if string(Rows(i).suite) == string(Row.suite) && ...
                string(Rows(i).variant) == string(Row.variant) && ...
                string(Rows(i).outDir) == string(Row.outDir)
            hit = i;
            break;
        end
    end
    if isempty(hit)
        Rows(end+1,1) = Row;
    else
        Rows(hit) = Row;
    end
end

function value = readManifestString_BDG(Header,Data,name)
    idx = find(Header == string(name),1);
    if isempty(idx) || idx > numel(Data) || isempty(Data{idx})
        value = "";
    else
        value = string(Data{idx});
    end
end

function value = readManifestDouble_BDG(Header,Data,name)
    x = readManifestString_BDG(Header,Data,name);
    value = str2double(x);
end

function Row = emptyManifestRow_BDG()
    Row = struct( ...
        'suite',"", ...
        'variant',"", ...
        'problems',"", ...
        'problem_count',NaN, ...
        'run_count',NaN, ...
        'worker_count',NaN, ...
        'outDir',"", ...
        'summary_file',"", ...
        'metrics_file',"", ...
        'status',"");
end
