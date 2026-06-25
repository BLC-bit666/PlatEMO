function [Summary,outDir,ByProblem,Raw,FigureManifest,VariantManifest] = ...
    run_CBS_CGAN_loss_ablation_lir_online(outDir,Options)
%RUN_CBS_CGAN_LOSS_ABLATION_LIR_ONLINE Online validation for fix2 variants.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_CGAN', ...
            ['loss_ablation_lir_online_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(Options)
        Options = struct();
    end
    Options = normalizeOnlineLossOptions(Options);
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    Variants = onlineLossVariants(Options.variantNames);
    Summary = table();
    Raw = table();
    History = table();
    FigureManifest = table();
    VariantRows = repmat(emptyOnlineVariantRow(),numel(Variants),1);

    for i = 1 : numel(Variants)
        variant = Variants(i);
        variantDir = fullfile(outDir,char(variant.variant));
        RunnerOptions = struct( ...
            'plotRun',Options.plotRun, ...
            'conditionMode',variant.conditionMode, ...
            'visualDiagnostics',Options.visualDiagnostics, ...
            'plotDiagnosticTrends',Options.plotDiagnosticTrends, ...
            'algorithmParams',{Options.algorithmParams}, ...
            'algorithmClass',Options.algorithmClass, ...
            'advWeight',variant.advWeight, ...
            'reconstructionWeight',variant.reconstructionWeight, ...
            'trainZMode',Options.trainZMode, ...
            'trainMode',Options.trainMode, ...
            'sampleZMode',Options.sampleZMode);

        [RunSummary,~,RunStage,RunFigures] = ...
            run_CBS_CGAN_boundary_quality_experiments(variantDir, ...
            Options.workerCount,Options.problemNames,Options.N,Options.D, ...
            Options.maxFE,Options.runIds,Options.targets,RunnerOptions);
        RunHistory = readOptionalOnlineTable(fullfile(variantDir, ...
            'history_metrics_all.csv'));

        RunSummary = addOnlineVariantColumns(RunSummary,variant);
        RunStage = addOnlineVariantColumns(RunStage,variant);
        RunHistory = addOnlineVariantColumns(RunHistory,variant);
        RunFigures = addOnlineVariantColumns(RunFigures,variant);

        writetable(selectFix2OnlineColumns(RunStage),fullfile(variantDir, ...
            'stage_metrics_all.csv'));
        writetable(selectFix2OnlineColumns(RunHistory),fullfile(variantDir, ...
            'history_metrics_all.csv'));

        Summary = appendOnlineTable(Summary,RunSummary);
        Raw = appendOnlineTable(Raw,RunStage);
        History = appendOnlineTable(History,RunHistory);
        FigureManifest = appendOnlineTable(FigureManifest,RunFigures);
        VariantRows(i) = makeOnlineVariantRow(variant,variantDir, ...
            RunSummary,RunStage,RunHistory,RunFigures);
    end

    VariantManifest = struct2table(VariantRows);
    Raw = selectFix2OnlineColumns(Raw);
    History = selectFix2OnlineColumns(History);
    FinalRaw = finalStageRows(Raw);
    ByProblem = aggregateOnlineMetrics(FinalRaw,{'variant','problem'});

    writetable(Summary,fullfile(outDir,'online_loss_ablation_summary.csv'));
    writetable(ByProblem,fullfile(outDir, ...
        'online_loss_ablation_by_problem.csv'));
    writetable(Raw,fullfile(outDir,'online_loss_ablation_raw.csv'));
    writetable(FigureManifest,fullfile(outDir,'figure_manifest.csv'));
    writetable(VariantManifest,fullfile(outDir, ...
        'online_loss_ablation_variant_manifest.csv'));
    writetable(History,fullfile(outDir,'online_loss_ablation_history.csv'));
end

function Options = normalizeOnlineLossOptions(Options)
    Options = ensureOnlineField(Options,'workerCount',8);
    Options = ensureOnlineField(Options,'problemNames', ...
        ["LIRCMOP5_BC";"LIRCMOP7_BC"; ...
        "LIRCMOP8_BC";"LIRCMOP10_BC"]);
    Options = ensureOnlineField(Options,'variantNames', ...
        ["A_ref_only_adv","B_ref_only_adv_huber","C_ref_tau_adv"]);
    Options = ensureOnlineField(Options,'N',100);
    Options = ensureOnlineField(Options,'D',[]);
    Options = ensureOnlineField(Options,'maxFE',100000);
    Options = ensureOnlineField(Options,'runIds',1:3);
    Options = ensureOnlineField(Options,'targets', ...
        [10000 30000 50000 70000 100000]);
    Options = ensureOnlineField(Options,'plotRun',1);
    Options = ensureOnlineField(Options,'conditionMode',"");
    Options = ensureOnlineField(Options,'trainZMode',"zero");
    Options = ensureOnlineField(Options,'trainMode',"iter");
    Options = ensureOnlineField(Options,'sampleZMode',"zero");
    Options = ensureOnlineField(Options,'visualDiagnostics',true);
    Options = ensureOnlineField(Options,'plotDiagnosticTrends',false);
    Options = ensureOnlineField(Options,'algorithmClass',"CBS_CGAN");
    Options = ensureOnlineField(Options,'algorithmParams', ...
        {1,1,20,2,50,32,1e-4,2e-4,0,1,1,4,3,2,1,0.10,1});

    Options.workerCount = max(1,round(double(Options.workerCount)));
    Options.problemNames = string(Options.problemNames(:));
    Options.variantNames = string(Options.variantNames(:));
    Options.N = max(1,round(double(Options.N)));
    if ~isempty(Options.D)
        Options.D = max(1,round(double(Options.D)));
    end
    Options.maxFE = max(1,round(double(Options.maxFE)));
    Options.runIds = unique(double(Options.runIds(:)'),'stable');
    Options.targets = unique(double(Options.targets(:)'),'stable');
    Options.targets = Options.targets(isfinite(Options.targets) & ...
        Options.targets > 0);
    Options.plotRun = round(double(Options.plotRun));
    Options.conditionMode = lower(strtrim(string(Options.conditionMode)));
    Options.trainZMode = lower(strtrim(string(Options.trainZMode)));
    Options.trainMode = lower(strtrim(string(Options.trainMode)));
    Options.sampleZMode = lower(strtrim(string(Options.sampleZMode)));
    Options.visualDiagnostics = logical(Options.visualDiagnostics);
    Options.plotDiagnosticTrends = logical(Options.plotDiagnosticTrends);
    Options.algorithmClass = string(Options.algorithmClass);
end

function S = ensureOnlineField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function Variants = onlineLossVariants(names)
    All = struct( ...
        'variant',{"A_ref_only_adv","B_ref_only_adv_huber", ...
            "C_ref_tau_adv"}, ...
        'conditionMode',{"ref_only","ref_only","ref_tau"}, ...
        'advWeight',{1,1,1}, ...
        'reconstructionWeight',{0,1,0});
    names = string(names(:));
    Variants = All([]);
    for i = 1 : numel(names)
        idx = find(string({All.variant}) == names(i),1,'first');
        if isempty(idx)
            error('CBSCGANOnlineLoss:UnknownVariant', ...
                'Unknown online loss variant: %s',names(i));
        end
        Variants(end+1) = All(idx); %#ok<AGROW>
    end
end

function T = readOptionalOnlineTable(fileName)
    if isfile(fileName)
        T = readtable(fileName,'TextType','string','Delimiter',',', ...
            'VariableNamingRule','preserve');
    else
        T = table();
    end
end

function T = addOnlineVariantColumns(T,variant)
    if isempty(T)
        T = table();
    end
    n = height(T);
    T.variant = repmat(string(variant.variant),n,1);
    T.condition_mode = repmat(string(variant.conditionMode),n,1);
    T.advWeight = repmat(double(variant.advWeight),n,1);
    T.reconstructionWeight = repmat(double(variant.reconstructionWeight),n,1);
    leading = {'variant','condition_mode','advWeight', ...
        'reconstructionWeight'};
    if width(T) > numel(leading)
        T = movevars(T,leading,'Before',1);
    end
end

function A = appendOnlineTable(A,B)
    if isempty(A) || width(A) == 0
        A = B;
    elseif ~isempty(B)
        A = [A;B];
    end
end

function Row = emptyOnlineVariantRow()
    Row = struct( ...
        'variant',"", ...
        'condition_mode',"", ...
        'advWeight',NaN, ...
        'reconstructionWeight',NaN, ...
        'variant_folder',"", ...
        'run_count',0, ...
        'ok_count',0, ...
        'failed_count',0, ...
        'stage_count',0, ...
        'history_count',0, ...
        'figure_count',0);
end

function Row = makeOnlineVariantRow(variant,variantDir,Summary,Stage, ...
        History,Figures)
    Row = emptyOnlineVariantRow();
    Row.variant = string(variant.variant);
    Row.condition_mode = string(variant.conditionMode);
    Row.advWeight = double(variant.advWeight);
    Row.reconstructionWeight = double(variant.reconstructionWeight);
    Row.variant_folder = string(variantDir);
    Row.run_count = height(Summary);
    if height(Summary) > 0 && ismember('status',Summary.Properties.VariableNames)
        Row.ok_count = sum(string(Summary.status) == "ok");
        Row.failed_count = sum(string(Summary.status) ~= "ok");
    end
    Row.stage_count = height(Stage);
    Row.history_count = height(History);
    Row.figure_count = height(Figures);
end

function T = finalStageRows(T)
    if isempty(T) || height(T) == 0 || ~ismember('target_FE',T.Properties.VariableNames)
        return;
    end
    keys = unique(T(:,{'variant','problem','run'}),'rows','stable');
    keep = false(height(T),1);
    for i = 1 : height(keys)
        idx = string(T.variant) == string(keys.variant(i)) & ...
            string(T.problem) == string(keys.problem(i)) & ...
            double(T.run) == double(keys.run(i));
        rows = find(idx);
        [~,best] = max(double(T.target_FE(rows)));
        keep(rows(best)) = true;
    end
    T = T(keep,:);
end

function T = aggregateOnlineMetrics(Raw,groupNames)
    if isempty(Raw) || height(Raw) == 0
        T = table();
        return;
    end
    metrics = fix2OnlineMetricNames();
    metrics = metrics(ismember(metrics,Raw.Properties.VariableNames));
    T = groupsummary(Raw,groupNames,'median',metrics);
end

function names = fix2OnlineMetricNames()
    names = {'train_x_rec90','train_y_rec90','boundary_dist90', ...
        'segment_width90','feasible_rate','side_rate','ref_cover', ...
        'train_tau_range','train_tau_nonzero_rate'};
end

function T = selectFix2OnlineColumns(T)
    if isempty(T) || width(T) == 0
        return;
    end
    idNames = {'variant','condition_mode','advWeight','reconstructionWeight', ...
        'problem','run','target_FE','actual_FE','gen','sample_z_mode'};
    keep = [idNames,fix2OnlineMetricNames()];
    keep = keep(ismember(keep,T.Properties.VariableNames));
    T = T(:,keep);
end
