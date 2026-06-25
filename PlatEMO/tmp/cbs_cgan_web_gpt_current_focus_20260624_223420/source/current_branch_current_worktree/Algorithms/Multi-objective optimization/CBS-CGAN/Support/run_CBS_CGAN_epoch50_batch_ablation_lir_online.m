function [Manifest,outDir,Summary,Raw,FigureManifest] = ...
    run_CBS_CGAN_epoch50_batch_ablation_lir_online(outDir,Options)
%RUN_CBS_CGAN_EPOCH50_BATCH_ABLATION_LIR_ONLINE Compare CBS-CGAN epoch batches.
%   Runs A1/A2 only: epoch=50 with batch size 16 and 32, V0/V4, run1.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_CGAN', ...
            ['loss_ablation_lir_online_epoch50_batches_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(Options)
        Options = struct();
    end
    Options = normalizeEpochBatchOptions(Options);
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    Settings = epochBatchSettings();
    Summary = table();
    Raw = table();
    FigureManifest = table();
    ManifestRows = repmat(emptyEpochBatchManifestRow(),numel(Settings),1);

    for i = 1 : numel(Settings)
        setting = Settings(i);
        settingDir = fullfile(outDir,char(setting.setting));
        RunOptions = makeEpochBatchRunOptions(Options,setting);
        ManifestRows(i) = makeEpochBatchManifestRow(setting,settingDir, ...
            RunOptions);
        try
            [RunSummary,~,~,RunRaw,RunFigures] = ...
                run_CBS_CGAN_loss_ablation_lir_online(settingDir,RunOptions);
            RunSummary = addEpochBatchColumns(RunSummary,setting);
            RunRaw = addEpochBatchColumns(RunRaw,setting);
            RunFigures = addEpochBatchColumns(RunFigures,setting);
            Summary = appendEpochBatchTable(Summary,RunSummary);
            Raw = appendEpochBatchTable(Raw,RunRaw);
            FigureManifest = appendEpochBatchTable(FigureManifest,RunFigures);
            ManifestRows(i).status = "ok";
            ManifestRows(i).summary_file = string(fullfile(settingDir, ...
                'online_loss_ablation_summary.csv'));
            ManifestRows(i).raw_file = string(fullfile(settingDir, ...
                'online_loss_ablation_raw.csv'));
            ManifestRows(i).figure_manifest_file = string(fullfile( ...
                settingDir,'figure_manifest.csv'));
            ManifestRows(i).figure_count = height(RunFigures);
        catch err
            ManifestRows(i).status = "failed";
            ManifestRows(i).error_message = string(getReport(err, ...
                'extended','hyperlinks','off'));
        end
    end

    Manifest = struct2table(ManifestRows);
    writetable(Manifest,fullfile(outDir, ...
        'online_epoch50_batch_ablation_manifest.csv'));
    writetable(Summary,fullfile(outDir, ...
        'online_epoch50_batch_ablation_summary.csv'));
    writetable(Raw,fullfile(outDir, ...
        'online_epoch50_batch_ablation_raw.csv'));
    writetable(FigureManifest,fullfile(outDir, ...
        'online_epoch50_batch_ablation_figure_manifest.csv'));
end

function Options = normalizeEpochBatchOptions(Options)
    Options = ensureEpochBatchField(Options,'workerCount',10);
    Options = ensureEpochBatchField(Options,'problemNames', ...
        ["LIRCMOP5_BC";"LIRCMOP6_BC";"LIRCMOP7_BC"; ...
        "LIRCMOP8_BC";"LIRCMOP9_BC";"LIRCMOP10_BC"]);
    Options = ensureEpochBatchField(Options,'variantNames', ...
        ["A_ref_only_adv","B_ref_only_adv_huber","C_ref_tau_adv"]);
    Options = ensureEpochBatchField(Options,'runIds',1);
    Options = ensureEpochBatchField(Options,'N',100);
    Options = ensureEpochBatchField(Options,'D',[]);
    Options = ensureEpochBatchField(Options,'maxFE',100000);
    Options = ensureEpochBatchField(Options,'targets', ...
        [10000 30000 50000 70000 100000]);
    Options = ensureEpochBatchField(Options,'conditionMode',"");
    Options = ensureEpochBatchField(Options,'trainZMode',"zero");
    Options = ensureEpochBatchField(Options,'sampleZMode',"zero");
    Options = ensureEpochBatchField(Options,'visualDiagnostics',false);
    Options = ensureEpochBatchField(Options,'plotDiagnosticTrends',false);

    Options.workerCount = max(1,round(double(Options.workerCount)));
    Options.problemNames = string(Options.problemNames(:));
    Options.variantNames = string(Options.variantNames(:));
    Options.runIds = unique(double(Options.runIds(:)'),'stable');
    Options.N = max(1,round(double(Options.N)));
    if ~isempty(Options.D)
        Options.D = max(1,round(double(Options.D)));
    end
    Options.maxFE = max(1,round(double(Options.maxFE)));
    Options.targets = unique(double(Options.targets(:)'),'stable');
    Options.targets = Options.targets(isfinite(Options.targets) & ...
        Options.targets > 0);
    Options.conditionMode = lower(strtrim(string(Options.conditionMode)));
    Options.trainZMode = lower(strtrim(string(Options.trainZMode)));
    Options.sampleZMode = lower(strtrim(string(Options.sampleZMode)));
    Options.visualDiagnostics = logical(Options.visualDiagnostics);
    Options.plotDiagnosticTrends = logical(Options.plotDiagnosticTrends);
end

function Settings = epochBatchSettings()
    Settings = struct( ...
        'setting',{"A1_epoch50_batch16","A2_epoch50_batch32"}, ...
        'trainMode',{"epoch","epoch"}, ...
        'ganIter',{50,50}, ...
        'ganMiniBatch',{16,32});
end

function RunOptions = makeEpochBatchRunOptions(Options,setting)
    RunOptions = struct( ...
        'workerCount',Options.workerCount, ...
        'problemNames',Options.problemNames, ...
        'variantNames',Options.variantNames, ...
        'N',Options.N, ...
        'D',Options.D, ...
        'maxFE',Options.maxFE, ...
        'runIds',Options.runIds, ...
        'targets',Options.targets, ...
        'plotRun',1, ...
        'conditionMode',Options.conditionMode, ...
        'trainZMode',Options.trainZMode, ...
        'trainMode',setting.trainMode, ...
        'sampleZMode',Options.sampleZMode, ...
        'visualDiagnostics',Options.visualDiagnostics, ...
        'plotDiagnosticTrends',Options.plotDiagnosticTrends, ...
        'algorithmClass',"CBS_CGAN", ...
        'algorithmParams',{algorithmParamsForBatch(setting.ganMiniBatch)});
end

function Params = algorithmParamsForBatch(batchSize)
    Params = {1,1,20,2,50,round(double(batchSize)), ...
        1e-4,2e-4,0,1,1,4,3,2,1,0.10,1};
end

function T = addEpochBatchColumns(T,setting)
    n = height(T);
    T.setting = repmat(string(setting.setting),n,1);
    T.trainMode = repmat(string(setting.trainMode),n,1);
    T.ganIter = repmat(double(setting.ganIter),n,1);
    T.ganMiniBatch = repmat(double(setting.ganMiniBatch),n,1);
    leading = {'setting','trainMode','ganIter','ganMiniBatch'};
    trailing = setdiff(T.Properties.VariableNames,leading,'stable');
    T = T(:,[leading,trailing]);
end

function T = appendEpochBatchTable(T,Extra)
    if isempty(T) || width(T) == 0
        T = Extra;
    elseif ~isempty(Extra) && width(Extra) > 0
        T = [T;Extra];
    end
end

function Row = makeEpochBatchManifestRow(setting,settingDir,RunOptions)
    Row = emptyEpochBatchManifestRow();
    Row.setting = string(setting.setting);
    Row.trainMode = string(setting.trainMode);
    Row.ganIter = double(setting.ganIter);
    Row.ganMiniBatch = double(setting.ganMiniBatch);
    Row.workerCount = double(RunOptions.workerCount);
    Row.problem_count = double(numel(RunOptions.problemNames));
    Row.run_count = double(numel(RunOptions.runIds));
    Row.variant_count = double(numel(RunOptions.variantNames));
    Row.visualDiagnostics = logical(RunOptions.visualDiagnostics);
    Row.setting_folder = string(settingDir);
end

function Row = emptyEpochBatchManifestRow()
    Row = struct( ...
        'setting',"", ...
        'trainMode',"", ...
        'ganIter',NaN, ...
        'ganMiniBatch',NaN, ...
        'workerCount',NaN, ...
        'problem_count',NaN, ...
        'run_count',NaN, ...
        'variant_count',NaN, ...
        'visualDiagnostics',false, ...
        'setting_folder',"", ...
        'summary_file',"", ...
        'raw_file',"", ...
        'figure_manifest_file',"", ...
        'figure_count',0, ...
        'status',"", ...
        'error_message',"");
end

function S = ensureEpochBatchField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end
