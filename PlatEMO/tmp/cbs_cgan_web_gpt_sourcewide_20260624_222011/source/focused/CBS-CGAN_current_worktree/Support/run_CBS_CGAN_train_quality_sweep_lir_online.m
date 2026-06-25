function [Manifest,outDir,Summary,StageMetrics,FigureManifest] = ...
    run_CBS_CGAN_train_quality_sweep_lir_online(outDir,Options)
%RUN_CBS_CGAN_TRAIN_QUALITY_SWEEP_LIR_ONLINE Train-figure CGAN sweeps.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_CGAN', ...
            ['train_quality_sweep_lir_online_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(Options)
        Options = struct();
    end
    Options = normalizeSweepOptions(Options);
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    Summary = table();
    StageMetrics = table();
    FigureManifest = table();
    ManifestRows = repmat(emptySweepManifestRow(),numel(Options.settings),1);

    for i = 1 : numel(Options.settings)
        setting = Options.settings(i);
        settingDir = fullfile(outDir,char(setting.setting));
        RunOptions = makeBoundaryRunnerOptions(Options,setting);
        ManifestRows(i) = makeSweepManifestRow(setting,settingDir, ...
            RunOptions);
        try
            [RunSummary,~,RunStage,RunFigures] = ...
                run_CBS_CGAN_boundary_quality_experiments(settingDir, ...
                Options.workerCount,Options.problemNames,Options.N, ...
                Options.D,Options.maxFE,Options.runIds,Options.targets, ...
                RunOptions);
            RunSummary = addSweepColumns(RunSummary,setting,RunOptions);
            RunStage = addSweepColumns(RunStage,setting,RunOptions);
            RunFigures = addSweepColumns(RunFigures,setting,RunOptions);
            Summary = appendSweepTable(Summary,RunSummary);
            StageMetrics = appendSweepTable(StageMetrics,RunStage);
            FigureManifest = appendSweepTable(FigureManifest,RunFigures);
            ManifestRows(i).status = "ok";
            ManifestRows(i).run_count = height(RunSummary);
            ManifestRows(i).stage_count = height(RunStage);
            ManifestRows(i).figure_count = height(RunFigures);
            ManifestRows(i).summary_file = string(fullfile(settingDir, ...
                'run_summary.csv'));
            ManifestRows(i).stage_metrics_file = string(fullfile( ...
                settingDir,'stage_metrics_all.csv'));
            ManifestRows(i).figure_manifest_file = string(fullfile( ...
                settingDir,'figure_manifest.csv'));
        catch err
            ManifestRows(i).status = "failed";
            ManifestRows(i).error_message = string(getReport(err, ...
                'extended','hyperlinks','off'));
        end
    end

    Manifest = struct2table(ManifestRows);
    writetable(Manifest,fullfile(outDir,'train_quality_sweep_manifest.csv'));
    writetable(Summary,fullfile(outDir,'train_quality_sweep_summary.csv'));
    writetable(StageMetrics,fullfile(outDir, ...
        'train_quality_sweep_stage_metrics.csv'));
    writetable(FigureManifest,fullfile(outDir, ...
        'train_quality_sweep_figure_manifest.csv'));
end

function Options = normalizeSweepOptions(Options)
    Options = ensureSweepField(Options,'workerCount',10);
    Options = ensureSweepField(Options,'problemNames', ...
        ["LIRCMOP5_BC";"LIRCMOP6_BC";"LIRCMOP7_BC"; ...
        "LIRCMOP8_BC";"LIRCMOP9_BC";"LIRCMOP10_BC"]);
    Options = ensureSweepField(Options,'runIds',1);
    Options = ensureSweepField(Options,'N',100);
    Options = ensureSweepField(Options,'D',[]);
    Options = ensureSweepField(Options,'maxFE',100000);
    Options = ensureSweepField(Options,'targets', ...
        [10000 30000 50000 70000 100000]);
    Options = ensureSweepField(Options,'plotRun',1);
    Options = ensureSweepField(Options,'visualDiagnostics',true);
    Options = ensureSweepField(Options,'plotDiagnosticTrends',false);
    Options = ensureSweepField(Options,'figureKinds','train_reconstruction');
    Options = ensureSweepField(Options,'settings',defaultSweepSettings());

    Options.workerCount = max(1,round(double(Options.workerCount)));
    Options.problemNames = string(Options.problemNames(:));
    Options.runIds = unique(double(Options.runIds(:)'),'stable');
    Options.runIds = Options.runIds(isfinite(Options.runIds));
    Options.N = max(1,round(double(Options.N)));
    if ~isempty(Options.D)
        Options.D = max(1,round(double(Options.D)));
    end
    Options.maxFE = max(1,round(double(Options.maxFE)));
    Options.targets = unique(double(Options.targets(:)'),'stable');
    Options.targets = Options.targets(isfinite(Options.targets) & ...
        Options.targets > 0);
    Options.plotRun = round(double(Options.plotRun));
    Options.visualDiagnostics = logical(Options.visualDiagnostics);
    Options.plotDiagnosticTrends = logical(Options.plotDiagnosticTrends);
    Options.figureKinds = string(Options.figureKinds);
    Options.settings = normalizeSweepSettings(Options.settings);
end

function S = ensureSweepField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function Settings = defaultSweepSettings()
    Settings = [ ...
        makeDefaultSetting("A_ref_only_adv","ref_only",1,0); ...
        makeDefaultSetting("B_ref_only_adv_huber","ref_only",1,1); ...
        makeDefaultSetting("C_ref_tau_adv","ref_tau",1,0); ...
        makeDefaultSetting("D_ref_tau_adv_huber","ref_tau",1,1)];
end

function S = makeDefaultSetting(name,conditionMode,advWeight,reconstructionWeight)
    S = struct( ...
        'setting',string(name), ...
        'variant',string(name), ...
        'conditionMode',string(conditionMode), ...
        'advWeight',double(advWeight), ...
        'reconstructionWeight',double(reconstructionWeight), ...
        'nGen',30, ...
        'zDim',2, ...
        'ganIter',50, ...
        'ganMiniBatch',32, ...
        'trainMode',"iter", ...
        'trainZMode',"zero", ...
        'sampleZMode',"zero", ...
        'maxCandidatePairsPerRef',3, ...
        'boundaryTargetMode',"thin_support", ...
        'networkPreset',"default", ...
        'generatorHidden',[64 64 64], ...
        'discriminatorHidden',[64 64 32]);
end

function Settings = normalizeSweepSettings(Settings)
    if isempty(Settings)
        Settings = defaultSweepSettings();
        return;
    end
    Defaults = makeDefaultSetting("A_ref_only_adv","ref_only",1,0);
    for i = 1 : numel(Settings)
        hasGeneratorHidden = isfield(Settings,'generatorHidden') && ...
            ~isempty(Settings(i).generatorHidden);
        hasDiscriminatorHidden = isfield(Settings,'discriminatorHidden') && ...
            ~isempty(Settings(i).discriminatorHidden);
        fields = fieldnames(Defaults);
        for f = 1 : numel(fields)
            name = fields{f};
            if ~isfield(Settings,name) || isempty(Settings(i).(name))
                Settings(i).(name) = Defaults.(name);
            end
        end
        Settings(i).setting = string(Settings(i).setting);
        Settings(i).variant = string(Settings(i).variant);
        Settings(i).conditionMode = lower(strtrim(string( ...
            Settings(i).conditionMode)));
        Settings(i).advWeight = max(0,double(Settings(i).advWeight));
        Settings(i).reconstructionWeight = max(0,double( ...
            Settings(i).reconstructionWeight));
        Settings(i).nGen = max(0,round(double(Settings(i).nGen)));
        Settings(i).zDim = max(0,round(double(Settings(i).zDim)));
        Settings(i).ganIter = max(0,round(double(Settings(i).ganIter)));
        Settings(i).ganMiniBatch = max(1,round(double( ...
            Settings(i).ganMiniBatch)));
        Settings(i).trainMode = lower(strtrim(string(Settings(i).trainMode)));
        Settings(i).trainZMode = lower(strtrim(string( ...
            Settings(i).trainZMode)));
        Settings(i).sampleZMode = lower(strtrim(string( ...
            Settings(i).sampleZMode)));
        Settings(i).maxCandidatePairsPerRef = max(1,round(double( ...
            Settings(i).maxCandidatePairsPerRef)));
        Settings(i).boundaryTargetMode = lower(strtrim(string( ...
            Settings(i).boundaryTargetMode)));
        Settings(i).networkPreset = lower(strtrim(string( ...
            Settings(i).networkPreset)));
        [gHidden,dHidden] = hiddenFromPreset(Settings(i).networkPreset);
        if ~hasGeneratorHidden
            Settings(i).generatorHidden = gHidden;
        end
        if ~hasDiscriminatorHidden
            Settings(i).discriminatorHidden = dHidden;
        end
        Settings(i).generatorHidden = normalizeHidden( ...
            Settings(i).generatorHidden,gHidden);
        Settings(i).discriminatorHidden = normalizeHidden( ...
            Settings(i).discriminatorHidden,dHidden);
    end
end

function [gHidden,dHidden] = hiddenFromPreset(preset)
    switch lower(strtrim(string(preset)))
        case {"small","compact"}
            gHidden = [32 32];
            dHidden = [32 16];
        case {"wide","large"}
            gHidden = [128 128 128];
            dHidden = [128 128 64];
        otherwise
            gHidden = [64 64 64];
            dHidden = [64 64 32];
    end
end

function hidden = normalizeHidden(hidden,defaultValue)
    hidden = double(hidden(:)');
    hidden = hidden(isfinite(hidden) & hidden > 0);
    if isempty(hidden)
        hidden = defaultValue;
    else
        hidden = max(1,round(hidden));
    end
end

function RunOptions = makeBoundaryRunnerOptions(Options,setting)
    RunOptions = struct( ...
        'plotRun',Options.plotRun, ...
        'conditionMode',setting.conditionMode, ...
        'visualDiagnostics',Options.visualDiagnostics, ...
        'plotDiagnosticTrends',Options.plotDiagnosticTrends, ...
        'figureKinds',Options.figureKinds, ...
        'algorithmClass',"CBS_CGAN", ...
        'algorithmParams',{algorithmParamsFromSetting(setting)}, ...
        'advWeight',setting.advWeight, ...
        'reconstructionWeight',setting.reconstructionWeight, ...
        'trainZMode',setting.trainZMode, ...
        'trainMode',setting.trainMode, ...
        'sampleZMode',setting.sampleZMode, ...
        'generatorHidden',setting.generatorHidden, ...
        'discriminatorHidden',setting.discriminatorHidden, ...
        'boundaryTargetMode',setting.boundaryTargetMode);
end

function Params = algorithmParamsFromSetting(setting)
    Params = {1,1,setting.nGen,setting.zDim,setting.ganIter, ...
        setting.ganMiniBatch,1e-4,2e-4,0,1,1,4, ...
        setting.maxCandidatePairsPerRef,2,1,0.10, ...
        setting.reconstructionWeight};
end

function T = addSweepColumns(T,setting,RunOptions)
    if ~istable(T)
        T = table();
    end
    n = height(T);
    T.setting = repmat(string(setting.setting),n,1);
    T.variant = repmat(string(setting.variant),n,1);
    T.condition_mode = repmat(string(setting.conditionMode),n,1);
    T.advWeight = repmat(double(setting.advWeight),n,1);
    T.reconstructionWeight = repmat(double(setting.reconstructionWeight),n,1);
    T.nGen = repmat(double(setting.nGen),n,1);
    T.zDim = repmat(double(setting.zDim),n,1);
    T.ganIter = repmat(double(setting.ganIter),n,1);
    T.ganMiniBatch = repmat(double(setting.ganMiniBatch),n,1);
    T.trainMode = repmat(string(setting.trainMode),n,1);
    T.trainZMode = repmat(string(setting.trainZMode),n,1);
    T.sampleZMode = repmat(string(setting.sampleZMode),n,1);
    T.maxCandidatePairsPerRef = repmat(double( ...
        setting.maxCandidatePairsPerRef),n,1);
    T.boundaryTargetMode = repmat(string(setting.boundaryTargetMode),n,1);
    T.networkPreset = repmat(string(setting.networkPreset),n,1);
    T.generatorHidden = repmat(hiddenText(RunOptions.generatorHidden),n,1);
    T.discriminatorHidden = repmat(hiddenText( ...
        RunOptions.discriminatorHidden),n,1);
    leading = {'setting','variant','condition_mode','advWeight', ...
        'reconstructionWeight','nGen','zDim','ganIter','ganMiniBatch', ...
        'trainMode','trainZMode','sampleZMode', ...
        'maxCandidatePairsPerRef','boundaryTargetMode','networkPreset', ...
        'generatorHidden','discriminatorHidden'};
    trailing = setdiff(T.Properties.VariableNames,leading,'stable');
    T = T(:,[leading,trailing]);
end

function T = appendSweepTable(T,Extra)
    if isempty(T) || width(T) == 0
        T = Extra;
    elseif ~isempty(Extra) && width(Extra) > 0
        T = [T;Extra]; %#ok<AGROW>
    end
end

function Row = makeSweepManifestRow(setting,settingDir,RunOptions)
    Row = emptySweepManifestRow();
    Row.setting = string(setting.setting);
    Row.variant = string(setting.variant);
    Row.condition_mode = string(setting.conditionMode);
    Row.advWeight = double(setting.advWeight);
    Row.reconstructionWeight = double(setting.reconstructionWeight);
    Row.nGen = double(setting.nGen);
    Row.zDim = double(setting.zDim);
    Row.ganIter = double(setting.ganIter);
    Row.ganMiniBatch = double(setting.ganMiniBatch);
    Row.trainMode = string(setting.trainMode);
    Row.trainZMode = string(setting.trainZMode);
    Row.sampleZMode = string(setting.sampleZMode);
    Row.maxCandidatePairsPerRef = double(setting.maxCandidatePairsPerRef);
    Row.boundaryTargetMode = string(setting.boundaryTargetMode);
    Row.networkPreset = string(setting.networkPreset);
    Row.generatorHidden = hiddenText(RunOptions.generatorHidden);
    Row.discriminatorHidden = hiddenText(RunOptions.discriminatorHidden);
    Row.setting_folder = string(settingDir);
end

function Row = emptySweepManifestRow()
    Row = struct( ...
        'setting',"", ...
        'variant',"", ...
        'condition_mode',"", ...
        'advWeight',NaN, ...
        'reconstructionWeight',NaN, ...
        'nGen',NaN, ...
        'zDim',NaN, ...
        'ganIter',NaN, ...
        'ganMiniBatch',NaN, ...
        'trainMode',"", ...
        'trainZMode',"", ...
        'sampleZMode',"", ...
        'maxCandidatePairsPerRef',NaN, ...
        'boundaryTargetMode',"", ...
        'networkPreset',"", ...
        'generatorHidden',"", ...
        'discriminatorHidden',"", ...
        'setting_folder',"", ...
        'run_count',0, ...
        'stage_count',0, ...
        'figure_count',0, ...
        'summary_file',"", ...
        'stage_metrics_file',"", ...
        'figure_manifest_file',"", ...
        'status',"pending", ...
        'error_message',"");
end

function text = hiddenText(hidden)
    text = strjoin(string(double(hidden(:)')),':');
end
