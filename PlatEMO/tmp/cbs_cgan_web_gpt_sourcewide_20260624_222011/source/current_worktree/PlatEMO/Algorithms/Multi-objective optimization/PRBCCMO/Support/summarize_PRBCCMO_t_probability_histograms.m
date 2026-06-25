function HistTable = summarize_PRBCCMO_t_probability_histograms(inputPath,outDir)
% Summarize per-sample MLP probability decile histograms from core_metrics.csv.

    if nargin < 1 || isempty(inputPath)
        rootDir = fileparts(which('platemo'));
        inputPath = fullfile(rootDir,'Data','PRBCCMO_t');
    end
    if nargin < 2
        outDir = "";
    end

    coreFiles = resolveCoreFiles(inputPath);
    Scopes = probabilityHistogramScopes();
    Bins = probabilityHistogramBins();
    Counts = zeros(numel(Scopes),numel(Bins));

    for i = 1 : numel(coreFiles)
        Core = PRBCCMOUtils.numericizeTable(readtable(coreFiles{i},'TextType','string'));
        for s = 1 : numel(Scopes)
            Columns = probabilityHistogramColumns(Scopes(s).prefix);
            PRBCCMOUtils.requireColumns(Core,Columns,coreFiles{i});
            for b = 1 : numel(Bins)
                Counts(s,b) = Counts(s,b) + PRBCCMOUtils.sumFinite(Core.(Columns{b}));
            end
        end
    end

    RowCount = numel(Scopes)*numel(Bins);
    HistTable = table('Size',[RowCount,4], ...
        'VariableTypes',{'string','string','double','double'}, ...
        'VariableNames',{'sample_scope','bin','count','fraction'});
    r = 0;
    for s = 1 : numel(Scopes)
        ScopeTotal = sum(Counts(s,:));
        for b = 1 : numel(Bins)
            r = r + 1;
            HistTable.sample_scope(r) = Scopes(s).name;
            HistTable.bin(r) = Bins(b);
            HistTable.count(r) = Counts(s,b);
            HistTable.fraction(r) = Counts(s,b) / max(ScopeTotal,1);
        end
    end

    if strlength(string(outDir)) > 0
        if ~isfolder(outDir)
            mkdir(outDir);
        end
        writetable(HistTable,fullfile(char(string(outDir)),'metric_mlp_probability_hist_counts.csv'));
    end
end

function coreFiles = resolveCoreFiles(inputPath)
    if istable(inputPath)
        T = inputPath;
        if PRBCCMOUtils.hasColumns(T,{'analysis_core_csv'})
            coreFiles = cellstr(string(T.analysis_core_csv(:)));
        else
            PRBCCMOUtils.requireColumns(T,{'analysis_folder'});
            coreFiles = fullfile(cellstr(string(T.analysis_folder(:))),'core_metrics.csv');
        end
        coreFiles = coreFiles(cellfun(@isfile,coreFiles));
        return;
    end

    if ischar(inputPath) || (isstring(inputPath) && isscalar(inputPath))
        folder = char(string(inputPath));
        if isfolder(folder) && isfile(fullfile(folder,'core_metrics.csv'))
            coreFiles = {fullfile(folder,'core_metrics.csv')};
            return;
        end
        assert(isfolder(folder), ...
            'summarize_PRBCCMO_t_probability_histograms:MissingInputFolder', ...
            'Input path not found: %s', folder);
        dirs = dir(folder);
        dirs = dirs([dirs.isdir]);
        dirs = dirs(~ismember({dirs.name},{'.','..'}));
        candidates = fullfile({dirs.folder},{dirs.name},'core_metrics.csv');
        coreFiles = reshape(candidates(cellfun(@isfile,candidates)),[],1);
    else
        folders = cellstr(string(inputPath(:)));
        coreFiles = fullfile(folders,'core_metrics.csv');
        coreFiles = coreFiles(cellfun(@isfile,coreFiles));
    end

    assert(~isempty(coreFiles), ...
        'summarize_PRBCCMO_t_probability_histograms:NoCoreFiles', ...
        'No PRBCCMO_t core_metrics.csv files were found.');
end

function Scopes = probabilityHistogramScopes()
    Scopes = struct( ...
        'name',{"train","bcore_feasible","bcore_infeasible","curr_feasible","curr_infeasible"}, ...
        'prefix',{"mlp_prob_train_hist","mlp_prob_bcore_feasible_hist", ...
            "mlp_prob_bcore_infeasible_hist","mlp_prob_curr_feasible_hist", ...
            "mlp_prob_curr_infeasible_hist"});
end

function Bins = probabilityHistogramBins()
    Bins = ["0.0-0.1","0.1-0.2","0.2-0.3","0.3-0.4","0.4-0.5", ...
        "0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"];
end

function Columns = probabilityHistogramColumns(Prefix)
    Suffixes = {'00_10','10_20','20_30','30_40','40_50', ...
        '50_60','60_70','70_80','80_90','90_100'};
    Columns = strcat(char(Prefix),'_',Suffixes);
end
