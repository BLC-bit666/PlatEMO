function OutputFile = plot_PRBCCMO_t_generation_region_evidence(runFolder,outDir,generation)
% Plot the three PRBCCMO_t core observations for one run.

    if nargin < 1 || isempty(runFolder)
        runFolder = latestCoreTraceFolder();
    end
    runFolder = char(string(runFolder));
    coreFile = fullfile(runFolder,'core_metrics.csv');
    metaFile = fullfile(runFolder,'run_meta.csv');
    assert(isfile(coreFile) && isfile(metaFile), ...
        'plot_PRBCCMO_t_generation_region_evidence:MissingCsv', ...
        'Run folder must contain run_meta.csv and core_metrics.csv: %s', runFolder);

    if nargin < 2 || isempty(outDir)
        outDir = fullfile(runFolder,'core_metric_figures');
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    Core = numericizeTable(readtable(coreFile,'TextType','string'));
    Meta = readtable(metaFile,'TextType','string');
    requireColumns(Core,{ ...
        'generation','fe','b_mean_pair_gap','b_p90_pair_gap', ...
        'mlp_train_acc','train_size','train_src_refinement','train_src_boundary_band', ...
        'inf_prob_gain','inf_carry_prob_gain'});
    if nargin < 3 || isempty(generation)
        generation = double(Core.generation(end));
    end

    idx = nearestGeneration(Core,generation);
    generation = double(Core.generation(idx));
    problemName = metaString(Meta,'problem',"PRBCCMO_t");
    runId = metaDouble(Meta,'run',NaN);

    fig = figure('Visible','off','Color','w','Position',[100 100 920 720]);
    ax1 = subplot(3,1,1,'Parent',fig);
    hold(ax1,'on');
    plotMetric(ax1,Core,'b_mean_pair_gap','-','mean B gap');
    plotMetric(ax1,Core,'b_p90_pair_gap','--','p90 B gap');
    markGeneration(ax1,Core,idx);
    ylabel(ax1,'B gap');
    title(ax1,sprintf('%s run %.0f, generation %.0f, FE %.0f', ...
        char(problemName),runId,generation,double(Core.fe(idx))), ...
        'Interpreter','none');
    grid(ax1,'on');
    legend(ax1,'Location','best');

    ax2 = subplot(3,1,2,'Parent',fig);
    yyaxis(ax2,'left');
    plotMetric(ax2,Core,'mlp_train_acc','-','training accuracy');
    markGeneration(ax2,Core,idx);
    ylabel(ax2,'MLP acc');
    ylim(ax2,[0 1]);
    yyaxis(ax2,'right');
    hold(ax2,'on');
    plotMetric(ax2,Core,'train_src_refinement','--','refinement samples');
    plotMetric(ax2,Core,'train_src_boundary_band',':','boundary-band samples');
    ylabel(ax2,'T source count');
    grid(ax2,'on');
    legend(ax2,'Location','best');

    ax3 = subplot(3,1,3,'Parent',fig);
    hold(ax3,'on');
    plotMetric(ax3,Core,'inf_prob_gain','--','fill probability gain');
    plotMetric(ax3,Core,'inf_carry_prob_gain','-','carry probability gain');
    markGeneration(ax3,Core,idx);
    xlabel(ax3,'generation');
    ylabel(ax3,'infeasible gain');
    grid(ax3,'on');
    legend(ax3,'Location','best');

    OutputFile = string(fullfile(outDir,sprintf('%s_run%.0f_gen%06.0f_core_metrics.png', ...
        char(problemName),runId,generation)));
    saveFigure(fig,OutputFile);
    close(fig);
end

function folder = latestCoreTraceFolder()
    rootDir = fileparts(which('platemo'));
    baseDir = fullfile(rootDir,'Data','PRBCCMO_t');
    dirs = dir(baseDir);
    dirs = dirs([dirs.isdir]);
    dirs = dirs(~ismember({dirs.name},{'.','..'}));
    folders = strings(0,1);
    times = zeros(0,1);
    for i = 1 : numel(dirs)
        candidate = fullfile(dirs(i).folder,dirs(i).name);
        if isfile(fullfile(candidate,'core_metrics.csv'))
            folders(end+1,1) = string(candidate); %#ok<AGROW>
            times(end+1,1) = dirs(i).datenum; %#ok<AGROW>
        end
    end
    assert(~isempty(folders), ...
        'plot_PRBCCMO_t_generation_region_evidence:NoCoreTrace', ...
        'No PRBCCMO_t core trace folder found under %s.', baseDir);
    [~,idx] = max(times);
    folder = char(folders(idx));
end

function idx = nearestGeneration(Core,generation)
    [~,idx] = min(abs(double(Core.generation) - double(generation)));
end

function plotMetric(ax,T,Name,LineStyle,Label)
    plot(ax,double(T.generation),double(T.(Name)),LineStyle, ...
        'LineWidth',1.5,'DisplayName',Label);
end

function markGeneration(ax,T,idx)
    X = double(T.generation(idx));
    yl = ylim(ax);
    line(ax,[X X],yl,'Color',[0.25 0.25 0.25],'LineStyle','-.', ...
        'LineWidth',1.0,'HandleVisibility','off');
end

function saveFigure(fig,OutputFile)
    try
        exportgraphics(fig,char(OutputFile),'Resolution',220);
    catch
        saveas(fig,char(OutputFile));
    end
end

function requireColumns(T,Names)
    Missing = setdiff(string(Names),string(T.Properties.VariableNames));
    assert(isempty(Missing), ...
        'plot_PRBCCMO_t_generation_region_evidence:MissingColumns', ...
        'Missing columns: %s', strjoin(cellstr(Missing), ', '));
end

function value = metaString(Meta,Name,Default)
    if hasColumns(Meta,{Name}) && height(Meta) > 0
        value = string(Meta.(Name)(1));
    else
        value = string(Default);
    end
end

function value = metaDouble(Meta,Name,Default)
    value = Default;
    if hasColumns(Meta,{Name}) && height(Meta) > 0
        raw = Meta.(Name);
        parsed = str2double(string(raw(1)));
        if ~isnan(parsed)
            value = parsed;
        elseif isnumeric(raw)
            value = double(raw(1));
        end
    end
end

function Flag = hasColumns(T,Names)
    Flag = all(ismember(string(Names),string(T.Properties.VariableNames)));
end

function T = numericizeTable(T)
    Names = T.Properties.VariableNames;
    for i = 1 : numel(Names)
        Value = T.(Names{i});
        if iscell(Value) || isstring(Value)
            Num = str2double(string(Value));
            if any(~isnan(Num))
                T.(Names{i}) = Num;
            end
        elseif islogical(Value)
            T.(Names{i}) = double(Value);
        end
    end
end
