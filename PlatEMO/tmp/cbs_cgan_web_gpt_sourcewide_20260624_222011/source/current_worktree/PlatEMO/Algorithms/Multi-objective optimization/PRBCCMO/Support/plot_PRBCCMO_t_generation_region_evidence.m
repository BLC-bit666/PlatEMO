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

    Core = PRBCCMOUtils.numericizeTable(readtable(coreFile,'TextType','string'));
    Meta = readtable(metaFile,'TextType','string');
    PRBCCMOUtils.requireColumns(Core,{ ...
        'generation','fe','b_mean_pair_gap','b_p90_pair_gap','b_mean_mid_scalar','b_p90_mid_scalar', ...
        'mlp_trained','train_size','train_b_core_count','train_refinement_count','train_current_pop_count', ...
        'mlp_prob_bcore_feasible','mlp_prob_bcore_infeasible', ...
        'mlp_prob_curr_feasible','mlp_prob_curr_infeasible', ...
        'prob_accepted_refinement','prob_rejected_refinement', ...
        'mlp_two_inf_tournament_rate','mlp_effective_win_rate', ...
        'mlp_primary_resolution_rate','b_fallback_rate', ...
        'inf_ranked_count','inf_selected','inf_utility_gain'});
    if nargin < 3 || isempty(generation)
        generation = double(Core.generation(end));
    end

    idx = nearestGeneration(Core,generation);
    generation = double(Core.generation(idx));
    problemName = PRBCCMOUtils.metaString(Meta,'problem',"PRBCCMO_t");
    runId = PRBCCMOUtils.metaDouble(Meta,'run',NaN);

    fig = figure('Visible','off','Color','w','Position',[100 100 920 720]);
    ax1 = subplot(3,1,1,'Parent',fig);
    hold(ax1,'on');
    plotMetric(ax1,Core,'b_mean_pair_gap','-','mean B gap');
    plotMetric(ax1,Core,'b_p90_pair_gap','--','p90 B gap');
    plotMetric(ax1,Core,'b_mean_mid_scalar',':','mean mid scalar');
    markGeneration(ax1,Core,idx);
    ylabel(ax1,'B objective key');
    title(ax1,sprintf('%s run %.0f, generation %.0f, FE %.0f', ...
        char(problemName),runId,generation,double(Core.fe(idx))), ...
        'Interpreter','none');
    grid(ax1,'on');
    legend(ax1,'Location','best');

    ax2 = subplot(3,1,2,'Parent',fig);
    yyaxis(ax2,'left');
    plotMetric(ax2,Core,'mlp_trained','-','MLP trained event');
    markGeneration(ax2,Core,idx);
    ylabel(ax2,'MLP event');
    ylim(ax2,[0 1]);
    yyaxis(ax2,'right');
    hold(ax2,'on');
    plotMetric(ax2,Core,'train_size','-','T size');
    plotMetric(ax2,Core,'train_b_core_count','--','B-core samples');
    plotMetric(ax2,Core,'train_refinement_count',':','refinement outcome samples');
    plotMetric(ax2,Core,'train_current_pop_count','-.','current-pop samples');
    ylabel(ax2,'T source count');
    grid(ax2,'on');
    legend(ax2,'Location','best');

    ax3 = subplot(3,1,3,'Parent',fig);
    hold(ax3,'on');
    plotMetric(ax3,Core,'inf_ranked_count',':','ranked infeasible');
    plotMetric(ax3,Core,'inf_selected','-','selected infeasible');
    plotMetric(ax3,Core,'inf_utility_gain','--','selection utility gain');
    plotMetric(ax3,Core,'mlp_prob_curr_feasible','-.','current feasible prob');
    plotMetric(ax3,Core,'mlp_prob_curr_infeasible','-','current infeasible prob');
    plotMetric(ax3,Core,'prob_accepted_refinement',':','accepted refinement prob');
    plotMetric(ax3,Core,'prob_rejected_refinement','--','rejected refinement prob');
    plotMetric(ax3,Core,'mlp_two_inf_tournament_rate',':','two-infeasible tournament rate');
    plotMetric(ax3,Core,'mlp_effective_win_rate','--','MLP effective win rate');
    plotMetric(ax3,Core,'mlp_primary_resolution_rate','-','MLP primary resolution rate');
    plotMetric(ax3,Core,'b_fallback_rate','-.','B fallback rate');
    markGeneration(ax3,Core,idx);
    xlabel(ax3,'generation');
    ylabel(ax3,'I-I utility selection');
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
    folders = strings(numel(dirs),1);
    times = NaN(numel(dirs),1);
    count = 0;
    for i = 1 : numel(dirs)
        candidate = fullfile(dirs(i).folder,dirs(i).name);
        if isfile(fullfile(candidate,'core_metrics.csv'))
            count = count + 1;
            folders(count) = string(candidate);
            times(count) = dirs(i).datenum;
        end
    end
    folders = folders(1:count);
    times = times(1:count);
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
