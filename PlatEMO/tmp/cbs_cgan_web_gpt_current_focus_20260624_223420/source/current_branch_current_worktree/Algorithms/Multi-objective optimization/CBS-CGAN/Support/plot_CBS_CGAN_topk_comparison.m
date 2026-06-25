function FigureManifest = plot_CBS_CGAN_topk_comparison(rootDir)
%PLOT_CBS_CGAN_TOPK_COMPARISON Plot K3/K5 CBS-CGAN top-K comparison.

    if nargin < 1 || isempty(rootDir)
        rootDir = fullfile(pwd,'Data','CBS_CGAN', ...
            'sync_topk_K3_K5_runs3_noplot_20260622_215123');
    end
    rootDir = char(rootDir);
    analysisFile = fullfile(rootDir,'comparison_analysis_summary.csv');
    stageFile = fullfile(rootDir,'comparison_stage_metrics_all.csv');
    assert(isfile(analysisFile), ...
        'CBSTopKPlot:MissingAnalysis', ...
        'Missing comparison_analysis_summary.csv in %s.',rootDir);
    assert(isfile(stageFile), ...
        'CBSTopKPlot:MissingStageMetrics', ...
        'Missing comparison_stage_metrics_all.csv in %s.',rootDir);

    Analysis = readtable(analysisFile,'TextType','string', ...
        'VariableNamingRule','preserve');
    Stage = readtable(stageFile,'TextType','string', ...
        'VariableNamingRule','preserve');
    outDir = fullfile(rootDir,'plots_topk_comparison');
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    Rows = repmat(emptyFigureRow(),3,1);
    Rows(1) = saveProblemSummaryFigure(Analysis,outDir);
    Rows(2) = saveStageTrendFigure(Stage,outDir);
    Rows(3) = savePipelineTrendFigure(Stage,outDir);
    FigureManifest = struct2table(Rows);
    writetable(FigureManifest,fullfile(outDir,'figure_manifest.csv'));
end

function Row = saveProblemSummaryFigure(Analysis,outDir)
    variants = ["pair3","pair5"];
    labels = ["K3","K5"];
    problems = orderedProblems(Analysis.problem);
    x = categorical(problems);
    x = reordercats(x,cellstr(problems));

    train = valuesByProblemVariant(Analysis,problems,variants, ...
        'train_count_med');
    feasible = valuesByProblemVariant(Analysis,problems,variants, ...
        'feasible_rate_med');
    dist90 = valuesByProblemVariant(Analysis,problems,variants, ...
        'boundary_dist90_med');

    fig = figure('Visible','off','Color','w','Position',[100 100 1500 900]);
    layout = tiledlayout(fig,3,1,'TileSpacing','compact', ...
        'Padding','compact');
    title(layout,'CBS-CGAN synchronized top-K comparison by problem', ...
        'Interpreter','none','FontWeight','normal');

    ax = nexttile(layout);
    bar(ax,x,train,'grouped');
    ylabel(ax,'median train count');
    title(ax,'Training data size');
    decorateProblemAxis(ax,labels);

    ax = nexttile(layout);
    bar(ax,x,feasible,'grouped');
    ylabel(ax,'median feasible rate');
    ylim(ax,[0 1]);
    title(ax,'Generated feasible rate');
    decorateProblemAxis(ax,labels);

    ax = nexttile(layout);
    bar(ax,x,dist90,'grouped');
    set(ax,'YScale','log');
    ylabel(ax,'median boundary dist p90 (log)');
    title(ax,'Generated-to-boundary distance');
    decorateProblemAxis(ax,labels);

    Row = exportFigure(fig,outDir,'topk_problem_summary');
end

function Row = saveStageTrendFigure(Stage,outDir)
    variants = ["pair3","pair5"];
    labels = ["K3","K5"];
    focus = ~ismember(Stage.problem,["DASCMOP4_BC","DASCMOP5_BC"]);
    T = Stage(focus,:);
    targets = unique(double(T.target_FE));
    targets = targets(isfinite(targets));
    targets = sort(targets(:));

    fig = figure('Visible','off','Color','w','Position',[100 100 1400 760]);
    layout = tiledlayout(fig,1,3,'TileSpacing','compact', ...
        'Padding','compact');
    title(layout, ...
        'CBS-CGAN top-K stage trends, excluding DASCMOP4/5', ...
        'Interpreter','none','FontWeight','normal');

    plotTrendTile(nexttile(layout),T,targets,variants,labels, ...
        'train_count','median train count');
    plotTrendTile(nexttile(layout),T,targets,variants,labels, ...
        'feasible_rate','median feasible rate');
    plotTrendTile(nexttile(layout),T,targets,variants,labels, ...
        'boundary_dist90','median boundary dist p90');

    Row = exportFigure(fig,outDir,'topk_stage_trends_focus');
end

function Row = savePipelineTrendFigure(Stage,outDir)
    variants = ["pair3","pair5"];
    labels = ["K3","K5"];
    focus = ~ismember(Stage.problem,["DASCMOP4_BC","DASCMOP5_BC"]);
    T = Stage(focus,:);
    targets = unique(double(T.target_FE));
    targets = targets(isfinite(targets));
    targets = sort(targets(:));

    fig = figure('Visible','off','Color','w','Position',[100 100 1400 760]);
    layout = tiledlayout(fig,1,3,'TileSpacing','compact', ...
        'Padding','compact');
    title(layout, ...
        'CBS-CGAN top-K pipeline counts, excluding DASCMOP4/5', ...
        'Interpreter','none','FontWeight','normal');

    plotTrendTile(nexttile(layout),T,targets,variants,labels, ...
        'boundary_pair_appended_count','accepted F/I pairs');
    plotTrendTile(nexttile(layout),T,targets,variants,labels, ...
        'boundary_candidate_final_count','final BMem rows');
    plotTrendTile(nexttile(layout),T,targets,variants,labels, ...
        'train_count','training rows');

    Row = exportFigure(fig,outDir,'topk_pipeline_trends_focus');
end

function values = valuesByProblemVariant(T,problems,variants,fieldName)
    values = nan(numel(problems),numel(variants));
    for i = 1 : numel(problems)
        for j = 1 : numel(variants)
            idx = T.problem == problems(i) & T.variant == variants(j);
            if any(idx)
                values(i,j) = finiteMedian(T.(fieldName)(idx));
            end
        end
    end
end

function plotTrendTile(ax,T,targets,variants,labels,fieldName,labelText)
    hold(ax,'on');
    colors = [0.12 0.35 0.64;0.79 0.29 0.13];
    for j = 1 : numel(variants)
        y = nan(size(targets));
        for i = 1 : numel(targets)
            idx = T.variant == variants(j) & ...
                double(T.target_FE) == targets(i);
            y(i) = finiteMedian(T.(fieldName)(idx));
        end
        plot(ax,targets,y,'-o','LineWidth',1.6,'MarkerSize',5, ...
            'Color',colors(j,:),'DisplayName',labels(j));
    end
    hold(ax,'off');
    grid(ax,'on');
    box(ax,'on');
    xlabel(ax,'target FE');
    ylabel(ax,labelText,'Interpreter','none');
    title(ax,labelText,'Interpreter','none','FontWeight','normal');
    legend(ax,'Location','best','Box','off');
end

function decorateProblemAxis(ax,labels)
    grid(ax,'on');
    box(ax,'on');
    legend(ax,labels,'Location','best','Box','off');
    ax.TickLabelInterpreter = 'none';
    xtickangle(ax,35);
end

function problems = orderedProblems(problemColumn)
    preferred = ["DASCMOP1_BC","DASCMOP2_BC","DASCMOP4_BC", ...
        "DASCMOP5_BC","LIRCMOP5_BC","LIRCMOP6_BC", ...
        "LIRCMOP7_BC","LIRCMOP8_BC","LIRCMOP9_BC","LIRCMOP10_BC"];
    present = unique(string(problemColumn),'stable');
    problems = preferred(ismember(preferred,present));
    extra = present(~ismember(present,problems));
    problems = [problems(:);extra(:)];
end

function value = finiteMedian(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = median(x);
    end
end

function Row = exportFigure(fig,outDir,name)
    pngFile = fullfile(outDir,[name,'.png']);
    pdfFile = fullfile(outDir,[name,'.pdf']);
    exportgraphics(fig,pngFile,'Resolution',300);
    exportgraphics(fig,pdfFile,'ContentType','vector');
    close(fig);
    Row = emptyFigureRow();
    Row.name = string(name);
    Row.png_file = string(pngFile);
    Row.pdf_file = string(pdfFile);
end

function Row = emptyFigureRow()
    Row = struct('name',"",'png_file',"",'pdf_file',"");
end
