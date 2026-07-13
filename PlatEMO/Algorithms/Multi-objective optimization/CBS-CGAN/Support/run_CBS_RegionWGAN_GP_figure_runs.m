function [Summary,FigureManifest,RunSummaryManifest,outDir] = ...
    run_CBS_RegionWGAN_GP_figure_runs(outDir,workerCount,runIds,Config)
%RUN_CBS_REGIONWGAN_GP_FIGURE_RUNS Run and plot the current mainline.
%   Each problem/run produces one four-stage trajectory figure.  The latest
%   captured stage of all problems is then assembled into one summary figure
%   per run.  Plotting only reads saved stage snapshots and never evaluates
%   additional solutions.

    rootDir = locateRootDir();
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_RegionGAN_compare', ...
            ['mainline_figures_runs2_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 10;
    end
    if nargin < 3 || isempty(runIds)
        runIds = 1 : 2;
    end
    if nargin < 4 || isempty(Config)
        Config = struct();
    end
    Config = normalizeFigureRunConfig(Config);
    runIds = unique(round(double(runIds(:)')),'stable');

    Options = struct( ...
        'captureRun',runIds, ...
        'stageTargets',Config.stageTargets, ...
        'captureWGANTrainHistory',false, ...
        'wganMappingDiagnostics',false, ...
        'bmemLearnabilityDiagnostics',false, ...
        'bmemMode',"legacy", ...
        'trainDedupMode',"off", ...
        'structuredZMode',"off", ...
        'schemaVersion',"cbs_region_wgan_mainline_figures_v1");

    if Config.renderOnly
        summaryFile = fullfile(outDir,'run_summary.csv');
        if ~isfile(summaryFile)
            error('CBSRegionGAN:MissingFigureRunSummary', ...
                'Cannot redraw figures without %s.',summaryFile);
        end
        Summary = readtable(summaryFile,'TextType','string', ...
            'Delimiter',',','VariableNamingRule','preserve');
    else
        [Summary,~,~,outDir] = run_CBS_RegionWGAN_GP_mainline( ...
            outDir,workerCount,Config.problemNames,Config.N,Config.D, ...
            Config.maxFE,runIds,Options);
    end
    validateCompletedRuns(Summary,Config,runIds);
    [FigureManifest,RunSummaryManifest] = renderSavedRunFigures( ...
        Summary,outDir,Config,runIds);
    writetable(FigureManifest,fullfile(outDir,'figure_manifest.csv'));
    writetable(RunSummaryManifest,fullfile(outDir, ...
        'run_summary_figure_manifest.csv'));
end

function Config = normalizeFigureRunConfig(Config)
    Config = defaultField(Config,'problemNames', ...
        ["LIRCMOP5_BC";"LIRCMOP6_BC";"LIRCMOP7_BC"; ...
        "LIRCMOP8_BC";"LIRCMOP9_BC";"LIRCMOP10_BC"]);
    Config = defaultField(Config,'N',100);
    Config = defaultField(Config,'D',30);
    Config = defaultField(Config,'maxFE',100000);
    Config = defaultField(Config,'stageTargets',[20000 40000 60000 80000]);
    Config = defaultField(Config,'renderOnly',false);
    Config.problemNames = string(Config.problemNames(:));
    Config.N = max(1,round(double(Config.N)));
    Config.D = max(1,round(double(Config.D)));
    Config.maxFE = max(1,round(double(Config.maxFE)));
    Config.renderOnly = logical(Config.renderOnly);
    if ~isscalar(Config.renderOnly)
        error('CBSRegionGAN:BadFigureRenderOnly', ...
            'Config.renderOnly must be scalar.');
    end
    Config.stageTargets = unique(round(double(Config.stageTargets(:)')), ...
        'stable');
    Config.stageTargets = Config.stageTargets( ...
        isfinite(Config.stageTargets) & Config.stageTargets > 0 & ...
        Config.stageTargets <= Config.maxFE);
    if isempty(Config.problemNames) || isempty(Config.stageTargets)
        error('CBSRegionGAN:BadFigureRunConfig', ...
            'At least one problem and one valid stage target are required.');
    end
end

function S = defaultField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function validateCompletedRuns(Summary,Config,runIds)
    expected = numel(Config.problemNames)*numel(runIds);
    if height(Summary) ~= expected
        error('CBSRegionGAN:IncompleteFigureRuns', ...
            'Expected %d runs but found %d.',expected,height(Summary));
    end
    if any(string(Summary.status) ~= "ok")
        bad = Summary(string(Summary.status) ~= "ok",:);
        error('CBSRegionGAN:FailedFigureRuns', ...
            'At least one mainline run failed: %s.', ...
            strjoin(cellstr(string(bad.problem) + " run=" + ...
            string(bad.run)),', '));
    end
    if any(double(Summary.finalFE) ~= Config.maxFE)
        error('CBSRegionGAN:InexactFigureRunFE', ...
            'All figure runs must terminate at finalFE=maxFE.');
    end
end

function [FigureManifest,RunSummaryManifest] = renderSavedRunFigures( ...
        Summary,outDir,Config,runIds)
    FigureRows = repmat(emptyFigureRow(),height(Summary),1);
    for i = 1 : height(Summary)
        [Problem,Snapshots] = loadRunPlotInputs(Summary(i,:),Config);
        runFolder = char(Summary.run_folder(i));
        figureFolder = fullfile(runFolder,'figures');
        if ~isfolder(figureFolder)
            mkdir(figureFolder);
        end
        figureFile = fullfile(figureFolder,sprintf( ...
            '%s_run%d_stage_trajectory.png',char(Summary.problem(i)), ...
            round(Summary.run(i))));
        plotProblemTrajectory(Problem,Snapshots,Summary.problem(i), ...
            Summary.run(i),figureFile,Config);
        Snapshot = latestGeneratedSnapshot(Snapshots);
        panelFile = fullfile(figureFolder,sprintf( ...
            '%s_run%d_latest_panel.png',char(Summary.problem(i)), ...
            round(Summary.run(i))));
        plotLatestSnapshotPanel(Problem,Snapshot,Summary.problem(i), ...
            Summary.run(i),panelFile);
        FigureRows(i).problem = string(Summary.problem(i));
        FigureRows(i).run = double(Summary.run(i));
        FigureRows(i).stage_count = numel(Snapshots);
        FigureRows(i).latest_target_FE = max([Snapshots.target_FE]);
        FigureRows(i).latest_actual_FE = max([Snapshots.actual_FE]);
        FigureRows(i).figure_file = string(figureFile);
        FigureRows(i).summary_panel_file = string(panelFile);
    end
    FigureManifest = struct2table(FigureRows);

    SummaryRows = repmat(emptyRunSummaryRow(),numel(runIds),1);
    for i = 1 : numel(runIds)
        runId = runIds(i);
        panelRows = FigureManifest(double(FigureManifest.run) == runId,:);
        panelRows = orderProblemRows(panelRows,Config.problemNames);
        figureFile = fullfile(outDir,sprintf( ...
            'CBS_RegionWGAN_GP_run%d_summary.png',runId));
        tileRunTrajectoryFigures(panelRows,figureFile);
        SummaryRows(i).run = runId;
        SummaryRows(i).problem_count = height(panelRows);
        SummaryRows(i).target_FE = NaN;
        SummaryRows(i).figure_file = string(figureFile);
    end
    RunSummaryManifest = struct2table(SummaryRows);
end

function Rows = orderProblemRows(Rows,problemNames)
    order = zeros(height(Rows),1);
    for i = 1 : height(Rows)
        idx = find(problemNames == string(Rows.problem(i)),1);
        if isempty(idx)
            idx = numel(problemNames) + i;
        end
        order(i) = idx;
    end
    [~,idx] = sort(order);
    Rows = Rows(idx,:);
end

function [Problem,Snapshots] = loadRunPlotInputs(Row,Config)
    Data = load(char(Row.metric_file),'metric');
    if ~isfield(Data,'metric') || ...
            ~isfield(Data.metric,'region_gan_stage_snapshots') || ...
            isempty(Data.metric.region_gan_stage_snapshots)
        error('CBSRegionGAN:MissingFigureSnapshots', ...
            'No stage snapshots found for %s run=%d.', ...
            char(Row.problem),round(Row.run));
    end
    Snapshots = Data.metric.region_gan_stage_snapshots;
    [~,idx] = sort([Snapshots.target_FE]);
    Snapshots = Snapshots(idx);
    Constructor = str2func(char(Row.problem));
    Problem = Constructor('N',Config.N,'D',Config.D, ...
        'maxFE',Config.maxFE);
end

function plotProblemTrajectory(Problem,Snapshots,problemName,runId, ...
        figureFile,Config)
    count = numel(Snapshots);
    columns = 2;
    rows = ceil(count/columns);
    fig = figure('Visible','off','Color','w', ...
        'Position',[50 50 1500 max(720,600*rows)]);
    layout = tiledlayout(fig,rows,columns,'TileSpacing','compact', ...
        'Padding','compact');
    [xLimits,yLimits] = commonObjectiveLimits(Problem,Snapshots);
    for i = 1 : count
        ax = nexttile(layout);
        plotSnapshotAxes(ax,Problem,Snapshots(i),xLimits,yLimits,true);
        title(ax,sprintf('target FE %d | actual %d | gen %d', ...
            round(Snapshots(i).target_FE),round(Snapshots(i).actual_FE), ...
            round(Snapshots(i).generation)), ...
            'FontWeight','normal','FontSize',13);
    end
    title(layout,sprintf( ...
        '%s | current mainline | run %d | N=%d, D=%d, maxFE=%d', ...
        char(problemName),round(runId),Config.N,Config.D,Config.maxFE), ...
        'Interpreter','none','FontWeight','bold','FontSize',16);
    exportgraphics(fig,figureFile,'Resolution',240, ...
        'BackgroundColor','white');
    close(fig);
end

function plotLatestSnapshotPanel(Problem,Snapshot,problemName,runId, ...
        figureFile)
    fig = figure('Visible','off','Color','w','Position',[80 80 900 700]);
    ax = axes(fig);
    [xLimits,yLimits] = commonObjectiveLimits(Problem,Snapshot);
    plotSnapshotAxes(ax,Problem,Snapshot,xLimits,yLimits,true);
    title(ax,sprintf('%s | run %d | FE %d | d_{50}=%.3g | w=%.3g', ...
        char(problemName),round(runId),round(Snapshot.actual_FE), ...
        Snapshot.bdist50_true,Snapshot.bwidth90_10_true), ...
        'Interpreter','tex','FontWeight','normal','FontSize',14);
    exportgraphics(fig,figureFile,'Resolution',220, ...
        'BackgroundColor','white');
    close(fig);
end

function tileRunTrajectoryFigures(Rows,figureFile)
    count = height(Rows);
    columns = 2;
    rows = ceil(count/columns);
    files = cellstr(string(Rows.figure_file));
    Tile = imtile(files,'GridSize',[rows columns], ...
        'BorderSize',[16 16],'BackgroundColor','white');
    imwrite(Tile,figureFile);
end

function Snapshot = latestGeneratedSnapshot(Snapshots)
    idx = numel(Snapshots);
    for i = numel(Snapshots) : -1 : 1
        if isfield(Snapshots(i),'generated_objs') && ...
                ~isempty(Snapshots(i).generated_objs)
            idx = i;
            break;
        end
    end
    Snapshot = Snapshots(idx);
end

function plotSnapshotAxes(ax,Problem,Snapshot,xLimits,yLimits,showLegend)
    hold(ax,'on');
    set(ax,'FontName','Times New Roman','FontSize',11,'Box','on', ...
        'Layer','top','View',[0 90],'Color',[0.98 0.94 0.88]);
    plotProblemDomain(ax,Problem);
    TrainObj = finiteTwoColumns(Snapshot.train_objs);
    if ~isempty(TrainObj)
        scatter(ax,TrainObj(:,1),TrainObj(:,2),24,'s', ...
            'MarkerFaceColor',[1.00 0.66 0.16], ...
            'MarkerEdgeColor',[0.30 0.20 0.06], ...
            'LineWidth',0.35,'MarkerFaceAlpha',0.78, ...
            'DisplayName','BMem training anchors');
    end
    GeneratedObj = finiteTwoColumns(Snapshot.generated_objs);
    feasible = snapshotFeasibleMask(Snapshot,size(GeneratedObj,1));
    if any(feasible)
        scatter(ax,GeneratedObj(feasible,1),GeneratedObj(feasible,2), ...
            30,'o','MarkerFaceColor',[0.12 0.42 0.82], ...
            'MarkerEdgeColor','none','MarkerFaceAlpha',0.72, ...
            'DisplayName','GAN feasible');
    end
    if any(~feasible)
        scatter(ax,GeneratedObj(~feasible,1),GeneratedObj(~feasible,2), ...
            30,'o','MarkerFaceColor',[0.88 0.16 0.18], ...
            'MarkerEdgeColor','none','MarkerFaceAlpha',0.72, ...
            'DisplayName','GAN infeasible');
    end
    xlim(ax,xLimits);
    ylim(ax,yLimits);
    xlabel(ax,'f_1');
    ylabel(ax,'f_2');
    grid(ax,'on');
    ax.GridAlpha = 0.10;
    if showLegend
        legend(ax,'Location','best','Box','off','FontSize',9);
    end
    hold(ax,'off');
end

function plotProblemDomain(ax,Problem)
    plot(ax,nan,nan,'s','MarkerSize',8, ...
        'MarkerFaceColor',[0.98 0.94 0.88], ...
        'MarkerEdgeColor',[0.78 0.63 0.50], ...
        'DisplayName','Infeasible domain');
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 3
        feasibleColor = [0.76 0.90 0.76];
        plot(ax,nan,nan,'s','MarkerSize',8, ...
            'MarkerFaceColor',feasibleColor,'MarkerEdgeColor','none', ...
            'DisplayName','Feasible domain');
        mask = isfinite(PF{3});
        [~,h] = contourf(ax,PF{1},PF{2},double(mask),[0.5 0.5], ...
            'LineStyle','none');
        h.FaceColor = feasibleColor;
        h.HandleVisibility = 'off';
    elseif isnumeric(PF) && size(PF,2) >= 2
        plot(ax,PF(:,1),PF(:,2),'-','Color',[0.16 0.52 0.18], ...
            'LineWidth',1.3,'DisplayName','Feasible boundary');
    end
end

function feasible = snapshotFeasibleMask(Snapshot,count)
    feasible = false(count,1);
    if count == 0
        return;
    end
    if isfield(Snapshot,'generated_feasible') && ...
            numel(Snapshot.generated_feasible) == count
        feasible = logical(Snapshot.generated_feasible(:));
    elseif isfield(Snapshot,'generated_cons') && ...
            size(Snapshot.generated_cons,1) == count
        feasible = all(double(Snapshot.generated_cons) <= 0,2);
    end
end

function [xLimits,yLimits] = commonObjectiveLimits(Problem,Snapshots)
    Obj = domainPoints(Problem);
    for i = 1 : numel(Snapshots)
        Obj = [Obj;finiteTwoColumns(Snapshots(i).train_objs); ...
            finiteTwoColumns(Snapshots(i).generated_objs)]; %#ok<AGROW>
    end
    Obj = Obj(all(isfinite(Obj),2),:);
    if isempty(Obj)
        xLimits = [0 1];
        yLimits = [0 1];
        return;
    end
    xLimits = paddedLimits(Obj(:,1));
    yLimits = paddedLimits(Obj(:,2));
end

function Obj = domainPoints(Problem)
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 3
        mask = isfinite(PF{3});
        Obj = [double(PF{1}(mask)),double(PF{2}(mask))];
    elseif isnumeric(PF) && size(PF,2) >= 2
        Obj = double(PF(:,1:2));
    else
        Obj = zeros(0,2);
    end
end

function Limits = paddedLimits(x)
    Limits = [min(x),max(x)];
    span = diff(Limits);
    if ~isfinite(span) || span <= eps
        center = mean(Limits);
        Limits = center + [-0.5 0.5];
    else
        Limits = Limits + 0.04*span*[-1 1];
    end
end

function Obj = finiteTwoColumns(Obj)
    Obj = double(Obj);
    if isempty(Obj) || size(Obj,2) < 2
        Obj = zeros(0,2);
        return;
    end
    Obj = Obj(:,1:2);
    Obj = Obj(all(isfinite(Obj),2),:);
end

function Row = emptyFigureRow()
    Row = struct('problem',"",'run',NaN,'stage_count',0, ...
        'latest_target_FE',NaN,'latest_actual_FE',NaN,'figure_file',"", ...
        'summary_panel_file',"");
end

function Row = emptyRunSummaryRow()
    Row = struct('run',NaN,'problem_count',0,'target_FE',NaN, ...
        'figure_file',"");
end

function rootDir = locateRootDir()
    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
end
