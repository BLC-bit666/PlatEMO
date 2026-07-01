function Manifest = redraw_CBS_RegionCGAN_domain_figures(expDir,outDir)
%REDRAW_CBS_REGIONCGAN_DOMAIN_FIGURES Redraw RegionCGAN snapshots with domains.
%   Reuses saved metric.mat snapshots. No algorithm run or new evaluations.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(expDir)
        error('CBSRegionCGAN:MissingExperimentDir', ...
            'Experiment directory is required.');
    end
    if nargin < 2 || isempty(outDir)
        outDir = fullfile(expDir,'domain_figures_all');
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    files = dir(fullfile(expDir,'*_run*','metric.mat'));
    Rows = repmat(emptyDomainFigureRow(),0,1);
    for i = 1 : numel(files)
        metricFile = fullfile(files(i).folder,files(i).name);
        S = load(metricFile,'metric','problem','run','N','maxFE');
        if ~isfield(S,'metric') || ...
                ~isfield(S.metric,'region_gan_stage_snapshots') || ...
                isempty(S.metric.region_gan_stage_snapshots)
            continue;
        end
        problemName = inferProblemName(S,files(i).folder);
        runId = inferRunId(S,files(i).folder);
        Problem = makePlotProblem(problemName,S);
        Snapshots = S.metric.region_gan_stage_snapshots;
        for j = 1 : numel(Snapshots)
            Snapshot = Snapshots(j);
            targetFE = getSnapshotNumber(Snapshot,'target_FE');
            figureFile = fullfile(outDir,sprintf( ...
                '%s_run%d_FE%06d_domain_boundary.png', ...
                char(problemName),round(runId),round(targetFE)));
            plotSingleDomainFigure(Problem,Snapshot,problemName,runId, ...
                figureFile);
            Row = emptyDomainFigureRow();
            Row.problem = string(problemName);
            Row.run = double(runId);
            Row.target_FE = targetFE;
            Row.actual_FE = getSnapshotNumber(Snapshot,'actual_FE');
            Row.generation = getSnapshotNumber(Snapshot,'generation');
            Row.figure_file = string(figureFile);
            Rows(end+1,1) = Row; %#ok<AGROW>
        end
    end
    Manifest = struct2table(Rows);
    writetable(Manifest,fullfile(outDir,'domain_figure_manifest.csv'));
end

function problemName = inferProblemName(S,runFolder)
    if isfield(S,'problem') && ~isempty(S.problem)
        problemName = string(S.problem);
        return;
    end
    [~,name] = fileparts(runFolder);
    problemName = regexprep(string(name),'_run\d+$','');
end

function runId = inferRunId(S,runFolder)
    if isfield(S,'run') && ~isempty(S.run)
        runId = double(S.run);
        return;
    end
    [~,name] = fileparts(runFolder);
    token = regexp(name,'_run(\d+)$','tokens','once');
    if isempty(token)
        runId = NaN;
    else
        runId = str2double(token{1});
    end
end

function Problem = makePlotProblem(problemName,S)
    N = 100;
    maxFE = 100000;
    if isfield(S,'N') && ~isempty(S.N)
        N = max(1,round(double(S.N)));
    end
    if isfield(S,'maxFE') && ~isempty(S.maxFE)
        maxFE = max(1,round(double(S.maxFE)));
    end
    Constructor = str2func(char(problemName));
    Problem = Constructor('N',N,'maxFE',maxFE);
end

function plotSingleDomainFigure(Problem,S,problemName,runId,figureFile)
    [xLimits,yLimits] = objectiveLimitsForDomainPlot(Problem,S);
    fig = figure('Visible','off','Color','w','Position',[100 100 1120 760]);
    ax = axes(fig);
    hold(ax,'on');
    set(ax,'FontName','Times New Roman','FontSize',16,'Box','on', ...
        'Layer','top','View',[0 90],'Color',[0.98 0.94 0.88]);
    plotFeasibleInfeasibleDomain(ax,Problem);
    plotTrainingSet(ax,getSnapshotObj(S,'train_objs'));
    plotGeneratedSet(ax,getSnapshotObj(S,'generated_objs'));
    xlim(ax,xLimits);
    ylim(ax,yLimits);
    xlabel(ax,'f_1');
    ylabel(ax,'f_2');
    title(ax,sprintf('%s run=%d targetFE=%d actualFE=%d gen=%d', ...
        char(problemName),round(runId), ...
        round(getSnapshotNumber(S,'target_FE')), ...
        round(getSnapshotNumber(S,'actual_FE')), ...
        round(getSnapshotNumber(S,'generation'))), ...
        'Interpreter','none','FontWeight','normal','FontSize',18);
    legend(ax,'Location','northeastoutside','Box','off','FontSize',15);
    hold(ax,'off');
    exportgraphics(fig,figureFile,'Resolution',300);
    close(fig);
end

function plotFeasibleInfeasibleDomain(ax,Problem)
    plot(ax,nan,nan,'s','MarkerSize',10, ...
        'MarkerFaceColor',[0.98 0.94 0.88], ...
        'MarkerEdgeColor',[0.78 0.63 0.50], ...
        'DisplayName','Infeasible domain');
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 3
        surf(ax,PF{1},PF{2},PF{3},'EdgeColor','none', ...
            'FaceColor',[0.82 0.92 0.82],'FaceAlpha',0.72, ...
            'DisplayName','Feasible domain');
    elseif isnumeric(PF) && size(PF,2) >= 2
        plot(ax,PF(:,1),PF(:,2),'-','Color',[0.20 0.55 0.20], ...
            'LineWidth',1.5,'DisplayName','Feasible domain');
    else
        plot(ax,nan,nan,'s','MarkerSize',10, ...
            'MarkerFaceColor',[0.82 0.92 0.82], ...
            'MarkerEdgeColor','none','DisplayName','Feasible domain');
    end
end

function plotTrainingSet(ax,TrainObj)
    TrainObj = firstTwoFiniteColumns(TrainObj);
    if isempty(TrainObj)
        scatter(ax,nan,nan,38,'s', ...
            'MarkerFaceColor',[1.00 0.68 0.20], ...
            'MarkerEdgeColor',[0.28 0.18 0.05], ...
            'DisplayName','Training set');
        return;
    end
    scatter(ax,TrainObj(:,1),TrainObj(:,2),38,'s', ...
        'MarkerFaceColor',[1.00 0.68 0.20], ...
        'MarkerEdgeColor',[0.28 0.18 0.05], ...
        'LineWidth',0.6,'DisplayName','Training set');
end

function plotGeneratedSet(ax,GeneratedObj)
    GeneratedObj = firstTwoFiniteColumns(GeneratedObj);
    if isempty(GeneratedObj)
        scatter(ax,nan,nan,42,'o', ...
            'MarkerFaceColor',[0.90 0.18 0.20], ...
            'MarkerEdgeColor','none','DisplayName','GAN generated');
        return;
    end
    scatter(ax,GeneratedObj(:,1),GeneratedObj(:,2),42,'o', ...
        'MarkerFaceColor',[0.90 0.18 0.20], ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.78, ...
        'DisplayName','GAN generated');
end

function [xLimits,yLimits] = objectiveLimitsForDomainPlot(Problem,S)
    Obj = [domainPFPoints(Problem); ...
        firstTwoFiniteColumns(getSnapshotObj(S,'train_objs')); ...
        firstTwoFiniteColumns(getSnapshotObj(S,'generated_objs'))];
    Obj = Obj(all(isfinite(Obj),2),:);
    if isempty(Obj)
        xLimits = [0 1];
        yLimits = [0 1];
        return;
    end
    xLimits = [min(Obj(:,1)),max(Obj(:,1))];
    yLimits = [min(Obj(:,2)),max(Obj(:,2))];
    xPad = max(0.05,0.04*max(diff(xLimits),eps));
    yPad = max(0.05,0.04*max(diff(yLimits),eps));
    xLimits = xLimits + [-xPad xPad];
    yLimits = yLimits + [-yPad yPad];
end

function Obj = domainPFPoints(Problem)
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 2
        Obj = [PF{1}(:),PF{2}(:)];
    elseif isnumeric(PF) && size(PF,2) >= 2
        Obj = PF(:,1:2);
    else
        Obj = zeros(0,2);
    end
    Obj = Obj(all(isfinite(Obj),2),:);
end

function Obj = getSnapshotObj(S,name)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        Obj = S.(name);
    else
        Obj = zeros(0,2);
    end
end

function value = getSnapshotNumber(S,name)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = double(S.(name));
    else
        value = NaN;
    end
end

function Obj = firstTwoFiniteColumns(Obj)
    if isempty(Obj) || size(Obj,2) < 2
        Obj = zeros(0,2);
        return;
    end
    Obj = double(Obj(:,1:2));
    Obj = Obj(all(isfinite(Obj),2),:);
end

function Row = emptyDomainFigureRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'target_FE',NaN, ...
        'actual_FE',NaN, ...
        'generation',NaN, ...
        'figure_file',"");
end
