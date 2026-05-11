function [OutputFiles,RunFolder,Manifest] = plot_PRBCCMO_B_boundary_relation(problemName,N,maxFE,runId,outDir)
% Plot B archive endpoints against the feasible/infeasible boundary.

    if nargin < 1 || isempty(problemName)
        problemName = 'DASCMOP1_BC';
    end
    if nargin < 2 || isempty(N)
        N = 100;
    end
    if nargin < 3 || isempty(maxFE)
        maxFE = 200000;
    end
    if nargin < 4 || isempty(runId)
        runId = 1;
    end
    if nargin < 5 || isempty(outDir)
        rootDir = fileparts(which('platemo'));
        outDir = fullfile(rootDir,'Data','PRBCCMO_t','boundary_relation_figures');
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    rootDir = fileparts(which('platemo'));
    addpath(genpath(rootDir));

    rng(runId,'twister');
    Algorithm = PRBCCMO_t( ...
        'save',0, ...
        'run',runId, ...
        'outputFcn',@(varargin)[], ...
        'parameter',{40,80,0.05,4,0.1,20,150,1});
    ProblemConstructor = str2func(problemName);
    Problem = ProblemConstructor('N',N,'maxFE',maxFE);
    Algorithm.Solve(Problem);

    assert(isfield(Algorithm.metric,'analysis_boundary_manifest_csv') && ...
        isfile(Algorithm.metric.analysis_boundary_manifest_csv), ...
        'plot_PRBCCMO_B_boundary_relation:MissingBoundaryManifest', ...
        'Boundary snapshots were not written for %s.', problemName);

    RunFolder = string(Algorithm.metric.analysis_folder);
    Manifest = readtable(Algorithm.metric.analysis_boundary_manifest_csv, ...
        'TextType','string','Delimiter',',');
    Manifest = numericizeTable(Manifest);
    SnapshotFiles = string(Manifest.snapshot_file);
    Boundaries = cell(height(Manifest),1);
    for i = 1 : height(Manifest)
        Boundaries{i} = readtable(SnapshotFiles(i),'TextType','string','Delimiter',',');
    end

    [xLimits,yLimits] = objectiveAxisLimits(Problem,Boundaries);
    SnapshotOutputFiles = strings(height(Manifest),1);
    for i = 1 : height(Manifest)
        SnapshotOutputFiles(i) = plotSingleSnapshot( ...
            Problem,Boundaries{i},Manifest(i,:),problemName,N,maxFE,runId,outDir,xLimits,yLimits);
    end

    ProgressionFile = plotProgressionSnapshots( ...
        Problem,Boundaries,Manifest,problemName,N,maxFE,runId,outDir,xLimits,yLimits);
    OutputFiles = [ProgressionFile; SnapshotOutputFiles];
end

function OutputFile = plotSingleSnapshot( ...
    Problem,Boundary,Snapshot,problemName,N,maxFE,runId,outDir,xLimits,yLimits)
    fig = figure('Visible','off','Color','w','Position',[100 100 760 620]);
    ax = axes(fig);
    Draw(ax);
    drawBoundaryAxes(ax,Problem,Boundary,snapshotTitle(Snapshot),true,xLimits,yLimits);
    title(ax,sprintf('%s: B archive and feasible/infeasible boundary',problemName), ...
        'Interpreter','none','FontWeight','normal');
    subtitle(ax,sprintf('N=%d, maxFE=%d, run=%d',N,maxFE,runId),'Interpreter','none');

    OutputFile = string(fullfile(outDir, ...
        sprintf('%s_N%d_FE%06d_run%d_B_boundary_relation.png', ...
        problemName,N,round(double(Snapshot.target_fe)),runId)));
    exportgraphics(fig,OutputFile,'Resolution',300);
    close(fig);
end

function OutputFile = plotProgressionSnapshots( ...
    Problem,Boundaries,Manifest,problemName,N,maxFE,runId,outDir,xLimits,yLimits)
    fig = figure('Visible','off','Color','w','Position',[80 80 1220 920]);
    tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
    PanelCount = min(4,numel(Boundaries));
    LegendPanel = min(2,PanelCount);
    for i = 1 : PanelCount
        ax = nexttile;
        Draw(ax);
        drawBoundaryAxes(ax,Problem,Boundaries{i},snapshotTitle(Manifest(i,:)), ...
            i == LegendPanel,xLimits,yLimits,'northeast');
    end
    sgtitle(fig,sprintf('%s: B boundary fitting progression (N=%d, maxFE=%d, run=%d)', ...
        problemName,N,maxFE,runId),'Interpreter','none','FontWeight','normal');

    OutputFile = string(fullfile(outDir, ...
        sprintf('%s_N%d_FE%d_run%d_B_boundary_progression.png',problemName,N,maxFE,runId)));
    exportgraphics(fig,OutputFile,'Resolution',300);
    close(fig);
end

function drawBoundaryAxes(ax,Problem,Boundary,panelTitle,showLegend,xLimits,yLimits,legendLocation)
    if nargin < 8 || isempty(legendLocation)
        legendLocation = 'northeastoutside';
    end
    set(ax,'FontName','Times New Roman','FontSize',13,'NextPlot','add', ...
        'Box','on','View',[0 90],'Layer','top','YScale','linear');
    plotFeasibleRegion(ax,Problem);
    plotBoundaryArchive(ax,Boundary);
    grid(ax,'on');
    ax.GridAlpha = 0.15;
    xlabel(ax,'{\it f\rm_1}');
    ylabel(ax,'{\it f\rm_2}');
    title(ax,panelTitle,'Interpreter','none','FontWeight','normal');
    xlim(ax,xLimits);
    ylim(ax,yLimits);
    if showLegend
        legend(ax,'Location',legendLocation,'Box','off');
    end
end

function plotFeasibleRegion(ax,Problem)
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 3
        X = PF{1};
        Y = PF{2};
        Z = PF{3};
        surf(ax,X,Y,Z,'EdgeColor','none','FaceColor',[.85 .85 .85], ...
            'FaceAlpha',0.58,'DisplayName','Feasible objective region');
        Mask = double(isfinite(Z));
        contour(ax,X,Y,Mask,[0.5 0.5],'Color',[0 .25 .45], ...
            'LineWidth',1.7,'DisplayName','Feasible/infeasible boundary');
    elseif isnumeric(PF) && size(PF,2) >= 2
        plot(ax,PF(:,1),PF(:,2),'-','Color',[0 .25 .45], ...
            'LineWidth',1.7,'DisplayName','Reference front');
    end
end

function plotBoundaryArchive(ax,Boundary)
    Obj = [numericColumn(Boundary,'obj1'),numericColumn(Boundary,'obj2')];
    PairIndex = numericColumn(Boundary,'pair_index');
    Side = string(Boundary.pair_side);
    FeasibleEndpoint = Side == "feasible_endpoint";
    InfeasibleEndpoint = Side == "infeasible_endpoint";

    Pairs = unique(PairIndex(isfinite(PairIndex)));
    for i = 1 : numel(Pairs)
        current = PairIndex == Pairs(i);
        f = Obj(current & FeasibleEndpoint,:);
        u = Obj(current & InfeasibleEndpoint,:);
        if ~isempty(f) && ~isempty(u)
            plot(ax,[f(1,1),u(1,1)],[f(1,2),u(1,2)],'-', ...
                'Color',[.55 .55 .55],'LineWidth',0.75, ...
                'HandleVisibility','off');
        end
    end

    scatter(ax,Obj(FeasibleEndpoint,1),Obj(FeasibleEndpoint,2),32,'o', ...
        'MarkerFaceColor',[.15 .6 .2],'MarkerEdgeColor',[0 .25 .45], ...
        'LineWidth',0.8,'DisplayName','B feasible endpoint');
    scatter(ax,Obj(InfeasibleEndpoint,1),Obj(InfeasibleEndpoint,2),32,'s', ...
        'MarkerFaceColor',[.9 .5 .2],'MarkerEdgeColor',[0 .25 .45], ...
        'LineWidth',0.8,'DisplayName','B infeasible endpoint');
end

function TitleText = snapshotTitle(Snapshot)
    TitleText = sprintf('target FE=%d, actual FE=%d, |B pairs|=%d', ...
        round(double(Snapshot.target_fe)),round(double(Snapshot.actual_fe)), ...
        round(double(Snapshot.b_pair_count)));
end

function [xLimits,yLimits] = objectiveAxisLimits(Problem,Boundaries)
    X = [];
    Y = [];
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 2
        X = [X; PF{1}(:)];
        Y = [Y; PF{2}(:)];
    elseif isnumeric(PF) && size(PF,2) >= 2
        X = [X; PF(:,1)];
        Y = [Y; PF(:,2)];
    end
    for i = 1 : numel(Boundaries)
        if ~isempty(Boundaries{i}) && all(ismember({'obj1','obj2'},Boundaries{i}.Properties.VariableNames))
            X = [X; numericColumn(Boundaries{i},'obj1')]; %#ok<AGROW>
            Y = [Y; numericColumn(Boundaries{i},'obj2')]; %#ok<AGROW>
        end
    end
    xLimits = paddedLimits(X);
    yLimits = paddedLimits(Y);
end

function Limits = paddedLimits(Values)
    Values = Values(isfinite(Values));
    if isempty(Values)
        Limits = [0 1];
        return;
    end
    Limits = [min(Values),max(Values)];
    Pad = 0.03*max(diff(Limits),eps);
    Limits = Limits + [-Pad,Pad];
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

function Values = numericColumn(T,Name)
    Values = T.(Name);
    if iscell(Values) || isstring(Values)
        Values = str2double(string(Values));
    elseif islogical(Values)
        Values = double(Values);
    else
        Values = double(Values);
    end
end
