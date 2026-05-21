function [OutputFiles,RunFolder,Manifest] = plot_PRBCCMO_B_boundary_relation(problemName,N,maxFE,runId,outDir,sourceRunFolder)
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
    if nargin < 6
        sourceRunFolder = "";
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    rootDir = fileparts(which('platemo'));
    addpath(genpath(rootDir));

    ProblemConstructor = str2func(problemName);
    Problem = ProblemConstructor('N',N,'maxFE',maxFE);
    if strlength(string(sourceRunFolder)) > 0
        RunFolder = string(sourceRunFolder);
        BoundaryManifestFile = fullfile(char(RunFolder),'boundary_snapshots.csv');
        assert(isfile(BoundaryManifestFile), ...
            'plot_PRBCCMO_B_boundary_relation:MissingBoundaryManifest', ...
            'Boundary snapshot manifest does not exist: %s.', BoundaryManifestFile);
    else
        rng(runId,'twister');
        Algorithm = PRBCCMO_t( ...
            'save',0, ...
            'run',runId, ...
            'outputFcn',@(varargin)[], ...
            'parameter',{64,200,1e-3,3,0.1,10,0,1,1});
        Algorithm.Solve(Problem);

        assert(isfield(Algorithm.metric,'analysis_boundary_manifest_csv') && ...
            isfile(Algorithm.metric.analysis_boundary_manifest_csv), ...
            'plot_PRBCCMO_B_boundary_relation:MissingBoundaryManifest', ...
            'Boundary snapshots were not written for %s.', problemName);
        RunFolder = string(Algorithm.metric.analysis_folder);
        BoundaryManifestFile = Algorithm.metric.analysis_boundary_manifest_csv;
    end

    Manifest = readtable(BoundaryManifestFile, ...
        'TextType','string','Delimiter',',');
    Manifest = PRBCCMOUtils.numericizeTable(Manifest);
    SnapshotFiles = string(Manifest.snapshot_file);
    Boundaries = cell(height(Manifest),1);
    for i = 1 : height(Manifest)
        Boundaries{i} = readtable(SnapshotFiles(i),'TextType','string','Delimiter',',');
    end
    Manifest = appendBoundaryFitMetrics(Problem,Manifest,Boundaries);
    FitMetricsFile = string(fullfile(outDir, ...
        sprintf('%s_N%d_FE%d_run%d_B_boundary_fit_metrics.csv',problemName,N,maxFE,runId)));
    Manifest.fit_metrics_csv = repmat(FitMetricsFile,height(Manifest),1);
    writetable(Manifest,FitMetricsFile);

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
    SingleTitle = sprintf('%s: B archive and feasible/infeasible boundary\n%s', ...
        problemName,snapshotTitle(Snapshot));
    drawBoundaryAxes(ax,Problem,Boundary,SingleTitle,true,xLimits,yLimits);
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
    if Problem.M > 2
        text(ax,0.02,0.04,'f_1-f_2 projection','Units','normalized', ...
            'FontSize',10,'Color',[.25 .25 .25],'Interpreter','tex', ...
            'BackgroundColor','w','Margin',2);
    end
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
        if min(Mask(:)) < max(Mask(:))
            contour(ax,X,Y,Mask,[0.5 0.5],'Color',[0 .25 .45], ...
                'LineWidth',1.7,'DisplayName','Feasible/infeasible boundary');
        end
    elseif isnumeric(PF) && size(PF,2) >= 2
        plot(ax,PF(:,1),PF(:,2),'-','Color',[0 .25 .45], ...
            'LineWidth',1.7,'DisplayName','Reference boundary');
    end
end

function plotBoundaryArchive(ax,Boundary)
    Obj = [numericColumn(Boundary,'obj1'),numericColumn(Boundary,'obj2')];
    PairIndex = numericColumn(Boundary,'pair_index');
    Side = string(Boundary.pair_side);
    FeasibleEndpoint = Side == "feasible_endpoint";
    InfeasibleEndpoint = Side == "infeasible_endpoint";

    Pairs = unique(PairIndex(isfinite(PairIndex)));
    Midpoints = NaN(numel(Pairs),2);
    MidpointCount = 0;
    for i = 1 : numel(Pairs)
        current = PairIndex == Pairs(i);
        f = Obj(current & FeasibleEndpoint,:);
        u = Obj(current & InfeasibleEndpoint,:);
        if ~isempty(f) && ~isempty(u)
            plot(ax,[f(1,1),u(1,1)],[f(1,2),u(1,2)],'-', ...
                'Color',[.55 .55 .55],'LineWidth',0.75, ...
                'HandleVisibility','off');
            MidpointCount = MidpointCount + 1;
            Midpoints(MidpointCount,:) = 0.5*(f(1,:) + u(1,:));
        end
    end
    Midpoints = Midpoints(1:MidpointCount,:);

    scatter(ax,Obj(FeasibleEndpoint,1),Obj(FeasibleEndpoint,2),32,'o', ...
        'MarkerFaceColor',[.15 .6 .2],'MarkerEdgeColor',[0 .25 .45], ...
        'LineWidth',0.8,'DisplayName','B feasible endpoint');
    scatter(ax,Obj(InfeasibleEndpoint,1),Obj(InfeasibleEndpoint,2),32,'s', ...
        'MarkerFaceColor',[.9 .5 .2],'MarkerEdgeColor',[0 .25 .45], ...
        'LineWidth',0.8,'DisplayName','B infeasible endpoint');
    if ~isempty(Midpoints)
        scatter(ax,Midpoints(:,1),Midpoints(:,2),20,'d', ...
            'MarkerFaceColor',[.45 .2 .75],'MarkerEdgeColor',[.2 .1 .4], ...
            'LineWidth',0.7,'DisplayName','B pair midpoint');
    end
end

function TitleText = snapshotTitle(Snapshot)
    TitleText = sprintf('target FE=%d, actual FE=%d, |B pairs|=%d', ...
        round(double(Snapshot.target_fe)),round(double(Snapshot.actual_fe)), ...
        round(double(Snapshot.b_pair_count)));
    FitText = snapshotFitText(Snapshot);
    if strlength(FitText) > 0
        TitleText = sprintf('%s\n%s',TitleText,char(FitText));
    end
end

function [xLimits,yLimits] = objectiveAxisLimits(Problem,Boundaries)
    XParts = cell(numel(Boundaries)+1,1);
    YParts = cell(numel(Boundaries)+1,1);
    part = 0;
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 2
        part = part + 1;
        XParts{part} = PF{1}(:);
        YParts{part} = PF{2}(:);
    elseif isnumeric(PF) && size(PF,2) >= 2
        part = part + 1;
        XParts{part} = PF(:,1);
        YParts{part} = PF(:,2);
    end
    for i = 1 : numel(Boundaries)
        if ~isempty(Boundaries{i}) && all(ismember({'obj1','obj2'},Boundaries{i}.Properties.VariableNames))
            part = part + 1;
            XParts{part} = numericColumn(Boundaries{i},'obj1');
            YParts{part} = numericColumn(Boundaries{i},'obj2');
        end
    end
    if part == 0
        X = [];
        Y = [];
    else
        X = vertcat(XParts{1:part});
        Y = vertcat(YParts{1:part});
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

function Manifest = appendBoundaryFitMetrics(Problem,Manifest,Boundaries)
    BoundaryPoints = extractObjectiveBoundaryPoints(Problem);
    Count = height(Manifest);
    boundaryPointCount = repmat(size(BoundaryPoints,1),Count,1);
    meanMidDist = NaN(Count,1);
    p90MidDist = NaN(Count,1);
    meanInfeasibleDist = NaN(Count,1);
    p90InfeasibleDist = NaN(Count,1);
    maxInfeasibleDist = NaN(Count,1);
    farInfeasibleRatio = NaN(Count,1);

    for i = 1 : Count
        Metrics = boundaryFitMetrics(Boundaries{i},BoundaryPoints);
        meanMidDist(i) = Metrics.mean_mid_boundary_dist;
        p90MidDist(i) = Metrics.p90_mid_boundary_dist;
        meanInfeasibleDist(i) = Metrics.mean_infeasible_boundary_dist;
        p90InfeasibleDist(i) = Metrics.p90_infeasible_boundary_dist;
        maxInfeasibleDist(i) = Metrics.max_infeasible_boundary_dist;
        farInfeasibleRatio(i) = Metrics.far_infeasible_boundary_ratio;
    end

    Manifest.boundary_point_count = boundaryPointCount;
    Manifest.b_mean_mid_boundary_dist = meanMidDist;
    Manifest.b_p90_mid_boundary_dist = p90MidDist;
    Manifest.b_mean_infeasible_boundary_dist = meanInfeasibleDist;
    Manifest.b_p90_infeasible_boundary_dist = p90InfeasibleDist;
    Manifest.b_max_infeasible_boundary_dist = maxInfeasibleDist;
    Manifest.b_far_infeasible_boundary_ratio = farInfeasibleRatio;
end

function Metrics = boundaryFitMetrics(Boundary,BoundaryPoints)
    Metrics = struct( ...
        'mean_mid_boundary_dist',NaN, ...
        'p90_mid_boundary_dist',NaN, ...
        'mean_infeasible_boundary_dist',NaN, ...
        'p90_infeasible_boundary_dist',NaN, ...
        'max_infeasible_boundary_dist',NaN, ...
        'far_infeasible_boundary_ratio',NaN);
    if isempty(BoundaryPoints) || isempty(Boundary) || ...
            ~all(ismember({'obj1','obj2','pair_index','pair_side'},Boundary.Properties.VariableNames))
        return;
    end

    Obj = [numericColumn(Boundary,'obj1'),numericColumn(Boundary,'obj2')];
    PairIndex = numericColumn(Boundary,'pair_index');
    Side = string(Boundary.pair_side);
    FeasibleEndpoint = Side == "feasible_endpoint";
    InfeasibleEndpoint = Side == "infeasible_endpoint";

    InfeasibleObj = Obj(InfeasibleEndpoint,:);
    MidObj = pairMidpoints(Obj,PairIndex,FeasibleEndpoint,InfeasibleEndpoint);
    MidDist = distanceToPointSet(MidObj,BoundaryPoints);
    InfeasibleDist = distanceToPointSet(InfeasibleObj,BoundaryPoints);
    FarThreshold = objectiveBoundaryFarThreshold([BoundaryPoints;Obj(all(isfinite(Obj),2),:)]);

    Metrics.mean_mid_boundary_dist = meanFinite(MidDist);
    Metrics.p90_mid_boundary_dist = percentileFinite(MidDist,90);
    Metrics.mean_infeasible_boundary_dist = meanFinite(InfeasibleDist);
    Metrics.p90_infeasible_boundary_dist = percentileFinite(InfeasibleDist,90);
    Metrics.max_infeasible_boundary_dist = maxFinite(InfeasibleDist);
    if isfinite(FarThreshold)
        Metrics.far_infeasible_boundary_ratio = mean(double(InfeasibleDist(isfinite(InfeasibleDist)) > FarThreshold));
    end
end

function Midpoints = pairMidpoints(Obj,PairIndex,FeasibleEndpoint,InfeasibleEndpoint)
    Pairs = unique(PairIndex(isfinite(PairIndex)));
    Midpoints = NaN(numel(Pairs),2);
    Count = 0;
    for i = 1 : numel(Pairs)
        current = PairIndex == Pairs(i);
        f = Obj(current & FeasibleEndpoint,:);
        u = Obj(current & InfeasibleEndpoint,:);
        if ~isempty(f) && ~isempty(u) && all(isfinite([f(1,:),u(1,:)]))
            Count = Count + 1;
            Midpoints(Count,:) = 0.5*(f(1,:) + u(1,:));
        end
    end
    Midpoints = Midpoints(1:Count,:);
end

function Points = extractObjectiveBoundaryPoints(Problem)
    Points = zeros(0,2);
    try
        PF = Problem.PF;
    catch
        return;
    end
    if iscell(PF)
        Points = extractObjectiveBoundaryPointsFromCell(PF);
    elseif isnumeric(PF) && size(PF,2) >= 2
        Points = double(PF(:,1:2));
        Points = Points(all(isfinite(Points),2),:);
    end
    if ~isempty(Points)
        Points = unique(Points,'rows');
    end
end

function Points = extractObjectiveBoundaryPointsFromCell(PF)
    Points = zeros(0,2);
    if numel(PF) < 3
        return;
    end
    X = double(PF{1});
    Y = double(PF{2});
    Z = double(PF{3});
    if isempty(X) || isempty(Y) || isempty(Z) || ~isequal(size(X),size(Y),size(Z))
        return;
    end
    Mask = isfinite(Z);
    Edge = false(size(Mask));
    if size(Mask,1) > 1
        Change = Mask(1:end-1,:) ~= Mask(2:end,:);
        Edge(1:end-1,:) = Edge(1:end-1,:) | Change;
        Edge(2:end,:) = Edge(2:end,:) | Change;
    end
    if size(Mask,2) > 1
        Change = Mask(:,1:end-1) ~= Mask(:,2:end);
        Edge(:,1:end-1) = Edge(:,1:end-1) | Change;
        Edge(:,2:end) = Edge(:,2:end) | Change;
    end
    Points = [X(Edge),Y(Edge)];
    Points = Points(all(isfinite(Points),2),:);
end

function Dist = distanceToPointSet(Points,Reference)
    Points = double(Points);
    Reference = double(Reference);
    Dist = NaN(size(Points,1),1);
    if isempty(Points) || isempty(Reference)
        return;
    end
    for i = 1 : size(Points,1)
        Delta = bsxfun(@minus,Reference,Points(i,:));
        Dist(i) = sqrt(min(sum(Delta.^2,2)));
    end
end

function Threshold = objectiveBoundaryFarThreshold(Points)
    Points = Points(all(isfinite(Points),2),:);
    if isempty(Points)
        Threshold = inf;
        return;
    end
    Span = max(Points,[],1) - min(Points,[],1);
    Threshold = 0.05*sqrt(sum(Span.^2));
    if Threshold <= 0 || ~isfinite(Threshold)
        Threshold = inf;
    end
end

function Value = meanFinite(Value)
    Value = finiteValues(Value);
    if isempty(Value)
        Value = NaN;
    else
        Value = mean(Value);
    end
end

function Value = maxFinite(Value)
    Value = finiteValues(Value);
    if isempty(Value)
        Value = NaN;
    else
        Value = max(Value);
    end
end

function Value = percentileFinite(Value,Percentile)
    Value = sort(finiteValues(Value));
    if isempty(Value)
        Value = NaN;
    else
        idx = max(1,min(numel(Value),ceil(Percentile/100*numel(Value))));
        Value = Value(idx);
    end
end

function Value = finiteValues(Value)
    Value = double(Value(:));
    Value = Value(isfinite(Value));
end

function FitText = snapshotFitText(Snapshot)
    FitText = "";
    if ~hasTableColumn(Snapshot,'b_mean_infeasible_boundary_dist') || ...
            ~hasTableColumn(Snapshot,'b_p90_infeasible_boundary_dist') || ...
            ~hasTableColumn(Snapshot,'b_far_infeasible_boundary_ratio')
        return;
    end
    MeanDist = double(Snapshot.b_mean_infeasible_boundary_dist);
    P90Dist = double(Snapshot.b_p90_infeasible_boundary_dist);
    FarRatio = double(Snapshot.b_far_infeasible_boundary_ratio);
    if isfinite(MeanDist) && isfinite(P90Dist) && isfinite(FarRatio)
        FitText = string(sprintf('d(B infeasible,boundary): mean=%.3g, p90=%.3g, far=%.1f%%', ...
            MeanDist,P90Dist,100*FarRatio));
    end
end

function Flag = hasTableColumn(T,Name)
    Flag = istable(T) && ismember(Name,T.Properties.VariableNames);
end
