function OutputFile = plot_PRBCCMO_t_generation_region_evidence(runFolder,outDir,generation)
% Plot one PRBCCMO_t generation over the problem feasible/infeasible region.

    if nargin < 1 || isempty(runFolder)
        runFolder = latestObjectiveTraceFolder();
    end
    runFolder = char(string(runFolder));
    objectiveFile = fullfile(runFolder,'objective_snapshot.csv');
    metaFile = fullfile(runFolder,'run_meta.csv');
    assert(isfile(objectiveFile) && isfile(metaFile), ...
        'plot_PRBCCMO_t_generation_region_evidence:MissingCsv', ...
        'Run folder must contain run_meta.csv and objective_snapshot.csv: %s', runFolder);

    if nargin < 2 || isempty(outDir)
        outDir = fullfile(runFolder,'generation_region_figures');
    end
    outDir = char(string(outDir));
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    T = readtable(objectiveFile,'TextType','string');
    Meta = readtable(metaFile,'TextType','string');
    ObjNames = objectiveColumnNames(T);
    assert(numel(ObjNames) == 2, ...
        'plot_PRBCCMO_t_generation_region_evidence:OnlyBiObjectiveSupported', ...
        'Generation region evidence plots require a bi-objective run.');

    if nargin < 3 || isempty(generation)
        generation = chooseEvidenceGeneration(T);
    end
    generation = double(generation);
    Stage = T(T.generation == generation,:);
    assert(~isempty(Stage), ...
        'plot_PRBCCMO_t_generation_region_evidence:MissingGeneration', ...
        'Generation %.0f is not available in objective_snapshot.csv.', generation);
    requireStageRoles(Stage,{'pop_c','pop_u'});

    Problem = feval(char(Meta.problem(1)),'N',max(100,height(Stage)));
    PF = Problem.GetPF();

    fig = figure('Visible','off','Color','w','Position',[100 100 920 720]);
    ax = axes(fig);
    hold(ax,'on');
    plotProblemRegions(ax,PF,Stage,ObjNames,char(Meta.problem(1)));
    plotSegmentRootEvidence(ax,Stage,ObjNames);
    plotGenerationRole(ax,Stage,ObjNames,"pop_c",[0.08 0.32 0.86],34,'P_C constraint population','o');
    plotGenerationRole(ax,Stage,ObjNames,"pop_u",[0.90 0.42 0.08],34,'P_U unconstrained population','o');
    plotNearestRootConnectors(ax,Stage,ObjNames,"archive_b",[0.10 0.10 0.10]);
    plotNearestRootConnectors(ax,Stage,ObjNames,"boundary_evidence",[0.45 0.45 0.45]);
    plotNearestRootConnectors(ax,Stage,ObjNames,"boundary_off",[0.62 0.10 0.78]);
    plotArchiveBoundaryRole(ax,Stage,ObjNames);
    plotGenerationRole(ax,Stage,ObjNames,"boundary_evidence",[0.55 0.55 0.55],46,'Evaluated boundary evidence','d');
    plotGenerationRole(ax,Stage,ObjNames,"boundary_off",[0.62 0.10 0.78],86,'MLP-selected boundary child','^');
    finishRegionAxes(ax,Meta,Stage,ObjNames,generation);

    OutputFile = string(fullfile(outDir,sprintf('%s_run%d_gen%06.0f_region.png', ...
        char(Meta.problem(1)),double(Meta.run(1)),generation)));
    exportgraphics(fig,char(OutputFile),'Resolution',220);
    close(fig);
end

function plotSegmentRootEvidence(ax,Stage,ObjNames)
    F = Stage(Stage.role == "pair_feasible_endpoint",:);
    U = Stage(Stage.role == "pair_infeasible_endpoint",:);
    R = Stage(Stage.role == "segment_root",:);
    Count = min([height(F),height(U),height(R)]);
    if Count <= 0
        return;
    end
    for i = 1 : Count
        plot(ax,[double(F.(ObjNames{1})(i)),double(U.(ObjNames{1})(i))], ...
            [double(F.(ObjNames{2})(i)),double(U.(ObjNames{2})(i))], ...
            '-','Color',[0.16 0.55 0.22],'LineWidth',0.9,'HandleVisibility','off');
    end
    scatter(ax,double(F.(ObjNames{1})(1:Count)),double(F.(ObjNames{2})(1:Count)),34, ...
        [0.12 0.55 0.26],'o','filled','MarkerEdgeColor',[0.05 0.25 0.08], ...
        'DisplayName','Feasible pair endpoint');
    scatter(ax,double(U.(ObjNames{1})(1:Count)),double(U.(ObjNames{2})(1:Count)),34, ...
        [0.88 0.22 0.18],'o','filled','MarkerEdgeColor',[0.35 0.05 0.05], ...
        'DisplayName','Infeasible pair endpoint');
    scatter(ax,double(R.(ObjNames{1})(1:Count)),double(R.(ObjNames{2})(1:Count)),92, ...
        [0.05 0.74 0.18],'p','filled','MarkerEdgeColor',[0.02 0.20 0.05], ...
        'LineWidth',0.8,'DisplayName','Bisection segment root');
end

function plotNearestRootConnectors(ax,Stage,ObjNames,Role,Color)
    Data = Stage(Stage.role == Role,:);
    if isempty(Data)
        return;
    end
    RootXName = "root_obj1";
    RootYName = "root_obj2";
    if ~any(string(Data.Properties.VariableNames) == RootXName) || ...
            ~any(string(Data.Properties.VariableNames) == RootYName)
        return;
    end
    Limit = min(height(Data),24);
    for i = 1 : Limit
        x = double(Data.(ObjNames{1})(i));
        y = double(Data.(ObjNames{2})(i));
        rx = double(Data.(RootXName)(i));
        ry = double(Data.(RootYName)(i));
        if all(isfinite([x,y,rx,ry]))
            plot(ax,[x,rx],[y,ry],':','Color',Color,'LineWidth',0.9,'HandleVisibility','off');
        end
    end
end

function folder = latestObjectiveTraceFolder()
    rootDir = fileparts(which('platemo'));
    baseDir = fullfile(rootDir,'Data','PRBCCMO_t');
    dirs = dir(baseDir);
    dirs = dirs([dirs.isdir]);
    dirs = dirs(~ismember({dirs.name},{'.','..'}));
    folders = strings(0,1);
    times = zeros(0,1);
    for i = 1 : numel(dirs)
        candidate = fullfile(dirs(i).folder,dirs(i).name);
        if isfile(fullfile(candidate,'objective_snapshot.csv'))
            folders(end+1,1) = string(candidate); %#ok<AGROW>
            times(end+1,1) = dirs(i).datenum; %#ok<AGROW>
        end
    end
    assert(~isempty(folders), ...
        'plot_PRBCCMO_t_generation_region_evidence:NoObjectiveTrace', ...
        'No PRBCCMO_t objective trace folder found under %s.', baseDir);
    [~,idx] = max(times);
    folder = char(folders(idx));
end

function ObjNames = objectiveColumnNames(T)
    Names = string(T.Properties.VariableNames);
    Mask = ~cellfun(@isempty,regexp(cellstr(Names),'^obj\d+$','once'));
    ObjNames = cellstr(Names(Mask));
end

function generation = chooseEvidenceGeneration(T)
    Generations = unique(double(T.generation),'stable');
    BestGeneration = Generations(end);
    BestCount = -1;
    for g = reshape(Generations,1,[])
        Stage = T(T.generation == g,:);
        HasContext = any(Stage.role == "pop_c") && any(Stage.role == "pop_u") && any(Stage.role == "archive_b");
        BoundaryCount = sum(Stage.role == "boundary_evidence") + sum(Stage.role == "boundary_off");
        if HasContext && BoundaryCount > BestCount
            BestGeneration = g;
            BestCount = BoundaryCount;
        end
    end
    generation = BestGeneration;
end

function requireStageRoles(Stage,Roles)
    for i = 1 : numel(Roles)
        assert(any(Stage.role == string(Roles{i})), ...
            'plot_PRBCCMO_t_generation_region_evidence:MissingStageRole', ...
            'Selected generation must contain role %s.', Roles{i});
    end
end

function plotProblemRegions(ax,PF,Stage,ObjNames,ProblemName)
    [XLim,YLim] = resolveRegionLimits(PF,Stage,ObjNames);
    patch(ax,[XLim(1) XLim(2) XLim(2) XLim(1)],[YLim(1) YLim(1) YLim(2) YLim(2)], ...
        [1.00 0.93 0.93],'EdgeColor','none','DisplayName','Infeasible objective region');

    PF = ensureRegionGrid(PF,XLim,YLim,ProblemName);
    if iscell(PF) && numel(PF) >= 3 && ~isempty(PF{1})
        FeasibleMask = isfinite(PF{3});
        h = imagesc(ax,PF{1}(1,:),PF{2}(:,1),FeasibleMask);
        set(h,'AlphaData',0.22*double(FeasibleMask));
        set(h,'HandleVisibility','off');
        colormap(ax,[1.00 0.93 0.93;0.74 0.74 0.74]);
        set(ax,'YDir','normal');
        patch(ax,nan,nan,[0.74 0.74 0.74], ...
            'EdgeColor','none','DisplayName','Feasible objective region');
        if any(FeasibleMask(:)) && any(~FeasibleMask(:))
            contour(ax,PF{1},PF{2},double(FeasibleMask),[0.5 0.5], ...
                'Color',[0.05 0.05 0.05],'LineWidth',2.0,'DisplayName','True boundary contour');
        end
    end
    xlim(ax,XLim);
    ylim(ax,YLim);
end

function PF = ensureRegionGrid(PF,XLim,YLim,ProblemName)
    if iscell(PF) && numel(PF) >= 3 && ~isempty(PF{1})
        return;
    end
    Grid = buildLIRCMOP14RegionGrid(ProblemName,XLim,YLim);
    if ~isempty(Grid)
        PF = Grid;
    end
end

function PF = buildLIRCMOP14RegionGrid(ProblemName,XLim,YLim)
    Tokens = regexp(char(ProblemName),'^LIRCMOP([1-4])_BC$','tokens','once');
    if isempty(Tokens)
        PF = {};
        return;
    end
    ProblemId = str2double(Tokens{1});
    [X,Y] = meshgrid(linspace(XLim(1),XLim(2),420),linspace(YLim(1),YLim(2),420));
    Mask = false(size(X));
    x1 = linspace(0,1,1200);
    if any(ProblemId == [1 3])
        base2 = 1 - x1.^2;
    else
        base2 = 1 - sqrt(x1);
    end
    if any(ProblemId == [3 4])
        x1 = x1(sin(20*pi*x1) >= 0.5);
        if any(ProblemId == [1 3])
            base2 = 1 - x1.^2;
        else
            base2 = 1 - sqrt(x1);
        end
    end
    dx = abs(X(1,2)-X(1,1));
    dy = abs(Y(2,1)-Y(1,1));
    TolX = max(0.002,0.55*dx);
    TolY = max(0.002,0.55*dy);
    for i = 1 : numel(x1)
        InObj1 = X >= x1(i) + 0.5 - TolX & X <= x1(i) + 0.51 + TolX;
        InObj2 = Y >= base2(i) + 0.5 - TolY & Y <= base2(i) + 0.51 + TolY;
        Mask = Mask | (InObj1 & InObj2);
    end
    Z = nan(size(X));
    Z(Mask) = 0;
    PF = {X,Y,Z};
end

function plotArchiveBoundaryRole(ax,Stage,ObjNames)
    Data = Stage(Stage.role == "archive_b",:);
    if isempty(Data)
        return;
    end
    if any(string(Data.Properties.VariableNames) == "true_boundary_dist")
        C = double(Data.true_boundary_dist);
        scatter(ax,double(Data.(ObjNames{1})),double(Data.(ObjNames{2})),70,C, ...
            's','filled','MarkerEdgeColor',[0.08 0.08 0.08], ...
            'LineWidth',0.6,'DisplayName','Boundary archive B');
        cb = colorbar(ax);
        cb.Label.String = 'dist to true boundary';
    else
        plotGenerationRole(ax,Stage,ObjNames,"archive_b",[0.02 0.02 0.02],62,'Boundary archive B','s');
    end
end

function [XLim,YLim] = resolveRegionLimits(PF,Stage,ObjNames)
    X = double(Stage.(ObjNames{1}));
    Y = double(Stage.(ObjNames{2}));
    if iscell(PF) && numel(PF) >= 2 && ~isempty(PF{1})
        X = [X;PF{1}(:)];
        Y = [Y;PF{2}(:)];
    end
    X = X(isfinite(X));
    Y = Y(isfinite(Y));
    XLim = paddedLimits(X);
    YLim = paddedLimits(Y);
end

function Limits = paddedLimits(Value)
    if isempty(Value)
        Limits = [0 1];
        return;
    end
    Limits = [min(Value),max(Value)];
    Pad = 0.06*max(Limits(2)-Limits(1),1e-6);
    Limits = Limits + [-Pad,Pad];
end

function plotGenerationRole(ax,Stage,ObjNames,Role,Color,Size,Label,Marker)
    Data = Stage(Stage.role == Role,:);
    if isempty(Data)
        return;
    end
    scatter(ax,double(Data.(ObjNames{1})),double(Data.(ObjNames{2})),Size, ...
        'Marker',Marker,'MarkerFaceColor',Color,'MarkerEdgeColor',[0.08 0.08 0.08], ...
        'LineWidth',0.5,'DisplayName',Label);
end

function finishRegionAxes(ax,Meta,Stage,ObjNames,generation)
    grid(ax,'on');
    box(ax,'on');
    xlabel(ax,ObjNames{1},'Interpreter','none');
    ylabel(ax,ObjNames{2},'Interpreter','none');
    title(ax,sprintf('%s run %d, generation %.0f, FE %.0f', ...
        char(Meta.problem(1)),double(Meta.run(1)),generation,double(Stage.fe(1))), ...
        'Interpreter','none');
    annotateBoundaryStats(ax,Stage);
    legend(ax,'Location','bestoutside');
end

function annotateBoundaryStats(ax,Stage)
    B = Stage(Stage.role == "archive_b",:);
    BoundaryEvidence = Stage(Stage.role == "boundary_evidence",:);
    BoundaryOff = Stage(Stage.role == "boundary_off",:);
    MeanDist = NaN;
    P90Dist = NaN;
    LowMarginRatio = NaN;
    if ~isempty(B) && any(string(B.Properties.VariableNames) == "true_boundary_dist")
        Dist = double(B.true_boundary_dist);
        Dist = Dist(isfinite(Dist));
        if ~isempty(Dist)
            Dist = sort(Dist);
            MeanDist = mean(Dist);
            P90Dist = Dist(max(1,ceil(0.90*numel(Dist))));
        end
        Margin = double(B.margin);
        LowMarginRatio = sum(Margin <= 0.10 & isfinite(Margin))/max(sum(isfinite(Margin)),1);
    end
    Text = sprintf('|B|=%d\nevidence=%d\nboundary_off=%d\nmean/p90 dist=%.3g / %.3g\nlowmargin=%.2f', ...
        height(B),height(BoundaryEvidence),height(BoundaryOff),MeanDist,P90Dist,LowMarginRatio);
    x = ax.XLim(1) + 0.98*(ax.XLim(2)-ax.XLim(1));
    y = ax.YLim(1) + 0.98*(ax.YLim(2)-ax.YLim(1));
    text(ax,x,y,Text,'HorizontalAlignment','right','VerticalAlignment','top', ...
        'BackgroundColor',[1 1 1],'EdgeColor',[0.35 0.35 0.35], ...
        'Margin',6,'FontSize',9,'Interpreter','none');
end
