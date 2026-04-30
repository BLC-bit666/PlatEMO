function OutputFiles = plot_PRBCCMO_t_objective_evidence(runFolder,outDir,generations)
% Plot objective-space evidence from PRBCCMO_t objective_snapshot.csv.

    if nargin < 1 || isempty(runFolder)
        runFolder = latestPRBCCMOTraceFolder();
    end
    runFolder = char(string(runFolder));
    objectiveFile = fullfile(runFolder,'objective_snapshot.csv');
    assert(isfile(objectiveFile), ...
        'plot_PRBCCMO_t_objective_evidence:MissingObjectiveCsv', ...
        'Missing objective snapshot CSV: %s', objectiveFile);

    if nargin < 2 || isempty(outDir)
        outDir = fullfile(runFolder,'objective_figures');
    end
    outDir = char(string(outDir));
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    T = readtable(objectiveFile,'TextType','string');
    assert(~isempty(T), ...
        'plot_PRBCCMO_t_objective_evidence:EmptyObjectiveCsv', ...
        'Objective snapshot CSV has no rows: %s', objectiveFile);
    ObjNames = objectiveColumnNames(T);
    assert(numel(ObjNames) >= 2, ...
        'plot_PRBCCMO_t_objective_evidence:InsufficientObjectives', ...
        'At least obj1 and obj2 are required for objective-space plots.');

    if nargin < 3 || isempty(generations)
        generations = defaultPlotGenerations(T);
    else
        generations = double(generations(:)');
    end

    OutputFiles = strings(0,1);
    for g = generations
        Stage = T(T.generation == g,:);
        if isempty(Stage)
            continue;
        end
        fig = figure('Visible','off','Color','w','Position',[100 100 1500 460]);
        layout = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
        title(layout,sprintf('PRBCCMO_t objective evidence, generation %.0f',g), ...
            'Interpreter','none');

        nexttile(layout);
        plotStageByRole(Stage,ObjNames);
        title('Population, archive, boundary offspring');

        nexttile(layout);
        plotBoundaryMetric(Stage,ObjNames,'margin','MLP margin');
        title('Selected boundary offspring: margin');

        nexttile(layout);
        plotBoundaryMetric(Stage,ObjNames,'opp_dist','Opposite-side distance');
        title('Selected boundary offspring: opposite distance');

        filePath = fullfile(outDir,sprintf('objective_snapshot_gen%06.0f.png',g));
        exportObjectiveFigure(fig,filePath);
        close(fig);
        OutputFiles(end+1,1) = string(filePath); %#ok<AGROW>
    end

    boundaryFile = plotAllBoundaryOffspring(T,ObjNames,outDir);
    if strlength(boundaryFile) > 0
        OutputFiles(end+1,1) = boundaryFile;
    end
end

function folder = latestPRBCCMOTraceFolder()
    rootDir = fileparts(which('platemo'));
    baseDir = fullfile(rootDir,'Data','PRBCCMO_t');
    assert(isfolder(baseDir), ...
        'plot_PRBCCMO_t_objective_evidence:MissingTraceRoot', ...
        'Trace root not found: %s', baseDir);
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
        'plot_PRBCCMO_t_objective_evidence:NoTraceFolders', ...
        'No PRBCCMO_t trace folder with objective_snapshot.csv found in %s.', baseDir);
    [~,idx] = max(times);
    folder = char(folders(idx));
end

function ObjNames = objectiveColumnNames(T)
    Names = string(T.Properties.VariableNames);
    Mask = ~cellfun(@isempty,regexp(cellstr(Names),'^obj\d+$','once'));
    ObjNames = cellstr(Names(Mask));
end

function generations = defaultPlotGenerations(T)
    Context = T.role ~= "boundary_off";
    generations = unique(double(T.generation(Context))','stable');
    if isempty(generations)
        generations = unique(double(T.generation)','stable');
    end
    if numel(generations) > 6
        idx = unique(round(linspace(1,numel(generations),6)));
        generations = generations(idx);
    end
end

function plotStageByRole(T,ObjNames)
    hold on;
    Handles = gobjects(0);
    Labels = strings(0,1);
    [Handles,Labels] = plotRole(T,ObjNames,"pop_c",[0.25 0.45 0.85],24,Handles,Labels,"Population C");
    [Handles,Labels] = plotRole(T,ObjNames,"pop_u",[0.88 0.45 0.18],24,Handles,Labels,"Population U");
    [Handles,Labels] = plotRole(T,ObjNames,"archive_b",[0.05 0.05 0.05],44,Handles,Labels,"Boundary archive");
    [Handles,Labels] = plotRole(T,ObjNames,"boundary_off",[0.15 0.70 0.35],58,Handles,Labels,"MLP-selected child");
    finishAxes(ObjNames);
    if ~isempty(Handles)
        legend(Handles,cellstr(Labels),'Location','best');
    end
end

function [Handles,Labels] = plotRole(T,ObjNames,Role,Color,Size,Handles,Labels,Label)
    Mask = T.role == Role;
    if ~any(Mask)
        return;
    end
    h = scatterObjective(T(Mask,:),ObjNames,Size,Color,true);
    set(h,'DisplayName',Label);
    Handles(end+1,1) = h;
    Labels(end+1,1) = string(Label);
end

function plotBoundaryMetric(T,ObjNames,MetricName,MetricLabel)
    hold on;
    plotContext(T,ObjNames);
    Mask = T.role == "boundary_off";
    if any(Mask)
        Values = double(T.(MetricName)(Mask));
        h = scatterObjective(T(Mask,:),ObjNames,64,Values,true);
        set(h,'MarkerEdgeColor',[0.05 0.05 0.05],'LineWidth',0.5);
        colormap(gca,parula);
        cb = colorbar;
        ylabel(cb,MetricLabel,'Interpreter','none');
    else
        text(0.5,0.5,'No boundary offspring at this generation', ...
            'Units','normalized','HorizontalAlignment','center');
    end
    finishAxes(ObjNames);
end

function plotContext(T,ObjNames)
    scatterObjective(T(T.role == "pop_c",:),ObjNames,18,[0.70 0.78 0.92],true);
    scatterObjective(T(T.role == "pop_u",:),ObjNames,18,[0.95 0.74 0.58],true);
    scatterObjective(T(T.role == "archive_b",:),ObjNames,34,[0.15 0.15 0.15],true);
end

function boundaryFile = plotAllBoundaryOffspring(T,ObjNames,outDir)
    boundaryFile = strings(0,1);
    Boundary = T(T.role == "boundary_off",:);
    if isempty(Boundary)
        return;
    end
    ContextGen = unique(double(T.generation(T.role ~= "boundary_off")),'stable');
    Context = table();
    if ~isempty(ContextGen)
        Context = T(T.generation == ContextGen(end) & T.role ~= "boundary_off",:);
    end

    fig = figure('Visible','off','Color','w','Position',[100 100 1050 460]);
    layout = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    title(layout,'PRBCCMO_t all MLP-selected boundary offspring','Interpreter','none');

    nexttile(layout);
    hold on;
    plotContext(Context,ObjNames);
    scatterObjective(Boundary,ObjNames,42,double(Boundary.margin),true);
    colormap(gca,parula);
    cb = colorbar;
    ylabel(cb,'MLP margin');
    title('All selected children by margin');
    finishAxes(ObjNames);

    nexttile(layout);
    hold on;
    plotContext(Context,ObjNames);
    scatterObjective(Boundary,ObjNames,42,double(Boundary.fe_ratio),true);
    colormap(gca,parula);
    cb = colorbar;
    ylabel(cb,'FE ratio');
    title('All selected children by stage');
    finishAxes(ObjNames);

    filePath = fullfile(outDir,'objective_boundary_offspring_all.png');
    exportObjectiveFigure(fig,filePath);
    close(fig);
    boundaryFile = string(filePath);
end

function h = scatterObjective(T,ObjNames,Size,Color,Filled)
    if isempty(T)
        h = gobjects(0);
        return;
    end
    X = double(T.(ObjNames{1}));
    Y = double(T.(ObjNames{2}));
    if numel(ObjNames) >= 3
        Z = double(T.(ObjNames{3}));
        if Filled
            h = scatter3(X,Y,Z,Size,Color,'filled');
        else
            h = scatter3(X,Y,Z,Size,Color);
        end
    else
        if Filled
            h = scatter(X,Y,Size,Color,'filled');
        else
            h = scatter(X,Y,Size,Color);
        end
    end
end

function finishAxes(ObjNames)
    grid on;
    box on;
    xlabel(ObjNames{1},'Interpreter','none');
    ylabel(ObjNames{2},'Interpreter','none');
    if numel(ObjNames) >= 3
        zlabel(ObjNames{3},'Interpreter','none');
        view(45,25);
    end
end

function exportObjectiveFigure(fig,filePath)
    try
        exportgraphics(fig,filePath,'Resolution',180);
    catch
        saveas(fig,filePath);
    end
end
