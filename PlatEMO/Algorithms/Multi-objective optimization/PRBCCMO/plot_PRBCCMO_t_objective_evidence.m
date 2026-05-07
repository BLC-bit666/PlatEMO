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
        title('Population, archive, boundary evidence');

        nexttile(layout);
        plotBoundaryMetric(Stage,ObjNames,'margin','MLP margin');
        title('Boundary evidence: margin');

        nexttile(layout);
        plotBoundaryMetric(Stage,ObjNames,'true_boundary_dist','Distance to true boundary');
        title('Boundary evidence: true-boundary distance');

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
    Context = T.role ~= "boundary_off" & T.role ~= "boundary_evidence";
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
    [Handles,Labels] = plotRole(T,ObjNames,"boundary_evidence",[0.55 0.55 0.55],34,Handles,Labels,"Evaluated boundary evidence");
    [Handles,Labels] = plotRole(T,ObjNames,"boundary_off",[0.15 0.70 0.35],58,Handles,Labels,"MLP-selected child");
    [Handles,Labels] = plotSegmentRootEvidence(T,ObjNames,Handles,Labels);
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

function [Handles,Labels] = plotSegmentRootEvidence(T,ObjNames,Handles,Labels)
    F = T(T.role == "pair_feasible_endpoint",:);
    U = T(T.role == "pair_infeasible_endpoint",:);
    R = T(T.role == "segment_root",:);
    Count = min([height(F),height(U),height(R)]);
    if Count <= 0
        return;
    end

    for i = 1 : Count
        plotPairSegment(F,U,ObjNames,i,[0.40 0.40 0.40],'-');
        plotPairSegmentToRoot(F,R,ObjNames,i,[0.15 0.55 0.25]);
        plotPairSegmentToRoot(U,R,ObjNames,i,[0.15 0.55 0.25]);
    end
    hF = scatterObjective(F(1:Count,:),ObjNames,34,[0.20 0.55 0.95],true);
    hU = scatterObjective(U(1:Count,:),ObjNames,34,[0.95 0.35 0.22],true);
    hR = scatterObjective(R(1:Count,:),ObjNames,92,[0.05 0.70 0.25],true);
    set(hF,'MarkerEdgeColor','w','LineWidth',0.4,'DisplayName','Feasible pair endpoint');
    set(hU,'MarkerEdgeColor','w','LineWidth',0.4,'DisplayName','Infeasible pair endpoint');
    set(hR,'MarkerEdgeColor',[0.02 0.20 0.08],'LineWidth',0.8,'DisplayName','Segment root');
    Handles = [Handles;hF;hU;hR];
    Labels = [Labels;"Feasible pair endpoint";"Infeasible pair endpoint";"Segment root"];
end

function plotPairSegment(A,B,ObjNames,i,Color,Style)
    X = [double(A.(ObjNames{1})(i)),double(B.(ObjNames{1})(i))];
    Y = [double(A.(ObjNames{2})(i)),double(B.(ObjNames{2})(i))];
    if numel(ObjNames) >= 3
        Z = [double(A.(ObjNames{3})(i)),double(B.(ObjNames{3})(i))];
        plot3(X,Y,Z,Style,'Color',Color,'LineWidth',0.8,'HandleVisibility','off');
    else
        plot(X,Y,Style,'Color',Color,'LineWidth',0.8,'HandleVisibility','off');
    end
end

function plotPairSegmentToRoot(A,R,ObjNames,i,Color)
    plotPairSegment(A,R,ObjNames,i,Color,':');
end

function plotBoundaryMetric(T,ObjNames,MetricName,MetricLabel)
    hold on;
    plotContext(T,ObjNames);
    Mask = T.role == "boundary_evidence" | T.role == "boundary_off";
    if any(Mask)
        Values = double(T.(MetricName)(Mask));
        h = scatterObjective(T(Mask,:),ObjNames,64,Values,true);
        set(h,'MarkerEdgeColor',[0.05 0.05 0.05],'LineWidth',0.5);
        colormap(gca,parula);
        cb = colorbar;
        ylabel(cb,MetricLabel,'Interpreter','none');
    else
        text(0.5,0.5,'No boundary evidence at this generation', ...
            'Units','normalized','HorizontalAlignment','center');
    end
    finishAxes(ObjNames);
end

function plotContext(T,ObjNames)
    scatterObjective(T(T.role == "pop_c",:),ObjNames,18,[0.70 0.78 0.92],true);
    scatterObjective(T(T.role == "pop_u",:),ObjNames,18,[0.95 0.74 0.58],true);
    scatterObjective(T(T.role == "archive_b",:),ObjNames,34,[0.15 0.15 0.15],true);
    scatterObjective(T(T.role == "boundary_evidence",:),ObjNames,24,[0.68 0.68 0.68],true);
end

function boundaryFile = plotAllBoundaryOffspring(T,ObjNames,outDir)
    boundaryFile = strings(0,1);
    Boundary = T(T.role == "boundary_evidence" | T.role == "boundary_off",:);
    if isempty(Boundary)
        return;
    end
    ContextGen = unique(double(T.generation(T.role ~= "boundary_off" & T.role ~= "boundary_evidence")),'stable');
    Context = table();
    if ~isempty(ContextGen)
        Context = T(T.generation == ContextGen(end) & T.role ~= "boundary_off" & T.role ~= "boundary_evidence",:);
    end

    fig = figure('Visible','off','Color','w','Position',[100 100 1050 460]);
    layout = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    title(layout,'PRBCCMO_t evaluated boundary evidence','Interpreter','none');

    nexttile(layout);
    hold on;
    plotContext(Context,ObjNames);
    scatterObjective(Boundary,ObjNames,42,double(Boundary.margin),true);
    colormap(gca,parula);
    cb = colorbar;
    ylabel(cb,'MLP margin');
    title('All evaluated boundary evidence by margin');
    finishAxes(ObjNames);

    nexttile(layout);
    hold on;
    scatter(double(Boundary.margin),double(Boundary.true_boundary_dist),42,double(Boundary.fe_ratio),'filled', ...
        'MarkerEdgeColor',[0.05 0.05 0.05],'LineWidth',0.4);
    grid on;
    box on;
    xlabel('calibrated margin');
    ylabel('dist to true boundary');
    colormap(gca,parula);
    cb = colorbar;
    ylabel(cb,'FE ratio');
    title('MLP margin vs true-boundary distance');

    filePath = fullfile(outDir,'objective_boundary_evidence_all.png');
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
