function OutputFile = plot_PRBCCMO_t_archive_boundary_relation(runFolder,outDir,generation)
% Plot the relation between archive B and the true boundary.

    if nargin < 1 || isempty(runFolder)
        runFolder = latestObjectiveTraceFolder();
    end
    runFolder = char(string(runFolder));
    objectiveFile = fullfile(runFolder,'objective_snapshot.csv');
    metaFile = fullfile(runFolder,'run_meta.csv');
    assert(isfile(objectiveFile) && isfile(metaFile), ...
        'plot_PRBCCMO_t_archive_boundary_relation:MissingCsv', ...
        'Run folder must contain run_meta.csv and objective_snapshot.csv: %s', runFolder);

    if nargin < 2 || isempty(outDir)
        outDir = fullfile(runFolder,'archive_boundary_relation_figures');
    end
    outDir = char(string(outDir));
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    T = readtable(objectiveFile,'TextType','string');
    Meta = readtable(metaFile,'TextType','string');
    ObjNames = objectiveColumnNames(T);
    assert(numel(ObjNames) == 2, ...
        'plot_PRBCCMO_t_archive_boundary_relation:OnlyBiObjectiveSupported', ...
        'Archive-boundary relation plots require a bi-objective run.');

    if nargin < 3 || isempty(generation)
        generation = chooseArchiveGeneration(T);
    end
    generation = double(generation);
    Stage = T(T.generation == generation,:);
    assert(~isempty(Stage), ...
        'plot_PRBCCMO_t_archive_boundary_relation:MissingGeneration', ...
        'Generation %.0f is not available in objective_snapshot.csv.', generation);

    Problem = feval(char(Meta.problem(1)),'N',max(100,height(Stage)));
    PF = Problem.GetPF();
    B = Stage(Stage.role == "archive_b",:);

    fig = figure('Visible','off','Color','w','Position',[100 100 760 620]);
    ax = Draw(axes(fig));
    hold(ax,'on');
    [hRegion,hBoundary] = drawPlatEMORegion(ax,PF);
    hArchive = drawBoundaryArchive(ax,B,ObjNames,PF);
    finishArchiveBoundaryAxes(ax,Meta,Stage,ObjNames,generation);

    handles = [hRegion,hBoundary,hArchive];
    labels = {'Feasible region','True boundary','Boundary archive B'};
    valid = isgraphics(handles);
    legend(ax,handles(valid),labels(valid),'Location','best','Box','on');

    OutputFile = string(fullfile(outDir,sprintf('%s_run%d_gen%06.0f_archive_boundary.png', ...
        char(Meta.problem(1)),double(Meta.run(1)),generation)));
    exportgraphics(fig,char(OutputFile),'Resolution',220);
    close(fig);
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
        'plot_PRBCCMO_t_archive_boundary_relation:NoObjectiveTrace', ...
        'No PRBCCMO_t objective trace folder found under %s.', baseDir);
    [~,idx] = max(times);
    folder = char(folders(idx));
end

function ObjNames = objectiveColumnNames(T)
    Names = string(T.Properties.VariableNames);
    Mask = ~cellfun(@isempty,regexp(cellstr(Names),'^obj\d+$','once'));
    ObjNames = cellstr(Names(Mask));
end

function generation = chooseArchiveGeneration(T)
    Generations = unique(double(T.generation),'stable');
    BestGeneration = Generations(end);
    BestCount = -1;
    for g = reshape(Generations,1,[])
        Count = sum(T.generation == g & (T.role == "archive_b" | T.role == "boundary_evidence"));
        if Count > BestCount
            BestGeneration = g;
            BestCount = Count;
        end
    end
    generation = BestGeneration;
end

function [hRegion,hBoundary] = drawPlatEMORegion(ax,PF)
    hRegion = gobjects(1);
    hBoundary = gobjects(1);
    if iscell(PF) && numel(PF) >= 3 && ~isempty(PF{1})
        hRegion = surf(ax,PF{1},PF{2},PF{3}, ...
            'EdgeColor','none','FaceColor',[.85 .85 .85], ...
            'FaceAlpha',0.70,'DisplayName','Feasible region');
        set(ax,'Children',ax.Children(flip(1:end)));
        Mask = isfinite(PF{3});
        if any(Mask(:)) && any(~Mask(:))
            [~,hBoundary] = contour(ax,PF{1},PF{2},double(Mask),[0.5 0.5], ...
                '-k','LineWidth',1.1,'DisplayName','True boundary');
        end
    elseif isnumeric(PF) && size(PF,2) >= 2
        hBoundary = plot(ax,PF(:,1),PF(:,2),'-k','LineWidth',1.1,'DisplayName','True boundary');
    end
end

function hArchive = drawBoundaryArchive(ax,B,ObjNames,PF)
    hArchive = gobjects(1);
    if isempty(B)
        return;
    end
    X = double(B.(ObjNames{1}));
    Y = double(B.(ObjNames{2}));
    hArchive = plot(ax,X,Y,'o', ...
        'MarkerSize',5.5,'LineStyle','none', ...
        'MarkerFaceColor',[0.0000 0.4470 0.7410], ...
        'MarkerEdgeColor',[0.00 0.20 0.40], ...
        'LineWidth',0.7,'DisplayName','Boundary archive B');
end

function finishArchiveBoundaryAxes(ax,Meta,Stage,ObjNames,generation)
    axis(ax,'tight');
    grid(ax,'on');
    box(ax,'on');
    view(ax,[0 90]);
    set(ax,'FontName','Times New Roman','FontSize',13,'Layer','top');
    xlabel(ax,ObjNames{1},'Interpreter','none');
    ylabel(ax,ObjNames{2},'Interpreter','none');
    title(ax,sprintf('%s run %d, generation %.0f, FE %.0f', ...
        char(Meta.problem(1)),double(Meta.run(1)),generation,double(Stage.fe(1))), ...
        'Interpreter','none','FontWeight','normal');
end
