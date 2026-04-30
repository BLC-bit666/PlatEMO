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
    requireStageRoles(Stage,{'pop_c','pop_u','archive_b'});

    Problem = feval(char(Meta.problem(1)),'N',max(100,height(Stage)));
    PF = Problem.GetPF();

    fig = figure('Visible','off','Color','w','Position',[100 100 920 720]);
    ax = axes(fig);
    hold(ax,'on');
    plotProblemRegions(ax,PF,Stage,ObjNames);
    plotGenerationRole(ax,Stage,ObjNames,"pop_c",[0.08 0.32 0.86],34,'P_C constraint population','o');
    plotGenerationRole(ax,Stage,ObjNames,"pop_u",[0.90 0.42 0.08],34,'P_U unconstrained population','o');
    plotGenerationRole(ax,Stage,ObjNames,"archive_b",[0.02 0.02 0.02],62,'Boundary archive B','s');
    plotGenerationRole(ax,Stage,ObjNames,"boundary_off",[0.62 0.10 0.78],86,'MLP-selected boundary child','^');
    finishRegionAxes(ax,Meta,Stage,ObjNames,generation);

    OutputFile = string(fullfile(outDir,sprintf('%s_run%d_gen%06.0f_region.png', ...
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
        BoundaryCount = sum(Stage.role == "boundary_off");
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

function plotProblemRegions(ax,PF,Stage,ObjNames)
    [XLim,YLim] = resolveRegionLimits(PF,Stage,ObjNames);
    patch(ax,[XLim(1) XLim(2) XLim(2) XLim(1)],[YLim(1) YLim(1) YLim(2) YLim(2)], ...
        [1.00 0.93 0.93],'EdgeColor','none','DisplayName','Infeasible objective region');

    if iscell(PF) && numel(PF) >= 3 && ~isempty(PF{1})
        FeasibleMask = isfinite(PF{3});
        h = imagesc(ax,PF{1}(1,:),PF{2}(:,1),FeasibleMask);
        set(h,'AlphaData',0.72*double(FeasibleMask));
        set(h,'HandleVisibility','off');
        colormap(ax,[1.00 0.93 0.93;0.74 0.74 0.74]);
        set(ax,'YDir','normal');
        patch(ax,nan,nan,[0.74 0.74 0.74], ...
            'EdgeColor','none','DisplayName','Feasible objective region');
        contour(ax,PF{1},PF{2},double(FeasibleMask),[0.5 0.5], ...
            'Color',[0.45 0.45 0.45],'LineWidth',0.8,'HandleVisibility','off');
    end
    xlim(ax,XLim);
    ylim(ax,YLim);
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
    legend(ax,'Location','bestoutside');
end
