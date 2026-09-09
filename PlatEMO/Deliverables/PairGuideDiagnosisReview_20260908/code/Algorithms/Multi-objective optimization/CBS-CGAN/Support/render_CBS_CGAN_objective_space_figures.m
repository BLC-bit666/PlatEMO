function Summary = render_CBS_CGAN_objective_space_figures( ...
        rootPath,campaignName)
%RENDER_CBS_CGAN_OBJECTIVE_SPACE_FIGURES Render 10 FE panels per problem.

    if nargin < 1 || isempty(rootPath)
        rootPath = fileparts(which('platemo'));
    end
    if nargin < 2 || isempty(campaignName)
        campaignName = "CBS_CGAN_objective_space_run1_10problems";
    end
    rootPath = char(rootPath);
    campaignDir = fullfile(rootPath,'Data',char(campaignName));
    Manifest = load(fullfile(campaignDir,'campaign_manifest.mat'), ...
        'Protocol','Tasks');
    Protocol = Manifest.Protocol;
    Tasks = Manifest.Tasks;
    addCBSPaths(rootPath);

    figureCount = 0;
    for t = 1 : numel(Tasks)
        Data = load(Tasks(t).outputFile,'Record','Snapshots');
        problemConstructor = str2func(char(Tasks(t).problem));
        Problem = problemConstructor('N',Protocol.popSize, ...
            'D',Protocol.dimension,'maxFE',Protocol.maxFE);
        outputDir = fullfile(campaignDir,'figures',char(Tasks(t).problem));
        ensureFolder(outputDir);
        for s = 1 : numel(Data.Snapshots)
            Snapshot = Data.Snapshots(s);
            outputFile = fullfile(outputDir,sprintf('%s_FE%06d.png', ...
                Tasks(t).problem,Snapshot.targetFE));
            if validPNG(outputFile)
                figureCount = figureCount+1;
                continue;
            elseif exist(outputFile,'file') == 2
                error('CBSRegionGAN:InvalidExistingObjectiveSpaceFigure', ...
                    'An invalid figure exists and will not be overwritten: %s', ...
                    outputFile);
            end
            renderOne(Problem,Tasks(t).problem,Snapshot,outputFile);
            figureCount = figureCount+1;
        end
    end
    Summary = struct('schemaVersion',Protocol.schemaVersion, ...
        'campaignName',string(campaignName), ...
        'problemCount',numel(Tasks),'figureCount',figureCount, ...
        'expectedFigureCount',numel(Tasks)*numel(Protocol.targetFE), ...
        'finishedAt',string(datetime('now')));
    if Summary.figureCount ~= Summary.expectedFigureCount
        error('CBSRegionGAN:ObjectiveSpaceFigureCount', ...
            'Rendered %d figures; expected %d.',Summary.figureCount, ...
            Summary.expectedFigureCount);
    end
    save(fullfile(campaignDir,'figure_summary.mat'),'Summary');
end

function renderOne(Problem,problemName,S,outputFile)
    Figure = figure('Visible','off','Color','w', ...
        'Position',[100,100,1400,1050]);
    cleanup = onCleanup(@()close(Figure));
    Ax = axes(Figure,'Position',[0.085,0.105,0.86,0.80]);
    hold(Ax,'on');
    colors = figureColors();

    if Problem.M == 2
        [hFeasible,hInfeasible] = draw_CBS_CGAN_objective_region( ...
            Ax,Problem,problemName,colors);
        hRaw = scatter(Ax,S.rawObjs(:,1),S.rawObjs(:,2),22, ...
            colors.raw,'filled','o','MarkerFaceAlpha',0.58, ...
            'MarkerEdgeColor','w','MarkerEdgeAlpha',0.35, ...
            'LineWidth',0.35);
        hGuided = scatter(Ax,S.guidedObjs(:,1),S.guidedObjs(:,2),82, ...
            colors.guided,'filled','d','MarkerEdgeColor',[0.12,0.12,0.12], ...
            'LineWidth',0.8);
        xlabel(Ax,'f_1','Interpreter','tex');
        ylabel(Ax,'f_2','Interpreter','tex');
        axis(Ax,'tight');
    else
        [hFeasible,hInfeasible] = draw_CBS_CGAN_objective_region( ...
            Ax,Problem,problemName,colors);
        hRaw = scatter3(Ax,S.rawObjs(:,1),S.rawObjs(:,2), ...
            S.rawObjs(:,3),18,colors.raw,'filled','o', ...
            'MarkerFaceAlpha',0.62,'MarkerEdgeColor','none');
        hGuided = scatter3(Ax,S.guidedObjs(:,1),S.guidedObjs(:,2), ...
            S.guidedObjs(:,3),88,colors.guided,'filled','d', ...
            'MarkerEdgeColor',[0.12,0.12,0.12],'LineWidth',0.8);
        xlabel(Ax,'f_1','Interpreter','tex');
        ylabel(Ax,'f_2','Interpreter','tex');
        zlabel(Ax,'f_3','Interpreter','tex');
        axis(Ax,'tight');
        axis(Ax,'vis3d');
        view(Ax,42,25);
        Ax.Position = [0.11,0.21,0.61,0.56];
    end
    grid(Ax,'on');
    box(Ax,'on');
    Ax.FontName = 'Helvetica';
    Ax.FontSize = 13;
    Ax.LineWidth = 0.9;
    titleLine = sprintf( ...
        '%s  |  target FE %s  |  event FE %s  |  pool FE %s', ...
        problemName,commaNumber(S.targetFE),commaNumber(S.actualFE), ...
        commaNumber(S.poolFE));
    annotation(Figure,'textbox',[0.05,0.95,0.90,0.035], ...
        'String',titleLine,'Interpreter','none','EdgeColor','none', ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontName','Helvetica','FontWeight','bold','FontSize',15);
    annotation(Figure,'textbox',[0.05,0.915,0.90,0.03], ...
        'String',sprintf( ...
        'Raw CGAN candidates: %d     CGAN-guided offspring: %d', ...
        S.rawCount,S.guidedCount),'Interpreter','none','EdgeColor','none', ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontName','Helvetica','FontSize',12);
    Legend = legend(Ax,[hFeasible,hInfeasible,hRaw,hGuided], ...
        {'Feasible objective region','Infeasible objective region', ...
         sprintf('Raw CGAN candidates (%d)',S.rawCount), ...
         sprintf('CGAN-guided offspring (%d)',S.guidedCount)}, ...
        'Location','northeastoutside','FontSize',11);
    if Problem.M == 3
        Legend.Units = 'normalized';
        Legend.Position = [0.765,0.735,0.225,0.14];
    end

    partialFile = [outputFile,'.partial.png'];
    exportgraphics(Figure,partialFile,'Resolution',180, ...
        'BackgroundColor','white');
    if ~validPNG(partialFile)
        error('CBSRegionGAN:InvalidObjectiveSpacePNG', ...
            'Rendered PNG failed validation: %s',partialFile);
    end
    [moved,message] = movefile(partialFile,outputFile);
    if ~moved
        error('CBSRegionGAN:ObjectiveSpacePNGMoveFailed','%s',message);
    end
    clear cleanup;
end

function C = figureColors()
    C = struct('feasible',[0.36,0.72,0.47], ...
        'infeasible',[0.90,0.43,0.40], ...
        'raw',[0.13,0.43,0.82], ...
        'guided',[1.00,0.66,0.12]);
end

function textValue = commaNumber(value)
    textValue = regexprep(sprintf('%.0f',value), ...
        '(?<!\d)(\d{1,3})(?=(\d{3})+(?!\d))','$1,');
end

function valid = validPNG(file)
    valid = false;
    if exist(file,'file') ~= 2
        return;
    end
    try
        Info = imfinfo(file);
        valid = strcmpi(Info.Format,'png') && ...
            Info.Width >= 1000 && Info.Height >= 700;
    catch
        valid = false;
    end
end

function ensureFolder(folder)
    if ~isfolder(folder)
        mkdir(folder);
    end
end
