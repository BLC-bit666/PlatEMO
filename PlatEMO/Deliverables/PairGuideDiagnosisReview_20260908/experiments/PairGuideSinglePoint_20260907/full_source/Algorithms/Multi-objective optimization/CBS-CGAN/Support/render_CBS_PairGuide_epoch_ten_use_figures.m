function Summary = render_CBS_PairGuide_epoch_ten_use_figures( ...
        rootPath,campaignName)
%RENDER_CBS_PAIRGUIDE_EPOCH_TEN_USE_FIGURES Plot one best trajectory/combo.

    if nargin < 1 || isempty(rootPath)
        rootPath = fileparts(which('platemo'));
    end
    if nargin < 2 || isempty(campaignName)
        campaignName = "CBS_PairGuide_ten_use_epoch_v3_20260901";
    end
    rootPath = char(rootPath);
    campaignDir = fullfile(rootPath,'Data',char(campaignName));
    Manifest = load(fullfile(campaignDir,'campaign_manifest.mat'), ...
        'Protocol','Tasks');
    P = Manifest.Protocol;
    Tasks = Manifest.Tasks;
    BestRuns = readtable(fullfile(campaignDir,'analysis','best_runs.csv'), ...
        'TextType','string');
    addCBSPaths(rootPath);
    expectedTrajectories = numel(P.problems)*numel(P.epochs);
    if height(BestRuns) ~= expectedTrajectories
        error('CBSPairGuide:BestRunCount', ...
            'Found %d best runs; expected %d.', ...
            height(BestRuns),expectedTrajectories);
    end

    Index = repmat(struct('problem',"",'epoch',0,'run',0, ...
        'eventIndex',0,'poolFE',NaN,'useFE',NaN,'guidedCount',0, ...
        'rawInFocus',0,'rawCount',0,'file',""), ...
        expectedTrajectories*P.captureUses,1);
    row = 0;
    for best = 1 : height(BestRuns)
        problem = BestRuns.problem(best);
        epoch = BestRuns.epoch(best);
        run = BestRuns.run(best);
        task = find([Tasks.problem] == problem & ...
            [Tasks.epoch] == epoch & [Tasks.run] == run,1);
        if isempty(task)
            error('CBSPairGuide:MissingBestRunTask', ...
                'No task for %s epoch %d run %d.',problem,epoch,run);
        end
        Data = load(Tasks(task).outputFile,'Snapshots');
        constructor = str2func(char(problem));
        Problem = constructor('N',P.popSize,'D',P.dimension, ...
            'maxFE',1e9,'maxRuntime',Inf);
        Limits = trajectoryLimits(Data.Snapshots,Problem.M);
        outputDir = fullfile(campaignDir,'figures',char(problem), ...
            sprintf('epoch_%04d',epoch),sprintf('best_run_%02d',run));
        ensureFolder(outputDir);
        for event = 1 : P.captureUses
            S = Data.Snapshots(event);
            outputFile = fullfile(outputDir,sprintf( ...
                '%s_epoch%04d_best_run%02d_use%02d_FE%06d.png', ...
                problem,epoch,run,event,S.useFE));
            rawInFocus = pointsInLimits(S.rawObjs,Limits.focusLower, ...
                Limits.focusUpper);
            if ~validPNG(outputFile)
                if exist(outputFile,'file') == 2
                    error('CBSPairGuide:InvalidExistingTenUseFigure', ...
                        'Invalid figure exists and will not be overwritten: %s', ...
                        outputFile);
                end
                renderOne(Problem,problem,epoch,run,event, ...
                    P.captureUses,S,Limits,rawInFocus,outputFile);
            end
            row = row+1;
            Index(row) = struct('problem',problem,'epoch',epoch, ...
                'run',run,'eventIndex',event,'poolFE',S.poolFE, ...
                'useFE',S.useFE,'guidedCount',S.guidedCount, ...
                'rawInFocus',rawInFocus,'rawCount',S.rawCount, ...
                'file',string(outputFile));
        end
    end
    FigureIndex = struct2table(Index);
    writetable(FigureIndex,fullfile(campaignDir,'figure_index.csv'));
    Summary = struct('schemaVersion',P.schemaVersion, ...
        'campaignName',string(campaignName), ...
        'trajectoryCount',expectedTrajectories, ...
        'figureCount',height(FigureIndex), ...
        'expectedFigureCount',expectedTrajectories*P.captureUses, ...
        'finishedAt',string(datetime('now')));
    if Summary.figureCount ~= Summary.expectedFigureCount
        error('CBSPairGuide:TenUseFigureCount', ...
            'Rendered %d figures; expected %d.',Summary.figureCount, ...
            Summary.expectedFigureCount);
    end
    save(fullfile(campaignDir,'figure_summary.mat'),'Summary');
end

function renderOne(Problem,problem,epoch,run,event,eventCount,S,L, ...
        rawInFocus,outputFile)
    Figure = figure('Visible','off','Color','w', ...
        'Position',[100,100,1600,1050]);
    cleanup = onCleanup(@()close(Figure));
    C = figureColors();
    Main = axes(Figure,'Position',[0.07,0.10,0.68,0.80]);
    hold(Main,'on');
    [hFeasible,hInfeasible] = draw_CBS_CGAN_objective_region( ...
        Main,Problem,problem,C);
    [hRaw,hGuided] = drawPoints(Main,Problem.M,S,C,true);
    setLimits(Main,L.focusLower,L.focusUpper,Problem.M);
    styleAxes(Main,Problem.M);

    Overview = axes(Figure,'Position',[0.785,0.105,0.19,0.24]);
    hold(Overview,'on');
    draw_CBS_CGAN_objective_region(Overview,Problem,problem,C);
    drawPoints(Overview,Problem.M,S,C,false);
    setLimits(Overview,L.fullLower,L.fullUpper,Problem.M);
    styleOverview(Overview,Problem.M);
    title(Overview,'全量概览','FontName','Helvetica', ...
        'FontSize',11,'FontWeight','normal');
    if Problem.M == 2
        rectangle(Overview,'Position',[L.focusLower(1),L.focusLower(2), ...
            L.focusUpper(1)-L.focusLower(1), ...
            L.focusUpper(2)-L.focusLower(2)], ...
            'EdgeColor',[0.1 0.1 0.1],'LineWidth',1.1,'LineStyle','--');
    else
        drawBox3(Overview,L.focusLower,L.focusUpper);
    end

    titleLine = sprintf( ...
        '%s  |  Epoch %d  |  best run %d  |  training use %d/%d', ...
        problem,epoch,run,event,eventCount);
    annotation(Figure,'textbox',[0.04,0.955,0.92,0.03], ...
        'String',titleLine,'Interpreter','none','EdgeColor','none', ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontName','Helvetica','FontWeight','bold','FontSize',15);
    detailLine = sprintf( ...
        'pool FE %s     use FE %s     raw candidates %d (%d in focus)     guided offspring %d/%d', ...
        commaNumber(S.poolFE),commaNumber(S.useFE),S.rawCount, ...
        rawInFocus,S.guidedCount,S.requestedCount);
    annotation(Figure,'textbox',[0.04,0.918,0.92,0.03], ...
        'String',detailLine,'Interpreter','none','EdgeColor','none', ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontName','Helvetica','FontSize',12);
    Legend = legend(Main,[hFeasible,hInfeasible,hRaw,hGuided], ...
        {'可行目标域','不可行目标域', ...
         sprintf('CGAN 原始候选 (%d)',S.rawCount), ...
         sprintf('真实引导子代 (%d/%d)', ...
         S.guidedCount,S.requestedCount)}, ...
        'Interpreter','none','FontSize',11,'Box','off');
    Legend.Units = 'normalized';
    Legend.Position = [0.78,0.69,0.20,0.17];

    partialFile = [outputFile,'.partial.png'];
    exportgraphics(Figure,partialFile,'Resolution',180, ...
        'BackgroundColor','white');
    if ~validPNG(partialFile)
        error('CBSPairGuide:InvalidTenUsePNG', ...
            'Rendered PNG failed validation: %s',partialFile);
    end
    [moved,message] = movefile(partialFile,outputFile);
    if ~moved
        error('CBSPairGuide:TenUsePNGMoveFailed','%s',message);
    end
    clear cleanup;
end

function [hRaw,hGuided] = drawPoints(Ax,M,S,C,large)
    if large
        rawSize = 25;
        guidedSize = 92;
    else
        rawSize = 10;
        guidedSize = 34;
    end
    if M == 2
        hRaw = scatter(Ax,S.rawObjs(:,1),S.rawObjs(:,2),rawSize, ...
            C.raw,'filled','o','MarkerFaceAlpha',0.62, ...
            'MarkerEdgeColor','none');
        hGuided = scatter(Ax,S.guidedObjs(:,1),S.guidedObjs(:,2), ...
            guidedSize,C.guided,'filled','d', ...
            'MarkerEdgeColor',[0.10,0.10,0.10],'LineWidth',0.8);
    else
        hRaw = scatter3(Ax,S.rawObjs(:,1),S.rawObjs(:,2), ...
            S.rawObjs(:,3),rawSize,C.raw,'filled','o', ...
            'MarkerFaceAlpha',0.62,'MarkerEdgeColor','none');
        hGuided = scatter3(Ax,S.guidedObjs(:,1),S.guidedObjs(:,2), ...
            S.guidedObjs(:,3),guidedSize,C.guided,'filled','d', ...
            'MarkerEdgeColor',[0.10,0.10,0.10],'LineWidth',0.8);
    end
end

function L = trajectoryLimits(Snapshots,M)
    Raw = zeros(0,M);
    Guided = zeros(0,M);
    for event = 1 : numel(Snapshots)
        Raw = [Raw;double(Snapshots(event).rawObjs)]; %#ok<AGROW>
        Guided = [Guided;double(Snapshots(event).guidedObjs)]; %#ok<AGROW>
    end
    Raw = Raw(all(isfinite(Raw),2),:);
    Guided = Guided(all(isfinite(Guided),2),:);
    Values = [Raw;Guided];
    if isempty(Values)
        error('CBSPairGuide:EmptyTrajectoryObjectives', ...
            'Best-run trajectory contains no finite objectives.');
    end
    fullLower = min(Values,[],1);
    fullUpper = max(Values,[],1);
    focusLower = prctile(Raw,1,1);
    focusUpper = prctile(Raw,99,1);
    if ~isempty(Guided)
        focusLower = min(focusLower,min(Guided,[],1));
        focusUpper = max(focusUpper,max(Guided,[],1));
    end
    [focusLower,focusUpper] = paddedLimits(focusLower,focusUpper,0.08);
    [fullLower,fullUpper] = paddedLimits(fullLower,fullUpper,0.04);
    focusLower = max(focusLower,fullLower);
    focusUpper = min(focusUpper,fullUpper);
    L = struct('focusLower',focusLower,'focusUpper',focusUpper, ...
        'fullLower',fullLower,'fullUpper',fullUpper);
end

function [lower,upper] = paddedLimits(lower,upper,fraction)
    span = upper-lower;
    fallback = max(1,max(abs([lower;upper]),[],1));
    span(span <= eps(fallback)) = fallback(span <= eps(fallback));
    lower = lower-fraction*span;
    upper = upper+fraction*span;
end

function count = pointsInLimits(Values,lower,upper)
    count = sum(all(isfinite(Values) & Values >= lower & ...
        Values <= upper,2));
end

function setLimits(Ax,lower,upper,M)
    xlim(Ax,[lower(1),upper(1)]);
    ylim(Ax,[lower(2),upper(2)]);
    if M == 3
        zlim(Ax,[lower(3),upper(3)]);
        view(Ax,42,25);
    end
end

function styleAxes(Ax,M)
    xlabel(Ax,'f_1','Interpreter','tex');
    ylabel(Ax,'f_2','Interpreter','tex');
    if M == 3
        zlabel(Ax,'f_3','Interpreter','tex');
    end
    grid(Ax,'on');
    box(Ax,'on');
    Ax.FontName = 'Helvetica';
    Ax.FontSize = 13;
    Ax.LineWidth = 0.9;
end

function styleOverview(Ax,M)
    grid(Ax,'on');
    box(Ax,'on');
    Ax.FontName = 'Helvetica';
    Ax.FontSize = 8;
    Ax.LineWidth = 0.7;
    if M == 3
        view(Ax,42,25);
    end
end

function drawBox3(Ax,lower,upper)
    corners = [lower;upper(1),lower(2),lower(3); ...
        upper(1),upper(2),lower(3);lower(1),upper(2),lower(3); ...
        lower(1),lower(2),upper(3);upper(1),lower(2),upper(3); ...
        upper;lower(1),upper(2),upper(3)];
    edges = [1 2;2 3;3 4;4 1;5 6;6 7;7 8;8 5;1 5;2 6;3 7;4 8];
    for edge = 1 : size(edges,1)
        rows = edges(edge,:);
        plot3(Ax,corners(rows,1),corners(rows,2),corners(rows,3), ...
            '--','Color',[0.1 0.1 0.1],'LineWidth',0.8);
    end
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
            Info.Width >= 1200 && Info.Height >= 700;
    catch
        valid = false;
    end
end

function ensureFolder(folder)
    if ~isfolder(folder), mkdir(folder); end
end
