function FigureManifest = plot_CCMO_GAN_BDG_fixmd_pairt_run_figures( ...
        outRoot,suiteName,runId,targetFE)
% Plot one-run matched FixMD pair+t diagnostic objective comparisons.

    rootDir = fileparts(which('platemo'));
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outRoot)
        error('plot_CCMO_GAN_BDG_fixmd_pairt_run_figures:MissingOutRoot', ...
            'outRoot is required.');
    end
    if nargin < 2 || isempty(suiteName)
        suiteName = "matched_global_ref_goal10";
    end
    if nargin < 3 || isempty(runId)
        runId = 1;
    end
    if nargin < 4 || isempty(targetFE)
        targetFE = 100000;
    end

    suiteName = string(suiteName);
    runId = round(double(runId));
    targetFE = round(double(targetFE));
    manifestFile = fullfile(outRoot,'suite_manifest.csv');
    if ~isfile(manifestFile)
        error('plot_CCMO_GAN_BDG_fixmd_pairt_run_figures:MissingManifest', ...
            'Missing suite_manifest.csv in %s.',outRoot);
    end

    Manifest = readtable(manifestFile,'TextType','string','Delimiter',',', ...
        'VariableNamingRule','preserve');
    Manifest = Manifest(string(Manifest.suite) == suiteName,:);
    if isempty(Manifest)
        error('plot_CCMO_GAN_BDG_fixmd_pairt_run_figures:MissingSuite', ...
            'Suite %s was not found in %s.',suiteName,manifestFile);
    end
    variants = unique(string(Manifest.variant),'stable');
    problems = manifestProblems_BDG(Manifest);
    figureRoot = fullfile(outRoot,'figures', ...
        sprintf('%s_run%d',safeName_BDG(suiteName),runId));
    if ~isfolder(figureRoot)
        mkdir(figureRoot);
    end

    Rows = repmat(emptyFigureRow_BDG(),0,1);
    for p = 1 : numel(problems)
        problemName = problems(p);
        Panel = repmat(emptyPanelData_BDG(),numel(variants),1);
        for v = 1 : numel(variants)
            rowIdx = manifestRowForProblem_BDG(Manifest,variants(v), ...
                problemName);
            Panel(v) = loadPanelData_BDG(Manifest(rowIdx,:), ...
                problemName,runId,targetFE);
        end
        figureFile = fullfile(figureRoot, ...
            sprintf('%s_run%d_FE%06d_ref_vs_global.png', ...
            safeName_BDG(problemName),runId,targetFE));
        plotProblemComparison_BDG(problemName,variants,Panel, ...
            runId,targetFE,figureFile);
        Rows(end+1,1) = emptyFigureRow_BDG(); %#ok<AGROW>
        Rows(end).suite = suiteName;
        Rows(end).problem = problemName;
        Rows(end).run = runId;
        Rows(end).target_FE = targetFE;
        Rows(end).variant_count = numel(variants);
        Rows(end).figure_file = string(figureFile);
    end

    FigureManifest = struct2table(Rows);
    writetable(FigureManifest,fullfile(figureRoot, ...
        'fixmd_pairt_run_figure_manifest.csv'));
end

function rowIdx = manifestRowForProblem_BDG(Manifest,variant,problemName)
    variantHit = find(string(Manifest.variant) == string(variant));
    rowIdx = [];
    for i = reshape(variantHit,1,[])
        if manifestRowHasProblem_BDG(Manifest(i,:),problemName)
            rowIdx = i;
            return;
        end
    end
    if ~isempty(variantHit)
        rowIdx = variantHit(1);
    else
        error('plot_CCMO_GAN_BDG_fixmd_pairt_run_figures:MissingVariant', ...
            'Variant %s was not found in the manifest.',variant);
    end
end

function flag = manifestRowHasProblem_BDG(Row,problemName)
    vars = string(Row.Properties.VariableNames);
    if ~ismember("problems",vars)
        flag = true;
        return;
    end
    parts = string(strsplit(char(string(Row.problems(1))),';'));
    flag = any(parts == string(problemName));
end

function problems = manifestProblems_BDG(Manifest)
    text = string(Manifest.problems(1));
    parts = string(strsplit(char(text),';'));
    problems = parts(strlength(parts) > 0);
end

function Panel = loadPanelData_BDG(ManifestRow,problemName,runId,targetFE)
    Panel = emptyPanelData_BDG();
    Panel.variant = string(ManifestRow.variant);
    Panel.outDir = string(ManifestRow.outDir);
    taskName = sprintf('%s_run%d',char(problemName),runId);
    snapshotManifest = fullfile(char(Panel.outDir),'snapshots', ...
        taskName,'archive_snapshot_manifest.csv');
    diagnosticManifest = fullfile(char(Panel.outDir),'diagnostics', ...
        taskName,'gan_diagnostic_manifest.csv');
    Panel.snapshot_manifest_file = string(snapshotManifest);
    Panel.diagnostic_manifest_file = string(diagnosticManifest);

    if isfile(snapshotManifest)
        S = readtable(snapshotManifest,'TextType','string','Delimiter',',', ...
            'VariableNamingRule','preserve');
        if ~isempty(S)
            idx = selectTargetRow_BDG(S,targetFE);
            Panel.actual_FE = tableNumber_BDG(S,'actual_FE',idx);
            Panel.gen = tableNumber_BDG(S,'gen',idx);
            Panel.snapshot_file = string(S.snapshot_file(idx));
            if isfile(Panel.snapshot_file)
                Panel.Boundary = readtable(Panel.snapshot_file, ...
                    'TextType','string','Delimiter',',', ...
                    'VariableNamingRule','preserve');
            end
        end
    end

    if isfile(diagnosticManifest)
        D = readtable(diagnosticManifest,'TextType','string', ...
            'Delimiter',',','VariableNamingRule','preserve');
        if ~isempty(D)
            idx = selectTargetRow_BDG(D,targetFE);
            Panel.sample_file = string(D.sample_file(idx));
            if isfile(Panel.sample_file)
                Panel.Samples = readtable(Panel.sample_file, ...
                    'TextType','string','Delimiter',',', ...
                    'VariableNamingRule','preserve');
            end
        end
    end
end

function idx = selectTargetRow_BDG(T,targetFE)
    target = numericColumn_BDG(T,'target_FE');
    if isempty(target)
        idx = height(T);
        return;
    end
    exact = find(round(target) == round(targetFE),1,'last');
    if ~isempty(exact)
        idx = exact;
    else
        [~,idx] = min(abs(target - targetFE));
    end
end

function plotProblemComparison_BDG(problemName,variants,Panel,runId, ...
        targetFE,figureFile)
    ProblemConstructor = str2func(char(problemName));
    Problem = ProblemConstructor('N',100,'maxFE',100000);
    [xLimits,yLimits] = objectiveLimits_BDG(Problem,Panel);

    fig = figure('Visible','off','Color','w','Position',[80 80 1480 660]);
    layout = tiledlayout(fig,1,numel(variants), ...
        'Padding','compact','TileSpacing','compact');
    title(layout,sprintf('%s | run=%d | target FE=%d', ...
        char(problemName),runId,targetFE), ...
        'Interpreter','none','FontWeight','normal');
    for v = 1 : numel(variants)
        ax = nexttile(layout);
        drawPanel_BDG(ax,Problem,Panel(v),xLimits,yLimits);
    end
    exportgraphics(fig,figureFile,'Resolution',300);
    close(fig);
end

function drawPanel_BDG(ax,Problem,Panel,xLimits,yLimits)
    hold(ax,'on');
    set(ax,'FontName','Times New Roman','FontSize',10,'Box','on', ...
        'Layer','top','Color',[0.985 0.96 0.93]);
    plotFeasibleRegion_BDG(ax,Problem);
    plotBoundarySnapshot_BDG(ax,Panel.Boundary,xLimits,yLimits);
    plotDiagnosticSamples_BDG(ax,Panel.Samples,xLimits,yLimits);
    grid(ax,'on');
    ax.GridAlpha = 0.14;
    xlim(ax,xLimits);
    ylim(ax,yLimits);
    xlabel(ax,'f_1');
    ylabel(ax,'f_2');
    title(ax,sprintf('%s | FE=%s | gen=%s',char(Panel.variant), ...
        numberText_BDG(Panel.actual_FE),numberText_BDG(Panel.gen)), ...
        'Interpreter','none','FontWeight','normal','FontSize',11);
    legend(ax,'Location','northeastoutside','Box','off');
    hold(ax,'off');
end

function plotFeasibleRegion_BDG(ax,Problem)
    plot(ax,nan,nan,'s','MarkerSize',8,'MarkerFaceColor',[0.985 0.96 0.93], ...
        'MarkerEdgeColor',[0.76 0.60 0.50], ...
        'DisplayName','Infeasible region');
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 3
        X = PF{1};
        Y = PF{2};
        Z = PF{3};
        surf(ax,X,Y,Z,'EdgeColor','none','FaceColor',[0.84 0.93 0.86], ...
            'FaceAlpha',0.62,'DisplayName','Feasible region');
        Mask = double(isfinite(Z));
        if min(Mask(:)) < max(Mask(:))
            contour(ax,X,Y,Mask,[0.5 0.5],'Color',[0.04 0.28 0.42], ...
                'LineWidth',1.5,'DisplayName','Boundary');
        end
        view(ax,[0 90]);
    elseif isnumeric(PF) && size(PF,2) >= 2
        plot(ax,PF(:,1),PF(:,2),'-','Color',[0.04 0.28 0.42], ...
            'LineWidth',1.6,'DisplayName','Reference boundary');
    end
end

function plotBoundarySnapshot_BDG(ax,Boundary,xLimits,yLimits)
    if isempty(Boundary) || height(Boundary) == 0 || ...
            ~all(ismember(["obj1","obj2"],string(Boundary.Properties.VariableNames)))
        text(ax,mean(xLimits),mean(yLimits),'missing snapshot', ...
            'HorizontalAlignment','center','Color',[0.45 0.05 0.05]);
        return;
    end
    Obj = [numericColumn_BDG(Boundary,'obj1'), ...
        numericColumn_BDG(Boundary,'obj2')];
    Role = string(Boundary.archive_role);
    Side = string(Boundary.pair_side);
    inside = insideLimits_BDG(Obj,xLimits,yLimits);
    constrained = Role == "constrained_population";
    unconstrained = Role == "unconstrained_population";
    afTrain = Role == "AF_train" & Side == "feasible_endpoint";
    aiTrain = Role == "AI_train" & Side == "infeasible_endpoint";

    scatter(ax,Obj(constrained & inside,1),Obj(constrained & inside,2), ...
        11,'o','MarkerFaceColor',[0.18 0.42 0.86], ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.18, ...
        'DisplayName','Constrained pop');
    scatter(ax,Obj(unconstrained & inside,1),Obj(unconstrained & inside,2), ...
        13,'^','MarkerFaceColor',[0.43 0.25 0.62], ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.18, ...
        'DisplayName','Unconstrained pop');
    plotBoundaryPairs_BDG(ax,Boundary,Obj,afTrain,aiTrain,inside);
    scatter(ax,Obj(afTrain & inside,1),Obj(afTrain & inside,2), ...
        28,'s','MarkerFaceColor',[1.00 0.70 0.18], ...
        'MarkerEdgeColor',[0.24 0.14 0.02], ...
        'LineWidth',0.6,'DisplayName','AF endpoint');
    scatter(ax,Obj(aiTrain & inside,1),Obj(aiTrain & inside,2), ...
        30,'d','MarkerFaceColor',[1.00 0.88 0.35], ...
        'MarkerEdgeColor',[0.24 0.14 0.02], ...
        'LineWidth',0.6,'DisplayName','AI endpoint');
end

function plotBoundaryPairs_BDG(ax,Boundary,Obj,afTrain,aiTrain,inside)
    pairIndex = numericColumn_BDG(Boundary,'pair_index');
    if isempty(pairIndex)
        return;
    end
    plot(ax,nan,nan,'-','Color',[0.86 0.46 0.02], ...
        'LineWidth',1.0,'DisplayName','AF-AI pair');
    pairs = unique(pairIndex(afTrain | aiTrain),'stable');
    for p = reshape(pairs,1,[])
        af = find(afTrain & pairIndex == p & inside,1,'first');
        ai = find(aiTrain & pairIndex == p & inside,1,'first');
        if ~isempty(af) && ~isempty(ai)
            plot(ax,Obj([af ai],1),Obj([af ai],2),'-', ...
                'Color',[0.86 0.46 0.02], ...
                'LineWidth',0.65,'HandleVisibility','off');
        end
    end
end

function plotDiagnosticSamples_BDG(ax,Samples,xLimits,yLimits)
    if isempty(Samples) || height(Samples) == 0 || ...
            ~all(ismember(["obj1","obj2","mode"], ...
            string(Samples.Properties.VariableNames)))
        return;
    end
    Obj = [numericColumn_BDG(Samples,'obj1'), ...
        numericColumn_BDG(Samples,'obj2')];
    mode = string(Samples.mode);
    feasible = numericColumn_BDG(Samples,'feasible') > 0;
    inside = insideLimits_BDG(Obj,xLimits,yLimits);
    normal = mode == "normal";
    fixedz = mode == "fixedz";
    zsweep = mode == "zsweep";

    scatter(ax,Obj(normal & feasible & inside,1), ...
        Obj(normal & feasible & inside,2),18,'o', ...
        'MarkerFaceColor',[0.88 0.04 0.08], ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.52, ...
        'DisplayName','GAN normal feasible');
    scatter(ax,Obj(normal & ~feasible & inside,1), ...
        Obj(normal & ~feasible & inside,2),16,'x', ...
        'MarkerEdgeColor',[0.65 0.04 0.08], ...
        'LineWidth',0.75,'DisplayName','GAN normal infeasible');
    scatter(ax,Obj(fixedz & inside,1),Obj(fixedz & inside,2), ...
        24,'p','MarkerFaceColor',[0.05 0.55 0.42], ...
        'MarkerEdgeColor',[0.01 0.18 0.14], ...
        'MarkerFaceAlpha',0.60,'LineWidth',0.55, ...
        'DisplayName','GAN fixed-z');
    if any(zsweep & inside)
        scatter(ax,Obj(zsweep & inside,1),Obj(zsweep & inside,2), ...
            10,'.','MarkerEdgeColor',[0.08 0.08 0.08], ...
            'DisplayName','GAN z-sweep');
    end
end

function [xLimits,yLimits] = objectiveLimits_BDG(Problem,Panel)
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
    for i = 1 : numel(Panel)
        if ~isempty(Panel(i).Boundary) && height(Panel(i).Boundary) > 0
            X = [X; numericColumn_BDG(Panel(i).Boundary,'obj1')]; %#ok<AGROW>
            Y = [Y; numericColumn_BDG(Panel(i).Boundary,'obj2')]; %#ok<AGROW>
        end
        if ~isempty(Panel(i).Samples) && height(Panel(i).Samples) > 0
            X = [X; numericColumn_BDG(Panel(i).Samples,'obj1')]; %#ok<AGROW>
            Y = [Y; numericColumn_BDG(Panel(i).Samples,'obj2')]; %#ok<AGROW>
        end
    end
    xLimits = finiteLimits_BDG(X);
    yLimits = finiteLimits_BDG(Y);
end

function limits = finiteLimits_BDG(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        limits = [0 1];
        return;
    end
    lo = min(x);
    hi = max(x);
    if lo == hi
        pad = max(1,abs(lo)) * 0.05;
    else
        pad = (hi - lo) * 0.06;
    end
    limits = [lo - pad, hi + pad];
end

function inside = insideLimits_BDG(Obj,xLimits,yLimits)
    inside = isfinite(Obj(:,1)) & isfinite(Obj(:,2)) & ...
        Obj(:,1) >= xLimits(1) & Obj(:,1) <= xLimits(2) & ...
        Obj(:,2) >= yLimits(1) & Obj(:,2) <= yLimits(2);
end

function value = tableNumber_BDG(T,name,idx)
    values = numericColumn_BDG(T,name);
    if isempty(values) || idx > numel(values)
        value = NaN;
    else
        value = double(values(idx));
    end
end

function values = numericColumn_BDG(T,name)
    vars = string(T.Properties.VariableNames);
    if isempty(T) || ~ismember(string(name),vars)
        values = zeros(height(T),1);
        return;
    end
    values = T.(char(name));
    if iscell(values)
        values = str2double(string(values));
    elseif isstring(values) || ischar(values)
        values = str2double(string(values));
    else
        values = double(values);
    end
    values = values(:);
end

function text = numberText_BDG(x)
    if ~isfinite(x)
        text = 'NA';
    else
        text = char(string(round(double(x))));
    end
end

function name = safeName_BDG(value)
    name = regexprep(char(string(value)),'[^A-Za-z0-9_=-]+','_');
end

function Row = emptyFigureRow_BDG()
    Row = struct('suite',"",'problem',"",'run',NaN,'target_FE',NaN, ...
        'variant_count',NaN,'figure_file',"");
end

function Panel = emptyPanelData_BDG()
    Panel = struct('variant',"",'outDir',"", ...
        'snapshot_manifest_file',"",'diagnostic_manifest_file',"", ...
        'snapshot_file',"",'sample_file',"", ...
        'actual_FE',NaN,'gen',NaN,'Boundary',table(),'Samples',table());
end
