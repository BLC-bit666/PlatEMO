function [Summary,outRoot,FigureManifest] = ...
        run_CCMO_GAN_BDG_fixmd_mainline_stage_figures(outRoot,workerCount)
% Run current FixMD global mainline and draw stage objective figures.
% Figures intentionally show only objective-domain feasibility, normal CGAN
% samples, and exported AF/AI training archive rows.

    rootDir = fileparts(which('platemo'));
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outRoot)
        outRoot = fullfile(rootDir,'Data','CCMO_GAN_BDG', ...
            ['fixmd_mainline_stage_figures_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 9;
    end

    problems = ["DASCMOP1_BC","DASCMOP2_BC", ...
        "DASCMOP4_BC","DASCMOP5_BC", ...
        "LIRCMOP5_BC","LIRCMOP6_BC", ...
        "LIRCMOP7_BC","LIRCMOP8_BC", ...
        "LIRCMOP9_BC","LIRCMOP10_BC"];
    targets = [5000 10000 20000 30000 50000 70000 100000];
    variant = "FixMD_GNDk60_nearseg_huber_z0";
    control = retainedMainlineControl_BDG(variant);
    algorithmParams = {1,20,0,5,0.20,50,0,200,0,1,1,1, ...
        "g2sl",64,0.9,1e-4,1e-4,"epoch"};
    diagOptions = struct( ...
        'conditionCount',30, ...
        'zPerCondition',10, ...
        'normalN',200, ...
        'seed',20260615, ...
        'variant',variant, ...
        'control',control, ...
        'algorithmParams',{algorithmParams});

    [Summary,outRoot] = run_CCMO_GAN_BDG_boundary_diagnostics( ...
        outRoot,workerCount,cellstr(problems),100,[],100000,1, ...
        targets,diagOptions);
    FigureManifest = drawStageFigures_BDG(Summary,outRoot,targets);
end

function Control = retainedMainlineControl_BDG(variant)
    Control = struct( ...
        'variant',string(variant), ...
        'archiveParetoFilterMode',"global_af_nd", ...
        'archivePairDirectionMode',"af_not_dominates_ai", ...
        'archiveSourceCapMode',"none", ...
        'archivePairRefMode',"neighbor4", ...
        'trainFilterMode',"condition_knn", ...
        'conditionKNNRetainRatio',0.60, ...
        'cganTrainMinRefCov',0, ...
        'cganTrainMinTargetTriples',0, ...
        'conditionMode',"yt_dt_t_ref", ...
        'targetMode',"near_segment_feasible", ...
        'nearSegmentTau',0.20, ...
        'nearSegmentMaxPerPair',5, ...
        'decisionInterpCount',5, ...
        'targetRealLabelMode',"binary", ...
        'generatorMode',"objective_target_conditioned", ...
        'ganCriticMode',"target_conditioned", ...
        'generatorLossMode',"conditional_adversarial_huber", ...
        'reconstructionWeight',250, ...
        'reconstructionHuberDelta',0.10);
end

function FigureManifest = drawStageFigures_BDG(Summary,outRoot,targets)
    figureRoot = fullfile(outRoot,'figures_domain_gan_train_only');
    if ~isfolder(figureRoot)
        mkdir(figureRoot);
    end
    Rows = repmat(emptyFigureRow_BDG(),0,1);
    for i = 1 : height(Summary)
        if string(Summary.status(i)) ~= "ok"
            continue;
        end
        problemName = string(Summary.problem(i));
        runId = round(double(Summary.run(i)));
        ProblemConstructor = str2func(char(problemName));
        Problem = ProblemConstructor('N',100,'maxFE',100000);
        snapshotManifest = string(Summary.snapshot_manifest_file(i));
        diagnosticManifest = string(Summary.diagnostic_manifest_file(i));
        Snap = readManifestIfExists_BDG(snapshotManifest);
        Diag = readManifestIfExists_BDG(diagnosticManifest);
        for t = reshape(targets,1,[])
            [Boundary,Sample,actualFE,gen] = loadStageTables_BDG(Snap,Diag,t);
            figFile = fullfile(figureRoot,sprintf( ...
                '%s_run%d_FE%06d_domain_gan_train.png', ...
                safeName_BDG(problemName),runId,round(t)));
            plotSingleStage_BDG(Problem,problemName,runId,t,actualFE,gen, ...
                Boundary,Sample,figFile);
            Rows(end+1,1) = emptyFigureRow_BDG(); %#ok<AGROW>
            Rows(end).problem = problemName;
            Rows(end).run = runId;
            Rows(end).target_FE = double(t);
            Rows(end).actual_FE = actualFE;
            Rows(end).gen = gen;
            Rows(end).figure_file = string(figFile);
        end
    end
    FigureManifest = struct2table(Rows);
    writetable(FigureManifest,fullfile(figureRoot, ...
        'stage_figure_manifest.csv'));
end

function T = readManifestIfExists_BDG(fileName)
    if strlength(fileName) == 0 || ~isfile(fileName)
        T = table();
    else
        T = readtable(fileName,'TextType','string','Delimiter',',', ...
            'VariableNamingRule','preserve');
    end
end

function [Boundary,Sample,actualFE,gen] = loadStageTables_BDG(Snap,Diag,targetFE)
    Boundary = table();
    Sample = table();
    actualFE = NaN;
    gen = NaN;
    if ~isempty(Snap) && height(Snap) > 0
        idx = selectTargetRow_BDG(Snap,targetFE);
        actualFE = numericAt_BDG(Snap,'actual_FE',idx);
        gen = numericAt_BDG(Snap,'gen',idx);
        snapshotFile = string(Snap.snapshot_file(idx));
        if isfile(snapshotFile)
            Boundary = readtable(snapshotFile,'TextType','string', ...
                'Delimiter',',','VariableNamingRule','preserve');
        end
    end
    if ~isempty(Diag) && height(Diag) > 0
        idx = selectTargetRow_BDG(Diag,targetFE);
        sampleFile = string(Diag.sample_file(idx));
        if isfile(sampleFile)
            Sample = readtable(sampleFile,'TextType','string', ...
                'Delimiter',',','VariableNamingRule','preserve');
        end
    end
end

function idx = selectTargetRow_BDG(T,targetFE)
    target = numericColumn_BDG(T,'target_FE');
    exact = find(round(target) == round(double(targetFE)),1,'last');
    if ~isempty(exact)
        idx = exact;
    elseif isempty(target)
        idx = 1;
    else
        [~,idx] = min(abs(target - double(targetFE)));
    end
end

function plotSingleStage_BDG(Problem,problemName,runId,targetFE,actualFE, ...
        gen,Boundary,Sample,figFile)
    [xLimits,yLimits] = objectiveLimits_BDG(Problem,Boundary,Sample);
    fig = figure('Visible','off','Color','w','Position',[100 100 900 900]);
    ax = axes(fig);
    hold(ax,'on');
    set(ax,'Box','on','Color',[0.94 0.91 0.86], ...
        'FontName','Times New Roman','FontSize',10);
    plotFeasibleDomain_BDG(ax,Problem);
    plotTrainingSet_BDG(ax,Boundary,xLimits,yLimits);
    plotNormalCGAN_BDG(ax,Sample,xLimits,yLimits);
    xlim(ax,xLimits);
    ylim(ax,yLimits);
    axis(ax,'equal');
    daspect(ax,[1 1 1]);
    pbaspect(ax,[1 1 1]);
    xlabel(ax,'f_1');
    ylabel(ax,'f_2');
    title(ax,sprintf('%s run=%d targetFE=%d actualFE=%s gen=%s', ...
        char(problemName),runId,round(targetFE),numberText_BDG(actualFE), ...
        numberText_BDG(gen)),'Interpreter','none','FontWeight','normal');
    legend(ax,'Location','bestoutside','Box','off');
    hold(ax,'off');
    exportgraphics(fig,figFile,'Resolution',300);
    close(fig);
end

function plotFeasibleDomain_BDG(ax,Problem)
    plot(ax,nan,nan,'s','MarkerSize',8, ...
        'MarkerFaceColor',[0.94 0.91 0.86], ...
        'MarkerEdgeColor',[0.72 0.64 0.54], ...
        'DisplayName','Infeasible domain');
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 3
        X = PF{1};
        Y = PF{2};
        Z = PF{3};
        mask = isfinite(Z);
        ZPlot = zeros(size(X));
        ZPlot(~mask) = NaN;
        surf(ax,X,Y,ZPlot,'EdgeColor','none', ...
            'FaceColor',[0.73 0.88 0.74], ...
            'FaceAlpha',0.58,'DisplayName','Feasible domain');
        view(ax,[0 90]);
    elseif isnumeric(PF) && size(PF,2) >= 2
        plot(ax,PF(:,1),PF(:,2),'.','Color',[0.25 0.62 0.32], ...
            'MarkerSize',7,'DisplayName','Feasible domain');
    else
        plot(ax,nan,nan,'.','Color',[0.25 0.62 0.32], ...
            'DisplayName','Feasible domain');
    end
end

function plotTrainingSet_BDG(ax,Boundary,xLimits,yLimits)
    if isempty(Boundary) || height(Boundary) == 0 || ...
            ~all(ismember(["obj1","obj2","archive_role"], ...
            string(Boundary.Properties.VariableNames)))
        scatter(ax,nan,nan,30,'s','MarkerFaceColor',[1.0 0.68 0.16], ...
            'MarkerEdgeColor',[0.25 0.14 0.03], ...
            'DisplayName','Training set');
        return;
    end
    Obj = [numericColumn_BDG(Boundary,'obj1'), ...
        numericColumn_BDG(Boundary,'obj2')];
    role = string(Boundary.archive_role);
    train = role == "AF_train" | role == "AI_train";
    inside = insideLimits_BDG(Obj,xLimits,yLimits);
    scatter(ax,Obj(train & inside,1),Obj(train & inside,2),34,'s', ...
        'MarkerFaceColor',[1.0 0.68 0.16], ...
        'MarkerEdgeColor',[0.25 0.14 0.03], ...
        'LineWidth',0.55,'MarkerFaceAlpha',0.78, ...
        'DisplayName','Training set');
end

function plotNormalCGAN_BDG(ax,Sample,xLimits,yLimits)
    if isempty(Sample) || height(Sample) == 0 || ...
            ~all(ismember(["obj1","obj2","mode"], ...
            string(Sample.Properties.VariableNames)))
        scatter(ax,nan,nan,20,'o','MarkerFaceColor',[0.84 0.06 0.10], ...
            'MarkerEdgeColor','none','DisplayName','CGAN generated');
        return;
    end
    Obj = [numericColumn_BDG(Sample,'obj1'), ...
        numericColumn_BDG(Sample,'obj2')];
    mode = string(Sample.mode);
    normal = mode == "normal";
    inside = insideLimits_BDG(Obj,xLimits,yLimits);
    scatter(ax,Obj(normal & inside,1),Obj(normal & inside,2),22,'o', ...
        'MarkerFaceColor',[0.84 0.06 0.10], ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.58, ...
        'DisplayName','CGAN generated');
end

function [xLimits,yLimits] = objectiveLimits_BDG(Problem,Boundary,Sample)
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
    if ~isempty(Boundary) && height(Boundary) > 0
        X = [X; numericColumn_BDG(Boundary,'obj1')]; %#ok<AGROW>
        Y = [Y; numericColumn_BDG(Boundary,'obj2')]; %#ok<AGROW>
    end
    if ~isempty(Sample) && height(Sample) > 0
        X = [X; numericColumn_BDG(Sample,'obj1')]; %#ok<AGROW>
        Y = [Y; numericColumn_BDG(Sample,'obj2')]; %#ok<AGROW>
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
        pad = (hi - lo) * 0.05;
    end
    limits = [lo - pad, hi + pad];
end

function inside = insideLimits_BDG(Obj,xLimits,yLimits)
    inside = isfinite(Obj(:,1)) & isfinite(Obj(:,2)) & ...
        Obj(:,1) >= xLimits(1) & Obj(:,1) <= xLimits(2) & ...
        Obj(:,2) >= yLimits(1) & Obj(:,2) <= yLimits(2);
end

function value = numericAt_BDG(T,name,idx)
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
    Row = struct('problem',"",'run',NaN,'target_FE',NaN, ...
        'actual_FE',NaN,'gen',NaN,'figure_file',"");
end
