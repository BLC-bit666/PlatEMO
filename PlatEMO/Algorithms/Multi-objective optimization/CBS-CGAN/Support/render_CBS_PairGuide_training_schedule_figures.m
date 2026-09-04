function Figures = render_CBS_PairGuide_training_schedule_figures( ...
        rootPath,campaignName)
%RENDER_CBS_PAIRGUIDE_TRAINING_SCHEDULE_FIGURES Render 3-by-10 sheets.

    if nargin < 1 || isempty(rootPath)
        rootPath = fileparts(which('platemo'));
    end
    if nargin < 2 || strlength(string(campaignName)) == 0
        error('CBSPairGuide:MissingScheduleCampaign', ...
            'rootPath and campaignName are required.');
    end
    rootPath = char(rootPath);
    campaignName = string(campaignName);
    campaignDir = fullfile(rootPath,'Data',char(campaignName));
    stages = ["ratio","epoch"];
    addCBSPaths(rootPath);

    Stage = repmat(struct('name',"",'directory',"", ...
        'Protocol',struct(),'Tasks',struct([]),'Analysis',[], ...
        'analysisFile',""),numel(stages),1);
    for s = 1 : numel(stages)
        stageDir = fullfile(campaignDir,char(stages(s)));
        manifestFile = fullfile(stageDir,'campaign_manifest.mat');
        requireFile(manifestFile,'stage campaign manifest');
        M = load(manifestFile,'Protocol','Tasks');
        if ~isfield(M,'Protocol') || ~isfield(M,'Tasks')
            error('CBSPairGuide:BadScheduleManifest', ...
                '%s must contain Protocol and Tasks.',manifestFile);
        end
        validateProtocolAndTasks(M.Protocol,M.Tasks,manifestFile);
        analysisFile = resolveAnalysisFile(stageDir);
        requireFile(analysisFile,'stage analysis');
        A = load(analysisFile,'Analysis');
        if ~isfield(A,'Analysis') || isempty(A.Analysis)
            error('CBSPairGuide:BadScheduleAnalysis', ...
                '%s must contain a nonempty Analysis value.',analysisFile);
        end
        Stage(s) = struct('name',stages(s),'directory',string(stageDir), ...
            'Protocol',M.Protocol,'Tasks',M.Tasks,'Analysis',A.Analysis, ...
            'analysisFile',string(analysisFile));
    end

    problems = string(Stage(1).Protocol.problems(:));
    if numel(problems) ~= 6 || numel(unique(problems)) ~= 6
        error('CBSPairGuide:ScheduleProblemCount', ...
            'Ratio stage must contain exactly six unique problems.');
    end
    epochProblems = string(Stage(2).Protocol.problems(:));
    if numel(epochProblems) ~= 6 || ...
            ~isequal(sort(problems),sort(epochProblems))
        error('CBSPairGuide:ScheduleProblemMismatch', ...
            'Ratio and epoch stages must contain the same six problems.');
    end

    representativeFile = fullfile(campaignDir,'ratio','analysis', ...
        'representative_runs.mat');
    requireFile(representativeFile,'ratio representative-run analysis');
    R = load(representativeFile,'RepresentativeRuns');
    if ~isfield(R,'RepresentativeRuns') || ~istable(R.RepresentativeRuns)
        error('CBSPairGuide:BadRepresentativeRuns', ...
            '%s must contain table RepresentativeRuns.',representativeFile);
    end
    requireTableColumns(R.RepresentativeRuns,["problem","run"], ...
        representativeFile);
    selectedRuns = representativeRuns(R.RepresentativeRuns,problems);

    Bundles = repmat(struct('armName',"",'armValues',zeros(1,0), ...
        'Snapshots',{{}},'files',strings(1,0),'M',NaN,'N',NaN,'D',NaN), ...
        numel(stages),numel(problems));
    for s = 1 : numel(stages)
        P = Stage(s).Protocol;
        arms = double(P.arms(:)');
        if numel(arms) ~= 3 || numel(unique(arms)) ~= 3
            error('CBSPairGuide:ScheduleArmCount', ...
                '%s stage must contain exactly three unique arms.',stages(s));
        end
        for p = 1 : numel(problems)
            Bundles(s,p) = loadProblemBundle(Stage(s),problems(p), ...
                selectedRuns(p),arms);
        end
    end

    outputDir = fullfile(campaignDir,'figures','training_schedule');
    ensureFolder(outputDir);
    Figures = repmat(struct('stage',"",'problem',"",'run',0, ...
        'armName',"",'armValues',zeros(1,3), ...
        'targetFE',zeros(1,10),'actualFE',zeros(3,10), ...
        'viewLower',zeros(1,3),'viewUpper',zeros(1,3), ...
        'snapshotCount',0,'analysisFile',"",'file',""),12,1);
    row = 0;
    for p = 1 : numel(problems)
        View = commonView(Bundles(:,p));
        Problem = makeProblem(problems(p),Bundles(1,p));
        for s = 1 : numel(stages)
            B = Bundles(s,p);
            outputFile = fullfile(outputDir,sprintf( ...
                '%s_%s_run%02d_3x10.png',stages(s),problems(p), ...
                selectedRuns(p)));
            if ~validPNG(outputFile)
                if exist(outputFile,'file') == 2
                    error('CBSPairGuide:InvalidExistingScheduleFigure', ...
                        'Invalid figure exists and will not be overwritten: %s', ...
                        outputFile);
                end
                renderSheet(Problem,problems(p),stages(s), ...
                    B.armName,B.armValues,selectedRuns(p), ...
                    B.Snapshots,View,outputFile);
            end
            row = row+1;
            targetFE = double([B.Snapshots{1}.targetFE]);
            actualFE = zeros(3,10);
            for arm = 1 : 3
                actualFE(arm,:) = double([B.Snapshots{arm}.actualFE]);
            end
            paddedLower = nan(1,3);
            paddedUpper = nan(1,3);
            paddedLower(1:Problem.M) = View.lower;
            paddedUpper(1:Problem.M) = View.upper;
            Figures(row) = struct('stage',stages(s), ...
                'problem',problems(p),'run',selectedRuns(p), ...
                'armName',B.armName,'armValues',B.armValues, ...
                'targetFE',targetFE,'actualFE',actualFE, ...
                'viewLower',paddedLower,'viewUpper',paddedUpper, ...
                'snapshotCount',30, ...
                'analysisFile',Stage(s).analysisFile, ...
                'file',string(outputFile));
        end
    end
    if row ~= 12
        error('CBSPairGuide:ScheduleFigureCount', ...
            'Rendered %d comparison sheets; expected 12.',row);
    end

    armValueText = strings(row,1);
    for i = 1 : row
        armValueText(i) = strjoin(compose('%g',Figures(i).armValues),',');
    end
    Manifest = table([Figures.stage]',[Figures.problem]',[Figures.run]', ...
        [Figures.armName]',armValueText, ...
        [Figures.snapshotCount]',[Figures.file]', ...
        'VariableNames',{'stage','problem','run','armName','armValues', ...
        'snapshotCount','file'});
    Summary = struct('campaignName',campaignName,'sheetCount',row, ...
        'expectedSheetCount',12,'panelsPerSheet',30, ...
        'representativeRunsFile',string(representativeFile), ...
        'finishedAt',string(datetime('now')));
    manifestMAT = fullfile(outputDir,'figure_manifest.mat');
    manifestCSV = fullfile(outputDir,'figure_manifest.csv');
    save(manifestMAT,'Figures','Manifest','Summary');
    writetable(Manifest,manifestCSV);
end

function validateProtocolAndTasks(P,Tasks,file)
    requiredProtocol = {'armName','arms','problems','targetFE'};
    missing = requiredProtocol(~isfield(P,requiredProtocol));
    if ~isstruct(P) || ~isscalar(P) || ~isempty(missing)
        error('CBSPairGuide:BadScheduleProtocol', ...
            '%s Protocol is missing fields: %s',file,strjoin(missing,', '));
    end
    requiredTasks = {'problem','armValue','run','outputFile'};
    missing = requiredTasks(~isfield(Tasks,requiredTasks));
    if ~isstruct(Tasks) || isempty(Tasks) || ~isempty(missing)
        error('CBSPairGuide:BadScheduleTasks', ...
            '%s Tasks is missing fields: %s',file,strjoin(missing,', '));
    end
    targets = double(P.targetFE(:)');
    if numel(targets) ~= 10
        error('CBSPairGuide:BadScheduleTargets', ...
            '%s targetFE must contain exactly 10 entries.',file);
    end
    expected = [targets(1),20000:10000:100000];
    if ~isequal(targets,expected) || targets(1) >= 20000
        error('CBSPairGuide:BadScheduleTargets', ...
            '%s targetFE must be first-use target plus 20K:10K:100K.',file);
    end
end

function requireTableColumns(T,names,file)
    present = string(T.Properties.VariableNames);
    missing = names(~ismember(names,present));
    if ~isempty(missing)
        error('CBSPairGuide:BadRepresentativeRuns', ...
            '%s RepresentativeRuns is missing columns: %s', ...
            file,strjoin(missing,', '));
    end
end

function runs = representativeRuns(T,problems)
    tableProblems = string(T.problem);
    tableRuns = double(T.run);
    runs = zeros(numel(problems),1);
    for p = 1 : numel(problems)
        values = unique(tableRuns(tableProblems == problems(p)));
        if numel(values) ~= 1 || ~isfinite(values) || ...
                values < 1 || values ~= round(values)
            error('CBSPairGuide:BadRepresentativeRun', ...
                'RepresentativeRuns must select one integer run for %s.', ...
                problems(p));
        end
        runs(p) = values;
    end
end

function B = loadProblemBundle(Stage,problem,run,arms)
    Tasks = Stage.Tasks;
    taskProblems = string({Tasks.problem});
    taskRuns = double([Tasks.run]);
    taskArms = double([Tasks.armValue]);
    snapshots = cell(1,3);
    files = strings(1,3);
    dimensions = nan(3,3);
    for arm = 1 : 3
        tolerance = 64*eps(max(1,abs(arms(arm))));
        rows = find(taskProblems == problem & taskRuns == run & ...
            abs(taskArms-arms(arm)) <= tolerance);
        if numel(rows) ~= 1
            error('CBSPairGuide:ScheduleTaskIdentity', ...
                '%s needs exactly one %s=%g run %d task; found %d.', ...
                problem,Stage.Protocol.armName,arms(arm),run,numel(rows));
        end
        resultFile = resolveResultFile(char(Tasks(rows).outputFile), ...
            char(Stage.directory));
        Data = loadScheduleResult(resultFile,problem,run, ...
            Stage.Protocol.targetFE);
        snapshots{arm} = Data.Snapshots;
        files(arm) = string(resultFile);
        dimensions(arm,:) = [double(Data.Record.M), ...
            double(Data.Record.N),double(Data.Record.D)];
    end
    if any(any(dimensions ~= dimensions(1,:)))
        error('CBSPairGuide:ScheduleDimensionMismatch', ...
            '%s stage %s arms have inconsistent M/N/D.', ...
            problem,Stage.name);
    end
    B = struct('armName',string(Stage.Protocol.armName), ...
        'armValues',arms,'Snapshots',{snapshots},'files',files, ...
        'M',dimensions(1,1),'N',dimensions(1,2),'D',dimensions(1,3));
end

function Data = loadScheduleResult(file,problem,run,targetFE)
    requireFile(file,'training-schedule result');
    variables = string({whos('-file',file).name});
    required = ["Record","Trajectory","Snapshots","Audit"];
    missing = required(~ismember(required,variables));
    if ~isempty(missing)
        error('CBSPairGuide:BadScheduleResultVariables', ...
            '%s is missing variables: %s',file,strjoin(missing,', '));
    end
    Data = load(file,'Record','Trajectory','Snapshots','Audit');
    if ~isstruct(Data.Record) || ~isscalar(Data.Record) || ...
            ~all(isfield(Data.Record,{'M','N','D'})) || ...
            ~isstruct(Data.Audit) || isempty(Data.Trajectory)
        error('CBSPairGuide:BadScheduleResult', ...
            '%s has invalid Record, Trajectory, or Audit.',file);
    end
    if isfield(Data.Record,'status') && string(Data.Record.status) ~= "ok"
        error('CBSPairGuide:IncompleteScheduleResult', ...
            '%s Record.status is not ok.',file);
    end
    if isfield(Data.Record,'problem') && ...
            string(Data.Record.problem) ~= problem
        error('CBSPairGuide:ScheduleResultProblem', ...
            '%s Record.problem does not match %s.',file,problem);
    end
    if isfield(Data.Record,'run') && double(Data.Record.run) ~= run
        error('CBSPairGuide:ScheduleResultRun', ...
            '%s Record.run does not match run %d.',file,run);
    end
    validateSnapshots(Data.Snapshots,double(Data.Record.M),targetFE,file);
end

function validateSnapshots(S,M,targetFE,file)
    required = {'targetFE','actualFE','rawObjs','rawCons', ...
        'targetObjs','targetCons','centerObjs','centerCons', ...
        'guidedObjs','guidedCons','population1Objs','population1Cons', ...
        'population2Objs','population2Cons'};
    missing = required(~isfield(S,required));
    if ~isstruct(S) || numel(S) ~= 10 || ~isempty(missing)
        error('CBSPairGuide:BadScheduleSnapshots', ...
            '%s needs 10 Snapshots and is missing fields: %s', ...
            file,strjoin(missing,', '));
    end
    capturedTargets = double([S.targetFE]);
    if ~isequal(capturedTargets,double(targetFE(:)')) || ...
            any(~isfinite([S.actualFE])) || ...
            any(diff(double([S.actualFE])) < 0)
        error('CBSPairGuide:BadScheduleSnapshotFE', ...
            '%s Snapshot FE schedule is invalid.',file);
    end
    layers = {'raw','target','center','guided','population1','population2'};
    for snapshot = 1 : 10
        for layer = 1 : numel(layers)
            objs = S(snapshot).([layers{layer},'Objs']);
            cons = S(snapshot).([layers{layer},'Cons']);
            if ~isnumeric(objs) || size(objs,2) ~= M || ...
                    ~isnumeric(cons) || size(cons,1) ~= size(objs,1)
                error('CBSPairGuide:BadScheduleSnapshotLayer', ...
                    '%s Snapshot %d layer %s has invalid objs/cons shape.', ...
                    file,snapshot,layers{layer});
            end
        end
        if size(S(snapshot).rawObjs,1) ~= 500
            error('CBSPairGuide:BadScheduleRawCount', ...
                '%s Snapshot %d has %d raw points; expected 500.', ...
                file,snapshot,size(S(snapshot).rawObjs,1));
        end
    end
end

function file = resolveResultFile(file,stageDir)
    if exist(file,'file') == 2
        return;
    end
    candidate = fullfile(stageDir,file);
    if exist(candidate,'file') == 2
        file = candidate;
    end
end

function file = resolveAnalysisFile(stageDir)
    candidates = {fullfile(stageDir,'analysis','analysis.mat'), ...
        fullfile(stageDir,'analysis','stage_analysis.mat')};
    file = candidates{1};
    for i = 1 : numel(candidates)
        if exist(candidates{i},'file') == 2
            file = candidates{i};
            return;
        end
    end
end

function Problem = makeProblem(problem,B)
    if ~all(isfinite([B.M,B.N,B.D])) || ...
            any([B.M,B.N,B.D] < 1)
        error('CBSPairGuide:BadScheduleDimensions', ...
            '%s has invalid M/N/D metadata.',problem);
    end
    constructor = str2func(char(problem));
    Problem = constructor('N',B.N,'D',B.D, ...
        'maxFE',1e9,'maxRuntime',Inf);
    if Problem.M ~= B.M
        error('CBSPairGuide:ScheduleProblemDimension', ...
            '%s constructor M=%d but result M=%d.',problem,Problem.M,B.M);
    end
end

function View = commonView(Bundles)
    M = Bundles(1).M;
    if any([Bundles.M] ~= M)
        error('CBSPairGuide:CrossStageDimensionMismatch', ...
            'Ratio and epoch results disagree on objective dimension.');
    end
    Raw = zeros(0,M);
    Required = zeros(0,M);
    for stage = 1 : numel(Bundles)
        for arm = 1 : 3
            S = Bundles(stage).Snapshots{arm};
            for snapshot = 1 : numel(S)
                Raw = [Raw;double(S(snapshot).rawObjs)]; %#ok<AGROW>
                Required = [Required;double(S(snapshot).population1Objs); ...
                    double(S(snapshot).population2Objs); ...
                    double(S(snapshot).targetObjs); ...
                    double(S(snapshot).centerObjs); ...
                    double(S(snapshot).guidedObjs)]; %#ok<AGROW>
            end
        end
    end
    Raw = Raw(all(isfinite(Raw),2),:);
    Required = Required(all(isfinite(Required),2),:);
    if isempty(Raw) && isempty(Required)
        error('CBSPairGuide:EmptyScheduleObjectives', ...
            'No finite objective values are available for plotting.');
    end
    lower = zeros(1,M);
    upper = zeros(1,M);
    for objective = 1 : M
        raw = Raw(:,objective);
        required = Required(:,objective);
        if isempty(raw)
            lo = min(required);
            hi = max(required);
        else
            lo = empiricalQuantile(raw,0.01);
            hi = empiricalQuantile(raw,0.99);
            if ~isempty(required)
                lo = min(lo,min(required));
                hi = max(hi,max(required));
            end
        end
        span = hi-lo;
        if span <= eps(max(1,max(abs([lo,hi]))))
            span = max(1,max(abs([lo,hi])));
        end
        lower(objective) = lo-0.05*span;
        upper(objective) = hi+0.05*span;
    end
    View = struct('lower',lower,'upper',upper);
end

function value = empiricalQuantile(X,probability)
    X = sort(X(:));
    if isscalar(X)
        value = X;
        return;
    end
    position = 1+(numel(X)-1)*probability;
    lo = floor(position);
    hi = ceil(position);
    fraction = position-lo;
    value = X(lo)+fraction*(X(hi)-X(lo));
end

function renderSheet(Problem,problem,stage,armName,arms,run, ...
        Snapshots,View,outputFile)
    Figure = figure('Visible','off','Color','w', ...
        'Position',[20,50,3600,1200]);
    cleanup = onCleanup(@()close(Figure));
    Layout = tiledlayout(Figure,3,10,'TileSpacing','compact', ...
        'Padding','compact');
    colors = figureColors();
    legendHandles = gobjects(1,0);
    legendAxes = gobjects(1,1);
    for arm = 1 : 3
        armLabel = sprintf('%s=%g',armName,arms(arm));
        for snapshot = 1 : 10
            Ax = nexttile(Layout,(arm-1)*10+snapshot);
            hold(Ax,'on');
            [hFeasible,hInfeasible] = draw_CBS_CGAN_objective_region( ...
                Ax,Problem,problem,colors,View);
            simplifyBackground(Ax,Problem.M);
            H = drawLayers(Ax,Problem.M,Snapshots{arm}(snapshot),colors);
            setView(Ax,View,Problem.M);
            styleAxes(Ax,Problem.M,arm,snapshot);
            if snapshot == 1
                timeLabel = 'first use';
            else
                timeLabel = sprintf('%gK target', ...
                    Snapshots{arm}(snapshot).targetFE/1000);
            end
            title(Ax,{armLabel,sprintf('%s | actual FE %s', ...
                timeLabel,commaNumber(Snapshots{arm}(snapshot).actualFE))}, ...
                'Interpreter','none','FontName','Helvetica', ...
                'FontSize',6,'FontWeight','normal');
            if arm == 1 && snapshot == 1
                legendAxes = Ax;
                legendHandles = [hFeasible,hInfeasible,H];
            end
        end
    end
    title(Layout,sprintf( ...
        '%s | %s stage | representative run %d | shared view across both stages', ...
        problem,stage,run),'Interpreter','none','FontName','Helvetica', ...
        'FontWeight','bold','FontSize',13);
    labels = {'feasible region','infeasible region','P1','P2', ...
        'raw 500','selected target','selected center','guided child'};
    Legend = legend(legendAxes,legendHandles,labels, ...
        'Orientation','horizontal','NumColumns',8,'FontSize',6, ...
        'Box','off','Interpreter','none');
    try
        Legend.Layout.Tile = 'south';
    catch
        Legend.Location = 'southoutside';
    end

    partialFile = [outputFile,'.partial.png'];
    exportgraphics(Figure,partialFile,'Resolution',96, ...
        'BackgroundColor','white');
    if ~validPNG(partialFile)
        error('CBSPairGuide:InvalidSchedulePNG', ...
            'Rendered PNG failed validation: %s',partialFile);
    end
    [moved,message] = movefile(partialFile,outputFile);
    if ~moved
        error('CBSPairGuide:SchedulePNGMoveFailed','%s',message);
    end
    clear cleanup;
end

function H = drawLayers(Ax,M,S,C)
    if M == 2
        hRaw = scatter(Ax,S.rawObjs(:,1),S.rawObjs(:,2),3,C.raw, ...
            'filled','o','MarkerFaceAlpha',0.33,'MarkerEdgeColor','none');
        hP1 = scatter(Ax,S.population1Objs(:,1), ...
            S.population1Objs(:,2),7,C.p1,'filled','o', ...
            'MarkerFaceAlpha',0.72,'MarkerEdgeColor','none');
        hP2 = scatter(Ax,S.population2Objs(:,1), ...
            S.population2Objs(:,2),8,C.p2,'filled','^', ...
            'MarkerFaceAlpha',0.72,'MarkerEdgeColor','none');
        hTarget = scatter(Ax,S.targetObjs(:,1),S.targetObjs(:,2), ...
            17,C.target,'filled','v','MarkerEdgeColor',[0.1,0.1,0.1], ...
            'LineWidth',0.25);
        hCenter = plot(Ax,S.centerObjs(:,1),S.centerObjs(:,2),'x', ...
            'Color',C.center,'MarkerSize',3.5,'LineWidth',0.75);
        hGuided = scatter(Ax,S.guidedObjs(:,1),S.guidedObjs(:,2), ...
            19,C.guided,'filled','d','MarkerEdgeColor',[0.1,0.1,0.1], ...
            'LineWidth',0.3);
    else
        hRaw = scatter3(Ax,S.rawObjs(:,1),S.rawObjs(:,2), ...
            S.rawObjs(:,3),3,C.raw,'filled','o', ...
            'MarkerFaceAlpha',0.33,'MarkerEdgeColor','none');
        hP1 = scatter3(Ax,S.population1Objs(:,1), ...
            S.population1Objs(:,2),S.population1Objs(:,3), ...
            7,C.p1,'filled','o','MarkerFaceAlpha',0.72, ...
            'MarkerEdgeColor','none');
        hP2 = scatter3(Ax,S.population2Objs(:,1), ...
            S.population2Objs(:,2),S.population2Objs(:,3), ...
            8,C.p2,'filled','^','MarkerFaceAlpha',0.72, ...
            'MarkerEdgeColor','none');
        hTarget = scatter3(Ax,S.targetObjs(:,1),S.targetObjs(:,2), ...
            S.targetObjs(:,3),17,C.target,'filled','v', ...
            'MarkerEdgeColor',[0.1,0.1,0.1],'LineWidth',0.25);
        hCenter = plot3(Ax,S.centerObjs(:,1),S.centerObjs(:,2), ...
            S.centerObjs(:,3),'x','Color',C.center,'MarkerSize',3.5, ...
            'LineWidth',0.75);
        hGuided = scatter3(Ax,S.guidedObjs(:,1),S.guidedObjs(:,2), ...
            S.guidedObjs(:,3),19,C.guided,'filled','d', ...
            'MarkerEdgeColor',[0.1,0.1,0.1],'LineWidth',0.3);
    end
    H = [hP1,hP2,hRaw,hTarget,hCenter,hGuided];
end

function simplifyBackground(Ax,M)
%SIMPLIFYBACKGROUND Match background detail to small comparison panels.

    if M == 2
        Images = findobj(Ax,'Type','image');
        for i = 1 : numel(Images)
            data = Images(i).CData;
            if size(data,1) > 120 || size(data,2) > 120
                rows = unique(round(linspace(1,size(data,1),120)));
                columns = unique(round(linspace(1,size(data,2),120)));
                Images(i).CData = data(rows,columns,:);
                alpha = Images(i).AlphaData;
                if ~isscalar(alpha)
                    Images(i).AlphaData = alpha(rows,columns);
                end
            end
        end
    end
end

function setView(Ax,View,M)
    xlim(Ax,[View.lower(1),View.upper(1)]);
    ylim(Ax,[View.lower(2),View.upper(2)]);
    if M == 3
        zlim(Ax,[View.lower(3),View.upper(3)]);
        axis(Ax,'vis3d');
        view(Ax,42,25);
    end
end

function styleAxes(Ax,M,arm,snapshot)
    grid(Ax,'on');
    box(Ax,'on');
    Ax.FontName = 'Helvetica';
    Ax.FontSize = 5;
    Ax.LineWidth = 0.45;
    if arm == 3
        xlabel(Ax,'f_1','Interpreter','tex','FontSize',6);
    end
    if snapshot == 1
        ylabel(Ax,'f_2','Interpreter','tex','FontSize',6);
        if M == 3
            zlabel(Ax,'f_3','Interpreter','tex','FontSize',6);
        end
    end
end

function C = figureColors()
    C = struct('feasible',[0.36,0.72,0.47], ...
        'infeasible',[0.90,0.43,0.40], ...
        'p1',[0.48,0.20,0.72],'p2',[0.00,0.58,0.62], ...
        'raw',[0.13,0.43,0.82],'target',[0.86,0.20,0.58], ...
        'center',[0.08,0.08,0.08],'guided',[1.00,0.66,0.12]);
end

function textValue = commaNumber(value)
    textValue = regexprep(sprintf('%.0f',value), ...
        '(?<!\d)(\d{1,3})(?=(\d{3})+(?!\d))','$1,');
end

function requireFile(file,description)
    if exist(file,'file') ~= 2
        error('CBSPairGuide:MissingScheduleFile', ...
            'Missing %s: %s',description,file);
    end
end

function valid = validPNG(file)
    valid = false;
    if exist(file,'file') ~= 2
        return;
    end
    try
        Info = imfinfo(file);
        valid = strcmpi(Info.Format,'png') && ...
            Info.Width >= 3200 && Info.Height >= 1000;
    catch
        valid = false;
    end
end

function ensureFolder(folder)
    if ~isfolder(folder), mkdir(folder); end
end
