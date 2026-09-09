function Summary = analyze_PairGuide_initial_validation(InputDir,OutputDir)
%ANALYZE_PAIRGUIDE_INITIAL_VALIDATION Export focused tables and run-1 plots.
%   SUMMARY = ANALYZE_PAIRGUIDE_INITIAL_VALIDATION(INPUTDIR,OUTPUTDIR)
%   reads task MAT files containing Record, Audit, and optional Snapshots.
%   Missing optional diagnostics produce NaN or empty, header-only tables.

    if nargin < 1 || isempty(InputDir)
        error('PairGuide:MissingValidationInput', ...
            'InputDir must contain the per-task MAT files.');
    end
    InputDir = char(InputDir);
    if nargin < 2 || isempty(OutputDir)
        OutputDir = fullfile(InputDir,'analysis');
    end
    OutputDir = char(OutputDir);
    if ~isfolder(InputDir)
        error('PairGuide:ValidationInputNotFound', ...
            'Input directory does not exist: %s',InputDir);
    end
    ensureFolder(OutputDir);

    Files = taskMATFiles(InputDir);
    if isempty(Files)
        error('PairGuide:NoValidationResults', ...
            'No MAT file containing Record was found under %s.',InputDir);
    end

    runParts = cell(numel(Files),1);
    trainingParts = cell(numel(Files),1);
    generationParts = cell(numel(Files),1);
    useParts = cell(numel(Files),1);
    checkpointParts = cell(numel(Files),1);
    figureCount = 0;
    for fileIndex = 1 : numel(Files)
        file = fullfile(Files(fileIndex).folder,Files(fileIndex).name);
        Data = loadTaskData(file);
        Record = Data.Record;
        Audit = optionalStruct(Data,'Audit');
        [problem,run,seed] = taskIdentity(Record,file);

        Trajectory = optionalStruct(Data,'Trajectory');
        FinalMetrics = optionalStruct(Data,'FinalMetrics');
        runParts{fileIndex} = runSummaryRow( ...
            Record,Audit,Trajectory,FinalMetrics,file,problem,run,seed);
        trainingParts{fileIndex} = trainingRows( ...
            Audit,problem,run,seed);
        generationParts{fileIndex} = generationRows( ...
            Audit,problem,run,seed);
        useParts{fileIndex} = useRows(Audit,problem,run,seed);
        checkpointParts{fileIndex} = checkpointRows( ...
            Audit,Trajectory,problem,run,seed);

        if run == 1 && isfield(Data,'Snapshots') && ...
                isstruct(Data.Snapshots) && ~isempty(Data.Snapshots)
            figureCount = figureCount+renderRunOneSnapshots( ...
                Data.Snapshots,Audit,problem,run,OutputDir);
        end
    end

    RunSummary = rowsToTable(runParts,emptyRunSummary());
    assertUniqueRuns(RunSummary);
    RunSummary = sortrows(RunSummary,{'problem','run'});
    TrainingEvents = rowsToTable(trainingParts,emptyTrainingRow());
    GenerationEvents = rowsToTable(generationParts,emptyGenerationRow());
    UseEvents = rowsToTable(useParts,emptyUseRow());
    Checkpoints = rowsToTable(checkpointParts,emptyCheckpointRow());
    if ~isempty(TrainingEvents)
        TrainingEvents = sortrows(TrainingEvents,{'problem','run','fe'});
    end
    if ~isempty(GenerationEvents)
        GenerationEvents = sortrows(GenerationEvents, ...
            {'problem','run','poolFE'});
    end
    if ~isempty(UseEvents)
        UseEvents = sortrows(UseEvents,{'problem','run','useFE'});
    end
    if ~isempty(Checkpoints)
        Checkpoints = sortrows(Checkpoints,{'problem','run','targetFE'});
    end

    writeTableAtomic(RunSummary,fullfile(OutputDir,'run_summary.csv'));
    writeTableAtomic(TrainingEvents, ...
        fullfile(OutputDir,'training_events.csv'));
    writeTableAtomic(GenerationEvents, ...
        fullfile(OutputDir,'generation_events.csv'));
    writeTableAtomic(UseEvents,fullfile(OutputDir,'use_events.csv'));
    writeTableAtomic(Checkpoints, ...
        fullfile(OutputDir,'checkpoint_metrics.csv'));

    Summary = struct('inputDir',string(InputDir), ...
        'outputDir',string(OutputDir),'taskCount',height(RunSummary), ...
        'trainingEventCount',height(TrainingEvents), ...
        'generationEventCount',height(GenerationEvents), ...
        'useEventCount',height(UseEvents), ...
        'checkpointCount',height(Checkpoints), ...
        'figureCount',figureCount, ...
        'finishedAt',string(datetime('now')));
end

function Files = taskMATFiles(InputDir)
    Candidates = dir(fullfile(InputDir,'**','*.mat'));
    keep = false(numel(Candidates),1);
    for i = 1 : numel(Candidates)
        file = fullfile(Candidates(i).folder,Candidates(i).name);
        if endsWith(file,'.partial.mat')
            finalFile = [erase(file,'.partial.mat'),'.mat'];
            if isfile(finalFile)
                continue;
            end
        end
        try
            variables = string({whos('-file',file).name});
            keep(i) = ismember("Record",variables);
        catch
            keep(i) = false;
        end
    end
    Files = Candidates(keep);
    if isempty(Files)
        return;
    end
    names = string(fullfile({Files.folder},{Files.name}));
    [~,order] = sort(names);
    Files = Files(order);
end

function Data = loadTaskData(file)
    variables = string({whos('-file',file).name});
    wanted = intersect(["Record","Audit","Trajectory","Snapshots", ...
        "FinalMetrics"], ...
        variables,'stable');
    names = cellstr(wanted);
    Data = load(file,names{:});
    if ~isfield(Data,'Record') || ~isstruct(Data.Record) || ...
            ~isscalar(Data.Record)
        error('PairGuide:BadValidationRecord', ...
            'Record must be a scalar struct: %s',file);
    end
end

function S = optionalStruct(Data,name)
    if isfield(Data,name) && isstruct(Data.(name)) && ...
            isscalar(Data.(name))
        S = Data.(name);
    else
        S = struct();
    end
end

function [problem,run,seed] = taskIdentity(Record,file)
    problem = textField(Record,"problem","");
    run = numberField(Record,"run",NaN);
    seed = numberField(Record,"seed",run);
    if strlength(problem) == 0 || ~isfinite(run)
        error('PairGuide:MissingValidationIdentity', ...
            'Record lacks problem or run: %s',file);
    end
end

function Row = runSummaryRow(Record,Audit,Trajectory,FinalMetrics, ...
        file,problem,run,seed)
    Row = emptyRunSummary();
    Row.sourceFile = string(file);
    Row.problem = problem;
    Row.run = run;
    Row.seed = seed;
    Row.status = textField(Record,"status","ok");
    Row.finalFE = numberField(Record,"finalFE", ...
        lastFinite(vectorField(Audit,"checkpointFE")));
    Row.wallClockSeconds = numberField(Record,"wallClockSeconds",NaN);
    Row.firstLegalPairFE = numberField(Audit,"firstLegalPairFE",NaN);
    Row.firstEligibleTrainingFE = firstEligibleFE(Audit);
    Row.firstTrainingFE = numberField(Audit, ...
        ["firstPairTrainingFE","firstTrainingFE"],NaN);
    Row.firstGuidedUseFE = numberField(Audit,"firstGuidedUseFE",NaN);
    Archive = latestLogEvent(Audit,"pairArchiveLog",Inf);
    Row.activePairs = numberField(Audit,"pairActiveLast", ...
        numberField(Archive,"active",NaN));
    Row.activeReferences = numberField(Audit,"pairRegionsLast", ...
        Row.activePairs);
    Row.gapMedian = numberField(Audit,"pairGapMedianLast", ...
        numberField(Archive,"pairGapMedian",NaN));
    Row.gapP90 = numberField(Audit,"pairGapP90Last", ...
        numberField(Archive,"pairGapP90",NaN));
    Row.pairsAdded = numberField(Audit,"pairArchiveAdded",NaN);
    Row.tightenedFeasible = numberField( ...
        Audit,"pairArchiveTightenedFeasible",NaN);
    Row.tightenedInfeasible = numberField( ...
        Audit,"pairArchiveTightenedInfeasible",NaN);
    Row.guidedTightened = sumKnown([ ...
        numberField(Audit,"pairGuidedTightenedFeasible",NaN), ...
        numberField(Audit,"pairGuidedTightenedInfeasible",NaN)]);
    Row.trainingEvents = numberField(Audit,"pairTrainingEvents", ...
        logLength(Audit,"pairTrainingLog"));
    Row.useEvents = numberField(Audit,"useEvents", ...
        logLength(Audit,"pairUseLog"));
    Row.guidedSelected = numberField(Audit,"guidedSelected",NaN);
    Row.guidedFallback = numberField(Audit,"guidedFallback",NaN);
    Row.guidedFeasibleRate = numberField( ...
        Audit,"guidedFeasibleRate",NaN);
    Row.guidedUsefulRate = numberField(Audit,"childUsefulRate",NaN);
    Row.guidedSurvivalRate = numberField( ...
        Audit,"guidedSurvivalRate",NaN);
    Row.objectiveOnlyFE = numberField( ...
        Audit,["ObjFE","pairObjectiveFE"],NaN);
    Row.guidedFullFE = numberField( ...
        Audit,["PairGuideFullFE","pairGuideFullFE"],NaN);
    Row.finalIGD = numberField(FinalMetrics,"IGD", ...
        lastFinite(vectorField(Trajectory,"IGD")));
    Row.finalHV = numberField(FinalMetrics,"HV", ...
        lastFinite(vectorField(Trajectory,"HV")));
    Row.finalP1Feasible = numberField(FinalMetrics,"feasibleCount", ...
        lastFinite(vectorField(Trajectory,"feasibleCount")));
    Row.finalP1NondominatedFeasible = numberField(FinalMetrics, ...
        "nondominatedFeasibleCount",lastFinite( ...
        vectorField(Trajectory,"nondominatedFeasibleCount")));
end

function Rows = trainingRows(Audit,problem,run,seed)
    Log = structLog(Audit,"pairTrainingLog");
    Rows = repmat(emptyTrainingRow(),numel(Log),1);
    fields = ["fe","nCritic","epochs","trainingPairs", ...
        "changedPairs","newRegions","updates","pairVisits", ...
        "batchesPerEpoch","trainingSeconds", ...
        "preAllEndpointRMSE","postAllEndpointRMSE", ...
        "preChangedEndpointRMSE","postChangedEndpointRMSE", ...
        "prePairDifferenceRMSE","postPairDifferenceRMSE", ...
        "preSameConditionThickness","postSameConditionThickness"];
    for i = 1 : numel(Log)
        Rows(i).problem = problem;
        Rows(i).run = run;
        Rows(i).seed = seed;
        Rows(i).kind = textField(Log(i),["kind","trainingKind"],"");
        for field = fields
            Rows(i).(field) = numberField(Log(i),field,NaN);
        end
    end
end

function Rows = generationRows(Audit,problem,run,seed)
    Log = structLog(Audit,"pairGenerationLog");
    Rows = repmat(emptyGenerationRow(),numel(Log),1);
    fields = ["modelVersion","rawCount","invalidCount","matchFailures", ...
        "sphereRejectCount","spherePassCount", ...
        "objectiveCandidateCount", ...
        "jointPassCount","keptCount","objectiveFE"];
    for i = 1 : numel(Log)
        Rows(i).problem = problem;
        Rows(i).run = run;
        Rows(i).seed = seed;
        Rows(i).poolFE = numberField(Log(i),["poolFE","fe"],NaN);
        for field = fields
            Rows(i).(field) = numberField(Log(i),field,NaN);
        end
        Rows(i).spherePassRate = safeRatio( ...
            Rows(i).spherePassCount,Rows(i).rawCount);
    end
end

function Rows = useRows(Audit,problem,run,seed)
    Log = structLog(Audit,"pairUseLog");
    Rows = repmat(emptyUseRow(),numel(Log),1);
    for i = 1 : numel(Log)
        Rows(i).problem = problem;
        Rows(i).run = run;
        Rows(i).seed = seed;
        Rows(i).modelVersion = numberField(Log(i),"modelVersion",NaN);
        Rows(i).poolFE = numberField(Log(i),"poolFE",NaN);
        Rows(i).useFE = numberField(Log(i),["useFE","fe"],NaN);
        Rows(i).requested = numberField(Log(i),"requested",NaN);
        Rows(i).selected = numberField(Log(i),"selected",NaN);
        Rows(i).fallback = numberField(Log(i),"fallback",NaN);
        Rows(i).feasible = numberField( ...
            Log(i),["feasible","feasibleChildren"],NaN);
        Rows(i).useful = numberField( ...
            Log(i),["useful","feasibleDominating","childUseful"],NaN);
        Rows(i).survivedP1 = numberField( ...
            Log(i),["survivedP1","survived"],NaN);
        Rows(i).selectionRate = safeRatio( ...
            Rows(i).selected,Rows(i).requested);
        Rows(i).feasibleRate = safeRatio( ...
            Rows(i).feasible,Rows(i).selected);
        Rows(i).usefulPerQuota = safeRatio( ...
            Rows(i).useful,Rows(i).requested);
        Rows(i).survivalRate = safeRatio( ...
            Rows(i).survivedP1,Rows(i).selected);
    end
end

function Rows = checkpointRows(Audit,Trajectory,problem,run,seed)
    targets = vectorField(Trajectory,"targetFE");
    fromTrajectory = ~isempty(targets);
    if ~fromTrajectory
        targets = vectorField(Audit,"checkpointTargets");
    end
    if isempty(targets)
        Rows = repmat(emptyCheckpointRow(),0,1);
        return;
    end
    wanted = targets >= 1e4 & targets <= 1e5 & ...
        abs(targets/1e4-round(targets/1e4)) <= 1e-12;
    indices = find(wanted);
    Rows = repmat(emptyCheckpointRow(),numel(indices),1);
    for row = 1 : numel(indices)
        i = indices(row);
        Rows(row).problem = problem;
        Rows(row).run = run;
        Rows(row).seed = seed;
        Rows(row).targetFE = targets(i);
        if fromTrajectory
            Rows(row).actualFE = vectorValue(Trajectory,"actualFE",i);
            Rows(row).IGD = vectorValue(Trajectory,"IGD",i);
            Rows(row).HV = vectorValue(Trajectory,"HV",i);
            Rows(row).p1Feasible = vectorValue( ...
                Trajectory,"feasibleCount",i);
            Rows(row).p1NondominatedFeasible = vectorValue( ...
                Trajectory,"nondominatedFeasibleCount",i);
        else
            Rows(row).actualFE = vectorValue(Audit,"checkpointFE",i);
            Rows(row).IGD = vectorValue(Audit,"checkpointIGD",i);
            Rows(row).HV = vectorValue(Audit,"checkpointHV",i);
            Rows(row).p1Feasible = vectorValue( ...
                Audit,"checkpointFeasibleCount",i);
            Rows(row).p1NondominatedFeasible = vectorValue( ...
                Audit,"checkpointNondominatedFeasibleCount",i);
        end
        Rows(row).trainingEvents = vectorValue( ...
            Audit,"checkpointTrainingEvents",i);
        Rows(row).guidedSelected = vectorValue( ...
            Audit,"checkpointGuidedSelected",i);
        Rows(row).guidedFeasible = vectorValue( ...
            Audit,"checkpointGuidedFeasible",i);
        Rows(row).guidedSurvived = vectorValue( ...
            Audit,"checkpointGuidedSurvived",i);
        Archive = latestLogEvent( ...
            Audit,"pairArchiveLog",Rows(row).actualFE);
        Rows(row).activePairs = numberField(Archive,"active",NaN);
        Rows(row).activeReferences = numberField( ...
            Archive,["regions","activeReferences"], ...
            Rows(row).activePairs);
        Rows(row).gapMedian = numberField( ...
            Archive,"pairGapMedian",NaN);
        Rows(row).gapP90 = numberField(Archive,"pairGapP90",NaN);
    end
end

function count = renderRunOneSnapshots(Snapshots,Audit,problem,run,OutputDir)
    targets = arrayField(Snapshots,"targetFE",NaN);
    actual = arrayField(Snapshots,["actualFE","useFE"],NaN);
    wanted = targets == 1 | (targets >= 1e4 & targets <= 1e5 & ...
        abs(targets/1e4-round(targets/1e4)) <= 1e-12);
    indices = find(wanted & isfinite(actual));
    if isempty(indices)
        count = 0;
        return;
    end
    [~,firstRows] = unique(actual(indices),'stable');
    indices = indices(firstRows);
    constructor = str2func(char(problem));
    Problem = constructor();
    if double(Problem.M) ~= 2
        warning('PairGuide:SkippedValidationFigure', ...
            'Only 2-objective validation plots are supported: %s.',problem);
        count = 0;
        return;
    end
    figureDir = fullfile(OutputDir,'figures',char(problem), ...
        sprintf('run_%02d',run));
    ensureFolder(figureDir);
    count = 0;
    for index = reshape(indices,1,[])
        S = Snapshots(index);
        if targets(index) == 1
            label = 'first_use';
        else
            label = sprintf('FE%06d',round(targets(index)));
        end
        outputFile = fullfile(figureDir,sprintf( ...
            '%s_%s_actualFE%06d.png',problem,label,round(actual(index))));
        renderTwoPanelFigure(Problem,problem,S,Audit, ...
            targets(index),actual(index),outputFile);
        count = count+1;
    end
end

function renderTwoPanelFigure(Problem,problem,S,Audit,targetFE,actualFE,file)
    P1 = matrixField(S,["population1Objs","constrainedPopulationObjs"]);
    P2 = matrixField(S,["population2Objs","unconstrainedPopulationObjs"]);
    Xf = matrixField(S,["archiveFeasibleObjs","pairFeasibleObjs"]);
    Xi = matrixField(S,["archiveInfeasibleObjs","pairInfeasibleObjs"]);
    Raw = matrixField(S,"rawObjs");
    Donor = matrixField(S,["targetObjs","donorObjs"]);
    Midpoint = matrixField(S,["centerObjs","midpointObjs","guidedObjs"]);
    View = objectiveView({P1,P2,Xf,Xi,Raw,Donor,Midpoint});
    C = figureColors();
    Figure = figure('Visible','off','Color','w', ...
        'Position',[100,100,1500,650]);
    cleanup = onCleanup(@()close(Figure));
    Layout = tiledlayout(Figure,1,2,'TileSpacing','compact', ...
        'Padding','compact');

    Search = nexttile(Layout,1);
    hold(Search,'on');
    [hFeasible,hInfeasible] = draw_CBS_CGAN_objective_region( ...
        Search,Problem,problem,C,View);
    pairCount = min(size(Xf,1),size(Xi,1));
    for pair = 1 : pairCount
        plot(Search,[Xf(pair,1),Xi(pair,1)], ...
            [Xf(pair,2),Xi(pair,2)],'-','Color',C.pair, ...
            'LineWidth',0.65);
    end
    hP1 = scatter2(Search,P1,23,C.p1,'o',0.72);
    hP2 = scatter2(Search,P2,24,C.p2,'^',0.72);
    hXf = scatter2(Search,Xf,32,C.xf,'o',0.95);
    hXi = scatter2(Search,Xi,32,C.xi,'s',0.95);
    applyView(Search,View);
    styleAxes(Search);
    title(Search,'Search state');
    legend(Search,[hFeasible,hInfeasible,hP1,hP2,hXf,hXi], ...
        {'feasible domain','infeasible domain','P1 constrained', ...
         'P2 unconstrained','active x_f','active x_i'}, ...
        'Location','best','Box','off','FontSize',8);

    Guide = nexttile(Layout,2);
    hold(Guide,'on');
    draw_CBS_CGAN_objective_region(Guide,Problem,problem,C,View);
    hRaw = scatter2(Guide,Raw,13,C.raw,'o',0.42);
    hDonor = scatter2(Guide,Donor,46,C.donor,'v',0.95);
    hMidpoint = scatter2(Guide,Midpoint,54,C.midpoint,'d',0.95);
    applyView(Guide,View);
    styleAxes(Guide);
    title(Guide,'Generated guidance');
    legend(Guide,[hRaw,hDonor,hMidpoint], ...
        {sprintf('raw (%d)',size(Raw,1)), ...
         sprintf('donor (%d)',size(Donor,1)), ...
         sprintf('midpoint child (%d)',size(Midpoint,1))}, ...
        'Location','best','Box','off','FontSize',8);

    Archive = latestLogEvent(Audit,"pairArchiveLog",actualFE);
    active = numberField(Archive,"active",pairCount);
    gapMedian = numberField(Archive,"pairGapMedian",NaN);
    if targetFE == 1
        targetText = 'first use';
    else
        targetText = sprintf('target FE %s',commaNumber(targetFE));
    end
    title(Layout,sprintf( ...
        '%s | run 1 | %s | actual FE %s | active pairs %.0f | gap median %.4g', ...
        problem,targetText,commaNumber(actualFE),active,gapMedian), ...
        'Interpreter','none','FontWeight','bold');
    exportPNGAtomic(Figure,file);
    clear cleanup;
end

function H = scatter2(Ax,X,sizeValue,color,marker,alpha)
    X = firstTwoFinite(X);
    H = scatter(Ax,X(:,1),X(:,2),sizeValue,color,'filled',marker, ...
        'MarkerFaceAlpha',alpha,'MarkerEdgeColor',[0.12,0.12,0.12], ...
        'LineWidth',0.35);
end

function View = objectiveView(Parts)
    Values = zeros(0,2);
    for i = 1 : numel(Parts)
        Values = [Values;firstTwoFinite(Parts{i})]; %#ok<AGROW>
    end
    if isempty(Values)
        lower = [0,0];
        upper = [1,1];
    else
        lower = min(Values,[],1);
        upper = max(Values,[],1);
        span = upper-lower;
        scale = max(1,max(abs([lower;upper]),[],1));
        span(span <= eps(scale)) = scale(span <= eps(scale));
        lower = lower-0.08*span;
        upper = upper+0.08*span;
    end
    View = struct('lower',lower,'upper',upper);
end

function X = firstTwoFinite(X)
    if ~isnumeric(X) || size(X,2) < 2
        X = zeros(0,2);
        return;
    end
    X = double(X(:,1:2));
    X = X(all(isfinite(X),2),:);
end

function applyView(Ax,View)
    xlim(Ax,[View.lower(1),View.upper(1)]);
    ylim(Ax,[View.lower(2),View.upper(2)]);
end

function styleAxes(Ax)
    xlabel(Ax,'f_1','Interpreter','tex');
    ylabel(Ax,'f_2','Interpreter','tex');
    grid(Ax,'on');
    box(Ax,'on');
    Ax.FontName = 'Helvetica';
    Ax.FontSize = 10;
    Ax.LineWidth = 0.8;
end

function C = figureColors()
    C = struct('feasible',[0.36,0.72,0.47], ...
        'infeasible',[0.90,0.43,0.40], ...
        'p1',[0.48,0.20,0.72],'p2',[0.00,0.58,0.62], ...
        'xf',[0.05,0.55,0.16],'xi',[0.78,0.13,0.14], ...
        'pair',[0.20,0.20,0.20], ...
        'raw',[0.13,0.43,0.82],'donor',[0.86,0.20,0.58], ...
        'midpoint',[1.00,0.66,0.12]);
end

function Row = emptyRunSummary()
    Row = struct('sourceFile',"",'problem',"",'run',NaN,'seed',NaN, ...
        'status',"",'finalFE',NaN,'wallClockSeconds',NaN, ...
        'firstLegalPairFE',NaN,'firstEligibleTrainingFE',NaN, ...
        'firstTrainingFE',NaN,'firstGuidedUseFE',NaN, ...
        'activePairs',NaN,'activeReferences',NaN, ...
        'gapMedian',NaN,'gapP90',NaN,'pairsAdded',NaN, ...
        'tightenedFeasible',NaN,'tightenedInfeasible',NaN, ...
        'guidedTightened',NaN,'trainingEvents',NaN,'useEvents',NaN, ...
        'guidedSelected',NaN,'guidedFallback',NaN, ...
        'guidedFeasibleRate',NaN,'guidedUsefulRate',NaN, ...
        'guidedSurvivalRate',NaN,'objectiveOnlyFE',NaN, ...
        'guidedFullFE',NaN,'finalIGD',NaN,'finalHV',NaN, ...
        'finalP1Feasible',NaN,'finalP1NondominatedFeasible',NaN);
end

function Row = emptyTrainingRow()
    Row = struct('problem',"",'run',NaN,'seed',NaN,'fe',NaN, ...
        'kind',"",'nCritic',NaN,'epochs',NaN,'trainingPairs',NaN, ...
        'changedPairs',NaN,'newRegions',NaN,'updates',NaN, ...
        'pairVisits',NaN,'batchesPerEpoch',NaN,'trainingSeconds',NaN, ...
        'preAllEndpointRMSE',NaN,'postAllEndpointRMSE',NaN, ...
        'preChangedEndpointRMSE',NaN,'postChangedEndpointRMSE',NaN, ...
        'prePairDifferenceRMSE',NaN,'postPairDifferenceRMSE',NaN, ...
        'preSameConditionThickness',NaN, ...
        'postSameConditionThickness',NaN);
end

function Row = emptyGenerationRow()
    Row = struct('problem',"",'run',NaN,'seed',NaN,'poolFE',NaN, ...
        'modelVersion',NaN,'rawCount',NaN,'invalidCount',NaN, ...
        'matchFailures',NaN,'sphereRejectCount',NaN, ...
        'spherePassCount',NaN,'spherePassRate',NaN, ...
        'objectiveCandidateCount',NaN,'jointPassCount',NaN, ...
        'keptCount',NaN,'objectiveFE',NaN);
end

function Row = emptyUseRow()
    Row = struct('problem',"",'run',NaN,'seed',NaN, ...
        'modelVersion',NaN,'poolFE',NaN,'useFE',NaN, ...
        'requested',NaN,'selected',NaN,'fallback',NaN, ...
        'feasible',NaN,'useful',NaN,'survivedP1',NaN, ...
        'selectionRate',NaN, ...
        'feasibleRate',NaN,'usefulPerQuota',NaN,'survivalRate',NaN);
end

function Row = emptyCheckpointRow()
    Row = struct('problem',"",'run',NaN,'seed',NaN, ...
        'targetFE',NaN,'actualFE',NaN,'IGD',NaN,'HV',NaN, ...
        'p1Feasible',NaN,'p1NondominatedFeasible',NaN, ...
        'activePairs',NaN,'activeReferences',NaN, ...
        'gapMedian',NaN,'gapP90',NaN,'trainingEvents',NaN, ...
        'guidedSelected',NaN,'guidedFeasible',NaN, ...
        'guidedSurvived',NaN);
end

function T = rowsToTable(Parts,Prototype)
    keep = ~cellfun(@isempty,Parts);
    if any(keep)
        Rows = vertcat(Parts{keep});
    else
        Rows = repmat(Prototype,0,1);
    end
    T = struct2table(Rows);
end

function assertUniqueRuns(T)
    keys = T.problem+"#"+compose('%.0f',T.run);
    if numel(unique(keys)) ~= height(T)
        error('PairGuide:DuplicateValidationRun', ...
            'Input contains duplicate problem/run records.');
    end
end

function Log = structLog(S,name)
    if isfield(S,name) && isstruct(S.(name))
        Log = S.(name)(:);
    else
        Log = struct([]);
    end
end

function count = logLength(S,name)
    count = numel(structLog(S,name));
end

function Event = latestLogEvent(Audit,name,fe)
    Log = structLog(Audit,name);
    Event = struct();
    if isempty(Log)
        return;
    end
    eventFE = arrayField(Log,["fe","actualFE","useFE"],NaN);
    rows = find(isfinite(eventFE) & eventFE <= fe);
    if ~isempty(rows)
        Event = Log(rows(end));
    end
end

function fe = firstEligibleFE(Audit)
    Log = structLog(Audit,"pairArchiveLog");
    if isempty(Log)
        fe = NaN;
        return;
    end
    active = arrayField(Log,"active",NaN);
    eventFE = arrayField(Log,"fe",NaN);
    row = find(active >= 32 & isfinite(eventFE),1);
    if isempty(row)
        fe = NaN;
    else
        fe = eventFE(row);
    end
end

function value = numberField(S,names,default)
    value = default;
    if ~isstruct(S) || isempty(S)
        return;
    end
    for name = string(names)
        field = char(name);
        if isfield(S,field)
            candidate = S.(field);
            if (isnumeric(candidate) || islogical(candidate)) && ...
                    isscalar(candidate)
                value = double(candidate);
                return;
            end
        end
    end
end

function value = textField(S,names,default)
    value = string(default);
    if ~isstruct(S) || isempty(S)
        return;
    end
    for name = string(names)
        field = char(name);
        if isfield(S,field) && isscalar(string(S.(field)))
            value = string(S.(field));
            return;
        end
    end
end

function values = vectorField(S,names)
    values = zeros(1,0);
    if ~isstruct(S) || isempty(S)
        return;
    end
    for name = string(names)
        field = char(name);
        if isfield(S,field) && isnumeric(S.(field))
            values = reshape(double(S.(field)),1,[]);
            return;
        end
    end
end

function value = vectorValue(S,names,index)
    values = vectorField(S,names);
    if index <= numel(values)
        value = values(index);
    else
        value = NaN;
    end
end

function values = arrayField(S,names,default)
    values = repmat(default,size(S));
    for i = 1 : numel(S)
        values(i) = numberField(S(i),names,default);
    end
    values = reshape(values,1,[]);
end

function X = matrixField(S,names)
    X = zeros(0,2);
    for name = string(names)
        field = char(name);
        if isfield(S,field) && isnumeric(S.(field)) && ...
                ~isempty(S.(field)) && size(S.(field),2) >= 2
            X = double(S.(field));
            return;
        end
    end
end

function value = lastFinite(values)
    row = find(isfinite(values),1,'last');
    if isempty(row)
        value = NaN;
    else
        value = values(row);
    end
end

function value = sumKnown(values)
    known = isfinite(values);
    if any(known)
        value = sum(values(known));
    else
        value = NaN;
    end
end

function value = safeRatio(numerator,denominator)
    if isfinite(numerator) && isfinite(denominator) && denominator > 0
        value = numerator/denominator;
    else
        value = NaN;
    end
end

function writeTableAtomic(T,file)
    partial = [file,'.partial.csv'];
    cleanup = onCleanup(@()deleteIfExists(partial));
    writetable(T,partial);
    [moved,message] = movefile(partial,file,'f');
    if ~moved
        error('PairGuide:ValidationCSVMoveFailed','%s',message);
    end
    clear cleanup;
end

function exportPNGAtomic(Figure,file)
    partial = [file,'.partial.png'];
    cleanup = onCleanup(@()deleteIfExists(partial));
    exportgraphics(Figure,partial,'Resolution',160, ...
        'BackgroundColor','white');
    Info = imfinfo(partial);
    if ~strcmpi(Info.Format,'png') || Info.Width < 1000 || Info.Height < 400
        error('PairGuide:InvalidValidationPNG', ...
            'Rendered PNG failed validation: %s',partial);
    end
    [moved,message] = movefile(partial,file,'f');
    if ~moved
        error('PairGuide:ValidationPNGMoveFailed','%s',message);
    end
    clear cleanup;
end

function textValue = commaNumber(value)
    textValue = regexprep(sprintf('%.0f',value), ...
        '(?<!\d)(\d{1,3})(?=(\d{3})+(?!\d))','$1,');
end

function ensureFolder(folder)
    if ~isfolder(folder)
        mkdir(folder);
    end
end

function deleteIfExists(file)
    if isfile(file)
        delete(file);
    end
end
