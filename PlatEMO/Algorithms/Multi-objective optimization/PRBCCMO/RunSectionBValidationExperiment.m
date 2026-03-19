function Results = RunSectionBValidationExperiment(varargin)
% Run Section B Gate 1/2/3 validation experiments for PRBCCMO on DASCMOP-BC.

    Options = ParseOptions(varargin{:});
    RepoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(genpath(RepoRoot));
    OutputDir = PrepareOutputDir(Options,RepoRoot);

    Results = struct();
    Results.options = Options;
    Results.outputDir = OutputDir;

    if Options.RunGate1
        Results.gate1 = RunGate1Suite(Options,fullfile(OutputDir,'gate1'));
    else
        Results.gate1 = struct();
    end
    if Options.RunGate23
        Results.gate23 = RunGate23Suite(Options,fullfile(OutputDir,'gate23'));
    else
        Results.gate23 = struct();
    end

    save(fullfile(OutputDir,'sectionB_results.mat'),'Results');
end

function Options = ParseOptions(varargin)
    Options.Problems = arrayfun(@(k)sprintf('DASCMOP%d_BC',k),1:9,'UniformOutput',false);
    Options.Runs = 20;
    Options.N = 100;
    Options.maxFE = 200000;
    Options.BinCount = 10;
    Options.SeedBase = 20260317;
    Options.BootstrapSamples = 2000;
    Options.OutputDir = '';
    Options.Parallel = false;
    Options.Workers = [];
    Options.Verbose = false;
    Options.RunGate1 = true;
    Options.RunGate23 = true;
    Options.SkipExisting = true;
    if mod(numel(varargin),2) ~= 0
        error('RunSectionBValidationExperiment expects name/value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        Value = varargin{i+1};
        switch lower(Name)
            case 'problems'
                Options.Problems = Value;
            case 'runs'
                Options.Runs = Value;
            case 'n'
                Options.N = Value;
            case 'maxfe'
                Options.maxFE = Value;
            case 'bincount'
                Options.BinCount = Value;
            case 'seedbase'
                Options.SeedBase = Value;
            case 'bootstrapsamples'
                Options.BootstrapSamples = Value;
            case 'outputdir'
                Options.OutputDir = Value;
            case 'parallel'
                Options.Parallel = logical(Value);
            case 'workers'
                Options.Workers = Value;
            case 'verbose'
                Options.Verbose = logical(Value);
            case 'rungate1'
                Options.RunGate1 = logical(Value);
            case 'rungate23'
                Options.RunGate23 = logical(Value);
            case 'skipexisting'
                Options.SkipExisting = logical(Value);
            otherwise
                error('Unknown option: %s',Name);
        end
    end
end

function OutputDir = PrepareOutputDir(Options,RepoRoot)
    if ~isempty(Options.OutputDir)
        OutputDir = Options.OutputDir;
    else
        Stamp = datestr(now,'yyyymmdd_HHMMSS');
        OutputDir = fullfile(RepoRoot,'Results','PRBCCMO', ...
            sprintf('sectionB_all9_%druns_fe%d_%s',Options.Runs,Options.maxFE,Stamp));
    end
    EnsureFolder(OutputDir);
end

function EnsureFolder(Folder)
    if exist(Folder,'dir') ~= 7
        mkdir(Folder);
    end
end

function Pool = OpenParallelPool(Workers)
    Cluster = parcluster('Processes');
    if ~isempty(Workers) && Workers > Cluster.NumWorkers
        Cluster.NumWorkers = Workers;
        saveProfile(Cluster);
    end

    Pool = gcp('nocreate');
    if isempty(Pool)
        if isempty(Workers)
            Pool = parpool(Cluster);
        else
            Pool = parpool(Cluster,Workers);
        end
        return;
    end
    if ~isempty(Workers) && Pool.NumWorkers ~= Workers
        delete(Pool);
        Pool = parpool(Cluster,Workers);
    end
end

function Variants = Gate1Variants()
    Variants = repmat(struct('name','','displayName','','parameters',{{}}),4,1);
    Variants(1) = InitVariant('raw_mlp','Raw MLP',ParameterOverride({12,1,13,1,14,0,21,1}));
    Variants(2) = InitVariant('raw_temperature','Raw + temperature',ParameterOverride({12,1,13,2,14,0,21,1}));
    Variants(3) = InitVariant('raw_sigmoid','Raw + sigmoid',ParameterOverride({12,1,13,3,14,0,21,1}));
    Variants(4) = InitVariant('ensemble_temperature','3-MLP ensemble + temperature',ParameterOverride({12,3,13,2,14,1,21,1}));
end

function Variants = Gate23Variants()
    Variants = repmat(struct('name','','displayName','','parameters',{{}}),4,1);
    Variants(1) = InitVariant('full','Full',ParameterOverride({19,1,20,1,21,1}));
    Variants(2) = InitVariant('uncertain_only','Uncertain-only',ParameterOverride({19,2,20,1,21,1}));
    Variants(3) = InitVariant('random_boundary','Random-boundary',ParameterOverride({19,3,20,1,21,1}));
    Variants(4) = InitVariant('no_local_label','No-local-label',ParameterOverride({19,1,20,2,21,1}));
end

function Variant = InitVariant(Name,DisplayName,Parameters)
    Variant = struct('name',Name,'displayName',DisplayName,'parameters',{Parameters});
end

function Params = ParameterOverride(Pairs)
    Params = repmat({[]},1,21);
    for i = 1 : 2 : numel(Pairs)
        Params{Pairs{i}} = Pairs{i+1};
    end
end

function EnsureVariantProblemFolders(GateDir,Variants,Problems)
    EnsureFolder(GateDir);
    for v = 1 : numel(Variants)
        EnsureFolder(fullfile(GateDir,Variants(v).name));
        for p = 1 : numel(Problems)
            EnsureFolder(fullfile(GateDir,Variants(v).name,Problems{p}));
        end
    end
end

function Tasks = BuildTasks(Variants,Options,GateDir,GateId)
    TaskCount = numel(Variants)*numel(Options.Problems)*Options.Runs;
    Tasks(TaskCount,1) = struct( ...
        'gateId',GateId, ...
        'variantIndex',0, ...
        'variantName','', ...
        'variantDisplay','', ...
        'parameters',{{}}, ...
        'problemIndex',0, ...
        'problemName','', ...
        'run',0, ...
        'seed',0, ...
        'problemDir','');
    Index = 0;
    for v = 1 : numel(Variants)
        for p = 1 : numel(Options.Problems)
            for r = 1 : Options.Runs
                Index = Index + 1;
                Tasks(Index).gateId = GateId;
                Tasks(Index).variantIndex = v;
                Tasks(Index).variantName = Variants(v).name;
                Tasks(Index).variantDisplay = Variants(v).displayName;
                Tasks(Index).parameters = Variants(v).parameters;
                Tasks(Index).problemIndex = p;
                Tasks(Index).problemName = Options.Problems{p};
                Tasks(Index).run = r;
                Tasks(Index).seed = Options.SeedBase + (GateId-1)*100000 + (v-1)*10000 + (p-1)*Options.Runs + r - 1;
                Tasks(Index).problemDir = fullfile(GateDir,Variants(v).name,Options.Problems{p});
            end
        end
    end
end

function Algorithm = BuildAlgorithm(Task,OutputFcn)
    Algorithm = PRBCCMO('save',0,'run',Task.run,'outputFcn',OutputFcn,'parameter',Task.parameters);
end

function T = StructRowsToTable(RowCells,Template)
    if isempty(RowCells)
        T = struct2table(repmat(Template,0,1));
        return;
    end
    Valid = ~cellfun(@isempty,RowCells);
    if ~any(Valid)
        T = struct2table(repmat(Template,0,1));
        return;
    end
    T = struct2table(vertcat(RowCells{Valid}));
end

function Value = MeanFinite(Data)
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
    else
        Value = mean(Data);
    end
end

function Value = MedianFinite(Data)
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
    else
        Value = median(Data);
    end
end

function Value = FieldOrDefault(S,Field,Default)
    if isstruct(S) && isfield(S,Field) && ~isempty(S.(Field))
        Value = S.(Field);
    else
        Value = Default;
    end
end

function SaveReliabilityFigure(Report,Filename,TitleText)
    Fig = figure('Visible','off','Color','w');
    hold on;
    plot([0,1],[0,1],'k--','LineWidth',1);
    Valid = Report.bin.count > 0;
    if any(Valid)
        MarkerSize = 40 + 200*Report.bin.weight(Valid);
        scatter(Report.bin.meanProb(Valid),Report.bin.feasibleRate(Valid), ...
            MarkerSize,'filled','MarkerFaceColor',[0.12,0.47,0.71]);
        plot(Report.bin.meanProb(Valid),Report.bin.feasibleRate(Valid), ...
            '-','Color',[0.12,0.47,0.71],'LineWidth',1.5);
    end
    xlabel('Predicted probability');
    ylabel('Empirical feasible rate');
    title(TitleText,'Interpreter','none');
    xlim([0,1]);
    ylim([0,1]);
    grid on;
    box on;
    saveas(Fig,Filename);
    close(Fig);
end

function Value = PercentileFromSorted(SortedData,Alpha)
    if isempty(SortedData)
        Value = NaN;
        return;
    end
    Alpha = min(max(Alpha,0),1);
    Position = 1 + Alpha*(numel(SortedData)-1);
    Lower = floor(Position);
    Upper = ceil(Position);
    if Lower == Upper
        Value = SortedData(Lower);
    else
        Weight = Position - Lower;
        Value = (1-Weight)*SortedData(Lower) + Weight*SortedData(Upper);
    end
end

function [Low,High] = BootstrapNearZoneCI(Prob,Label,SampleCount,Seed)
    Prob = Prob(:);
    Label = Label(:);
    Mask = Prob >= 0.4 & Prob <= 0.6;
    NearLabel = Label(Mask);
    if isempty(NearLabel)
        Low = NaN;
        High = NaN;
        return;
    end
    State = rng;
    Cleaner = onCleanup(@() rng(State)); %#ok<NASGU>
    rng(Seed,'twister');
    Boot = zeros(SampleCount,1);
    N = numel(NearLabel);
    for i = 1 : SampleCount
        Index = randi(N,N,1);
        Boot(i) = mean(NearLabel(Index));
    end
    Boot = sort(Boot);
    Low = PercentileFromSorted(Boot,0.025);
    High = PercentileFromSorted(Boot,0.975);
end

function Name = ResolveBoundarySourceName(Source)
    switch round(Source)
        case 1
            Name = 'bridge';
        case 2
            Name = 'local';
        otherwise
            Name = 'unknown';
    end
end

function SilentOutput(varargin) %#ok<INUSD>
end

function Gate = RunGate1Suite(Options,GateDir)
    Variants = Gate1Variants();
    EnsureVariantProblemFolders(GateDir,Variants,Options.Problems);
    Tasks = BuildTasks(Variants,Options,GateDir,1);
    TaskCount = numel(Tasks);
    TaskReports = cell(TaskCount,1);
    TaskSummary = repmat(InitGate1SummaryRow(),TaskCount,1);
    TaskBins = cell(TaskCount,1);
    TaskTrace = cell(TaskCount,1);

    Pool = [];
    if Options.Parallel
        Pool = OpenParallelPool(Options.Workers);
        parfor t = 1 : TaskCount
            [TaskReports{t},TaskSummary(t),TaskBins{t},TaskTrace{t}] = ExecuteGate1Run(Tasks(t),Options);
        end
    else
        for t = 1 : TaskCount
            [TaskReports{t},TaskSummary(t),TaskBins{t},TaskTrace{t}] = ExecuteGate1Run(Tasks(t),Options);
        end
    end

    Reports = cell(numel(Variants),numel(Options.Problems),Options.Runs);
    for t = 1 : TaskCount
        Task = Tasks(t);
        Reports{Task.variantIndex,Task.problemIndex,Task.run} = TaskReports{t};
    end

    SummaryTable = struct2table(TaskSummary);
    BinTable = StructRowsToTable(TaskBins,InitGate1BinRow());
    TraceTable = StructRowsToTable(TaskTrace,InitGate1TraceRow());
    PooledRows = repmat(InitGate1SummaryRow(),0,1);
    GateRows = repmat(InitGate1GateRow(),0,1);

    for v = 1 : numel(Variants)
        Variant = Variants(v);
        VariantProb = zeros(0,1);
        VariantLabel = zeros(0,1);
        VariantProblemRows = repmat(InitGate1SummaryRow(),0,1);
        for p = 1 : numel(Options.Problems)
            ProblemName = Options.Problems{p};
            ProblemProb = zeros(0,1);
            ProblemLabel = zeros(0,1);
            for r = 1 : Options.Runs
                Report = Reports{v,p,r};
                ProblemProb = [ProblemProb;Report.prob(:)]; %#ok<AGROW>
                ProblemLabel = [ProblemLabel;Report.label(:)]; %#ok<AGROW>
                VariantProb = [VariantProb;Report.prob(:)]; %#ok<AGROW>
                VariantLabel = [VariantLabel;Report.label(:)]; %#ok<AGROW>
            end
            PooledReport = SummarizeCalibrationProbabilities(ProblemProb,ProblemLabel,Options.BinCount);
            PooledReport.variant = Variant.displayName;
            PooledReport.problem = ProblemName;
            PooledReport.run = 0;
            ProblemDir = fullfile(GateDir,Variant.name,ProblemName);
            save(fullfile(ProblemDir,sprintf('%s_%s_pooled_report.mat',Variant.name,ProblemName)),'PooledReport');
            SaveReliabilityFigure(PooledReport, ...
                fullfile(ProblemDir,sprintf('%s_%s_reliability.png',Variant.name,ProblemName)), ...
                sprintf('%s | %s | pooled reliability',Variant.displayName,ProblemName));
            Row = BuildGate1SummaryRow(Variant.displayName,ProblemName,0,PooledReport);
            PooledRows(end+1,1) = Row; %#ok<AGROW>
            VariantProblemRows(end+1,1) = Row; %#ok<AGROW>
        end

        AggregateReport = SummarizeCalibrationProbabilities(VariantProb,VariantLabel,Options.BinCount);
        AggregateReport.variant = Variant.displayName;
        AggregateReport.problem = 'ALL';
        AggregateReport.run = 0;
        save(fullfile(GateDir,Variant.name,sprintf('%s_aggregate_report.mat',Variant.name)),'AggregateReport');
        SaveReliabilityFigure(AggregateReport, ...
            fullfile(GateDir,Variant.name,sprintf('%s_reliability_all.png',Variant.name)), ...
            sprintf('%s | ALL | pooled reliability',Variant.displayName));
        PooledRows(end+1,1) = BuildGate1SummaryRow(Variant.displayName,'ALL',0,AggregateReport); %#ok<AGROW>
        GateRows(end+1,1) = BuildGate1GateRow(Variant.displayName,AggregateReport,VariantProblemRows,Options); %#ok<AGROW>
    end

    PooledTable = struct2table(PooledRows);
    GateTable = struct2table(GateRows);
    writetable(SummaryTable,fullfile(GateDir,'gate1_summary.csv'));
    writetable(BinTable,fullfile(GateDir,'gate1_bin_stats.csv'));
    writetable(TraceTable,fullfile(GateDir,'gate1_trace.csv'));
    writetable(PooledTable,fullfile(GateDir,'gate1_pooled_summary.csv'));
    writetable(GateTable,fullfile(GateDir,'gate1_gatecheck.csv'));
    WriteGate1Overview(fullfile(GateDir,'gate1_overview.txt'),SummaryTable,PooledTable,GateTable,Options);

    Gate = struct();
    Gate.outputDir = GateDir;
    Gate.variants = Variants;
    Gate.summaryTable = SummaryTable;
    Gate.binTable = BinTable;
    Gate.traceTable = TraceTable;
    Gate.pooledTable = PooledTable;
    Gate.gateTable = GateTable;
    Gate.parallelWorkers = 0;
    if ~isempty(Pool)
        Gate.parallelWorkers = Pool.NumWorkers;
    end
    save(fullfile(GateDir,'gate1_results.mat'),'Gate');
end

function [Report,SummaryRow,BinRows,TraceRows] = ExecuteGate1Run(Task,Options)
    rng(Task.seed,'twister');
    if Options.Verbose
        fprintf('Gate1: %s | %s | run %d/%d\n',Task.variantDisplay,Task.problemName,Task.run,Options.Runs);
    end

    ReportFile = fullfile(Task.problemDir,sprintf('%s_%s_run%d_report.mat',Task.variantName,Task.problemName,Task.run));
    if Options.SkipExisting && exist(ReportFile,'file') == 2
        Data = load(ReportFile);
        Report = Data.Report;
        TraceRows = repmat(InitGate1TraceRow(),0,1);
        if isfield(Data,'TraceRows') && ~isempty(Data.TraceRows)
            TraceRows = Data.TraceRows;
        end
        SummaryRow = BuildGate1SummaryRow(Task.variantDisplay,Task.problemName,Task.run,Report);
        BinRows = BuildGate1BinRows(Task.variantDisplay,Task.problemName,Task.run,Report.bin);
        return;
    end

    Algorithm = BuildAlgorithm(Task,@SilentOutput);
    Problem = feval(Task.problemName,'N',Options.N,'maxFE',Options.maxFE);
    Algorithm.Solve(Problem);

    assert(isfield(Algorithm.metric,'boundaryCalibration'), ...
        'PRBCCMO did not populate metric.boundaryCalibration.');
    Report = Algorithm.metric.boundaryCalibration;
    Report.variant = Task.variantDisplay;
    Report.problem = Task.problemName;
    Report.run = Task.run;
    Report.seed = Task.seed;

    TraceRows = repmat(InitGate1TraceRow(),0,1);
    if isfield(Algorithm.metric,'sectionB') && isfield(Algorithm.metric.sectionB,'calibrationTrace')
        TraceRows = BuildGate1TraceRows(Task,Algorithm.metric.sectionB.calibrationTrace);
    end

    save(ReportFile,'Report','TraceRows');
    SummaryRow = BuildGate1SummaryRow(Task.variantDisplay,Task.problemName,Task.run,Report);
    BinRows = BuildGate1BinRows(Task.variantDisplay,Task.problemName,Task.run,Report.bin);
end

function Row = InitGate1SummaryRow()
    Row = struct( ...
        'variant','', ...
        'problem','', ...
        'run',0, ...
        'count',0, ...
        'feasible_rate',NaN, ...
        'mean_prob',NaN, ...
        'brier',NaN, ...
        'log_loss',NaN, ...
        'ece',NaN, ...
        'near_count',0, ...
        'near_mean_prob',NaN, ...
        'near_feasible_rate',NaN, ...
        'near_gap',NaN, ...
        'generation',NaN, ...
        'FE',NaN, ...
        'training_count',NaN, ...
        'calibration_count',NaN, ...
        'calibration_near_count',NaN, ...
        'test_count',NaN, ...
        'test_near_count',NaN);
end

function Row = BuildGate1SummaryRow(VariantName,ProblemName,RunIndex,Report)
    Row = InitGate1SummaryRow();
    Row.variant = VariantName;
    Row.problem = ProblemName;
    Row.run = RunIndex;
    Row.count = FieldOrDefault(Report,'count',0);
    Row.feasible_rate = FieldOrDefault(Report,'feasibleRate',NaN);
    Row.mean_prob = FieldOrDefault(Report,'meanProb',NaN);
    Row.brier = FieldOrDefault(Report,'brier',NaN);
    Row.log_loss = FieldOrDefault(Report,'logLoss',NaN);
    Row.ece = FieldOrDefault(Report,'ece',NaN);
    Row.near_count = FieldOrDefault(Report,'nearCount',0);
    Row.near_mean_prob = FieldOrDefault(Report,'nearMeanProb',NaN);
    Row.near_feasible_rate = FieldOrDefault(Report,'nearFeasibleRate',NaN);
    Row.near_gap = FieldOrDefault(Report,'nearGap',NaN);
    Row.generation = FieldOrDefault(Report,'generation',NaN);
    Row.FE = FieldOrDefault(Report,'FE',NaN);
    Row.training_count = FieldOrDefault(Report,'trainingCount',NaN);
    Row.calibration_count = FieldOrDefault(Report,'calibrationCount',NaN);
    Row.calibration_near_count = FieldOrDefault(Report,'calibrationNearCount',NaN);
    Row.test_count = FieldOrDefault(Report,'testCount',NaN);
    Row.test_near_count = FieldOrDefault(Report,'testNearCount',NaN);
end

function Row = InitGate1BinRow()
    Row = struct( ...
        'variant','', ...
        'problem','', ...
        'run',0, ...
        'bin_left',NaN, ...
        'bin_right',NaN, ...
        'bin_center',NaN, ...
        'count',0, ...
        'weight',NaN, ...
        'mean_prob',NaN, ...
        'feasible_rate',NaN, ...
        'gap',NaN);
end

function Rows = BuildGate1BinRows(VariantName,ProblemName,RunIndex,Bin)
    Count = numel(Bin.left);
    Rows = repmat(InitGate1BinRow(),Count,1);
    for i = 1 : Count
        Rows(i).variant = VariantName;
        Rows(i).problem = ProblemName;
        Rows(i).run = RunIndex;
        Rows(i).bin_left = Bin.left(i);
        Rows(i).bin_right = Bin.right(i);
        Rows(i).bin_center = Bin.center(i);
        Rows(i).count = Bin.count(i);
        Rows(i).weight = Bin.weight(i);
        Rows(i).mean_prob = Bin.meanProb(i);
        Rows(i).feasible_rate = Bin.feasibleRate(i);
        Rows(i).gap = Bin.gap(i);
    end
end

function Row = InitGate1TraceRow()
    Row = struct( ...
        'variant','', ...
        'problem','', ...
        'run',0, ...
        'generation',NaN, ...
        'FE',NaN, ...
        'count',0, ...
        'feasible_rate',NaN, ...
        'mean_prob',NaN, ...
        'brier',NaN, ...
        'log_loss',NaN, ...
        'ece',NaN, ...
        'near_count',0, ...
        'near_mean_prob',NaN, ...
        'near_feasible_rate',NaN, ...
        'near_gap',NaN, ...
        'training_count',NaN, ...
        'calibration_count',NaN, ...
        'calibration_near_count',NaN, ...
        'test_count',NaN, ...
        'test_near_count',NaN);
end

function Rows = BuildGate1TraceRows(Task,Trace)
    if isempty(Trace)
        Rows = repmat(InitGate1TraceRow(),0,1);
        return;
    end
    Rows = repmat(InitGate1TraceRow(),numel(Trace),1);
    for i = 1 : numel(Trace)
        Rows(i).variant = Task.variantDisplay;
        Rows(i).problem = Task.problemName;
        Rows(i).run = Task.run;
        Rows(i).generation = Trace(i).generation;
        Rows(i).FE = Trace(i).FE;
        Rows(i).count = Trace(i).count;
        Rows(i).feasible_rate = Trace(i).feasible_rate;
        Rows(i).mean_prob = Trace(i).mean_prob;
        Rows(i).brier = Trace(i).brier;
        Rows(i).log_loss = Trace(i).log_loss;
        Rows(i).ece = Trace(i).ece;
        Rows(i).near_count = Trace(i).near_count;
        Rows(i).near_mean_prob = Trace(i).near_mean_prob;
        Rows(i).near_feasible_rate = Trace(i).near_feasible_rate;
        Rows(i).near_gap = Trace(i).near_gap;
        Rows(i).training_count = Trace(i).training_count;
        Rows(i).calibration_count = Trace(i).calibration_count;
        Rows(i).calibration_near_count = Trace(i).calibration_near_count;
        Rows(i).test_count = Trace(i).test_count;
        Rows(i).test_near_count = Trace(i).test_near_count;
    end
end

function Row = InitGate1GateRow()
    Row = struct( ...
        'variant','', ...
        'pooled_near_gap',NaN, ...
        'pooled_ece',NaN, ...
        'problem_pass_count',0, ...
        'pooled_near_ci_low',NaN, ...
        'pooled_near_ci_high',NaN, ...
        'pooled_ci_covers_half',false, ...
        'gate1_pass',false);
end

function Row = BuildGate1GateRow(VariantName,AggregateReport,ProblemRows,Options)
    Row = InitGate1GateRow();
    Row.variant = VariantName;
    Row.pooled_near_gap = AggregateReport.nearGap;
    Row.pooled_ece = AggregateReport.ece;
    ValidProblem = ~strcmp({ProblemRows.problem}','ALL');
    NearGap = [ProblemRows(ValidProblem).near_gap]';
    Row.problem_pass_count = sum(NearGap <= 0.07);
    [Low,High] = BootstrapNearZoneCI(AggregateReport.prob,AggregateReport.label,Options.BootstrapSamples,Options.SeedBase + sum(double(VariantName)));
    Row.pooled_near_ci_low = Low;
    Row.pooled_near_ci_high = High;
    Row.pooled_ci_covers_half = isfinite(Low) && isfinite(High) && Low <= 0.5 && High >= 0.5;
    Row.gate1_pass = Row.pooled_near_gap <= 0.05 && Row.pooled_ece <= 0.05 ...
        && Row.problem_pass_count >= 7 && Row.pooled_ci_covers_half;
end

function WriteGate1Overview(Filename,SummaryTable,PooledTable,GateTable,Options)
    fid = fopen(Filename,'wt');
    assert(fid ~= -1,'Failed to open %s for writing.',Filename);
    Cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid,'PRBCCMO Section B Gate 1 validation\n');
    fprintf(fid,'Runs per problem: %d\n',Options.Runs);
    fprintf(fid,'Population size N: %d\n',Options.N);
    fprintf(fid,'Maximum FE: %d\n',Options.maxFE);
    fprintf(fid,'Probability bins: %d\n',Options.BinCount);
    fprintf(fid,'Bootstrap samples: %d\n',Options.BootstrapSamples);
    fprintf(fid,'Parallel: %d\n',Options.Parallel);
    if ~isempty(Options.Workers)
        fprintf(fid,'Requested workers: %d\n',Options.Workers);
    end
    fprintf(fid,'\nGate 1 pass criteria from fix2.txt\n');
    fprintf(fid,'1. pooled near_gap <= 0.05\n');
    fprintf(fid,'2. pooled ECE <= 0.05\n');
    fprintf(fid,'3. at least 7/9 problems with near_gap <= 0.07\n');
    fprintf(fid,'4. 95%% bootstrap CI on p in [0.4,0.6] covers 0.5\n\n');

    fprintf(fid,'Variant gate checks\n');
    for i = 1 : height(GateTable)
        fprintf(fid,'%s: pooled_near_gap=%.6f, pooled_ece=%.6f, problem_pass_count=%d, CI=[%.6f, %.6f], covers_0.5=%d, gate1_pass=%d\n', ...
            GateTable.variant{i},GateTable.pooled_near_gap(i),GateTable.pooled_ece(i), ...
            GateTable.problem_pass_count(i),GateTable.pooled_near_ci_low(i), ...
            GateTable.pooled_near_ci_high(i),GateTable.pooled_ci_covers_half(i), ...
            GateTable.gate1_pass(i));
    end

    fprintf(fid,'\nPooled summaries\n');
    for i = 1 : height(PooledTable)
        fprintf(fid,'%s | %s | run=%d | count=%d | brier=%.6f | log_loss=%.6f | ece=%.6f | near_count=%d | near_feasible_rate=%.6f | near_gap=%.6f\n', ...
            PooledTable.variant{i},PooledTable.problem{i},PooledTable.run(i), ...
            PooledTable.count(i),PooledTable.brier(i),PooledTable.log_loss(i), ...
            PooledTable.ece(i),PooledTable.near_count(i), ...
            PooledTable.near_feasible_rate(i),PooledTable.near_gap(i));
    end

    fprintf(fid,'\nPer-run summary rows: %d\n',height(SummaryTable));
end

function Gate = RunGate23Suite(Options,GateDir)
    Variants = Gate23Variants();
    EnsureVariantProblemFolders(GateDir,Variants,Options.Problems);
    Tasks = BuildTasks(Variants,Options,GateDir,2);
    TaskCount = numel(Tasks);
    TaskRunSummary = repmat(InitGate23RunRow(),TaskCount,1);

    Pool = [];
    if Options.Parallel
        Pool = OpenParallelPool(Options.Workers);
        parfor t = 1 : TaskCount
            TaskRunSummary(t) = ExecuteGate23Run(Tasks(t),Options);
        end
    else
        for t = 1 : TaskCount
            TaskRunSummary(t) = ExecuteGate23Run(Tasks(t),Options);
        end
    end

    TaskSeedRows = cell(TaskCount,1);
    TaskGainRows = cell(TaskCount,1);
    for t = 1 : TaskCount
        [TaskSeedRows{t},TaskGainRows{t},TaskRunSummary(t)] = LoadGate23Rows(Tasks(t),TaskRunSummary(t));
    end

    SeedTable = StructRowsToTable(TaskSeedRows,InitGate23SeedRow());
    GainTable = StructRowsToTable(TaskGainRows,InitGate23GainRow());
    RunSummaryTable = struct2table(TaskRunSummary);
    ProblemSummaryTable = BuildGate23ProblemSummary(SeedTable,RunSummaryTable);
    DBCurveTable = BuildGate23DBCurve(SeedTable);

    writetable(SeedTable,fullfile(GateDir,'gate23_seed_audit.csv'));
    writetable(RunSummaryTable,fullfile(GateDir,'gate23_run_summary.csv'));
    writetable(ProblemSummaryTable,fullfile(GateDir,'gate23_problem_summary.csv'));
    writetable(DBCurveTable,fullfile(GateDir,'gate23_db_curve.csv'));
    writetable(GainTable,fullfile(GateDir,'gate23_boundary_gain_trace.csv'));
    WriteGate23Overview(fullfile(GateDir,'gate23_overview.txt'),RunSummaryTable,ProblemSummaryTable,Options);

    Gate = struct();
    Gate.outputDir = GateDir;
    Gate.variants = Variants;
    Gate.seedTable = SeedTable;
    Gate.gainTable = GainTable;
    Gate.runSummaryTable = RunSummaryTable;
    Gate.problemSummaryTable = ProblemSummaryTable;
    Gate.dbCurveTable = DBCurveTable;
    Gate.parallelWorkers = 0;
    if ~isempty(Pool)
        Gate.parallelWorkers = Pool.NumWorkers;
    end
    save(fullfile(GateDir,'gate23_results.mat'),'Gate','-v7.3');
end

function RunRow = ExecuteGate23Run(Task,Options)
    rng(Task.seed,'twister');
    if Options.Verbose
        fprintf('Gate23: %s | %s | run %d/%d\n',Task.variantDisplay,Task.problemName,Task.run,Options.Runs);
    end

    ReportFile = GetGate23ReportFile(Task);
    RowsFile = GetGate23RowsFile(Task);
    if Options.SkipExisting && exist(RowsFile,'file') == 2
        Data = load(RowsFile,'RunRow');
        RunRow = Data.RunRow;
        return;
    end
    if Options.SkipExisting && exist(ReportFile,'file') == 2
        VarInfo = whos('-file',ReportFile);
        VarNames = {VarInfo.name};
        if any(strcmp(VarNames,'RunRow'))
            Data = load(ReportFile,'RunRow');
            RunRow = Data.RunRow;
        else
            [SeedRows,GainRows,Metric] = LoadGate23StoredRowsOrMetric(Task,ReportFile);
            RunRow = BuildGate23RunRow(Task,SeedRows,GainRows,Metric);
            save(ReportFile,'RunRow','-append');
        end
        if exist(RowsFile,'file') ~= 2
            [SeedRows,GainRows,~] = LoadGate23StoredRowsOrMetric(Task,ReportFile);
            save(RowsFile,'SeedRows','GainRows','RunRow','-v7.3');
        end
        return;
    end

    Algorithm = BuildAlgorithm(Task,@SilentOutput);
    Problem = feval(Task.problemName,'N',Options.N,'maxFE',Options.maxFE);
    Algorithm.Solve(Problem);

    assert(isfield(Algorithm.metric,'sectionB'), ...
        'PRBCCMO did not populate metric.sectionB.');
    Metric = Algorithm.metric.sectionB;
    SeedRows = BuildGate23SeedRows(Task,Metric);
    GainRows = BuildGate23GainRows(Task,Metric);
    RunRow = BuildGate23RunRow(Task,SeedRows,GainRows,Metric);

    save(RowsFile,'SeedRows','GainRows','RunRow','-v7.3');
    save(ReportFile,'RunRow','-v7.3');
end

function [SeedRows,GainRows,RunRow] = LoadGate23Rows(Task,RunRow)
    RowsFile = GetGate23RowsFile(Task);
    if exist(RowsFile,'file') ~= 2
        SeedRows = repmat(InitGate23SeedRow(),0,1);
        GainRows = repmat(InitGate23GainRow(),0,1);
        return;
    end

    Data = load(RowsFile);
    SeedRows = Data.SeedRows;
    GainRows = Data.GainRows;
    if isfield(Data,'RunRow') && ~isempty(Data.RunRow)
        RunRow = Data.RunRow;
    end
end

function ReportFile = GetGate23ReportFile(Task)
    ReportFile = fullfile(Task.problemDir,sprintf('%s_%s_run%d_report.mat',Task.variantName,Task.problemName,Task.run));
end

function RowsFile = GetGate23RowsFile(Task)
    RowsFile = fullfile(Task.problemDir,sprintf('%s_%s_run%d_rows.mat',Task.variantName,Task.problemName,Task.run));
end

function [SeedRows,GainRows,Metric] = LoadGate23StoredRowsOrMetric(Task,ReportFile)
    SeedRows = repmat(InitGate23SeedRow(),0,1);
    GainRows = repmat(InitGate23GainRow(),0,1);
    Metric = struct('externalArchiveCount',NaN);

    VarInfo = whos('-file',ReportFile);
    VarNames = {VarInfo.name};
    HasSeedRows = any(strcmp(VarNames,'SeedRows'));
    HasGainRows = any(strcmp(VarNames,'GainRows'));
    if HasSeedRows && HasGainRows
        Data = load(ReportFile,'SeedRows','GainRows');
        SeedRows = Data.SeedRows;
        GainRows = Data.GainRows;
        return;
    end

    if ~any(strcmp(VarNames,'Metric'))
        return;
    end
    Data = load(ReportFile,'Metric');
    Metric = Data.Metric;
    SeedRows = BuildGate23SeedRows(Task,Metric);
    GainRows = BuildGate23GainRows(Task,Metric);
    save(ReportFile,'SeedRows','GainRows','-append');
end

function Row = InitGate23SeedRow()
    Row = struct( ...
        'variant','', ...
        'problem','', ...
        'run',0, ...
        'generation',NaN, ...
        'FE',NaN, ...
        'source',NaN, ...
        'source_name','', ...
        'seed_feasible',false, ...
        'prob',NaN, ...
        'entropy',NaN, ...
        'hv_gain',NaN, ...
        'novelty',NaN, ...
        'penalty',NaN, ...
        'utility',NaN, ...
        'oracle_dB',NaN, ...
        'frr_success',false, ...
        'uby_success',false, ...
        'local_eval_count',0, ...
        'bracket_gap',NaN, ...
        'hard_negative_confirmed',false);
end

function Rows = BuildGate23SeedRows(Task,Metric)
    Rows = repmat(InitGate23SeedRow(),0,1);
    if ~isfield(Metric,'seedAudit') || isempty(Metric.seedAudit)
        return;
    end
    Audit = Metric.seedAudit;
    Count = numel(Audit);
    Rows = repmat(InitGate23SeedRow(),Count,1);
    Distance = NaN(Count,1);
    if Count > 0
        D = numel(Audit(1).seedDec);
        ChunkSize = 512;
        for StartIdx = 1 : ChunkSize : Count
            EndIdx = min(StartIdx + ChunkSize - 1,Count);
            ChunkDec = zeros(EndIdx-StartIdx+1,D);
            for j = StartIdx : EndIdx
                ChunkDec(j-StartIdx+1,:) = Audit(j).seedDec;
            end
            Distance(StartIdx:EndIdx) = ComputeOracleBoundaryDistance(Task.problemName,ChunkDec);
        end
    end
    for i = 1 : Count
        Rows(i).variant = Task.variantDisplay;
        Rows(i).problem = Task.problemName;
        Rows(i).run = Task.run;
        Rows(i).generation = Audit(i).generation;
        Rows(i).FE = Audit(i).FE;
        Rows(i).source = Audit(i).source;
        Rows(i).source_name = ResolveBoundarySourceName(Audit(i).source);
        Rows(i).seed_feasible = Audit(i).seedFeasible;
        Rows(i).prob = Audit(i).prob;
        Rows(i).entropy = Audit(i).entropy;
        Rows(i).hv_gain = Audit(i).hvGain;
        Rows(i).novelty = Audit(i).novelty;
        Rows(i).penalty = Audit(i).penalty;
        Rows(i).utility = Audit(i).utility;
        Rows(i).oracle_dB = Distance(i);
        Rows(i).frr_success = Audit(i).frrSuccess;
        Rows(i).uby_success = Audit(i).ubySuccess;
        Rows(i).local_eval_count = Audit(i).localEvalCount;
        Rows(i).bracket_gap = Audit(i).bracketGap;
        Rows(i).hard_negative_confirmed = Audit(i).hardNegativeConfirmed;
    end
end

function Row = InitGate23GainRow()
    Row = struct( ...
        'variant','', ...
        'problem','', ...
        'run',0, ...
        'generation',NaN, ...
        'FE',NaN, ...
        'boundary_gain',0, ...
        'cumulative_boundary_gain',0, ...
        'boundary_added_count',0, ...
        'external_archive_count',0);
end

function Rows = BuildGate23GainRows(Task,Metric)
    Rows = repmat(InitGate23GainRow(),0,1);
    if ~isfield(Metric,'boundaryGainTrace') || isempty(Metric.boundaryGainTrace)
        return;
    end
    Trace = Metric.boundaryGainTrace;
    Rows = repmat(InitGate23GainRow(),numel(Trace),1);
    for i = 1 : numel(Trace)
        Rows(i).variant = Task.variantDisplay;
        Rows(i).problem = Task.problemName;
        Rows(i).run = Task.run;
        Rows(i).generation = Trace(i).generation;
        Rows(i).FE = Trace(i).FE;
        Rows(i).boundary_gain = Trace(i).boundaryGain;
        Rows(i).cumulative_boundary_gain = Trace(i).cumulativeBoundaryGain;
        Rows(i).boundary_added_count = Trace(i).boundaryAddedCount;
        Rows(i).external_archive_count = Trace(i).externalArchiveCount;
    end
end

function Row = InitGate23RunRow()
    Row = struct( ...
        'variant','', ...
        'problem','', ...
        'run',0, ...
        'seed_count',0, ...
        'infeasible_seed_count',0, ...
        'mean_dB',NaN, ...
        'median_dB',NaN, ...
        'FRR',NaN, ...
        'UBY',NaN, ...
        'boundary_delta_hv',0, ...
        'external_archive_count',0);
end

function Row = BuildGate23RunRow(Task,SeedRows,GainRows,Metric)
    Row = InitGate23RunRow();
    Row.variant = Task.variantDisplay;
    Row.problem = Task.problemName;
    Row.run = Task.run;
    Row.seed_count = numel(SeedRows);
    Row.external_archive_count = FieldOrDefault(Metric,'externalArchiveCount',0);
    if isempty(SeedRows)
        return;
    end

    DB = [SeedRows.oracle_dB]';
    Row.mean_dB = MeanFinite(DB);
    Row.median_dB = MedianFinite(DB);
    InfeasibleMask = ~[SeedRows.seed_feasible]';
    Row.infeasible_seed_count = sum(InfeasibleMask);
    if any(InfeasibleMask)
        Row.FRR = mean(double([SeedRows(InfeasibleMask).frr_success]'));
    end
    Row.UBY = mean(double([SeedRows.uby_success]'));
    if isempty(GainRows)
        Row.boundary_delta_hv = 0;
    else
        Row.boundary_delta_hv = sum([GainRows.boundary_gain]');
    end
end

function SummaryTable = BuildGate23ProblemSummary(SeedTable,RunSummaryTable)
    Rows = repmat(InitGate23ProblemRow(),0,1);
    VariantNames = unique(RunSummaryTable.variant,'stable');
    ProblemNames = unique(RunSummaryTable.problem,'stable');
    for v = 1 : numel(VariantNames)
        Variant = VariantNames{v};
        VariantRunMask = strcmp(RunSummaryTable.variant,Variant);
        VariantSeedMask = strcmp(SeedTable.variant,Variant);
        Rows(end+1,1) = AggregateGate23ProblemRow(Variant,'ALL', ...
            RunSummaryTable(VariantRunMask,:),SeedTable(VariantSeedMask,:)); %#ok<AGROW>
        for p = 1 : numel(ProblemNames)
            Problem = ProblemNames{p};
            RunMask = VariantRunMask & strcmp(RunSummaryTable.problem,Problem);
            SeedMask = VariantSeedMask & strcmp(SeedTable.problem,Problem);
            Rows(end+1,1) = AggregateGate23ProblemRow(Variant,Problem, ...
                RunSummaryTable(RunMask,:),SeedTable(SeedMask,:)); %#ok<AGROW>
        end
    end
    SummaryTable = struct2table(Rows);
end

function Row = InitGate23ProblemRow()
    Row = struct( ...
        'variant','', ...
        'problem','', ...
        'run_count',0, ...
        'seed_count',0, ...
        'mean_dB',NaN, ...
        'median_dB',NaN, ...
        'FRR',NaN, ...
        'UBY',NaN, ...
        'mean_boundary_delta_hv',NaN, ...
        'total_boundary_delta_hv',NaN);
end

function Row = AggregateGate23ProblemRow(Variant,Problem,RunTable,SeedTable)
    Row = InitGate23ProblemRow();
    Row.variant = Variant;
    Row.problem = Problem;
    Row.run_count = height(RunTable);
    Row.seed_count = height(SeedTable);
    if ~isempty(SeedTable)
        Row.mean_dB = MeanFinite(SeedTable.oracle_dB);
        Row.median_dB = MedianFinite(SeedTable.oracle_dB);
        InfeasibleMask = ~SeedTable.seed_feasible;
        if any(InfeasibleMask)
            Row.FRR = mean(double(SeedTable.frr_success(InfeasibleMask)));
        end
        Row.UBY = mean(double(SeedTable.uby_success));
    end
    if ~isempty(RunTable)
        Row.mean_boundary_delta_hv = MeanFinite(RunTable.boundary_delta_hv);
        Row.total_boundary_delta_hv = sum(RunTable.boundary_delta_hv);
    end
end

function CurveTable = BuildGate23DBCurve(SeedTable)
    Rows = repmat(InitGate23DBCurveRow(),0,1);
    if isempty(SeedTable)
        CurveTable = struct2table(Rows);
        return;
    end
    VariantNames = unique(SeedTable.variant,'stable');
    ProblemNames = unique(SeedTable.problem,'stable');
    for v = 1 : numel(VariantNames)
        Variant = VariantNames{v};
        VariantMask = strcmp(SeedTable.variant,Variant);
        Rows = [Rows;CurveRowsForGroup(Variant,'ALL',SeedTable.oracle_dB(VariantMask))]; %#ok<AGROW>
        for p = 1 : numel(ProblemNames)
            Problem = ProblemNames{p};
            Mask = VariantMask & strcmp(SeedTable.problem,Problem);
            Rows = [Rows;CurveRowsForGroup(Variant,Problem,SeedTable.oracle_dB(Mask))]; %#ok<AGROW>
        end
    end
    CurveTable = struct2table(Rows);
end

function Row = InitGate23DBCurveRow()
    Row = struct( ...
        'variant','', ...
        'problem','', ...
        'tau',NaN, ...
        'count_lt_tau',0, ...
        'fraction_lt_tau',NaN, ...
        'total_count',0);
end

function Rows = CurveRowsForGroup(Variant,Problem,Distance)
    Distance = Distance(isfinite(Distance));
    if isempty(Distance)
        Rows = repmat(InitGate23DBCurveRow(),0,1);
        return;
    end
    MaxValue = max(Distance);
    if MaxValue <= 0
        Tau = zeros(101,1);
    else
        Tau = linspace(0,MaxValue,101)';
    end
    Rows = repmat(InitGate23DBCurveRow(),numel(Tau),1);
    Total = numel(Distance);
    for i = 1 : numel(Tau)
        Count = sum(Distance < Tau(i));
        Rows(i).variant = Variant;
        Rows(i).problem = Problem;
        Rows(i).tau = Tau(i);
        Rows(i).count_lt_tau = Count;
        Rows(i).fraction_lt_tau = Count/Total;
        Rows(i).total_count = Total;
    end
end

function WriteGate23Overview(Filename,RunSummaryTable,ProblemSummaryTable,Options)
    fid = fopen(Filename,'wt');
    assert(fid ~= -1,'Failed to open %s for writing.',Filename);
    Cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid,'PRBCCMO Section B Gate 2/3 validation\n');
    fprintf(fid,'Runs per problem: %d\n',Options.Runs);
    fprintf(fid,'Population size N: %d\n',Options.N);
    fprintf(fid,'Maximum FE: %d\n',Options.maxFE);
    fprintf(fid,'Parallel: %d\n',Options.Parallel);
    if ~isempty(Options.Workers)
        fprintf(fid,'Requested workers: %d\n',Options.Workers);
    end
    fprintf(fid,'\nGate 2 metrics\n');
    fprintf(fid,'1. median d_B\n');
    fprintf(fid,'2. mean d_B\n');
    fprintf(fid,'3. #(d_B < tau) cumulative curve\n');
    fprintf(fid,'Gate 3 metrics\n');
    fprintf(fid,'1. FRR\n');
    fprintf(fid,'2. UBY\n');
    fprintf(fid,'3. Boundary-induced Delta HV\n\n');

    fprintf(fid,'Variant/problem pooled summaries\n');
    for i = 1 : height(ProblemSummaryTable)
        fprintf(fid,'%s | %s | runs=%d | seeds=%d | mean_dB=%.6f | median_dB=%.6f | FRR=%.6f | UBY=%.6f | mean_delta_HV=%.6f | total_delta_HV=%.6f\n', ...
            ProblemSummaryTable.variant{i},ProblemSummaryTable.problem{i}, ...
            ProblemSummaryTable.run_count(i),ProblemSummaryTable.seed_count(i), ...
            ProblemSummaryTable.mean_dB(i),ProblemSummaryTable.median_dB(i), ...
            ProblemSummaryTable.FRR(i),ProblemSummaryTable.UBY(i), ...
            ProblemSummaryTable.mean_boundary_delta_hv(i),ProblemSummaryTable.total_boundary_delta_hv(i));
    end

    fprintf(fid,'\nPer-run summary rows: %d\n',height(RunSummaryTable));
end
