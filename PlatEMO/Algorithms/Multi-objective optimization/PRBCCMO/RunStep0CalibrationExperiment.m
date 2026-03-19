function Results = RunStep0CalibrationExperiment(varargin)
% Run Step 0 calibration sanity check for PRBCCMO on DASCMOP-BC.

    Options = ParseOptions(varargin{:});
    RepoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(genpath(RepoRoot));
    OutputDir = PrepareOutputDir(Options,RepoRoot);

    ProblemCount = numel(Options.Problems);
    TaskCount    = ProblemCount*Options.Runs;
    EnsureProblemFolders(OutputDir,Options.Problems);

    Reports     = cell(ProblemCount,Options.Runs);
    TaskReports = cell(TaskCount,1);
    TaskSummary = repmat(InitSummaryRow(),TaskCount,1);
    TaskBins    = cell(TaskCount,1);
    PooledRows  = repmat(InitSummaryRow(),0,1);
    AllProb     = zeros(0,1);
    AllLabel    = zeros(0,1);
    Tasks       = BuildTasks(Options,OutputDir);

    Pool = [];
    if Options.Parallel
        Pool = OpenParallelPool(Options.Workers);
        parfor t = 1 : TaskCount
            [TaskReports{t},TaskSummary(t),TaskBins{t}] = ExecuteSingleRun(Tasks(t),Options);
        end
    else
        for t = 1 : TaskCount
            [TaskReports{t},TaskSummary(t),TaskBins{t}] = ExecuteSingleRun(Tasks(t),Options);
        end
    end

    for t = 1 : TaskCount
        Task = Tasks(t);
        Reports{Task.problemIndex,Task.run} = TaskReports{t};
    end

    SummaryRows = TaskSummary;
    BinRows     = vertcat(TaskBins{:});

    for p = 1 : ProblemCount
        ProblemName  = Options.Problems{p};
        ProblemDir   = fullfile(OutputDir,ProblemName);
        ProblemProb  = zeros(0,1);
        ProblemLabel = zeros(0,1);
        for r = 1 : Options.Runs
            Report = Reports{p,r};
            ProblemProb  = [ProblemProb;Report.prob(:)];
            ProblemLabel = [ProblemLabel;Report.label(:)];
            AllProb      = [AllProb;Report.prob(:)];
            AllLabel     = [AllLabel;Report.label(:)];
        end

        PooledReport = SummarizeCalibrationProbabilities(ProblemProb,ProblemLabel,Options.BinCount);
        PooledReport.problem = ProblemName;
        PooledReport.run     = 0;
        save(fullfile(ProblemDir,sprintf('%s_pooled_report.mat',ProblemName)),'PooledReport');
        SaveReliabilityFigure(PooledReport, ...
            fullfile(ProblemDir,sprintf('%s_reliability.png',ProblemName)), ...
            sprintf('%s pooled reliability',ProblemName));
        PooledRows(end+1,1) = BuildSummaryRow(ProblemName,0,PooledReport); %#ok<AGROW>
    end

    AggregateReport = SummarizeCalibrationProbabilities(AllProb,AllLabel,Options.BinCount);
    AggregateReport.problem = 'ALL';
    AggregateReport.run     = 0;
    save(fullfile(OutputDir,'step0_aggregate_report.mat'),'AggregateReport');
    SaveReliabilityFigure(AggregateReport, ...
        fullfile(OutputDir,'step0_reliability_all.png'), ...
        'PRBCCMO Step 0 pooled reliability');

    SummaryTable = struct2table(SummaryRows);
    BinTable     = struct2table(BinRows);
    PooledTable  = struct2table([PooledRows;BuildSummaryRow('ALL',0,AggregateReport)]);
    writetable(SummaryTable,fullfile(OutputDir,'step0_summary.csv'));
    writetable(BinTable,fullfile(OutputDir,'step0_bin_stats.csv'));
    writetable(PooledTable,fullfile(OutputDir,'step0_pooled_summary.csv'));
    WriteOverview(fullfile(OutputDir,'step0_overview.txt'),SummaryTable,PooledTable,AggregateReport,Options);

    Results.options         = Options;
    Results.outputDir       = OutputDir;
    Results.summaryTable    = SummaryTable;
    Results.binTable        = BinTable;
    Results.pooledTable     = PooledTable;
    Results.aggregateReport = AggregateReport;
    Results.reports         = Reports;
    Results.parallelWorkers = 0;
    if ~isempty(Pool)
        Results.parallelWorkers = Pool.NumWorkers;
    end
    save(fullfile(OutputDir,'step0_results.mat'),'Results');
end

function Options = ParseOptions(varargin)
    Options.Problems = arrayfun(@(k)sprintf('DASCMOP%d_BC',k),1:9,'UniformOutput',false);
    Options.Runs     = 1;
    Options.N        = 100;
    Options.maxFE    = 10000;
    Options.BinCount = 10;
    Options.SeedBase = 20260317;
    Options.OutputDir = '';
    Options.AlgorithmParameters = {};
    Options.Parallel = false;
    Options.Workers  = [];
    Options.Verbose  = false;
    if mod(numel(varargin),2) ~= 0
        error('RunStep0CalibrationExperiment expects name/value pairs.');
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
            case 'outputdir'
                Options.OutputDir = Value;
            case 'algorithmparameters'
                Options.AlgorithmParameters = Value;
            case 'parallel'
                Options.Parallel = logical(Value);
            case 'workers'
                Options.Workers = Value;
            case 'verbose'
                Options.Verbose = logical(Value);
            otherwise
                error('Unknown option: %s',Name);
        end
    end
end

function Algorithm = BuildAlgorithm(Options,RunIndex)
    Args = {'save',0,'run',RunIndex,'outputFcn',@SilentOutput};
    if ~isempty(Options.AlgorithmParameters)
        Args = [Args,{'parameter',Options.AlgorithmParameters}];
    end
    Algorithm = PRBCCMO(Args{:});
end

function OutputDir = PrepareOutputDir(Options,RepoRoot)
    if ~isempty(Options.OutputDir)
        OutputDir = Options.OutputDir;
    else
        Stamp = datestr(now,'yyyymmdd_HHMMSS');
        OutputDir = fullfile(RepoRoot,'Results','PRBCCMO',['step0_calibration_',Stamp]);
    end
    EnsureFolder(OutputDir);
end

function EnsureFolder(Folder)
    if exist(Folder,'dir') ~= 7
        mkdir(Folder);
    end
end

function EnsureProblemFolders(OutputDir,ProblemNames)
    for i = 1 : numel(ProblemNames)
        EnsureFolder(fullfile(OutputDir,ProblemNames{i}));
    end
end

function Tasks = BuildTasks(Options,OutputDir)
    ProblemCount = numel(Options.Problems);
    TaskCount = ProblemCount*Options.Runs;
    Tasks(TaskCount,1) = struct( ...
        'problemIndex',0, ...
        'problemName','', ...
        'run',0, ...
        'seed',0, ...
        'problemDir','');
    Index = 0;
    for p = 1 : ProblemCount
        for r = 1 : Options.Runs
            Index = Index + 1;
            Tasks(Index).problemIndex = p;
            Tasks(Index).problemName  = Options.Problems{p};
            Tasks(Index).run          = r;
            Tasks(Index).seed         = Options.SeedBase + (p-1)*Options.Runs + r - 1;
            Tasks(Index).problemDir   = fullfile(OutputDir,Options.Problems{p});
        end
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

function [Report,SummaryRow,BinRows] = ExecuteSingleRun(Task,Options)
    rng(Task.seed,'twister');
    if Options.Verbose
        fprintf('Running Step 0 calibration: %s (run %d/%d)\n',Task.problemName,Task.run,Options.Runs);
    end

    Algorithm = BuildAlgorithm(Options,Task.run);
    Problem   = feval(Task.problemName,'N',Options.N,'maxFE',Options.maxFE);
    Algorithm.Solve(Problem);

    assert(isfield(Algorithm.metric,'boundaryCalibration'), ...
        'PRBCCMO did not populate metric.boundaryCalibration.');
    Report = Algorithm.metric.boundaryCalibration;
    Report.problem = Task.problemName;
    Report.run     = Task.run;
    Report.seed    = Task.seed;

    save(fullfile(Task.problemDir,sprintf('%s_run%d_report.mat',Task.problemName,Task.run)),'Report');
    SummaryRow = BuildSummaryRow(Task.problemName,Task.run,Report);
    BinRows    = BuildBinRows(Task.problemName,Task.run,Report.bin);
end

function Row = InitSummaryRow()
    Row = struct( ...
        'problem','', ...
        'run',0, ...
        'count',0, ...
        'feasible_rate',NaN, ...
        'mean_prob',NaN, ...
        'brier',NaN, ...
        'ece',NaN, ...
        'near_count',0, ...
        'near_mean_prob',NaN, ...
        'near_feasible_rate',NaN, ...
        'near_gap',NaN, ...
        'generation',NaN, ...
        'FE',NaN, ...
        'training_count',NaN, ...
        'calibration_count',NaN, ...
        'calibration_near_count',NaN);
end

function Row = BuildSummaryRow(ProblemName,RunIndex,Report)
    Row = InitSummaryRow();
    Row.problem = ProblemName;
    Row.run     = RunIndex;
    Row.count   = Report.count;
    Row.feasible_rate = Report.feasibleRate;
    Row.mean_prob     = Report.meanProb;
    Row.brier         = Report.brier;
    Row.ece           = Report.ece;
    Row.near_count         = Report.nearCount;
    Row.near_mean_prob     = Report.nearMeanProb;
    Row.near_feasible_rate = Report.nearFeasibleRate;
    Row.near_gap           = Report.nearGap;
    if isfield(Report,'generation'); Row.generation = Report.generation; end
    if isfield(Report,'FE'); Row.FE = Report.FE; end
    if isfield(Report,'trainingCount'); Row.training_count = Report.trainingCount; end
    if isfield(Report,'calibrationCount'); Row.calibration_count = Report.calibrationCount; end
    if isfield(Report,'calibrationNearCount'); Row.calibration_near_count = Report.calibrationNearCount; end
end

function Row = InitBinRow()
    Row = struct( ...
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

function Rows = BuildBinRows(ProblemName,RunIndex,Bin)
    Count = numel(Bin.left);
    Rows = repmat(InitBinRow(),Count,1);
    for i = 1 : Count
        Rows(i).problem       = ProblemName;
        Rows(i).run           = RunIndex;
        Rows(i).bin_left      = Bin.left(i);
        Rows(i).bin_right     = Bin.right(i);
        Rows(i).bin_center    = Bin.center(i);
        Rows(i).count         = Bin.count(i);
        Rows(i).weight        = Bin.weight(i);
        Rows(i).mean_prob     = Bin.meanProb(i);
        Rows(i).feasible_rate = Bin.feasibleRate(i);
        Rows(i).gap           = Bin.gap(i);
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

function WriteOverview(Filename,SummaryTable,PooledTable,AggregateReport,Options)
    fid = fopen(Filename,'wt');
    assert(fid ~= -1,'Failed to open %s for writing.',Filename);
    Cleaner = onCleanup(@() fclose(fid));

    fprintf(fid,'PRBCCMO Step 0 calibration sanity check\n');
    fprintf(fid,'Runs per problem: %d\n',Options.Runs);
    fprintf(fid,'Population size N: %d\n',Options.N);
    fprintf(fid,'Maximum FE: %d\n',Options.maxFE);
    fprintf(fid,'Probability bins: %d\n',Options.BinCount);
    fprintf(fid,'Parallel: %d\n',Options.Parallel);
    if ~isempty(Options.Workers)
        fprintf(fid,'Requested workers: %d\n',Options.Workers);
    end
    fprintf(fid,'\n');

    fprintf(fid,'Aggregate pooled report\n');
    fprintf(fid,'count: %d\n',AggregateReport.count);
    fprintf(fid,'brier: %.6f\n',AggregateReport.brier);
    fprintf(fid,'ece: %.6f\n',AggregateReport.ece);
    fprintf(fid,'near_count: %d\n',AggregateReport.nearCount);
    fprintf(fid,'near_mean_prob: %.6f\n',AggregateReport.nearMeanProb);
    fprintf(fid,'near_feasible_rate: %.6f\n',AggregateReport.nearFeasibleRate);
    fprintf(fid,'near_gap: %.6f\n\n',AggregateReport.nearGap);

    fprintf(fid,'Per-problem pooled near-zone summary\n');
    for i = 1 : height(PooledTable)
        fprintf(fid,'%s, run=%d, count=%d, brier=%.6f, ece=%.6f, near_count=%d, near_feasible_rate=%.6f, near_gap=%.6f\n', ...
            PooledTable.problem{i},PooledTable.run(i),PooledTable.count(i), ...
            PooledTable.brier(i),PooledTable.ece(i),PooledTable.near_count(i), ...
            PooledTable.near_feasible_rate(i),PooledTable.near_gap(i));
    end

    fprintf(fid,'\nPer-run summary rows: %d\n',height(SummaryTable));
end

function SilentOutput(varargin) %#ok<INUSD>
end
