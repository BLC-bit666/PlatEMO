function Result = FinalizeSectionBGate23FromRows(varargin)
% Build partial Gate 2/3 summaries from lightweight rows files.
% This implementation streams large row sets to CSV to avoid materializing
% all seed and gain rows in memory during final aggregation.

    Options = ParseOptions(varargin{:});
    RowFiles = DiscoverRowFiles(Options.GateDir,Options.Variants);
    Prefix = fullfile(Options.GateDir,sprintf('gate23_%s',Options.Tag));

    SeedAuditFile = [Prefix,'_seed_audit.csv'];
    RunSummaryFile = [Prefix,'_run_summary.csv'];
    ProblemSummaryFile = [Prefix,'_problem_summary.csv'];
    DBCurveFile = [Prefix,'_db_curve.csv'];
    GainTraceFile = [Prefix,'_boundary_gain_trace.csv'];
    PartialFile = [Prefix,'_partial.mat'];

    CleanupOutputFiles({SeedAuditFile,RunSummaryFile,ProblemSummaryFile,DBCurveFile, ...
        GainTraceFile,PartialFile});

    RunRows = repmat(InitGate23RunRow(),numel(RowFiles),1);
    SeedStats = repmat(InitGate23SeedStats(),0,1);
    SeedWritten = false;
    GainWritten = false;
    for i = 1 : numel(RowFiles)
        Data = load(RowFiles{i},'SeedRows','GainRows','RunRow');
        RunRows(i) = Data.RunRow;

        SeedChunk = StructArrayToTable(FieldOrDefault(Data,'SeedRows',repmat(InitGate23SeedRow(),0,1)),InitGate23SeedRow());
        GainChunk = StructArrayToTable(FieldOrDefault(Data,'GainRows',repmat(InitGate23GainRow(),0,1)),InitGate23GainRow());
        SeedWritten = AppendTableChunk(SeedChunk,SeedAuditFile,SeedWritten);
        GainWritten = AppendTableChunk(GainChunk,GainTraceFile,GainWritten);
        SeedStats = AccumulateSeedStats(SeedStats,FieldOrDefault(Data,'SeedRows',repmat(InitGate23SeedRow(),0,1)));
        clear Data SeedChunk GainChunk
    end

    EnsureCsvExists(SeedAuditFile,InitGate23SeedRow(),SeedWritten);
    EnsureCsvExists(GainTraceFile,InitGate23GainRow(),GainWritten);

    RunSummaryTable = struct2table(RunRows);
    ProblemSummaryTable = BuildGate23ProblemSummary(SeedStats,RunSummaryTable);
    DBCurveTable = BuildGate23DBCurve(SeedStats,RunSummaryTable);

    writetable(RunSummaryTable,RunSummaryFile);
    writetable(ProblemSummaryTable,ProblemSummaryFile);
    writetable(DBCurveTable,DBCurveFile);

    Result = struct();
    Result.tag = Options.Tag;
    Result.variants = Options.Variants;
    Result.seedAuditFile = SeedAuditFile;
    Result.runSummaryFile = RunSummaryFile;
    Result.problemSummaryFile = ProblemSummaryFile;
    Result.dbCurveFile = DBCurveFile;
    Result.gainTraceFile = GainTraceFile;
    Result.runSummaryTable = RunSummaryTable;
    Result.problemSummaryTable = ProblemSummaryTable;
    Result.dbCurveTable = DBCurveTable;
    save(PartialFile,'Result');
end

function Options = ParseOptions(varargin)
    Options.GateDir = '';
    Options.Variants = {};
    Options.Tag = 'partial';
    if mod(numel(varargin),2) ~= 0
        error('FinalizeSectionBGate23FromRows expects name/value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        Value = varargin{i+1};
        switch lower(Name)
            case 'gatedir'
                Options.GateDir = Value;
            case 'variants'
                Options.Variants = Value;
            case 'tag'
                Options.Tag = Value;
            otherwise
                error('Unknown option: %s',Name);
        end
    end
    if isempty(Options.GateDir)
        error('GateDir is required.');
    end
    if isempty(Options.Variants)
        DirInfo = dir(Options.GateDir);
        Options.Variants = {DirInfo([DirInfo.isdir] & ~startsWith({DirInfo.name},'.')).name};
        Options.Variants = Options.Variants(~startsWith(Options.Variants,'gate23_'));
    end
end

function Files = DiscoverRowFiles(GateDir,Variants)
    Files = {};
    for i = 1 : numel(Variants)
        VariantDir = fullfile(GateDir,Variants{i});
        if exist(VariantDir,'dir') ~= 7
            continue;
        end
        Info = dir(fullfile(VariantDir,'**','*_rows.mat'));
        for j = 1 : numel(Info)
            Files{end+1,1} = fullfile(Info(j).folder,Info(j).name); %#ok<AGROW>
        end
    end
end

function CleanupOutputFiles(Files)
    for i = 1 : numel(Files)
        if exist(Files{i},'file') == 2
            delete(Files{i});
        end
    end
end

function Value = FieldOrDefault(S,Field,DefaultValue)
    if isfield(S,Field)
        Value = S.(Field);
    else
        Value = DefaultValue;
    end
end

function T = StructArrayToTable(Rows,Template)
    if isempty(Rows)
        T = struct2table(repmat(Template,0,1));
    else
        T = struct2table(Rows);
    end
end

function Written = AppendTableChunk(T,Filename,Written)
    if isempty(T)
        return;
    end
    if ~Written
        writetable(T,Filename);
    else
        writetable(T,Filename,'WriteMode','append','WriteVariableNames',false);
    end
    Written = true;
end

function EnsureCsvExists(Filename,Template,Written)
    if Written
        return;
    end
    writetable(struct2table(repmat(Template,0,1)),Filename);
end

function Stats = InitGate23SeedStats()
    Stats = struct( ...
        'variant','', ...
        'problem','', ...
        'oracleParts',{{}}, ...
        'seedCount',0, ...
        'infeasibleSeedCount',0, ...
        'frrSuccessCount',0, ...
        'ubySuccessCount',0);
end

function Stats = AccumulateSeedStats(Stats,SeedRows)
    if isempty(SeedRows)
        return;
    end
    Variant = SeedRows(1).variant;
    Problem = SeedRows(1).problem;
    Index = FindSeedStatsIndex(Stats,Variant,Problem);
    if Index == 0
        Index = numel(Stats) + 1;
        Stats(Index) = InitGate23SeedStats();
        Stats(Index).variant = Variant;
        Stats(Index).problem = Problem;
    end

    Oracle = [SeedRows.oracle_dB]';
    Oracle = Oracle(isfinite(Oracle));
    if ~isempty(Oracle)
        Stats(Index).oracleParts{end+1,1} = Oracle; %#ok<AGROW>
    end

    SeedFeasible = logical([SeedRows.seed_feasible]');
    FRRSuccess = logical([SeedRows.frr_success]');
    UBYSuccess = logical([SeedRows.uby_success]');
    InfeasibleMask = ~SeedFeasible;

    Stats(Index).seedCount = Stats(Index).seedCount + numel(SeedRows);
    Stats(Index).infeasibleSeedCount = Stats(Index).infeasibleSeedCount + sum(InfeasibleMask);
    Stats(Index).frrSuccessCount = Stats(Index).frrSuccessCount + sum(FRRSuccess(InfeasibleMask));
    Stats(Index).ubySuccessCount = Stats(Index).ubySuccessCount + sum(UBYSuccess);
end

function Index = FindSeedStatsIndex(Stats,Variant,Problem)
    Index = 0;
    if isempty(Stats)
        return;
    end
    Match = strcmp({Stats.variant},Variant) & strcmp({Stats.problem},Problem);
    Found = find(Match,1,'first');
    if ~isempty(Found)
        Index = Found;
    end
end

function SummaryTable = BuildGate23ProblemSummary(SeedStats,RunSummaryTable)
    Rows = repmat(InitGate23ProblemRow(),0,1);
    VariantNames = unique(RunSummaryTable.variant,'stable');
    ProblemNames = unique(RunSummaryTable.problem,'stable');
    for v = 1 : numel(VariantNames)
        Variant = VariantNames{v};
        VariantRunMask = strcmp(RunSummaryTable.variant,Variant);
        Rows(end+1,1) = AggregateGate23ProblemRow(Variant,'ALL', ...
            RunSummaryTable(VariantRunMask,:),CollectSeedStats(SeedStats,Variant,'')); %#ok<AGROW>
        for p = 1 : numel(ProblemNames)
            Problem = ProblemNames{p};
            RunMask = VariantRunMask & strcmp(RunSummaryTable.problem,Problem);
            Rows(end+1,1) = AggregateGate23ProblemRow(Variant,Problem, ...
                RunSummaryTable(RunMask,:),CollectSeedStats(SeedStats,Variant,Problem)); %#ok<AGROW>
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

function Row = AggregateGate23ProblemRow(Variant,Problem,RunTable,SeedStat)
    Row = InitGate23ProblemRow();
    Row.variant = Variant;
    Row.problem = Problem;
    Row.run_count = height(RunTable);
    Row.seed_count = SeedStat.seedCount;
    if SeedStat.seedCount > 0
        Row.mean_dB = MeanFinite(SeedStat.oracle_dB);
        Row.median_dB = MedianFinite(SeedStat.oracle_dB);
        if SeedStat.infeasibleSeedCount > 0
            Row.FRR = SeedStat.frrSuccessCount / SeedStat.infeasibleSeedCount;
        end
        Row.UBY = SeedStat.ubySuccessCount / SeedStat.seedCount;
    end
    if ~isempty(RunTable)
        Row.mean_boundary_delta_hv = MeanFinite(RunTable.boundary_delta_hv);
        Row.total_boundary_delta_hv = sum(RunTable.boundary_delta_hv);
    end
end

function CurveTable = BuildGate23DBCurve(SeedStats,RunSummaryTable)
    Rows = repmat(InitGate23DBCurveRow(),0,1);
    VariantNames = unique(RunSummaryTable.variant,'stable');
    ProblemNames = unique(RunSummaryTable.problem,'stable');
    for v = 1 : numel(VariantNames)
        Variant = VariantNames{v};
        Rows = [Rows;CurveRowsForGroup(Variant,'ALL',CollectSeedStats(SeedStats,Variant,'').oracle_dB)]; %#ok<AGROW>
        for p = 1 : numel(ProblemNames)
            Problem = ProblemNames{p};
            Rows = [Rows;CurveRowsForGroup(Variant,Problem,CollectSeedStats(SeedStats,Variant,Problem).oracle_dB)]; %#ok<AGROW>
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

function SeedStat = CollectSeedStats(SeedStats,Variant,Problem)
    SeedStat = InitCollectedSeedStats();
    if isempty(SeedStats)
        return;
    end
    Mask = strcmp({SeedStats.variant},Variant);
    if ~isempty(Problem)
        Mask = Mask & strcmp({SeedStats.problem},Problem);
    end
    Selected = SeedStats(Mask);
    if isempty(Selected)
        return;
    end
    SeedStat.seedCount = sum([Selected.seedCount]);
    SeedStat.infeasibleSeedCount = sum([Selected.infeasibleSeedCount]);
    SeedStat.frrSuccessCount = sum([Selected.frrSuccessCount]);
    SeedStat.ubySuccessCount = sum([Selected.ubySuccessCount]);
    Parts = {};
    for i = 1 : numel(Selected)
        Parts = [Parts; Selected(i).oracleParts(:)]; %#ok<AGROW>
    end
    if ~isempty(Parts)
        SeedStat.oracle_dB = vertcat(Parts{:});
    end
end

function SeedStat = InitCollectedSeedStats()
    SeedStat = struct( ...
        'oracle_dB',zeros(0,1), ...
        'seedCount',0, ...
        'infeasibleSeedCount',0, ...
        'frrSuccessCount',0, ...
        'ubySuccessCount',0);
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
