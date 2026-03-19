function Result = MergeSectionBGate23Partials(varargin)
% Merge partial Gate 2/3 summaries into final outputs.
% This implementation concatenates partial CSV outputs directly so the
% final merge does not need to materialize the full seed/gain audit tables.

    Options = ParseOptions(varargin{:});
    Partials = DiscoverPartials(Options.GateDir,Options.Tags);
    assert(~isempty(Partials),'No partial files found under %s.',Options.GateDir);

    SeedAuditInputs = cell(numel(Partials),1);
    GainTraceInputs = cell(numel(Partials),1);
    RunSummaryTable = table();
    ProblemSummaryTable = table();
    DBCurveTable = table();
    for i = 1 : numel(Partials)
        Data = load(Partials{i},'Result');
        SeedAuditInputs{i} = Data.Result.seedAuditFile;
        GainTraceInputs{i} = Data.Result.gainTraceFile;
        RunSummaryTable = VertcatTables(RunSummaryTable,Data.Result.runSummaryTable);
        ProblemSummaryTable = VertcatTables(ProblemSummaryTable,Data.Result.problemSummaryTable);
        DBCurveTable = VertcatTables(DBCurveTable,Data.Result.dbCurveTable);
    end

    SeedAuditFile = fullfile(Options.GateDir,'gate23_seed_audit.csv');
    RunSummaryFile = fullfile(Options.GateDir,'gate23_run_summary.csv');
    ProblemSummaryFile = fullfile(Options.GateDir,'gate23_problem_summary.csv');
    DBCurveFile = fullfile(Options.GateDir,'gate23_db_curve.csv');
    GainTraceFile = fullfile(Options.GateDir,'gate23_boundary_gain_trace.csv');
    ResultsFile = fullfile(Options.GateDir,'gate23_results.mat');

    CleanupOutputFiles({SeedAuditFile,RunSummaryFile,ProblemSummaryFile,DBCurveFile, ...
        GainTraceFile,ResultsFile});
    ConcatCsvFiles(SeedAuditInputs,SeedAuditFile);
    ConcatCsvFiles(GainTraceInputs,GainTraceFile);
    writetable(RunSummaryTable,RunSummaryFile);
    writetable(ProblemSummaryTable,ProblemSummaryFile);
    writetable(DBCurveTable,DBCurveFile);
    WriteGate23Overview(fullfile(Options.GateDir,'gate23_overview.txt'),RunSummaryTable,ProblemSummaryTable);

    Result = struct();
    Result.seedAuditFile = SeedAuditFile;
    Result.gainTraceFile = GainTraceFile;
    Result.runSummaryTable = RunSummaryTable;
    Result.problemSummaryTable = ProblemSummaryTable;
    Result.dbCurveTable = DBCurveTable;
    save(ResultsFile,'Result');
end

function Options = ParseOptions(varargin)
    Options.GateDir = '';
    Options.Tags = {};
    if mod(numel(varargin),2) ~= 0
        error('MergeSectionBGate23Partials expects name/value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        Value = varargin{i+1};
        switch lower(Name)
            case 'gatedir'
                Options.GateDir = Value;
            case 'tags'
                Options.Tags = Value;
            otherwise
                error('Unknown option: %s',Name);
        end
    end
    if isempty(Options.GateDir)
        error('GateDir is required.');
    end
end

function Files = DiscoverPartials(GateDir,Tags)
    if isempty(Tags)
        Info = dir(fullfile(GateDir,'gate23_*_partial.mat'));
        Files = arrayfun(@(S) fullfile(S.folder,S.name),Info,'UniformOutput',false);
        return;
    end
    Files = cell(numel(Tags),1);
    for i = 1 : numel(Tags)
        Files{i} = fullfile(GateDir,sprintf('gate23_%s_partial.mat',Tags{i}));
    end
end

function T = VertcatTables(A,B)
    if isempty(A)
        T = B;
    elseif isempty(B)
        T = A;
    else
        T = [A;B];
    end
end

function CleanupOutputFiles(Files)
    for i = 1 : numel(Files)
        if exist(Files{i},'file') == 2
            delete(Files{i});
        end
    end
end

function ConcatCsvFiles(InputFiles,OutputFile)
    fidOut = fopen(OutputFile,'wt');
    assert(fidOut ~= -1,'Failed to open %s for writing.',OutputFile);
    Cleaner = onCleanup(@() fclose(fidOut)); %#ok<NASGU>

    HeaderWritten = false;
    for i = 1 : numel(InputFiles)
        InputFile = InputFiles{i};
        if exist(InputFile,'file') ~= 2
            continue;
        end
        fidIn = fopen(InputFile,'rt');
        assert(fidIn ~= -1,'Failed to open %s for reading.',InputFile);
        LineCleaner = onCleanup(@() fclose(fidIn)); %#ok<NASGU>
        LineNumber = 0;
        while true
            Line = fgetl(fidIn);
            if ~ischar(Line)
                break;
            end
            LineNumber = LineNumber + 1;
            if LineNumber == 1
                if HeaderWritten
                    continue;
                end
                HeaderWritten = true;
            end
            fprintf(fidOut,'%s\n',Line);
        end
        clear LineCleaner
    end
end

function WriteGate23Overview(Filename,RunSummaryTable,ProblemSummaryTable)
    fid = fopen(Filename,'wt');
    assert(fid ~= -1,'Failed to open %s for writing.',Filename);
    Cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid,'PRBCCMO Section B Gate 2/3 validation\n');
    fprintf(fid,'Gate 2 metrics\n');
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
