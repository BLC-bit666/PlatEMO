function [RunTable,ProblemTable,FamilyTable] = analyze_PRBCCMO_t_completed_traces(traceListFile,outDir)
% Summarize completed PRBCCMO_t core trace folders listed in a text file.

    if nargin < 1 || isempty(traceListFile)
        error('analyze_PRBCCMO_t_completed_traces:MissingTraceList', ...
            'traceListFile is required.');
    end
    if nargin < 2 || isempty(outDir)
        outDir = fileparts(char(string(traceListFile)));
        if isempty(outDir)
            outDir = pwd;
        end
    end

    traceListFile = char(string(traceListFile));
    outDir = char(string(outDir));
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    RunFolders = readTraceList(traceListFile);
    assert(~isempty(RunFolders), ...
        'analyze_PRBCCMO_t_completed_traces:EmptyTraceList', ...
        'No completed trace folders listed in %s.', traceListFile);

    [RunTable,ProblemTable,FamilyTable] = summarize_PRBCCMO_t_data(RunFolders,outDir);
    writetable(RunTable,fullfile(outDir,'completed_trace_runs.csv'));
    writetable(ProblemTable,fullfile(outDir,'completed_trace_problem_summary.csv'));
    writetable(FamilyTable,fullfile(outDir,'completed_trace_family_summary.csv'));
end

function RunFolders = readTraceList(traceListFile)
    Text = strtrim(string(fileread(traceListFile)));
    if strlength(Text) == 0
        RunFolders = {};
        return;
    end
    Lines = splitlines(Text);
    Lines = strtrim(Lines);
    Lines = Lines(strlength(Lines) > 0);
    RunFolders = cellstr(Lines);
end
