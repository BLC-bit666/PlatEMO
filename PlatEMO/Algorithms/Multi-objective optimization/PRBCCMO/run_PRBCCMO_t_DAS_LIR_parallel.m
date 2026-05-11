function [Results,RunTable,ProblemTable,FamilyTable] = run_PRBCCMO_t_DAS_LIR_parallel(runs,workerCount,N,maxFE,outDir)
% Run PRBCCMO_t on all DASCMOP_BC and LIRCMOP_BC problems with CSV outputs only.

    if nargin < 1 || isempty(runs)
        runs = 5;
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 10;
    end
    if nargin < 3 || isempty(N)
        N = 100;
    end
    if nargin < 4 || isempty(maxFE)
        maxFE = 200000;
    end
    if nargin < 5 || isempty(outDir)
        rootDir = fileparts(which('platemo'));
        outDir = fullfile(rootDir,'Data','PRBCCMO_t', ...
            sprintf('DAS_LIR_runs%d_workers%d_%s',runs,workerCount, ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))));
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    problemNames = [ ...
        arrayfun(@(i)sprintf('DASCMOP%d_BC',i),1:9,'UniformOutput',false), ...
        arrayfun(@(i)sprintf('LIRCMOP%d_BC',i),1:14,'UniformOutput',false)];
    runIds = 1 : runs;

    benchmarkCsv = fullfile(outDir,'benchmark_runs.csv');
    Results = benchmark_PRBCCMO_t_suite(problemNames,runIds,benchmarkCsv,N,maxFE,workerCount);

    traceListFile = fullfile(outDir,'trace_list.txt');
    writeTraceList(traceListFile,Results.analysis_folder);
    [RunTable,ProblemTable,FamilyTable] = summarize_PRBCCMO_t_data( ...
        cellstr(string(Results.analysis_folder)),outDir);
end

function writeTraceList(filePath,runFolders)
    fid = fopen(filePath,'w');
    cleaner = onCleanup(@()fclose(fid));
    runFolders = string(runFolders(:));
    for i = 1 : numel(runFolders)
        fprintf(fid,'%s\n',char(runFolders(i)));
    end
end
