function [Summary,SnapshotTable] = run_PRBCCMO_B_boundary_relation_all(N,maxFE,runId,outDir)
% Generate B-boundary fitting figures for all DASCMOP_BC and LIRCMOP_BC cases.

    if nargin < 1 || isempty(N)
        N = 100;
    end
    if nargin < 2 || isempty(maxFE)
        maxFE = 200000;
    end
    if nargin < 3 || isempty(runId)
        runId = 1;
    end
    if nargin < 4 || isempty(outDir)
        rootDir = fileparts(which('platemo'));
        outDir = fullfile(rootDir,'Data','PRBCCMO_t', ...
            ['boundary_relation_all_',char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    problemNames = allBoundaryProblemNames();
    Rows = repmat(emptySummaryRow(),numel(problemNames),1);
    SnapshotRows = repmat(emptySnapshotRow(),0,1);
    for i = 1 : numel(problemNames)
        problemName = problemNames{i};
        fprintf('[%02d/%02d] Plotting %s B-boundary snapshots...\n', ...
            i,numel(problemNames),problemName);
        problemOutDir = fullfile(outDir,problemName);
        try
            [OutputFiles,RunFolder,Manifest] = plot_PRBCCMO_B_boundary_relation( ...
                problemName,N,maxFE,runId,problemOutDir);
            Rows(i) = composeSummaryRow(problemName,RunFolder,OutputFiles,Manifest,"ok","");
            SnapshotRows = appendSnapshotRows( ...
                SnapshotRows,problemName,RunFolder,OutputFiles,Manifest); %#ok<AGROW>
        catch err
            Rows(i) = composeSummaryRow(problemName,"",strings(0,1),table(),"failed",err.message);
            fprintf(2,'[%02d/%02d] %s failed: %s\n', ...
                i,numel(problemNames),problemName,err.message);
        end
    end

    Summary = struct2table(Rows);
    SnapshotTable = struct2table(SnapshotRows);
    writetable(Summary,fullfile(outDir,'boundary_relation_all_summary.csv'));
    writetable(SnapshotTable,fullfile(outDir,'boundary_relation_all_snapshots.csv'));
end

function problemNames = allBoundaryProblemNames()
    problemNames = [ ...
        arrayfun(@(i)sprintf('DASCMOP%d_BC',i),1:9,'UniformOutput',false), ...
        arrayfun(@(i)sprintf('LIRCMOP%d_BC',i),1:14,'UniformOutput',false)];
end

function Row = composeSummaryRow(problemName,RunFolder,OutputFiles,Manifest,status,errorMessage)
    Row = emptySummaryRow();
    Row.problem = string(problemName);
    Row.family = problemFamily(problemName);
    Row.run_folder = string(RunFolder);
    Row.status = string(status);
    Row.error_message = string(errorMessage);
    if ~isempty(OutputFiles)
        Row.progression_png = string(OutputFiles(1));
    end
    if ~isempty(Manifest)
        Row.snapshot_count = height(Manifest);
        Row.final_b_pair_count = double(Manifest.b_pair_count(end));
        Row.final_actual_fe = double(Manifest.actual_fe(end));
    end
end

function Rows = appendSnapshotRows(Rows,problemName,RunFolder,OutputFiles,Manifest)
    for i = 1 : height(Manifest)
        Row = emptySnapshotRow();
        Row.problem = string(problemName);
        Row.family = problemFamily(problemName);
        Row.run_folder = string(RunFolder);
        Row.target_fe = double(Manifest.target_fe(i));
        Row.actual_fe = double(Manifest.actual_fe(i));
        Row.generation = double(Manifest.generation(i));
        Row.b_pair_count = double(Manifest.b_pair_count(i));
        Row.snapshot_csv = string(Manifest.snapshot_file(i));
        Row.snapshot_png = string(OutputFiles(i+1));
        Rows(end+1,1) = Row; %#ok<AGROW>
    end
end

function Family = problemFamily(problemName)
    if startsWith(string(problemName),"DASCMOP")
        Family = "DASCMOP_BC";
    else
        Family = "LIRCMOP_BC";
    end
end

function Row = emptySummaryRow()
    Row = struct( ...
        'problem',"", ...
        'family',"", ...
        'status',"", ...
        'error_message',"", ...
        'run_folder',"", ...
        'progression_png',"", ...
        'snapshot_count',0, ...
        'final_b_pair_count',NaN, ...
        'final_actual_fe',NaN);
end

function Row = emptySnapshotRow()
    Row = struct( ...
        'problem',"", ...
        'family',"", ...
        'run_folder',"", ...
        'target_fe',NaN, ...
        'actual_fe',NaN, ...
        'generation',NaN, ...
        'b_pair_count',NaN, ...
        'snapshot_csv',"", ...
        'snapshot_png',"");
end
