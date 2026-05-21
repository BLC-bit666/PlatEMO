function [Summary,SnapshotTable] = run_PRBCCMO_B_boundary_relation_all(N,maxFE,runId,outDir,sourceSummaryCsv)
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
    if nargin < 5
        sourceSummaryCsv = "";
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    problemNames = allBoundaryProblemNames();
    SourceRunFolders = sourceRunFolders(problemNames,sourceSummaryCsv);
    Rows = repmat(emptySummaryRow(),numel(problemNames),1);
    SnapshotChunks = cell(numel(problemNames),1);
    for i = 1 : numel(problemNames)
        problemName = problemNames{i};
        if strlength(SourceRunFolders(i)) > 0
            fprintf('[%02d/%02d] Replotting %s B-boundary snapshots from existing run...\n', ...
                i,numel(problemNames),problemName);
        else
            fprintf('[%02d/%02d] Plotting %s B-boundary snapshots...\n', ...
                i,numel(problemNames),problemName);
        end
        problemOutDir = fullfile(outDir,problemName);
        try
            [OutputFiles,RunFolder,Manifest] = plot_PRBCCMO_B_boundary_relation( ...
                problemName,N,maxFE,runId,problemOutDir,SourceRunFolders(i));
            Rows(i) = composeSummaryRow(problemName,RunFolder,OutputFiles,Manifest,"ok","");
            SnapshotChunks{i} = composeSnapshotRows(problemName,RunFolder,OutputFiles,Manifest);
        catch err
            Rows(i) = composeSummaryRow(problemName,"",strings(0,1),table(),"failed",err.message);
            fprintf(2,'[%02d/%02d] %s failed: %s\n', ...
                i,numel(problemNames),problemName,err.message);
        end
    end

    Summary = struct2table(Rows);
    SnapshotChunks = SnapshotChunks(~cellfun(@isempty,SnapshotChunks));
    if isempty(SnapshotChunks)
        SnapshotRows = repmat(emptySnapshotRow(),0,1);
    else
        SnapshotRows = vertcat(SnapshotChunks{:});
    end
    SnapshotTable = struct2table(SnapshotRows);
    writetable(Summary,fullfile(outDir,'boundary_relation_all_summary.csv'));
    writetable(SnapshotTable,fullfile(outDir,'boundary_relation_all_snapshots.csv'));
end

function problemNames = allBoundaryProblemNames()
    problemNames = [ ...
        arrayfun(@(i)sprintf('DASCMOP%d_BC',i),1:9,'UniformOutput',false), ...
        arrayfun(@(i)sprintf('LIRCMOP%d_BC',i),1:14,'UniformOutput',false)];
end

function SourceRunFolders = sourceRunFolders(problemNames,sourceSummaryCsv)
    SourceRunFolders = strings(numel(problemNames),1);
    sourceSummaryCsv = string(sourceSummaryCsv);
    if strlength(sourceSummaryCsv) == 0
        return;
    end
    assert(isfile(sourceSummaryCsv), ...
        'run_PRBCCMO_B_boundary_relation_all:MissingSourceSummary', ...
        'Source summary CSV does not exist: %s.', char(sourceSummaryCsv));
    Source = readtable(sourceSummaryCsv,'TextType','string','Delimiter',',');
    PRBCCMOUtils.requireColumns(Source,{'problem','run_folder'},sourceSummaryCsv);
    for i = 1 : numel(problemNames)
        Match = find(string(Source.problem) == string(problemNames{i}),1,'last');
        if ~isempty(Match)
            RunFolder = string(Source.run_folder(Match));
            if isfolder(RunFolder)
                SourceRunFolders(i) = RunFolder;
            end
        end
    end
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
        Row.final_b_mean_infeasible_boundary_dist = manifestValue(Manifest,'b_mean_infeasible_boundary_dist',height(Manifest));
        Row.final_b_p90_infeasible_boundary_dist = manifestValue(Manifest,'b_p90_infeasible_boundary_dist',height(Manifest));
        Row.final_b_far_infeasible_boundary_ratio = manifestValue(Manifest,'b_far_infeasible_boundary_ratio',height(Manifest));
        Row.fit_metrics_csv = manifestString(Manifest,'fit_metrics_csv',height(Manifest));
    end
end

function Rows = composeSnapshotRows(problemName,RunFolder,OutputFiles,Manifest)
    Rows = repmat(emptySnapshotRow(),height(Manifest),1);
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
        Row.boundary_point_count = manifestValue(Manifest,'boundary_point_count',i);
        Row.b_mean_mid_boundary_dist = manifestValue(Manifest,'b_mean_mid_boundary_dist',i);
        Row.b_p90_mid_boundary_dist = manifestValue(Manifest,'b_p90_mid_boundary_dist',i);
        Row.b_mean_infeasible_boundary_dist = manifestValue(Manifest,'b_mean_infeasible_boundary_dist',i);
        Row.b_p90_infeasible_boundary_dist = manifestValue(Manifest,'b_p90_infeasible_boundary_dist',i);
        Row.b_max_infeasible_boundary_dist = manifestValue(Manifest,'b_max_infeasible_boundary_dist',i);
        Row.b_far_infeasible_boundary_ratio = manifestValue(Manifest,'b_far_infeasible_boundary_ratio',i);
        Row.fit_metrics_csv = manifestString(Manifest,'fit_metrics_csv',i);
        Rows(i) = Row;
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
        'fit_metrics_csv',"", ...
        'snapshot_count',0, ...
        'final_b_pair_count',NaN, ...
        'final_actual_fe',NaN, ...
        'final_b_mean_infeasible_boundary_dist',NaN, ...
        'final_b_p90_infeasible_boundary_dist',NaN, ...
        'final_b_far_infeasible_boundary_ratio',NaN);
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
        'snapshot_png',"", ...
        'fit_metrics_csv',"", ...
        'boundary_point_count',NaN, ...
        'b_mean_mid_boundary_dist',NaN, ...
        'b_p90_mid_boundary_dist',NaN, ...
        'b_mean_infeasible_boundary_dist',NaN, ...
        'b_p90_infeasible_boundary_dist',NaN, ...
        'b_max_infeasible_boundary_dist',NaN, ...
        'b_far_infeasible_boundary_ratio',NaN);
end

function Value = manifestValue(Manifest,Name,Index)
    Value = NaN;
    if isempty(Manifest) || ~ismember(Name,Manifest.Properties.VariableNames) || Index < 1 || Index > height(Manifest)
        return;
    end
    Raw = Manifest.(Name);
    Value = double(Raw(Index));
end

function Value = manifestString(Manifest,Name,Index)
    Value = "";
    if isempty(Manifest) || ~ismember(Name,Manifest.Properties.VariableNames) || Index < 1 || Index > height(Manifest)
        return;
    end
    Raw = Manifest.(Name);
    Value = string(Raw(Index));
end
