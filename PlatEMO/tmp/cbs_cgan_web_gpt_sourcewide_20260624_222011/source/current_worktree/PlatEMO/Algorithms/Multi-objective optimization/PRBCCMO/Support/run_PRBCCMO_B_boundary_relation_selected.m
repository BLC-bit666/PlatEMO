function Summary = run_PRBCCMO_B_boundary_relation_selected(problemNames,N,maxFE,runId,outDir)
% Generate boundary-fitting progression figures for selected PRBCCMO_t cases.

    if nargin < 1 || isempty(problemNames)
        problemNames = {'DASCMOP1_BC','DASCMOP3_BC','LIRCMOP7_BC','LIRCMOP10_BC'};
    end
    if nargin < 2 || isempty(N)
        N = 100;
    end
    if nargin < 3 || isempty(maxFE)
        maxFE = 200000;
    end
    if nargin < 4 || isempty(runId)
        runId = 1;
    end
    if nargin < 5 || isempty(outDir)
        rootDir = fileparts(which('platemo'));
        outDir = fullfile(rootDir,'Data','PRBCCMO_t','boundary_relation_selected');
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    Rows = repmat(emptyRow(),numel(problemNames),1);
    for i = 1 : numel(problemNames)
        problemName = char(string(problemNames{i}));
        problemOutDir = fullfile(outDir,problemName);
        [OutputFiles,RunFolder,Manifest] = plot_PRBCCMO_B_boundary_relation( ...
            problemName,N,maxFE,runId,problemOutDir);
        Rows(i).problem = string(problemName);
        Rows(i).run_folder = string(RunFolder);
        Rows(i).progression_png = OutputFiles(1);
        Rows(i).snapshot_count = height(Manifest);
        Rows(i).final_b_pair_count = double(Manifest.b_pair_count(end));
        Rows(i).final_b_mean_infeasible_boundary_dist = manifestValue(Manifest,'b_mean_infeasible_boundary_dist',height(Manifest));
        Rows(i).final_b_p90_infeasible_boundary_dist = manifestValue(Manifest,'b_p90_infeasible_boundary_dist',height(Manifest));
        Rows(i).final_b_far_infeasible_boundary_ratio = manifestValue(Manifest,'b_far_infeasible_boundary_ratio',height(Manifest));
        Rows(i).fit_metrics_csv = manifestString(Manifest,'fit_metrics_csv',height(Manifest));
    end

    Summary = struct2table(Rows);
    writetable(Summary,fullfile(outDir,'boundary_relation_selected_summary.csv'));
end

function Row = emptyRow()
    Row = struct( ...
        'problem',"", ...
        'run_folder',"", ...
        'progression_png',"", ...
        'fit_metrics_csv',"", ...
        'snapshot_count',0, ...
        'final_b_pair_count',NaN, ...
        'final_b_mean_infeasible_boundary_dist',NaN, ...
        'final_b_p90_infeasible_boundary_dist',NaN, ...
        'final_b_far_infeasible_boundary_ratio',NaN);
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
