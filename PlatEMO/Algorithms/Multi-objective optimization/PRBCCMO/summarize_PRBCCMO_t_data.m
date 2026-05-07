function [RunTable,ProblemTable,FamilyTable] = summarize_PRBCCMO_t_data(inputPath,outDir)
% Summarize PRBCCMO_t trace folders into run/problem/family CSV tables.

    if nargin < 1 || isempty(inputPath)
        rootDir = fileparts(which('platemo'));
        inputPath = fullfile(rootDir,'Data','PRBCCMO_t');
    end
    if nargin < 2 || isempty(outDir)
        outDir = inputPath;
    end

    runFolders = resolveRunFolders(inputPath);
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    Summaries = repmat(summarize_PRBCCMO_t_run(runFolders{1}),numel(runFolders),1);
    for i = 1 : numel(runFolders)
        Summaries(i) = summarize_PRBCCMO_t_run(runFolders{i});
    end

    RunTable = struct2table(Summaries);
    writetable(RunTable,fullfile(outDir,'run_summary.csv'));

    ProblemTable = aggregateTraceTable(RunTable,"problem");
    writetable(ProblemTable,fullfile(outDir,'problem_summary.csv'));

    FamilyTable = aggregateTraceTable(RunTable,"family");
    writetable(FamilyTable,fullfile(outDir,'family_summary.csv'));
end

function runFolders = resolveRunFolders(inputPath)
    if ischar(inputPath) || (isstring(inputPath) && isscalar(inputPath))
        folder = char(string(inputPath));
        if isfolder(folder) && isfile(fullfile(folder,'run_meta.csv'))
            runFolders = {folder};
            return;
        end
        assert(isfolder(folder), ...
            'summarize_PRBCCMO_t_data:MissingInputFolder', ...
            'Input path not found: %s', folder);
        dirs = dir(folder);
        dirs = dirs([dirs.isdir]);
        dirs = dirs(~ismember({dirs.name},{'.','..'}));
        runFolders = {};
        for i = 1 : numel(dirs)
            candidate = fullfile(folder,dirs(i).name);
            if isfile(fullfile(candidate,'run_meta.csv'))
                runFolders{end+1,1} = candidate; %#ok<AGROW>
            end
        end
    else
        folders = cellstr(string(inputPath(:)));
        runFolders = folders(isfolder(folders));
        runFolders = runFolders(cellfun(@(p)isfile(fullfile(p,'run_meta.csv')),runFolders));
    end

    assert(~isempty(runFolders), ...
        'summarize_PRBCCMO_t_data:NoRunFolders', ...
        'No PRBCCMO_t run folders were found in the provided input.');
end

function SummaryTable = aggregateTraceTable(RunTable,groupField)
    keys = unique(RunTable.(groupField),'stable');
    MetricNames = { ...
        'final_boundary_lowmargin_pair_hit', ...
        'final_boundary_seg_cross_dist', ...
        'final_boundary_margin_oppdist_corr', ...
        'final_boundary_lowmargin_oppdist_gain', ...
        'final_boundary_survive_c_rate', ...
        'final_boundary_survive_u_rate', ...
        'final_boundary_off_count', ...
        'final_boundary_evidence_count', ...
        'final_boundary_evidence_feasible_ratio', ...
        'final_b_size', ...
        'final_b_lowmargin_ratio', ...
        'final_b_mean_dist_to_true_boundary', ...
        'final_b_p90_dist_to_true_boundary', ...
        'final_boundary_off_hit_rate_eps', ...
        'final_archive_hit_rate_eps', ...
        'final_b_margin_true_boundary_corr', ...
        'final_b_margin_oppdist_corr', ...
        'final_b_lowmargin_oppdist_gain', ...
        'final_inf_lowmargin_ratio_gain', ...
        'final_inf_margin_gain', ...
        'final_inf_obj_score_gain', ...
        'final_train_boundary_evidence_count', ...
        'final_train_pair_count', ...
        'final_train_bal_acc', ...
        'stable3_generation'};
    MetricColumns = traceAggregateColumnNames(MetricNames);
    if groupField == "problem"
        Names = [{'problem','family','num_runs'},MetricColumns];
        Types = [{'string','string'},repmat({'double'},1,numel(Names)-2)];
    else
        Names = [{'family','num_runs'},MetricColumns];
        Types = [{'string'},repmat({'double'},1,numel(Names)-1)];
    end
    SummaryTable = table('Size',[numel(keys),numel(Names)], ...
        'VariableTypes',Types,'VariableNames',Names);

    for i = 1 : numel(keys)
        idx = RunTable.(groupField) == keys(i);
        if groupField == "problem"
            SummaryTable.problem(i) = keys(i);
            SummaryTable.family(i)  = RunTable.family(find(idx,1,'first'));
        else
            SummaryTable.family(i) = keys(i);
        end
        SummaryTable.num_runs(i) = sum(idx);
        firstMetricColumn = 3 + double(groupField == "problem");
        for m = 1 : numel(MetricNames)
            Values = RunTable.(MetricNames{m})(idx);
            SummaryTable{ i, firstMetricColumn + 2*(m-1) } = mean(Values,'omitnan');
            SummaryTable{ i, firstMetricColumn + 2*(m-1) + 1 } = std(Values,0,'omitnan');
        end
    end
end

function Names = traceAggregateColumnNames(MetricNames)
    Names = cell(1,2*numel(MetricNames));
    for i = 1 : numel(MetricNames)
        Names{2*i-1} = ['avg_',MetricNames{i}];
        Names{2*i}   = ['std_',MetricNames{i}];
    end
end
