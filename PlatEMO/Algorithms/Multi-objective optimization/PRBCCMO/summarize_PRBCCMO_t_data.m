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
    if groupField == "problem"
        Names = { ...
            'problem','family','num_runs', ...
            'mean_final_boundary_selected','mean_final_boundary_sector_coverage','mean_final_boundary_lowmargin_pair_hit', ...
            'mean_final_boundary_seg_cross_dist','mean_final_b_size','mean_final_b_sector_coverage', ...
            'mean_final_b_lowmargin_pair_hit','mean_final_train_pair_count','mean_final_train_bal_acc', ...
            'mean_final_train_minus_boundary_bal_gap','mean_final_train_minus_b_bal_gap', ...
            'stable3_hit_rate','mean_first_can_train_generation','mean_first_trained_generation', ...
            'mean_helper_real_opp_ratio','mean_helper_skip_no_helper','mean_boundary_attempts'};
        Types = [{'string','string'},repmat({'double'},1,numel(Names)-2)];
    else
        Names = { ...
            'family','num_runs', ...
            'mean_final_boundary_selected','mean_final_boundary_sector_coverage','mean_final_boundary_lowmargin_pair_hit', ...
            'mean_final_boundary_seg_cross_dist','mean_final_b_size','mean_final_b_sector_coverage', ...
            'mean_final_b_lowmargin_pair_hit','mean_final_train_pair_count','mean_final_train_bal_acc', ...
            'mean_final_train_minus_boundary_bal_gap','mean_final_train_minus_b_bal_gap', ...
            'stable3_hit_rate','mean_first_can_train_generation','mean_first_trained_generation', ...
            'mean_helper_real_opp_ratio','mean_helper_skip_no_helper','mean_boundary_attempts'};
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
        SummaryTable.mean_final_boundary_selected(i) = mean(RunTable.final_boundary_selected(idx),'omitnan');
        SummaryTable.mean_final_boundary_sector_coverage(i) = mean(RunTable.final_boundary_sector_coverage(idx),'omitnan');
        SummaryTable.mean_final_boundary_lowmargin_pair_hit(i) = mean(RunTable.final_boundary_lowmargin_pair_hit(idx),'omitnan');
        SummaryTable.mean_final_boundary_seg_cross_dist(i) = mean(RunTable.final_boundary_seg_cross_dist(idx),'omitnan');
        SummaryTable.mean_final_b_size(i) = mean(RunTable.final_b_size(idx),'omitnan');
        SummaryTable.mean_final_b_sector_coverage(i) = mean(RunTable.final_b_sector_coverage(idx),'omitnan');
        SummaryTable.mean_final_b_lowmargin_pair_hit(i) = mean(RunTable.final_b_lowmargin_pair_hit(idx),'omitnan');
        SummaryTable.mean_final_train_pair_count(i) = mean(RunTable.final_train_pair_count(idx),'omitnan');
        SummaryTable.mean_final_train_bal_acc(i) = mean(RunTable.final_train_bal_acc(idx),'omitnan');
        SummaryTable.mean_final_train_minus_boundary_bal_gap(i) = mean(RunTable.final_train_minus_boundary_bal_gap(idx),'omitnan');
        SummaryTable.mean_final_train_minus_b_bal_gap(i) = mean(RunTable.final_train_minus_b_bal_gap(idx),'omitnan');
        SummaryTable.stable3_hit_rate(i) = mean(~isnan(RunTable.stable3_generation(idx)));
        SummaryTable.mean_first_can_train_generation(i) = mean(RunTable.first_can_train_generation(idx),'omitnan');
        SummaryTable.mean_first_trained_generation(i) = mean(RunTable.first_trained_generation(idx),'omitnan');
        SummaryTable.mean_helper_real_opp_ratio(i) = mean(RunTable.mean_helper_real_opp_ratio(idx),'omitnan');
        SummaryTable.mean_helper_skip_no_helper(i) = mean(RunTable.mean_helper_skip_no_helper(idx),'omitnan');
        SummaryTable.mean_boundary_attempts(i) = mean(RunTable.mean_boundary_attempts(idx),'omitnan');
    end
end
