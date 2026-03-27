function Results = merge_PRBCCMO_step2_selector_acceptance_shards(varargin)
% Merge selector-acceptance shard CSV outputs without MAT artifacts.

    Params = struct( ...
        'BaseDir',DefaultBaseDir(), ...
        'RunIndices',1:8, ...
        'BaseVariant','ParetoOnly', ...
        'TargetVariant','FullSoftTrust', ...
        'TargetDisplayName','FullNew');
    Params = ParseInputs(Params,varargin{:});

    BaseDir = char(Params.BaseDir);
    RunIndices = unique(round(double(Params.RunIndices(:)')),'stable');
    if isempty(RunIndices)
        error('PRBCCMO:SelectorAcceptanceMergeInput', ...
            'RunIndices must not be empty.');
    end

    QueryRuns = table();
    QueryUpdates = table();
    CalibratorRuns = table();
    for RunIndex = RunIndices
        ShardDir = fullfile(BaseDir,sprintf('shard_%02d',RunIndex));
        QueryRuns = AppendTable(QueryRuns,ReadShardTable(ShardDir,'step2_query_runs.csv'));
        QueryUpdates = AppendTable(QueryUpdates,ReadShardTable(ShardDir,'step2_query_updates.csv'));
        CalibratorRuns = AppendTable(CalibratorRuns,ReadShardTable(ShardDir,'step2_calibrator_runs.csv'));
    end

    RunBoard = BuildRunBoard(QueryRuns,QueryUpdates,Params);
    ProblemBoard = BuildProblemBoard(RunBoard,Params);
    FamilyBoard = BuildFamilyBoard(ProblemBoard,Params);

    writetable(QueryRuns,fullfile(BaseDir,'step2_query_runs.csv'));
    writetable(QueryUpdates,fullfile(BaseDir,'step2_query_updates.csv'));
    writetable(CalibratorRuns,fullfile(BaseDir,'step2_calibrator_runs.csv'));
    writetable(RunBoard,fullfile(BaseDir,'step2_selector_accept_run_board.csv'));
    writetable(ProblemBoard,fullfile(BaseDir,'step2_selector_accept_problem_board.csv'));
    writetable(FamilyBoard,fullfile(BaseDir,'step2_selector_accept_family_board.csv'));
    WriteDone(fullfile(BaseDir,'DONE.txt'));

    Results = struct();
    Results.params = Params;
    Results.queryRuns = QueryRuns;
    Results.queryUpdates = QueryUpdates;
    Results.calibratorRuns = CalibratorRuns;
    Results.runBoard = RunBoard;
    Results.problemBoard = ProblemBoard;
    Results.familyBoard = FamilyBoard;
end

function BaseDir = DefaultBaseDir()
    Here = fileparts(mfilename('fullpath'));
    BaseDir = fullfile(Here,'results_step2_selector_acceptance_r8_20260326');
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('PRBCCMO:SelectorAcceptanceMergeInput', ...
            'Inputs must be provided as name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~isfield(Params,Name)
            error('PRBCCMO:SelectorAcceptanceMergeInput', ...
                'Unknown parameter ''%s''.',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function Data = ReadShardTable(ShardDir,FileName)
    FilePath = fullfile(ShardDir,FileName);
    if exist(FilePath,'file') ~= 2
        error('PRBCCMO:SelectorAcceptanceMergeMissingShard', ...
            'Missing shard table: %s',FilePath);
    end
    Data = readtable(FilePath,'TextType','string');
end

function Out = AppendTable(Left,Right)
    if isempty(Left)
        Out = Right;
        return;
    end
    if isempty(Right)
        Out = Left;
        return;
    end
    Out = [Left; Right]; %#ok<AGROW>
end

function RunBoard = BuildRunBoard(QueryRuns,QueryUpdates,Params)
    TargetRows = QueryRuns(strcmp(QueryRuns.queryVariant,Params.TargetVariant),:);
    if isempty(TargetRows)
        RunBoard = InitRunBoardTable();
        return;
    end

    RunBoard = table();
    RunBoard.problem = TargetRows.problem;
    RunBoard.calibratorVariant = TargetRows.calibratorVariant;
    RunBoard.baseVariant = repmat(string(Params.BaseVariant),height(TargetRows),1);
    RunBoard.targetVariant = repmat(string(Params.TargetVariant),height(TargetRows),1);
    RunBoard.targetDisplayName = repmat(string(Params.TargetDisplayName),height(TargetRows),1);
    RunBoard.run = TargetRows.run;
    RunBoard.selectionDiffPoolRate = TargetRows.selectionDiffPoolRate;
    RunBoard.oracleAuditEnabledRate = TargetRows.oracleAuditEnabled;
    RunBoard.medianSelectionOverlapPareto = TargetRows.medianSelectionOverlapPareto;
    RunBoard.stopGoReason = TargetRows.stopGoReason;
    RunBoard.auditedUpdateCount = zeros(height(TargetRows),1);

    UpdateTarget = QueryUpdates(strcmp(QueryUpdates.queryVariant,Params.TargetVariant),:);
    for i = 1 : height(TargetRows)
        Mask = strcmp(UpdateTarget.problem,TargetRows.problem(i)) & ...
            strcmp(UpdateTarget.calibratorVariant,TargetRows.calibratorVariant(i)) & ...
            UpdateTarget.run == TargetRows.run(i);
        if any(Mask)
            RunBoard.auditedUpdateCount(i) = sum(UpdateTarget.oracleAuditEnabled(Mask) ~= 0);
        end
    end
end

function ProblemBoard = BuildProblemBoard(RunBoard,Params)
    ProblemBoard = InitProblemBoardTable();
    if isempty(RunBoard)
        return;
    end

    [Groups,Problems,Calibrators] = findgroups(RunBoard.problem,RunBoard.calibratorVariant);
    ProblemBoard = repmat(InitProblemBoardRow(),max(Groups),1);
    for g = 1 : max(Groups)
        Mask = Groups == g;
        Subset = RunBoard(Mask,:);
        Row = InitProblemBoardRow();
        Row.problem = char(Problems(g));
        Row.calibratorVariant = char(Calibrators(g));
        Row.baseVariant = char(Params.BaseVariant);
        Row.targetVariant = char(Params.TargetVariant);
        Row.targetDisplayName = char(Params.TargetDisplayName);
        Row.runCount = height(Subset);
        Row.medianSelectionDiffPoolRate = median(Subset.selectionDiffPoolRate,'omitnan');
        Row.oracleAuditEnabledRate = mean(Subset.oracleAuditEnabledRate,'omitnan');
        Row.medianSelectionOverlapPareto = median(Subset.medianSelectionOverlapPareto,'omitnan');
        Row.medianAuditedUpdateCount = median(Subset.auditedUpdateCount,'omitnan');
        Row.meanAuditedUpdateCount = mean(Subset.auditedUpdateCount,'omitnan');
        Row.auditedPositiveRunCount = sum(Subset.auditedUpdateCount > 0);
        Row.auditedUpdateCountSignrankP = SafeSignrankGreaterZero(Subset.auditedUpdateCount);
        Row.dominantStopGoReason = char(ModeString(Subset.stopGoReason));
        Row.passSelectionDiff = isfinite(Row.medianSelectionDiffPoolRate) && ...
            Row.medianSelectionDiffPoolRate > 0.05;
        Row.passOracleAudit = isfinite(Row.oracleAuditEnabledRate) && ...
            Row.oracleAuditEnabledRate > 0.30;
        Row.passAuditedUpdates = isfinite(Row.medianAuditedUpdateCount) && ...
            Row.medianAuditedUpdateCount > 0 && ...
            isfinite(Row.auditedUpdateCountSignrankP) && ...
            Row.auditedUpdateCountSignrankP < 0.05;
        Row.stopGoPass = Row.passSelectionDiff && Row.passOracleAudit && Row.passAuditedUpdates;
        ProblemBoard(g,1) = Row; %#ok<AGROW>
    end
    ProblemBoard = struct2table(ProblemBoard,'AsArray',true);
end

function FamilyBoard = BuildFamilyBoard(ProblemBoard,Params)
    FamilyBoard = InitFamilyBoardTable();
    if isempty(ProblemBoard)
        return;
    end

    [Groups,Calibrators] = findgroups(string(ProblemBoard.calibratorVariant));
    FamilyBoard = repmat(InitFamilyBoardRow(),max(Groups),1);
    for g = 1 : max(Groups)
        Mask = Groups == g;
        Subset = ProblemBoard(Mask,:);
        ProblemCount = height(Subset);
        Row = InitFamilyBoardRow();
        Row.calibratorVariant = char(Calibrators(g));
        Row.baseVariant = char(Params.BaseVariant);
        Row.targetVariant = char(Params.TargetVariant);
        Row.targetDisplayName = char(Params.TargetDisplayName);
        Row.problemCount = ProblemCount;
        Row.passSelectionDiffProblemCount = sum(Subset.passSelectionDiff);
        Row.passOracleAuditProblemCount = sum(Subset.passOracleAudit);
        Row.passAuditedProblemCount = sum(Subset.passAuditedUpdates);
        Row.stopGoPassProblemCount = sum(Subset.stopGoPass);
        Row.majorityStopGoPass = Row.stopGoPassProblemCount > floor(ProblemCount/2);
        FamilyBoard(g,1) = Row; %#ok<AGROW>
    end
    FamilyBoard = struct2table(FamilyBoard,'AsArray',true);
end

function Value = SafeSignrankGreaterZero(Data)
    Value = NaN;
    if isempty(Data) || exist('signrank','file') ~= 2
        return;
    end
    Data = Data(isfinite(Data));
    if isempty(Data)
        return;
    end
    if ~any(abs(Data) > eps(max(abs(Data),1)))
        Value = 1;
        return;
    end
    try
        Value = signrank(Data,0,'tail','right');
    catch
        Value = NaN;
    end
end

function Value = ModeString(Data)
    Data = string(Data);
    if isempty(Data)
        Value = "";
        return;
    end
    Data(ismissing(Data)) = "";
    Keys = unique(Data,'stable');
    Counts = zeros(numel(Keys),1);
    for i = 1 : numel(Keys)
        Counts(i) = sum(Data == Keys(i));
    end
    [~,Idx] = max(Counts);
    Value = Keys(Idx);
end

function T = InitRunBoardTable()
    T = table( ...
        strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),zeros(0,1), ...
        NaN(0,1),NaN(0,1),NaN(0,1),strings(0,1),zeros(0,1), ...
        'VariableNames',{'problem','calibratorVariant','baseVariant','targetVariant', ...
        'targetDisplayName','run','selectionDiffPoolRate','oracleAuditEnabledRate', ...
        'medianSelectionOverlapPareto','stopGoReason','auditedUpdateCount'});
end

function Row = InitProblemBoardRow()
    Row = struct( ...
        'problem','', ...
        'calibratorVariant','', ...
        'baseVariant','', ...
        'targetVariant','', ...
        'targetDisplayName','', ...
        'runCount',0, ...
        'medianSelectionDiffPoolRate',NaN, ...
        'oracleAuditEnabledRate',NaN, ...
        'medianSelectionOverlapPareto',NaN, ...
        'medianAuditedUpdateCount',NaN, ...
        'meanAuditedUpdateCount',NaN, ...
        'auditedPositiveRunCount',0, ...
        'auditedUpdateCountSignrankP',NaN, ...
        'dominantStopGoReason','', ...
        'passSelectionDiff',false, ...
        'passOracleAudit',false, ...
        'passAuditedUpdates',false, ...
        'stopGoPass',false);
end

function T = InitProblemBoardTable()
    T = struct2table(repmat(InitProblemBoardRow(),0,1),'AsArray',true);
end

function Row = InitFamilyBoardRow()
    Row = struct( ...
        'calibratorVariant','', ...
        'baseVariant','', ...
        'targetVariant','', ...
        'targetDisplayName','', ...
        'problemCount',0, ...
        'passSelectionDiffProblemCount',0, ...
        'passOracleAuditProblemCount',0, ...
        'passAuditedProblemCount',0, ...
        'stopGoPassProblemCount',0, ...
        'majorityStopGoPass',false);
end

function T = InitFamilyBoardTable()
    T = struct2table(repmat(InitFamilyBoardRow(),0,1),'AsArray',true);
end

function WriteDone(FilePath)
    FID = fopen(FilePath,'w');
    if FID < 0
        error('PRBCCMO:SelectorAcceptanceMergeIO', ...
            'Unable to open DONE file: %s',FilePath);
    end
    Cleanup = onCleanup(@() fclose(FID));
    fprintf(FID,'completed=%s\n',datestr(now,31));
    clear Cleanup;
end
