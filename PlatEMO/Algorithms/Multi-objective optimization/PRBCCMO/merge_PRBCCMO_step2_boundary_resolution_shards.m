function Results = merge_PRBCCMO_step2_boundary_resolution_shards(varargin)
% Merge boundary-resolution shard CSV outputs without MAT artifacts.

    Params = struct( ...
        'BaseDir',DefaultBaseDir(), ...
        'RunIndices',1:8, ...
        'LowBoundaryStdThreshold',0.05, ...
        'HighPositiveRankCorrelationThreshold',0.85, ...
        'HighTopOverlapThreshold',0.75);
    Params = ParseInputs(Params,varargin{:});

    BaseDir = char(Params.BaseDir);
    RunIndices = unique(round(double(Params.RunIndices(:)')),'stable');
    if isempty(RunIndices)
        error('PRBCCMO:BoundaryResolutionMergeInput', ...
            'RunIndices must not be empty.');
    end

    UpdateTable = table();
    RunTable = table();
    for RunIndex = RunIndices
        ShardDir = fullfile(BaseDir,sprintf('shard_%02d',RunIndex));
        UpdateTable = AppendTable(UpdateTable,ReadShardTable(ShardDir,'boundary_resolution_updates.csv'));
        RunTable = AppendTable(RunTable,ReadShardTable(ShardDir,'boundary_resolution_runs.csv'));
    end

    ProblemBoard = BuildProblemBoard(RunTable,Params);
    SummaryBoard = BuildSummaryBoard(ProblemBoard);

    writetable(UpdateTable,fullfile(BaseDir,'boundary_resolution_updates.csv'));
    writetable(RunTable,fullfile(BaseDir,'boundary_resolution_runs.csv'));
    writetable(ProblemBoard,fullfile(BaseDir,'boundary_resolution_problem_board.csv'));
    writetable(SummaryBoard,fullfile(BaseDir,'boundary_resolution_summary_board.csv'));
    WriteDone(fullfile(BaseDir,'DONE.txt'));

    Results = struct();
    Results.updateTable = UpdateTable;
    Results.runTable = RunTable;
    Results.problemBoard = ProblemBoard;
    Results.summaryBoard = SummaryBoard;
end

function BaseDir = DefaultBaseDir()
    Here = fileparts(mfilename('fullpath'));
    BaseDir = fullfile(Here,'results_step2_boundary_resolution_r8_20260326');
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('PRBCCMO:BoundaryResolutionMergeInput', ...
            'Inputs must be provided as name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~isfield(Params,Name)
            error('PRBCCMO:BoundaryResolutionMergeInput', ...
                'Unknown parameter ''%s''.',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function Data = ReadShardTable(ShardDir,FileName)
    FilePath = fullfile(ShardDir,FileName);
    if exist(FilePath,'file') ~= 2
        error('PRBCCMO:BoundaryResolutionMergeMissingShard', ...
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

function ProblemBoard = BuildProblemBoard(RunTable,Params)
    ProblemBoard = InitProblemBoardTable();
    if isempty(RunTable)
        return;
    end

    [Groups,Problems,Calibrators,Shortlists] = findgroups( ...
        string(RunTable.problem),string(RunTable.calibratorVariant),RunTable.shortlistFactor);
    Rows = repmat(InitProblemBoardRow(),max(Groups),1);
    for g = 1 : max(Groups)
        Mask = Groups == g;
        Subset = RunTable(Mask,:);
        Row = InitProblemBoardRow();
        Row.problem = char(Problems(g));
        Row.calibratorVariant = char(Calibrators(g));
        Row.shortlistFactor = Shortlists(g);
        Row.runCount = height(Subset);
        Row.medianRunBoundaryStdEligible = median(Subset.medianBoundaryStdEligible,'omitnan');
        Row.medianRunBoundaryStdShortlist = median(Subset.medianBoundaryStdShortlist,'omitnan');
        Row.medianRunQueryScoreStdShortlist = median(Subset.medianQueryScoreStdShortlist,'omitnan');
        Row.medianRunReliabilityStdShortlist = median(Subset.medianReliabilityStdShortlist,'omitnan');
        Row.medianRunDisagreementStdShortlist = median(Subset.medianDisagreementStdShortlist,'omitnan');
        Row.medianRunSpearmanBoundaryParetoShortlist = median(Subset.medianSpearmanBoundaryParetoShortlist,'omitnan');
        Row.medianRunAbsSpearmanBoundaryParetoShortlist = median(Subset.medianAbsSpearmanBoundaryParetoShortlist,'omitnan');
        Row.meanRunSpearmanBoundaryParetoShortlist = mean(Subset.meanSpearmanBoundaryParetoShortlist,'omitnan');
        Row.medianRunTopBoundaryParetoOverlap = median(Subset.medianTopBoundaryParetoOverlap,'omitnan');
        Row.meanRunTopBoundaryParetoOverlap = mean(Subset.meanTopBoundaryParetoOverlap,'omitnan');
        Row.meanRunTopBoundaryParetoDiffRate = mean(Subset.topBoundaryParetoDiffRate,'omitnan');
        Row.meanRunTrustGateRate = mean(Subset.trustGateRate,'omitnan');
        Row.meanRunECE = mean(Subset.meanECE,'omitnan');
        Row.meanRunCoreNearGap = mean(Subset.meanCoreNearGap,'omitnan');
        Row.lowBoundaryVariance = isfinite(Row.medianRunBoundaryStdShortlist) && ...
            Row.medianRunBoundaryStdShortlist <= Params.LowBoundaryStdThreshold;
        Row.highPositiveRankCorrelation = isfinite(Row.medianRunSpearmanBoundaryParetoShortlist) && ...
            Row.medianRunSpearmanBoundaryParetoShortlist >= Params.HighPositiveRankCorrelationThreshold;
        Row.highTopOverlap = isfinite(Row.medianRunTopBoundaryParetoOverlap) && ...
            Row.medianRunTopBoundaryParetoOverlap >= Params.HighTopOverlapThreshold;
        Row.weakResolution = Row.lowBoundaryVariance || ...
            (Row.highPositiveRankCorrelation && Row.highTopOverlap);
        Rows(g,1) = Row; %#ok<AGROW>
    end
    ProblemBoard = struct2table(Rows,'AsArray',true);
end

function SummaryBoard = BuildSummaryBoard(ProblemBoard)
    SummaryBoard = InitSummaryBoardTable();
    if isempty(ProblemBoard)
        return;
    end

    [Groups,Calibrators,Shortlists] = findgroups( ...
        string(ProblemBoard.calibratorVariant),ProblemBoard.shortlistFactor);
    Rows = repmat(InitSummaryBoardRow(),max(Groups),1);
    for g = 1 : max(Groups)
        Mask = Groups == g;
        Subset = ProblemBoard(Mask,:);
        Row = InitSummaryBoardRow();
        Row.calibratorVariant = char(Calibrators(g));
        Row.shortlistFactor = Shortlists(g);
        Row.problemCount = height(Subset);
        Row.lowBoundaryVarianceProblemCount = sum(Subset.lowBoundaryVariance);
        Row.highPositiveRankCorrelationProblemCount = sum(Subset.highPositiveRankCorrelation);
        Row.highTopOverlapProblemCount = sum(Subset.highTopOverlap);
        Row.weakResolutionProblemCount = sum(Subset.weakResolution);
        Row.medianProblemBoundaryStdShortlist = median(Subset.medianRunBoundaryStdShortlist,'omitnan');
        Row.medianProblemSpearmanBoundaryParetoShortlist = median(Subset.medianRunSpearmanBoundaryParetoShortlist,'omitnan');
        Row.medianProblemTopBoundaryParetoOverlap = median(Subset.medianRunTopBoundaryParetoOverlap,'omitnan');
        Row.meanProblemTopBoundaryParetoDiffRate = mean(Subset.meanRunTopBoundaryParetoDiffRate,'omitnan');
        Row.meanProblemTrustGateRate = mean(Subset.meanRunTrustGateRate,'omitnan');
        Rows(g,1) = Row; %#ok<AGROW>
    end
    SummaryBoard = struct2table(Rows,'AsArray',true);
end

function Row = InitProblemBoardRow()
    Row = struct( ...
        'problem','', ...
        'calibratorVariant','', ...
        'shortlistFactor',NaN, ...
        'runCount',0, ...
        'medianRunBoundaryStdEligible',NaN, ...
        'medianRunBoundaryStdShortlist',NaN, ...
        'medianRunQueryScoreStdShortlist',NaN, ...
        'medianRunReliabilityStdShortlist',NaN, ...
        'medianRunDisagreementStdShortlist',NaN, ...
        'medianRunSpearmanBoundaryParetoShortlist',NaN, ...
        'medianRunAbsSpearmanBoundaryParetoShortlist',NaN, ...
        'meanRunSpearmanBoundaryParetoShortlist',NaN, ...
        'medianRunTopBoundaryParetoOverlap',NaN, ...
        'meanRunTopBoundaryParetoOverlap',NaN, ...
        'meanRunTopBoundaryParetoDiffRate',NaN, ...
        'meanRunTrustGateRate',NaN, ...
        'meanRunECE',NaN, ...
        'meanRunCoreNearGap',NaN, ...
        'lowBoundaryVariance',false, ...
        'highPositiveRankCorrelation',false, ...
        'highTopOverlap',false, ...
        'weakResolution',false);
end

function T = InitProblemBoardTable()
    T = struct2table(repmat(InitProblemBoardRow(),0,1),'AsArray',true);
end

function Row = InitSummaryBoardRow()
    Row = struct( ...
        'calibratorVariant','', ...
        'shortlistFactor',NaN, ...
        'problemCount',0, ...
        'lowBoundaryVarianceProblemCount',0, ...
        'highPositiveRankCorrelationProblemCount',0, ...
        'highTopOverlapProblemCount',0, ...
        'weakResolutionProblemCount',0, ...
        'medianProblemBoundaryStdShortlist',NaN, ...
        'medianProblemSpearmanBoundaryParetoShortlist',NaN, ...
        'medianProblemTopBoundaryParetoOverlap',NaN, ...
        'meanProblemTopBoundaryParetoDiffRate',NaN, ...
        'meanProblemTrustGateRate',NaN);
end

function T = InitSummaryBoardTable()
    T = struct2table(repmat(InitSummaryBoardRow(),0,1),'AsArray',true);
end

function WriteDone(FilePath)
    FID = fopen(FilePath,'w');
    if FID < 0
        error('PRBCCMO:BoundaryResolutionMergeIO', ...
            'Unable to open DONE file: %s',FilePath);
    end
    Cleanup = onCleanup(@() fclose(FID));
    fprintf(FID,'completed=%s\n',datestr(now,31));
    clear Cleanup;
end
