function Results = merge_PRBCCMO_step2_phaseA_shards(varargin)
% Merge Phase A Step-2 shard outputs back into full paired statistics.

    Params = struct( ...
        'BaseDir',DefaultBaseDir(), ...
        'RunIndices',1:10, ...
        'DeleteShardMat',true);
    Params = ParseInputs(Params,varargin{:});

    BaseDir = char(Params.BaseDir);
    RunIndices = unique(round(double(Params.RunIndices(:)')),'stable');
    if isempty(RunIndices)
        error('PRBCCMO:PhaseAMergeInput', ...
            'RunIndices must not be empty.');
    end

    QueryReports = [];
    QueryUpdateReports = [];
    CalibratorReports = [];
    OracleMeta = [];
    Template = [];
    for RunIndex = RunIndices
        ShardDir = fullfile(BaseDir,sprintf('shard_%02d',RunIndex));
        ResultFile = fullfile(ShardDir,'tmp','step2_results.mat');
        if exist(ResultFile,'file') ~= 2
            error('PRBCCMO:PhaseAMergeMissingShard', ...
                'Missing shard result: %s',ResultFile);
        end
        Data = load(ResultFile,'Results');
        Shard = Data.Results;
        if isempty(Template)
            Template = Shard;
        end
        QueryReports = AppendStructArray(QueryReports,Shard.queryReport);
        QueryUpdateReports = AppendStructArray(QueryUpdateReports,Shard.queryUpdateReport);
        CalibratorReports = AppendStructArray(CalibratorReports,Shard.calibratorReport);
        OracleMeta = AppendOracleMeta(OracleMeta,Shard.oracle);
        if Params.DeleteShardMat
            delete(ResultFile);
            CleanupTmpDir(fullfile(ShardDir,'tmp'));
        end
    end

    Results = struct();
    Results.params = Template.params;
    Results.params.RunIndices = RunIndices;
    Results.oracle = OracleMeta;
    Results.queryReport = QueryReports;
    Results.queryUpdateReport = QueryUpdateReports;
    Results.calibratorReport = CalibratorReports;
    Results.querySummary = SummarizeQueryReports(QueryReports, ...
        Results.params.QueryVariants,Results.params.CalibratorVariants);
    Results.calibratorSummary = SummarizeCalibratorReports(CalibratorReports, ...
        Results.params.CalibratorVariants);
    Results.queryPaired = BuildQueryPairedSummary(QueryReports, ...
        Results.params.QueryVariants,Results.params.CalibratorVariants);
    Results.calibratorPaired = BuildCalibratorPairedSummary(CalibratorReports, ...
        Results.params.CalibratorVariants);

    writetable(struct2table(Results.queryReport,'AsArray',true), ...
        fullfile(BaseDir,'step2_query_runs.csv'));
    writetable(struct2table(Results.queryUpdateReport,'AsArray',true), ...
        fullfile(BaseDir,'step2_query_updates.csv'));
    writetable(struct2table(Results.calibratorReport,'AsArray',true), ...
        fullfile(BaseDir,'step2_calibrator_runs.csv'));
    writetable(struct2table(Results.querySummary,'AsArray',true), ...
        fullfile(BaseDir,'step2_query_summary.csv'));
    writetable(struct2table(Results.calibratorSummary,'AsArray',true), ...
        fullfile(BaseDir,'step2_calibrator_summary.csv'));
    writetable(struct2table(Results.queryPaired,'AsArray',true), ...
        fullfile(BaseDir,'step2_query_paired.csv'));
    writetable(struct2table(Results.calibratorPaired,'AsArray',true), ...
        fullfile(BaseDir,'step2_calibrator_paired.csv'));
    WriteDone(fullfile(BaseDir,'DONE.txt'));
end

function BaseDir = DefaultBaseDir()
    Here = fileparts(mfilename('fullpath'));
    BaseDir = fullfile(Here,'results_step2_dascmop9_r10_phaseA_parallel_20260326');
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('PRBCCMO:PhaseAMergeInput', ...
            'Inputs must be provided as name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~isfield(Params,Name)
            error('PRBCCMO:PhaseAMergeInput', ...
                'Unknown parameter ''%s''.',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function Data = AppendStructArray(Data,Block)
    if isempty(Block)
        if isempty(Data) && isstruct(Block)
            Data = Block;
        end
        return;
    end
    if isempty(Data)
        Data = Block;
    else
        Data(end + (1:numel(Block)),1) = Block;
    end
end

function OracleMeta = AppendOracleMeta(OracleMeta,Block)
    if isempty(Block)
        return;
    end
    if isempty(OracleMeta)
        OracleMeta = Block;
        return;
    end
    Existing = {OracleMeta.problem};
    for i = 1 : numel(Block)
        if ~any(strcmp(Existing,Block(i).problem))
            OracleMeta(end+1,1) = Block(i); %#ok<AGROW>
            Existing{end+1} = Block(i).problem; %#ok<AGROW>
        end
    end
end

function CleanupTmpDir(TmpDir)
    if exist(TmpDir,'dir') ~= 7
        return;
    end
    Files = dir(TmpDir);
    Names = {Files.name};
    Names = Names(~ismember(Names,{'.','..'}));
    if isempty(Names)
        rmdir(TmpDir);
    end
end

function Summary = SummarizeQueryReports(QueryReports,QueryVariants,CalibratorVariants)
    Summary = repmat(InitQuerySummaryRow(),0,1);
    for c = 1 : numel(CalibratorVariants)
        for q = 1 : numel(QueryVariants)
            Mask = strcmp({QueryReports.calibratorVariant},CalibratorVariants{c}) & ...
                strcmp({QueryReports.queryVariant},QueryVariants{q});
            Subset = QueryReports(Mask);
            if isempty(Subset)
                continue;
            end
            Row = InitQuerySummaryRow();
            Row.calibratorVariant = CalibratorVariants{c};
            Row.queryVariant = QueryVariants{q};
            Row.runCount = numel(Subset);
            Row.meanDB = MeanStructField(Subset,'meanDB');
            Row.medianDB = MeanStructField(Subset,'medianDB');
            Row.QPTau = MeanStructField(Subset,'QPTau');
            Row.rhoUtilityNegDB = MeanStructField(Subset,'rhoUtilityNegDB');
            Row.selectedCount = MeanStructField(Subset,'selectedCount');
            Row.oracleAuditEnabledRate = MeanStructField(Subset,'oracleAuditEnabled');
            Row.meanSelectionOverlapPareto = MeanStructField(Subset,'meanSelectionOverlapPareto');
            Row.medianSelectionOverlapPareto = MeanStructField(Subset,'medianSelectionOverlapPareto');
            Row.selectionDiffPoolCount = MeanStructField(Subset,'selectionDiffPoolCount');
            Row.selectionDiffPoolRate = MeanStructField(Subset,'selectionDiffPoolRate');
            Row.meanBoundaryScoreDispersion = MeanStructField(Subset,'meanBoundaryScoreDispersion');
            Row.medianBoundaryScoreDispersion = MeanStructField(Subset,'medianBoundaryScoreDispersion');
            Summary(end+1,1) = Row; %#ok<AGROW>
        end
    end
end

function Summary = SummarizeCalibratorReports(CalibratorReports,CalibratorVariants)
    Summary = repmat(InitCalibratorSummaryRow(),0,1);
    for c = 1 : numel(CalibratorVariants)
        Mask = strcmp({CalibratorReports.calibratorVariant},CalibratorVariants{c});
        Subset = CalibratorReports(Mask);
        if isempty(Subset)
            continue;
        end
        Row = InitCalibratorSummaryRow();
        Row.calibratorVariant = CalibratorVariants{c};
        Row.runCount = numel(Subset);
        Row.meanECE = MeanStructField(Subset,'meanECE');
        Row.meanCoreNearGap = MeanStructField(Subset,'meanCoreNearGap');
        Row.TWS = MeanStructField(Subset,'TWS');
        Row.TGP = MeanStructField(Subset,'TGP');
        Row.auditReadyUpdateCount = MeanStructField(Subset,'auditReadyUpdateCount');
        Summary(end+1,1) = Row; %#ok<AGROW>
    end
end

function Rows = BuildQueryPairedSummary(QueryReports,QueryVariants,CalibratorVariants)
    Metrics = {'meanSelectionOverlapPareto','medianSelectionOverlapPareto', ...
        'selectionDiffPoolRate','meanDB','medianDB','QPTau','rhoUtilityNegDB'};
    Problems = unique({QueryReports.problem},'stable');
    Rows = repmat(InitQueryPairedSummaryRow(),0,1);
    if numel(QueryVariants) < 2
        return;
    end

    BaseVariant = ResolveBaseQueryVariant(QueryVariants);
    for p = 1 : numel(Problems)
        ProblemName = Problems{p};
        for c = 1 : numel(CalibratorVariants)
            CalibratorVariant = CalibratorVariants{c};
            BaseRows = FilterQueryReports(QueryReports,ProblemName,CalibratorVariant,BaseVariant);
            if isempty(BaseRows)
                continue;
            end
            for q = 1 : numel(QueryVariants)
                CompareVariant = QueryVariants{q};
                if strcmp(CompareVariant,BaseVariant)
                    continue;
                end
                OtherRows = FilterQueryReports(QueryReports,ProblemName,CalibratorVariant,CompareVariant);
                CommonRuns = intersect([BaseRows.run],[OtherRows.run]);
                for m = 1 : numel(Metrics)
                    [A,B] = ExtractPairedMetric(BaseRows,OtherRows,CommonRuns,Metrics{m});
                    Row = InitQueryPairedSummaryRow();
                    Row.problem = ProblemName;
                    Row.calibratorVariant = CalibratorVariant;
                    Row.baseVariant = BaseVariant;
                    Row.compareVariant = CompareVariant;
                    Row.metric = Metrics{m};
                    Row.pairCount = numel(A);
                    Row.baseMedian = median(A,'omitnan');
                    Row.compareMedian = median(B,'omitnan');
                    Row.medianDiff = median(A-B,'omitnan');
                    Row.signrankP = SafeSignrank(A,B);
                    Rows(end+1,1) = Row; %#ok<AGROW>
                end
            end
        end
    end
end

function Rows = BuildCalibratorPairedSummary(CalibratorReports,CalibratorVariants)
    Metrics = {'meanECE','meanCoreNearGap','TWS','TGP'};
    Problems = unique({CalibratorReports.problem},'stable');
    Rows = repmat(InitCalibratorPairedSummaryRow(),0,1);
    if numel(CalibratorVariants) < 2
        return;
    end

    BaseVariant = 'raw';
    for p = 1 : numel(Problems)
        ProblemName = Problems{p};
        BaseRows = FilterCalibratorReports(CalibratorReports,ProblemName,BaseVariant);
        if isempty(BaseRows)
            continue;
        end
        for c = 1 : numel(CalibratorVariants)
            CompareVariant = CalibratorVariants{c};
            if strcmp(CompareVariant,BaseVariant)
                continue;
            end
            OtherRows = FilterCalibratorReports(CalibratorReports,ProblemName,CompareVariant);
            CommonRuns = intersect([BaseRows.run],[OtherRows.run]);
            for m = 1 : numel(Metrics)
                [A,B] = ExtractPairedMetric(BaseRows,OtherRows,CommonRuns,Metrics{m});
                Row = InitCalibratorPairedSummaryRow();
                Row.problem = ProblemName;
                Row.baseVariant = BaseVariant;
                Row.compareVariant = CompareVariant;
                Row.metric = Metrics{m};
                Row.pairCount = numel(A);
                Row.baseMedian = median(A,'omitnan');
                Row.compareMedian = median(B,'omitnan');
                Row.medianDiff = median(A-B,'omitnan');
                Row.signrankP = SafeSignrank(A,B);
                Rows(end+1,1) = Row; %#ok<AGROW>
            end
        end
    end
end

function Rows = FilterQueryReports(QueryReports,ProblemName,CalibratorVariant,QueryVariant)
    Mask = strcmp({QueryReports.problem},ProblemName) & ...
        strcmp({QueryReports.calibratorVariant},CalibratorVariant) & ...
        strcmp({QueryReports.queryVariant},QueryVariant);
    Rows = QueryReports(Mask);
end

function Rows = FilterCalibratorReports(CalibratorReports,ProblemName,CalibratorVariant)
    Mask = strcmp({CalibratorReports.problem},ProblemName) & ...
        strcmp({CalibratorReports.calibratorVariant},CalibratorVariant);
    Rows = CalibratorReports(Mask);
end

function [A,B] = ExtractPairedMetric(LeftRows,RightRows,CommonRuns,MetricName)
    A = zeros(0,1);
    B = zeros(0,1);
    for i = 1 : numel(CommonRuns)
        RunIndex = CommonRuns(i);
        Left = LeftRows([LeftRows.run] == RunIndex);
        Right = RightRows([RightRows.run] == RunIndex);
        if isempty(Left) || isempty(Right)
            continue;
        end
        A(end+1,1) = Left(1).(MetricName); %#ok<AGROW>
        B(end+1,1) = Right(1).(MetricName); %#ok<AGROW>
    end
end

function Value = SafeSignrank(A,B)
    Value = NaN;
    if isempty(A) || isempty(B) || numel(A) ~= numel(B) || exist('signrank','file') ~= 2
        return;
    end
    Valid = isfinite(A) & isfinite(B);
    A = A(Valid);
    B = B(Valid);
    if isempty(A)
        return;
    end
    try
        Value = signrank(A,B);
    catch
        Value = NaN;
    end
end

function Value = MeanStructField(Rows,Field)
    Value = NaN;
    if isempty(Rows)
        return;
    end
    Data = [Rows.(Field)];
    Value = mean(Data,'omitnan');
end

function Name = ResolveBaseQueryVariant(QueryVariants)
    Name = QueryVariants{1};
    if any(strcmp(QueryVariants,'ParetoOnly'))
        Name = 'ParetoOnly';
    end
end

function Row = InitQuerySummaryRow()
    Row = struct( ...
        'calibratorVariant','', ...
        'queryVariant','', ...
        'runCount',0, ...
        'meanDB',NaN, ...
        'medianDB',NaN, ...
        'QPTau',NaN, ...
        'rhoUtilityNegDB',NaN, ...
        'selectedCount',NaN, ...
        'oracleAuditEnabledRate',NaN, ...
        'meanSelectionOverlapPareto',NaN, ...
        'medianSelectionOverlapPareto',NaN, ...
        'selectionDiffPoolCount',NaN, ...
        'selectionDiffPoolRate',NaN, ...
        'meanBoundaryScoreDispersion',NaN, ...
        'medianBoundaryScoreDispersion',NaN);
end

function Row = InitCalibratorSummaryRow()
    Row = struct( ...
        'calibratorVariant','', ...
        'runCount',0, ...
        'meanECE',NaN, ...
        'meanCoreNearGap',NaN, ...
        'TWS',NaN, ...
        'TGP',NaN, ...
        'auditReadyUpdateCount',NaN);
end

function Row = InitQueryPairedSummaryRow()
    Row = struct( ...
        'problem','', ...
        'calibratorVariant','', ...
        'baseVariant','', ...
        'compareVariant','', ...
        'metric','', ...
        'pairCount',0, ...
        'baseMedian',NaN, ...
        'compareMedian',NaN, ...
        'medianDiff',NaN, ...
        'signrankP',NaN);
end

function Row = InitCalibratorPairedSummaryRow()
    Row = struct( ...
        'problem','', ...
        'baseVariant','', ...
        'compareVariant','', ...
        'metric','', ...
        'pairCount',0, ...
        'baseMedian',NaN, ...
        'compareMedian',NaN, ...
        'medianDiff',NaN, ...
        'signrankP',NaN);
end

function WriteDone(FilePath)
    FID = fopen(FilePath,'w');
    if FID < 0
        error('PRBCCMO:PhaseAMergeIO', ...
            'Unable to open DONE file: %s',FilePath);
    end
    Cleanup = onCleanup(@() fclose(FID));
    fprintf(FID,'completed=%s\n',datestr(now,31));
    clear Cleanup;
end
