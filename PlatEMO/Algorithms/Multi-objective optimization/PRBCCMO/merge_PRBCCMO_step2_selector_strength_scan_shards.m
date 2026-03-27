function Results = merge_PRBCCMO_step2_selector_strength_scan_shards(varargin)
% Merge selector-strength scan shard CSV outputs without MAT artifacts.

    Params = struct( ...
        'BaseDir',DefaultBaseDir(), ...
        'RunIndices',1:8, ...
        'BaseVariant','ParetoOnly', ...
        'CurrentVariant','FullSoftTrust', ...
        'StopGoOverlapMedianMax',0.99, ...
        'StopGoSelectionDiffRateMin',0.05, ...
        'StopGoDispersionMedianMin',1e-6, ...
        'OracleAuditEnabledRateMin',0.30, ...
        'PrimaryAuditOpenProblemMin',4, ...
        'PrimarySelectionDiffProblemMin',6, ...
        'SecondaryAuditedProblemMin',3);
    Params = ParseInputs(Params,varargin{:});

    BaseDir = char(Params.BaseDir);
    RunIndices = unique(round(double(Params.RunIndices(:)')),'stable');
    if isempty(RunIndices)
        error('PRBCCMO:SelectorStrengthScanMergeInput', ...
            'RunIndices must not be empty.');
    end

    QueryRuns = table();
    CalibratorRuns = table();
    for RunIndex = RunIndices
        ShardDir = fullfile(BaseDir,sprintf('shard_%02d',RunIndex));
        QueryRuns = AppendTable(QueryRuns,ReadShardTable(ShardDir,'step2_query_runs.csv'));
        CalibratorRuns = AppendTable(CalibratorRuns,ReadShardTable(ShardDir,'step2_calibrator_runs.csv'));
    end
    Manifest = ReadShardTable(fullfile(BaseDir,'shard_01'),'selector_scan_variants.csv');

    RunBoard = BuildRunBoard(QueryRuns,Manifest,Params);
    ProblemBoard = BuildProblemBoard(RunBoard,Params);
    VariantBoard = BuildVariantBoard(ProblemBoard,Params);

    writetable(QueryRuns,fullfile(BaseDir,'step2_query_runs.csv'));
    writetable(CalibratorRuns,fullfile(BaseDir,'step2_calibrator_runs.csv'));
    writetable(Manifest,fullfile(BaseDir,'selector_scan_variants.csv'));
    writetable(RunBoard,fullfile(BaseDir,'step2_selector_strength_scan_run_board.csv'));
    writetable(ProblemBoard,fullfile(BaseDir,'step2_selector_strength_scan_problem_board.csv'));
    writetable(VariantBoard,fullfile(BaseDir,'step2_selector_strength_scan_variant_board.csv'));
    WriteDone(fullfile(BaseDir,'DONE.txt'));

    Results = struct();
    Results.params = Params;
    Results.queryRuns = QueryRuns;
    Results.calibratorRuns = CalibratorRuns;
    Results.manifest = Manifest;
    Results.runBoard = RunBoard;
    Results.problemBoard = ProblemBoard;
    Results.variantBoard = VariantBoard;
end

function BaseDir = DefaultBaseDir()
    Here = fileparts(mfilename('fullpath'));
    BaseDir = fullfile(Here,'results_step2_selector_strength_scan_r8_20260326');
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('PRBCCMO:SelectorStrengthScanMergeInput', ...
            'Inputs must be provided as name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~isfield(Params,Name)
            error('PRBCCMO:SelectorStrengthScanMergeInput', ...
                'Unknown parameter ''%s''.',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function Data = ReadShardTable(ShardDir,FileName)
    FilePath = fullfile(ShardDir,FileName);
    if exist(FilePath,'file') ~= 2
        error('PRBCCMO:SelectorStrengthScanMergeMissingShard', ...
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

function RunBoard = BuildRunBoard(QueryRuns,Manifest,Params)
    ScanRows = QueryRuns(QueryRuns.queryVariant ~= string(Params.BaseVariant),:);
    if isempty(ScanRows)
        RunBoard = InitRunBoardTable();
        return;
    end

    ManifestLookup = containers.Map(cellstr(Manifest.queryVariant),num2cell(1:height(Manifest)));
    Rows = repmat(InitRunBoardRow(),height(ScanRows),1);
    for i = 1 : height(ScanRows)
        VariantKey = char(ScanRows.queryVariant(i));
        if ~isKey(ManifestLookup,VariantKey)
            error('PRBCCMO:SelectorStrengthScanMergeManifest', ...
                'Missing variant manifest entry for ''%s''.',VariantKey);
        end
        Meta = Manifest(ManifestLookup(VariantKey),:);
        Row = InitRunBoardRow();
        Row.problem = char(ScanRows.problem(i));
        Row.calibratorVariant = char(ScanRows.calibratorVariant(i));
        Row.baseVariant = char(Params.BaseVariant);
        Row.queryVariant = VariantKey;
        Row.variantGroup = char(Meta.variantGroup);
        Row.alphaLowMax = Meta.alphaLowMax;
        Row.shortlistFactor = Meta.shortlistFactor;
        Row.trustFallbackCap = Meta.trustFallbackCap;
        Row.isCurrentVariant = logical(Meta.isCurrentVariant);
        Row.matchesCurrentConfig = logical(Meta.matchesCurrentConfig);
        Row.run = ScanRows.run(i);
        Row.selectionDiffPoolRate = ScanRows.selectionDiffPoolRate(i);
        Row.medianSelectionOverlapPareto = ScanRows.medianSelectionOverlapPareto(i);
        Row.medianBoundaryScoreDispersion = ScanRows.medianBoundaryScoreDispersion(i);
        Row.poolCount = ScanRows.poolCount(i);
        Row.oracleAuditEnabled = EvaluateRunGate( ...
            ScanRows.medianSelectionOverlapPareto(i), ...
            ScanRows.selectionDiffPoolRate(i), ...
            ScanRows.medianBoundaryScoreDispersion(i),Params);
        if Row.oracleAuditEnabled
            Row.auditedUpdateCount = ScanRows.poolCount(i);
        else
            Row.auditedUpdateCount = 0;
        end
        Rows(i,1) = Row; %#ok<AGROW>
    end
    RunBoard = struct2table(Rows,'AsArray',true);
end

function Flag = EvaluateRunGate(MedianOverlap,SelectionDiffRate,MedianDispersion,Params)
    Flag = true;
    if ~isfinite(MedianOverlap) || MedianOverlap >= Params.StopGoOverlapMedianMax
        Flag = false;
    end
    if ~isfinite(SelectionDiffRate) || SelectionDiffRate < Params.StopGoSelectionDiffRateMin
        Flag = false;
    end
    if ~isfinite(MedianDispersion) || MedianDispersion <= Params.StopGoDispersionMedianMin
        Flag = false;
    end
end

function ProblemBoard = BuildProblemBoard(RunBoard,Params)
    ProblemBoard = InitProblemBoardTable();
    if isempty(RunBoard)
        return;
    end

    [Groups,Problems,Calibrators,Variants] = findgroups( ...
        string(RunBoard.problem),string(RunBoard.calibratorVariant),string(RunBoard.queryVariant));
    Rows = repmat(InitProblemBoardRow(),max(Groups),1);
    for g = 1 : max(Groups)
        Mask = Groups == g;
        Subset = RunBoard(Mask,:);
        Row = InitProblemBoardRow();
        Row.problem = char(Problems(g));
        Row.calibratorVariant = char(Calibrators(g));
        Row.baseVariant = char(Params.BaseVariant);
        Row.queryVariant = char(Variants(g));
        Row.variantGroup = char(string(Subset.variantGroup(1)));
        Row.alphaLowMax = Subset.alphaLowMax(1);
        Row.shortlistFactor = Subset.shortlistFactor(1);
        Row.trustFallbackCap = Subset.trustFallbackCap(1);
        Row.isCurrentVariant = logical(Subset.isCurrentVariant(1));
        Row.matchesCurrentConfig = logical(Subset.matchesCurrentConfig(1));
        Row.runCount = height(Subset);
        Row.medianSelectionDiffPoolRate = median(Subset.selectionDiffPoolRate,'omitnan');
        Row.oracleAuditEnabledRate = mean(Subset.oracleAuditEnabled,'omitnan');
        Row.medianSelectionOverlapPareto = median(Subset.medianSelectionOverlapPareto,'omitnan');
        Row.medianBoundaryScoreDispersion = median(Subset.medianBoundaryScoreDispersion,'omitnan');
        Row.medianAuditedUpdateCount = median(Subset.auditedUpdateCount,'omitnan');
        Row.meanAuditedUpdateCount = mean(Subset.auditedUpdateCount,'omitnan');
        Row.auditedPositiveRunCount = sum(Subset.auditedUpdateCount > 0);
        Row.auditedUpdateCountSignrankP = SafeSignrankGreaterZero(Subset.auditedUpdateCount);
        Row.passSelectionDiff = isfinite(Row.medianSelectionDiffPoolRate) && ...
            Row.medianSelectionDiffPoolRate > Params.StopGoSelectionDiffRateMin;
        Row.passOracleAudit = isfinite(Row.oracleAuditEnabledRate) && ...
            Row.oracleAuditEnabledRate > Params.OracleAuditEnabledRateMin;
        Row.passAuditedUpdates = isfinite(Row.medianAuditedUpdateCount) && ...
            Row.medianAuditedUpdateCount > 0 && ...
            isfinite(Row.auditedUpdateCountSignrankP) && ...
            Row.auditedUpdateCountSignrankP < 0.05;
        Rows(g,1) = Row; %#ok<AGROW>
    end
    ProblemBoard = struct2table(Rows,'AsArray',true);
end

function VariantBoard = BuildVariantBoard(ProblemBoard,Params)
    VariantBoard = InitVariantBoardTable();
    if isempty(ProblemBoard)
        return;
    end

    [Groups,Calibrators,Variants] = findgroups( ...
        string(ProblemBoard.calibratorVariant),string(ProblemBoard.queryVariant));
    Rows = repmat(InitVariantBoardRow(),max(Groups),1);
    for g = 1 : max(Groups)
        Mask = Groups == g;
        Subset = ProblemBoard(Mask,:);
        ProblemCount = height(Subset);
        Row = InitVariantBoardRow();
        Row.calibratorVariant = char(Calibrators(g));
        Row.baseVariant = char(Params.BaseVariant);
        Row.queryVariant = char(Variants(g));
        Row.variantGroup = char(string(Subset.variantGroup(1)));
        Row.alphaLowMax = Subset.alphaLowMax(1);
        Row.shortlistFactor = Subset.shortlistFactor(1);
        Row.trustFallbackCap = Subset.trustFallbackCap(1);
        Row.isCurrentVariant = logical(Subset.isCurrentVariant(1));
        Row.matchesCurrentConfig = logical(Subset.matchesCurrentConfig(1));
        Row.problemCount = ProblemCount;
        Row.selectionDiffPassProblemCount = sum(Subset.passSelectionDiff);
        Row.auditOpenProblemCount = sum(Subset.passOracleAudit);
        Row.auditedPositiveProblemCount = sum(Subset.passAuditedUpdates);
        Row.primaryPass = Row.auditOpenProblemCount >= Params.PrimaryAuditOpenProblemMin && ...
            Row.selectionDiffPassProblemCount >= Params.PrimarySelectionDiffProblemMin;
        Row.secondaryPass = Row.primaryPass && ...
            Row.auditedPositiveProblemCount >= Params.SecondaryAuditedProblemMin;
        Row.meanProblemOracleAuditEnabledRate = mean(Subset.oracleAuditEnabledRate,'omitnan');
        Row.meanProblemSelectionDiffPoolRate = mean(Subset.medianSelectionDiffPoolRate,'omitnan');
        Row.meanProblemMedianSelectionOverlapPareto = mean(Subset.medianSelectionOverlapPareto,'omitnan');
        Rows(g,1) = Row; %#ok<AGROW>
    end
    VariantBoard = struct2table(Rows,'AsArray',true);
    VariantBoard = sortrows(VariantBoard, ...
        {'primaryPass','secondaryPass','auditOpenProblemCount','selectionDiffPassProblemCount', ...
         'auditedPositiveProblemCount','calibratorVariant','queryVariant'}, ...
        {'descend','descend','descend','descend','descend','ascend','ascend'});
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

function Row = InitRunBoardRow()
    Row = struct( ...
        'problem','', ...
        'calibratorVariant','', ...
        'baseVariant','', ...
        'queryVariant','', ...
        'variantGroup','', ...
        'alphaLowMax',NaN, ...
        'shortlistFactor',NaN, ...
        'trustFallbackCap',NaN, ...
        'isCurrentVariant',false, ...
        'matchesCurrentConfig',false, ...
        'run',0, ...
        'selectionDiffPoolRate',NaN, ...
        'medianSelectionOverlapPareto',NaN, ...
        'medianBoundaryScoreDispersion',NaN, ...
        'poolCount',0, ...
        'oracleAuditEnabled',false, ...
        'auditedUpdateCount',0);
end

function T = InitRunBoardTable()
    T = struct2table(repmat(InitRunBoardRow(),0,1),'AsArray',true);
end

function Row = InitProblemBoardRow()
    Row = struct( ...
        'problem','', ...
        'calibratorVariant','', ...
        'baseVariant','', ...
        'queryVariant','', ...
        'variantGroup','', ...
        'alphaLowMax',NaN, ...
        'shortlistFactor',NaN, ...
        'trustFallbackCap',NaN, ...
        'isCurrentVariant',false, ...
        'matchesCurrentConfig',false, ...
        'runCount',0, ...
        'medianSelectionDiffPoolRate',NaN, ...
        'oracleAuditEnabledRate',NaN, ...
        'medianSelectionOverlapPareto',NaN, ...
        'medianBoundaryScoreDispersion',NaN, ...
        'medianAuditedUpdateCount',NaN, ...
        'meanAuditedUpdateCount',NaN, ...
        'auditedPositiveRunCount',0, ...
        'auditedUpdateCountSignrankP',NaN, ...
        'passSelectionDiff',false, ...
        'passOracleAudit',false, ...
        'passAuditedUpdates',false);
end

function T = InitProblemBoardTable()
    T = struct2table(repmat(InitProblemBoardRow(),0,1),'AsArray',true);
end

function Row = InitVariantBoardRow()
    Row = struct( ...
        'calibratorVariant','', ...
        'baseVariant','', ...
        'queryVariant','', ...
        'variantGroup','', ...
        'alphaLowMax',NaN, ...
        'shortlistFactor',NaN, ...
        'trustFallbackCap',NaN, ...
        'isCurrentVariant',false, ...
        'matchesCurrentConfig',false, ...
        'problemCount',0, ...
        'selectionDiffPassProblemCount',0, ...
        'auditOpenProblemCount',0, ...
        'auditedPositiveProblemCount',0, ...
        'primaryPass',false, ...
        'secondaryPass',false, ...
        'meanProblemOracleAuditEnabledRate',NaN, ...
        'meanProblemSelectionDiffPoolRate',NaN, ...
        'meanProblemMedianSelectionOverlapPareto',NaN);
end

function T = InitVariantBoardTable()
    T = struct2table(repmat(InitVariantBoardRow(),0,1),'AsArray',true);
end

function WriteDone(FilePath)
    FID = fopen(FilePath,'w');
    if FID < 0
        error('PRBCCMO:SelectorStrengthScanMergeIO', ...
            'Unable to open DONE file: %s',FilePath);
    end
    Cleanup = onCleanup(@() fclose(FID));
    fprintf(FID,'completed=%s\n',datestr(now,31));
    clear Cleanup;
end
