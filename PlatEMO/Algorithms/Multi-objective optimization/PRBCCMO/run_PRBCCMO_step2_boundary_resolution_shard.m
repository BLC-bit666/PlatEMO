function Results = run_PRBCCMO_step2_boundary_resolution_shard(RunIndex,varargin)
% Run one paired-seed shard for the boundary-resolution diagnostic.

    Params = struct( ...
        'BaseDir',DefaultBaseDir(), ...
        'Problems',{{ ...
            'DASCMOP1_BC','DASCMOP2_BC','DASCMOP3_BC','DASCMOP4_BC', ...
            'DASCMOP5_BC','DASCMOP6_BC','DASCMOP7_BC','DASCMOP8_BC','DASCMOP9_BC'}}, ...
        'Runs',8, ...
        'Population',100, ...
        'MaxFE',200000, ...
        'BridgeTopK',5, ...
        'ShortlistFactors',[2,3,4], ...
        'Verbose',true);
    Params = ParseInputs(Params,varargin{:});

    RunIndex = round(double(RunIndex));
    if ~isfinite(RunIndex) || RunIndex < 1
        error('PRBCCMO:BoundaryResolutionShardInput', ...
            'RunIndex must be a positive integer.');
    end
    Params.ShortlistFactors = NormalizeShortlistFactors(Params.ShortlistFactors);

    BaseDir = char(Params.BaseDir);
    ShardDir = fullfile(BaseDir,sprintf('shard_%02d',RunIndex));
    if exist(ShardDir,'dir') ~= 7
        mkdir(ShardDir);
    end

    UpdateRows = repmat(InitUpdateRow(),0,1);
    RunRows = repmat(InitRunRow(),0,1);
    Calibrators = {'raw','beta'};

    for p = 1 : numel(Params.Problems)
        ProblemName = Params.Problems{p};
        for c = 1 : numel(Calibrators)
            CalibratorVariant = Calibrators{c};
            rng(RunIndex,'twister');
            Problem = feval(ProblemName,'N',Params.Population,'maxFE',Params.MaxFE);
            Algorithm = PRBCCMO('parameter',BuildAlgorithmParameters(Params,CalibratorVariant), ...
                'save',0,'outputFcn',@NoOutput);
            Algorithm.Solve(Problem);

            Metric = Algorithm.metric;
            [RunReports,UpdateReports] = EvaluateBoundaryResolutionRun( ...
                Metric,ProblemName,CalibratorVariant,RunIndex,Params);
            UpdateRows = AppendStructRows(UpdateRows,UpdateReports);
            RunRows = AppendStructRows(RunRows,RunReports);

            if logical(Params.Verbose)
                PrintRunSummary(ProblemName,CalibratorVariant,RunIndex,RunReports);
            end
        end
    end

    writetable(struct2table(UpdateRows,'AsArray',true), ...
        fullfile(ShardDir,'boundary_resolution_updates.csv'));
    writetable(struct2table(RunRows,'AsArray',true), ...
        fullfile(ShardDir,'boundary_resolution_runs.csv'));
    WriteDone(fullfile(ShardDir,'DONE.txt'),RunIndex);

    Results = struct();
    Results.updateRows = UpdateRows;
    Results.runRows = RunRows;
end

function BaseDir = DefaultBaseDir()
    Here = fileparts(mfilename('fullpath'));
    BaseDir = fullfile(Here,'results_step2_boundary_resolution_r8_20260326');
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('PRBCCMO:BoundaryResolutionShardInput', ...
            'Inputs must be provided as name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~isfield(Params,Name)
            error('PRBCCMO:BoundaryResolutionShardInput', ...
                'Unknown parameter ''%s''.',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function Factors = NormalizeShortlistFactors(Factors)
    Factors = unique(round(double(Factors(:)')),'stable');
    Factors = Factors(isfinite(Factors) & Factors >= 1);
    if isempty(Factors)
        error('PRBCCMO:BoundaryResolutionShortlistFactor', ...
            'ShortlistFactors must contain at least one positive integer.');
    end
end

function ParamStruct = BuildAlgorithmParameters(Params,CalibratorVariant)
    RuntimeOverride = struct();
    RuntimeOverride.SelectionName = 'pareto_only';
    RuntimeOverride.BridgeTopK = Params.BridgeTopK;
    RuntimeOverride.DisableBridgeScan = true;
    RuntimeOverride.TraceFlag = true;
    RuntimeOverride.CalibratorCandidates = ResolveCalibratorCandidates(CalibratorVariant);
    ParamStruct = struct();
    ParamStruct.runtimeOptions = BuildBoundaryRuntimeOptions(RuntimeOverride);
end

function Candidates = ResolveCalibratorCandidates(VariantName)
    switch lower(strtrim(char(VariantName)))
        case 'raw'
            Candidates = {'raw'};
        case 'beta'
            Candidates = {'beta'};
        otherwise
            error('PRBCCMO:BoundaryResolutionCalibrator', ...
                'Unsupported calibrator variant ''%s''.',char(VariantName));
    end
end

function [RunReports,UpdateReports] = EvaluateBoundaryResolutionRun( ...
    Metric,ProblemName,CalibratorVariant,RunIndex,Params)

    SectionB = FieldOrDefault(Metric,'sectionB',struct());
    CandidateAudit = FieldOrDefault(SectionB,'candidateAudit',repmat(InitCandidateAuditRowLocal(),0,1));
    SelectionTrace = FieldOrDefault(SectionB,'selectionTrace',repmat(InitSelectionTraceRowLocal(),0,1));
    CalTrace = FieldOrDefault(SectionB,'calibrationTrace',repmat(InitCalibrationTraceRowLocal(),0,1));

    UpdateReports = repmat(InitUpdateRow(),0,1);
    for i = 1 : numel(SelectionTrace)
        SelectionRow = SelectionTrace(i);
        Budget = SafeScalarStructField(SelectionRow,'budget',0);
        if ~isfinite(Budget) || Budget <= 0
            continue;
        end
        Generation = SafeScalarStructField(SelectionRow,'generation',NaN);
        FE = SafeScalarStructField(SelectionRow,'FE',NaN);
        Mask = [CandidateAudit.generation]' == Generation & [CandidateAudit.FE]' == FE;
        if ~any(Mask)
            continue;
        end
        Rows = CandidateAudit(Mask);
        TrustContext = ResolveTrustContext(CalTrace,SelectionRow);
        for ShortlistFactor = Params.ShortlistFactors
            UpdateReports(end+1,1) = EvaluateShortlistResolution( ... %#ok<AGROW>
                Rows,ProblemName,CalibratorVariant,RunIndex,SelectionRow,TrustContext,ShortlistFactor);
        end
    end

    RunReports = repmat(InitRunRow(),numel(Params.ShortlistFactors),1);
    for i = 1 : numel(Params.ShortlistFactors)
        RunReports(i) = SummarizeRunUpdates( ...
            UpdateReports,ProblemName,CalibratorVariant,RunIndex,Params.ShortlistFactors(i));
    end
end

function Row = EvaluateShortlistResolution( ...
    Rows,ProblemName,CalibratorVariant,RunIndex,SelectionRow,TrustContext,ShortlistFactor)

    Row = InitUpdateRow();
    Row.problem = ProblemName;
    Row.calibratorVariant = CalibratorVariant;
    Row.run = RunIndex;
    Row.generation = SafeScalarStructField(SelectionRow,'generation',NaN);
    Row.FE = SafeScalarStructField(SelectionRow,'FE',NaN);
    Row.budget = round(SafeScalarStructField(SelectionRow,'budget',0));
    Row.shortlistFactor = ShortlistFactor;
    Row.candidateCount = numel(Rows);
    Row.trustGate = logical(FieldOrDefault(TrustContext,'trustGate',false));
    Row.trustWeight = FieldOrDefault(TrustContext,'trustWeight',NaN);
    Row.ece = FieldOrDefault(TrustContext,'ece',inf);
    Row.coreNearGap = FieldOrDefault(TrustContext,'coreNearGap',inf);

    Eligible = ExtractRowField(Rows,'eligible',false) ~= 0;
    ParetoValue = ExtractRowField(Rows,'paretoValue',NaN);
    BoundaryTrust = ExtractRowField(Rows,'boundaryTrust',NaN);
    QueryScore = ExtractRowField(Rows,'queryScore',NaN);
    Reliability = ExtractRowField(Rows,'reliability',NaN);
    Disagreement = ExtractRowField(Rows,'disagreement',NaN);

    EligibleIdx = find(Eligible & isfinite(ParetoValue));
    Row.eligibleCount = numel(EligibleIdx);
    Row.boundaryStdEligible = SafeStd(BoundaryTrust(EligibleIdx));
    if isempty(EligibleIdx) || Row.budget <= 0
        return;
    end

    ShortlistCount = min(numel(EligibleIdx),max(Row.budget,ceil(ShortlistFactor*Row.budget)));
    RankTable = [-ParetoValue(EligibleIdx),EligibleIdx];
    RankTable = sortrows(RankTable,[1 2]);
    ShortlistedIdx = RankTable(1:ShortlistCount,2);
    Row.shortlistCount = numel(ShortlistedIdx);

    ShortBoundary = BoundaryTrust(ShortlistedIdx);
    ShortPareto = ParetoValue(ShortlistedIdx);
    ShortQuery = QueryScore(ShortlistedIdx);
    ShortReliability = Reliability(ShortlistedIdx);
    ShortDisagreement = Disagreement(ShortlistedIdx);

    Row.boundaryStdShortlist = SafeStd(ShortBoundary);
    Row.queryScoreStdShortlist = SafeStd(ShortQuery);
    Row.reliabilityStdShortlist = SafeStd(ShortReliability);
    Row.disagreementStdShortlist = SafeStd(ShortDisagreement);
    Row.spearmanBoundaryParetoShortlist = SafeSpearman(ShortBoundary,ShortPareto);

    Keep = min(Row.budget,numel(ShortlistedIdx));
    if Keep <= 0
        return;
    end
    PickPareto = TopDescending(ShortPareto,Keep);
    PickBoundary = TopDescending(ShortBoundary,Keep);
    Row.topBoundaryParetoOverlap = ComputeSelectionOverlap(PickPareto,PickBoundary,Keep);
    Row.topBoundaryParetoDiffers = SelectionsDiffer(PickPareto,PickBoundary);
end

function Row = SummarizeRunUpdates(UpdateReports,ProblemName,CalibratorVariant,RunIndex,ShortlistFactor)
    Row = InitRunRow();
    Row.problem = ProblemName;
    Row.calibratorVariant = CalibratorVariant;
    Row.run = RunIndex;
    Row.shortlistFactor = ShortlistFactor;

    if isempty(UpdateReports)
        return;
    end
    Mask = [UpdateReports.shortlistFactor]' == ShortlistFactor;
    Subset = UpdateReports(Mask);
    if isempty(Subset)
        return;
    end

    Row.updateCount = numel(Subset);
    Row.trustGateRate = mean(double([Subset.trustGate]),'omitnan');
    Row.meanECE = mean([Subset.ece],'omitnan');
    Row.meanCoreNearGap = mean([Subset.coreNearGap],'omitnan');
    Row.meanBudget = mean([Subset.budget],'omitnan');
    Row.meanShortlistCount = mean([Subset.shortlistCount],'omitnan');
    Row.medianBoundaryStdEligible = median([Subset.boundaryStdEligible],'omitnan');
    Row.medianBoundaryStdShortlist = median([Subset.boundaryStdShortlist],'omitnan');
    Row.medianQueryScoreStdShortlist = median([Subset.queryScoreStdShortlist],'omitnan');
    Row.medianReliabilityStdShortlist = median([Subset.reliabilityStdShortlist],'omitnan');
    Row.medianDisagreementStdShortlist = median([Subset.disagreementStdShortlist],'omitnan');
    Row.medianSpearmanBoundaryParetoShortlist = median([Subset.spearmanBoundaryParetoShortlist],'omitnan');
    Row.medianAbsSpearmanBoundaryParetoShortlist = median(abs([Subset.spearmanBoundaryParetoShortlist]),'omitnan');
    Row.meanSpearmanBoundaryParetoShortlist = mean([Subset.spearmanBoundaryParetoShortlist],'omitnan');
    Row.medianTopBoundaryParetoOverlap = median([Subset.topBoundaryParetoOverlap],'omitnan');
    Row.meanTopBoundaryParetoOverlap = mean([Subset.topBoundaryParetoOverlap],'omitnan');
    Row.topBoundaryParetoDiffRate = mean(double([Subset.topBoundaryParetoDiffers]),'omitnan');
end

function Idx = TopDescending(Score,Keep)
    Score = Score(:);
    Keep = min(max(0,round(Keep)),numel(Score));
    if Keep <= 0
        Idx = zeros(0,1);
        return;
    end
    Table = [-Score,(1:numel(Score))'];
    Table = sortrows(Table,[1 2]);
    Idx = Table(1:Keep,2);
end

function Value = ComputeSelectionOverlap(BasePick,ComparePick,Budget)
    Value = NaN;
    Budget = round(double(Budget));
    if Budget <= 0
        return;
    end
    Value = numel(intersect(BasePick(:),ComparePick(:))) / Budget;
end

function Flag = SelectionsDiffer(BasePick,ComparePick)
    Flag = false;
    BasePick = sort(BasePick(:));
    ComparePick = sort(ComparePick(:));
    if numel(BasePick) ~= numel(ComparePick)
        Flag = true;
        return;
    end
    if isempty(BasePick)
        return;
    end
    Flag = any(BasePick ~= ComparePick);
end

function Context = ResolveTrustContext(CalTrace,SelectionRow)
    Context = struct( ...
        'ece',inf, ...
        'coreNearGap',inf, ...
        'trustGate',false, ...
        'trustWeight',NaN);
    if nargin < 2 || isempty(SelectionRow)
        return;
    end
    if isfield(SelectionRow,'trustGate') && ~isempty(SelectionRow.trustGate)
        Context.trustGate = logical(SelectionRow.trustGate);
    end
    if isempty(CalTrace)
        return;
    end

    TargetFE = SafeScalarStructField(SelectionRow,'FE',NaN);
    TargetGeneration = SafeScalarStructField(SelectionRow,'generation',NaN);
    FEValues = [CalTrace.FE]';
    GenValues = [CalTrace.generation]';
    Match = GenValues < TargetGeneration;
    if any(Match)
        Candidates = find(Match);
        [~,BestIdx] = max(FEValues(Candidates));
        Row = CalTrace(Candidates(BestIdx));
    else
        EarlierFE = FEValues < TargetFE;
        if any(EarlierFE)
            Candidates = find(EarlierFE);
            [~,BestIdx] = max(FEValues(Candidates));
            Row = CalTrace(Candidates(BestIdx));
        else
            Row = CalTrace(1);
        end
    end
    Context.ece = SafeScalarStructField(Row,'ece',inf);
    Context.coreNearGap = SafeScalarStructField(Row,'core_near_gap', ...
        SafeScalarStructField(Row,'near_gap',inf));
    Context.trustWeight = SafeScalarStructField(Row,'trust_weight',NaN);
end

function Value = ExtractRowField(Rows,Field,Default)
    Count = numel(Rows);
    Value = repmat(Default,Count,1);
    if isempty(Rows) || ~isfield(Rows,Field)
        return;
    end
    Raw = {Rows.(Field)};
    for i = 1 : Count
        Item = Raw{i};
        if isempty(Item)
            continue;
        end
        if islogical(Default)
            Value(i,1) = logical(Item(1));
        else
            Value(i,1) = double(Item(1));
        end
    end
end

function Value = SafeStd(Data)
    Value = NaN;
    Data = double(Data(:));
    Data = Data(isfinite(Data));
    if numel(Data) < 2
        return;
    end
    Value = std(Data,0,'omitnan');
end

function Value = SafeSpearman(X,Y)
    Value = NaN;
    X = X(:);
    Y = Y(:);
    Valid = isfinite(X) & isfinite(Y);
    X = X(Valid);
    Y = Y(Valid);
    if numel(X) < 2
        return;
    end
    RX = AverageRanks(X);
    RY = AverageRanks(Y);
    RX = RX - mean(RX);
    RY = RY - mean(RY);
    Den = sqrt(sum(RX.^2) * sum(RY.^2));
    if Den <= 0
        return;
    end
    Value = sum(RX .* RY) / Den;
end

function Rank = AverageRanks(X)
    [Sorted,Order] = sort(X);
    Rank = zeros(size(X));
    Start = 1;
    while Start <= numel(Sorted)
        Stop = Start;
        while Stop < numel(Sorted) && isequaln(Sorted(Stop+1),Sorted(Start))
            Stop = Stop + 1;
        end
        RankValue = 0.5 * (Start + Stop);
        Rank(Order(Start:Stop)) = RankValue;
        Start = Stop + 1;
    end
end

function Rows = AppendStructRows(Rows,AddRows)
    if isempty(AddRows)
        return;
    end
    if isempty(Rows)
        Rows = AddRows;
    else
        Rows = [Rows; AddRows]; %#ok<AGROW>
    end
end

function PrintRunSummary(ProblemName,CalibratorVariant,RunIndex,RunReports)
    if isempty(RunReports)
        return;
    end
    BestOverlap = min([RunReports.medianTopBoundaryParetoOverlap]);
    BestCorr = min([RunReports.medianAbsSpearmanBoundaryParetoShortlist]);
    fprintf('[boundary-resolution] %s | run=%d | cal=%s | bestOverlap=%.4f | bestAbsRho=%.4f\n', ...
        ProblemName,RunIndex,CalibratorVariant,BestOverlap,BestCorr);
end

function Value = FieldOrDefault(S,Field,Default)
    Value = Default;
    if isstruct(S) && isfield(S,Field) && ~isempty(S.(Field))
        Value = S.(Field);
    end
end

function Value = SafeScalarStructField(S,Field,Default)
    Value = Default;
    if ~isstruct(S) || ~isfield(S,Field) || isempty(S.(Field))
        return;
    end
    Data = S.(Field);
    if ischar(Data) || (isstring(Data) && isscalar(Data))
        Value = Data;
    else
        Value = double(Data(1));
    end
end

function Row = InitUpdateRow()
    Row = struct( ...
        'problem','', ...
        'calibratorVariant','', ...
        'run',0, ...
        'generation',NaN, ...
        'FE',NaN, ...
        'budget',0, ...
        'shortlistFactor',NaN, ...
        'candidateCount',0, ...
        'eligibleCount',0, ...
        'shortlistCount',0, ...
        'trustGate',false, ...
        'trustWeight',NaN, ...
        'ece',NaN, ...
        'coreNearGap',NaN, ...
        'boundaryStdEligible',NaN, ...
        'boundaryStdShortlist',NaN, ...
        'queryScoreStdShortlist',NaN, ...
        'reliabilityStdShortlist',NaN, ...
        'disagreementStdShortlist',NaN, ...
        'spearmanBoundaryParetoShortlist',NaN, ...
        'topBoundaryParetoOverlap',NaN, ...
        'topBoundaryParetoDiffers',false);
end

function Row = InitRunRow()
    Row = struct( ...
        'problem','', ...
        'calibratorVariant','', ...
        'run',0, ...
        'shortlistFactor',NaN, ...
        'updateCount',0, ...
        'trustGateRate',NaN, ...
        'meanECE',NaN, ...
        'meanCoreNearGap',NaN, ...
        'meanBudget',NaN, ...
        'meanShortlistCount',NaN, ...
        'medianBoundaryStdEligible',NaN, ...
        'medianBoundaryStdShortlist',NaN, ...
        'medianQueryScoreStdShortlist',NaN, ...
        'medianReliabilityStdShortlist',NaN, ...
        'medianDisagreementStdShortlist',NaN, ...
        'medianSpearmanBoundaryParetoShortlist',NaN, ...
        'medianAbsSpearmanBoundaryParetoShortlist',NaN, ...
        'meanSpearmanBoundaryParetoShortlist',NaN, ...
        'medianTopBoundaryParetoOverlap',NaN, ...
        'meanTopBoundaryParetoOverlap',NaN, ...
        'topBoundaryParetoDiffRate',NaN);
end

function Row = InitCandidateAuditRowLocal()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'eligible',false, ...
        'queryScore',NaN, ...
        'disagreement',NaN, ...
        'reliability',NaN, ...
        'paretoValue',NaN, ...
        'boundaryTrust',NaN, ...
        'trustWeight',NaN);
end

function Row = InitSelectionTraceRowLocal()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'budget',0, ...
        'trustGate',false);
end

function Row = InitCalibrationTraceRowLocal()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'ece',NaN, ...
        'near_gap',NaN, ...
        'core_near_gap',NaN, ...
        'trust_weight',NaN);
end

function NoOutput(varargin) %#ok<INUSD>
end

function WriteDone(FilePath,RunIndex)
    FID = fopen(FilePath,'w');
    if FID < 0
        error('PRBCCMO:BoundaryResolutionShardIO', ...
            'Unable to open DONE file: %s',FilePath);
    end
    Cleanup = onCleanup(@() fclose(FID));
    fprintf(FID,'run=%d\ncompleted=%s\n',RunIndex,datestr(now,31));
    clear Cleanup;
end
