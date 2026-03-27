function Results = check_PRBCCMO_step2_query_trust(varargin)
% Step 2 query + trust validation for PRBCCMO-Lite.
%
% Query variants are evaluated offline on the same candidate pools emitted
% by the Pareto-only runtime, so ParetoOnly / FullV2 / FullV2Current /
% FullNoTrust / FullSoftTrust / Uncertain-only / HighProb / Rand share
% identical bridge generation, boundary budget, and search trajectory
% inside each calibrator configuration.
%
% Optional name-value pairs:
%   'Problems'            : cellstr problem list, default DASCMOP1_BC:9_BC
%   'Runs'                : paired independent runs per calibrator, default 30
%   'RunIndices'          : explicit run indices to execute, default 1:Runs
%   'Population'          : population size, default 100
%   'MaxFE'               : maximum function evaluations, default 200000
%   'QueryVariants'       : subset of {'ParetoOnly','FullV2','FullV2Current', ...
%                           'FullNoTrust','FullSoftTrust','UncertainOnly', ...
%                           'HighProbBoundary','RandBoundary'}
%   'QueryVariantRuntimeOverrides' : cell array aligned with QueryVariants;
%                           each entry is a runtime-options struct overlay
%                           used only for that offline selector variant
%   'CalibratorVariants'  : subset of {'raw','beta','temperature','auto'}
%                           default keeps only raw/beta; temperature/auto
%                           remain optional audit branches
%   'BridgeTopK'          : top-k width for TopKPair, default 5
%   'SavePath'            : MAT output path, default 'prbccmo_step2_query_trust.mat'
%   'QueryCsv'            : query run-level CSV output path
%   'QueryUpdateCsv'      : query update-level CSV output path
%   'CalibratorCsv'       : calibrator run-level CSV output path
%   'QueryPairedCsv'      : paired query-summary CSV output path
%   'CalibratorPairedCsv' : paired calibrator-summary CSV output path
%   'Verbose'             : print concise per-run summaries, default true
%   'OracleSampleCount'   : offline uniform samples per raw problem, default 50000
%   'OracleScaleMode'     : {'std','mad','meanabs'}, default 'std'
%   'OracleTau'           : hit-rate threshold for QP_tau, default 0.05
%   'OracleScaleEps'      : floor for scale denominator, default 1e-12
%   'EnableStopGoGate'    : true to stop oracle audit when the target
%                           query variant still collapses to ParetoOnly,
%                           default true
%   'StopGoTargetVariant' : target variant for stop/go, default
%                           'FullSoftTrust' with safe fallback to the best
%                           available non-Pareto variant
%   'StopGoBaseVariant'   : base variant for stop/go, default 'ParetoOnly'
%   'StopGoOverlapMedianMax' : max allowed median O_t before stopping,
%                           default 0.99
%   'StopGoSelectionDiffRateMin' : minimum rate of genuinely different
%                           selections before continuing, default 0.05
%   'StopGoDispersionMedianMin' : minimum allowed median D_t before
%                           continuing, default 1e-6

    Params = struct( ...
        'Problems',{{ ...
            'DASCMOP1_BC','DASCMOP2_BC','DASCMOP3_BC','DASCMOP4_BC', ...
            'DASCMOP5_BC','DASCMOP6_BC','DASCMOP7_BC','DASCMOP8_BC','DASCMOP9_BC'}}, ...
        'Runs',30, ...
        'RunIndices',[], ...
        'Population',100, ...
        'MaxFE',200000, ...
        'QueryVariants',{{'ParetoOnly','FullV2','UncertainOnly','HighProbBoundary','RandBoundary'}}, ...
        'QueryVariantRuntimeOverrides',{{}}, ...
        'CalibratorVariants',{{'raw','beta'}}, ...
        'BridgeTopK',5, ...
        'SavePath','prbccmo_step2_query_trust.mat', ...
        'QueryCsv','prbccmo_step2_query_trust_query_runs.csv', ...
        'QueryUpdateCsv','prbccmo_step2_query_trust_query_updates.csv', ...
        'CalibratorCsv','prbccmo_step2_query_trust_calibrator_runs.csv', ...
        'QueryPairedCsv','prbccmo_step2_query_trust_query_paired.csv', ...
        'CalibratorPairedCsv','prbccmo_step2_query_trust_calibrator_paired.csv', ...
        'Verbose',true, ...
        'OracleSampleCount',50000, ...
        'OracleScaleMode','std', ...
        'OracleTau',0.05, ...
        'OracleScaleEps',1e-12, ...
        'EnableStopGoGate',true, ...
        'StopGoTargetVariant','FullSoftTrust', ...
        'StopGoBaseVariant','ParetoOnly', ...
        'StopGoOverlapMedianMax',0.99, ...
        'StopGoSelectionDiffRateMin',0.05, ...
        'StopGoDispersionMedianMin',1e-6);
    Params = ParseInputs(Params,varargin{:});
    Params.QueryVariants = NormalizeQueryVariantList(Params.QueryVariants);
    Params.QueryVariantRuntimeOverrides = NormalizeQueryVariantRuntimeOverrides( ...
        Params.QueryVariantRuntimeOverrides,numel(Params.QueryVariants));
    Params.CalibratorVariants = NormalizeCalibratorVariantList(Params.CalibratorVariants);
    Params.RunIndices = NormalizeRunIndices(Params.RunIndices,Params.Runs);
    Params.StopGoBaseVariant = CanonicalQueryVariant(Params.StopGoBaseVariant);
    Params.StopGoTargetVariant = ResolveStopGoTargetVariant( ...
        Params.QueryVariants,Params.StopGoTargetVariant,Params.StopGoBaseVariant);
    Params.EnableStopGoGate = logical(Params.EnableStopGoGate);
    Params.StopGoOverlapMedianMax = min(max(double(Params.StopGoOverlapMedianMax),0),1);
    Params.StopGoSelectionDiffRateMin = min(max(double(Params.StopGoSelectionDiffRateMin),0),1);
    Params.StopGoDispersionMedianMin = max(double(Params.StopGoDispersionMedianMin),0);

    QueryReports = repmat(InitQueryRunReport(),0,1);
    QueryUpdateReports = repmat(InitQueryUpdateReport(),0,1);
    CalibratorReports = repmat(InitCalibratorRunReport(),0,1);
    OracleCache = containers.Map('KeyType','char','ValueType','any');
    OracleMeta = repmat(InitOracleMetaRow(),0,1);
    QueryRow = 0;
    QueryUpdateRow = 0;
    CalRow = 0;

    for p = 1 : numel(Params.Problems)
        ProblemName = Params.Problems{p};
        [OracleContext,OracleMetaRow] = ResolveOracleContext(OracleCache,ProblemName,Params);
        OracleMeta(end+1,1) = OracleMetaRow; %#ok<AGROW>
        for r = Params.RunIndices(:)'
            for c = 1 : numel(Params.CalibratorVariants)
                CalibratorVariant = Params.CalibratorVariants{c};
                rng(r,'twister');
                Problem = feval(ProblemName,'N',Params.Population,'maxFE',Params.MaxFE);
                Algorithm = PRBCCMO('parameter',BuildAlgorithmParameters(Params,CalibratorVariant), ...
                    'save',0,'outputFcn',@NoOutput);
                Algorithm.Solve(Problem);

                Metric = Algorithm.metric;
                CalRow = CalRow + 1;
                CalibratorReports(CalRow,1) = EvaluateCalibratorRun( ...
                    Metric,ProblemName,CalibratorVariant,r); %#ok<AGROW>
                [RunQueryReports,RunQueryUpdateReports] = EvaluateQueryRuns( ...
                    Metric,ProblemName,CalibratorVariant,r,OracleContext,Params);
                QueryReports(QueryRow + (1:numel(RunQueryReports)),1) = RunQueryReports; %#ok<AGROW>
                QueryRow = QueryRow + numel(RunQueryReports);
                if ~isempty(RunQueryUpdateReports)
                    QueryUpdateReports(QueryUpdateRow + (1:numel(RunQueryUpdateReports)),1) = ...
                        RunQueryUpdateReports; %#ok<AGROW>
                    QueryUpdateRow = QueryUpdateRow + numel(RunQueryUpdateReports);
                end

                if Params.Verbose
                    PrintRunSummary(ProblemName,r,CalibratorReports(CalRow),RunQueryReports);
                end
            end
        end
    end

    Results = struct();
    Results.params = Params;
    Results.oracle = OracleMeta;
    Results.queryReport = QueryReports;
    Results.queryUpdateReport = QueryUpdateReports;
    Results.calibratorReport = CalibratorReports;
    Results.querySummary = SummarizeQueryReports(QueryReports,Params.QueryVariants,Params.CalibratorVariants);
    Results.calibratorSummary = SummarizeCalibratorReports(CalibratorReports,Params.CalibratorVariants);
    Results.queryPaired = BuildQueryPairedSummary(QueryReports,Params.QueryVariants,Params.CalibratorVariants);
    Results.calibratorPaired = BuildCalibratorPairedSummary(CalibratorReports,Params.CalibratorVariants);

    if ~isempty(Params.SavePath)
        save(Params.SavePath,'Results','-v7.3');
    end
    if ~isempty(Params.QueryCsv)
        writetable(struct2table(QueryReports,'AsArray',true),Params.QueryCsv);
    end
    if ~isempty(Params.QueryUpdateCsv)
        writetable(struct2table(QueryUpdateReports,'AsArray',true),Params.QueryUpdateCsv);
    end
    if ~isempty(Params.CalibratorCsv)
        writetable(struct2table(CalibratorReports,'AsArray',true),Params.CalibratorCsv);
    end
    if ~isempty(Params.QueryPairedCsv)
        writetable(struct2table(Results.queryPaired,'AsArray',true),Params.QueryPairedCsv);
    end
    if ~isempty(Params.CalibratorPairedCsv)
        writetable(struct2table(Results.calibratorPaired,'AsArray',true),Params.CalibratorPairedCsv);
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

function RuntimeOptions = ResolveQueryRuntimeOptions(Params)
    RuntimeOverride = struct();
    RuntimeOverride.BridgeTopK = Params.BridgeTopK;
    RuntimeOptions = BuildBoundaryRuntimeOptions(RuntimeOverride);
end

function VariantRuntimeOptions = ResolveVariantRuntimeOptions(Params)
    BaseRuntimeOptions = ResolveQueryRuntimeOptions(Params);
    VariantRuntimeOptions = repmat({BaseRuntimeOptions},numel(Params.QueryVariants),1);
    for i = 1 : numel(VariantRuntimeOptions)
        Override = Params.QueryVariantRuntimeOverrides{i};
        if isempty(Override)
            continue;
        end
        VariantRuntimeOptions{i} = BuildBoundaryRuntimeOptions(BaseRuntimeOptions,Override);
    end
end

function Candidates = ResolveCalibratorCandidates(VariantName)
    switch lower(strtrim(char(VariantName)))
        case 'raw'
            Candidates = {'raw'};
        case {'temperature','temp'}
            Candidates = {'temperature'};
        case 'beta'
            Candidates = {'beta'};
        case {'auto','auto_trust','auto_selected'}
            Candidates = {'raw','temperature','beta'};
        otherwise
            error('PRBCCMO:Step2CalibratorVariant', ...
                'Unsupported calibrator variant ''%s''.',VariantName);
    end
end

function [Context,Meta] = ResolveOracleContext(Cache,ProblemName,Params)
    Meta = InitOracleMetaRow();
    Meta.problem = ProblemName;
    Meta.rawProblem = regexprep(ProblemName,'_BC$','');
    if isKey(Cache,ProblemName)
        Context = Cache(ProblemName);
        Meta.available = Context.available;
        Meta.rawProblem = Context.rawProblem;
        Meta.constraintCount = Context.constraintCount;
        Meta.scaleMode = Context.scaleMode;
        Meta.sampleCount = Context.sampleCount;
        return;
    end

    Context = struct();
    Context.available = false;
    Context.problem = ProblemName;
    Context.rawProblem = regexprep(ProblemName,'_BC$','');
    Context.scaleMode = char(Params.OracleScaleMode);
    Context.sampleCount = Params.OracleSampleCount;
    Context.scale = [];
    Context.constraintCount = 0;

    if strcmp(Context.problem,Context.rawProblem) || exist(Context.rawProblem,'class') ~= 8
        Cache(ProblemName) = Context;
        Meta.available = false;
        return;
    end

    RawProblem = feval(Context.rawProblem,'N',Params.Population,'maxFE',Params.MaxFE);
    Stream = RandStream('mt19937ar','Seed',HashName(Context.rawProblem) + 7919);
    Range = RawProblem.upper - RawProblem.lower;
    Samples = repmat(RawProblem.lower,Params.OracleSampleCount,1) + ...
        rand(Stream,Params.OracleSampleCount,RawProblem.D).*repmat(Range,Params.OracleSampleCount,1);
    RawCon = RawProblem.CalCon(Samples);
    if isempty(RawCon)
        Cache(ProblemName) = Context;
        return;
    end

    Context.available = true;
    Context.lower = RawProblem.lower;
    Context.upper = RawProblem.upper;
    Context.constraintCount = size(RawCon,2);
    Context.scale = EstimateOracleScale(RawCon,Params.OracleScaleMode,Params.OracleScaleEps);
    Context.eps = Params.OracleScaleEps;
    Cache(ProblemName) = Context;

    Meta.available = true;
    Meta.constraintCount = Context.constraintCount;
    Meta.scaleMode = Context.scaleMode;
    Meta.sampleCount = Context.sampleCount;
end

function Scale = EstimateOracleScale(RawCon,Mode,ScaleEps)
    if nargin < 3 || isempty(ScaleEps)
        ScaleEps = 1e-12;
    end
    switch lower(char(Mode))
        case 'std'
            Scale = std(RawCon,0,1);
        case 'mad'
            Center = median(RawCon,1,'omitnan');
            Scale = median(abs(RawCon - repmat(Center,size(RawCon,1),1)),1,'omitnan');
        case 'meanabs'
            Scale = mean(abs(RawCon),1,'omitnan');
        otherwise
            error('PRBCCMO:Step2OracleScaleMode', ...
                'Unsupported OracleScaleMode ''%s''.',char(Mode));
    end
    Fallback = mean(abs(RawCon),1,'omitnan');
    Invalid = ~isfinite(Scale) | Scale < ScaleEps;
    Scale(Invalid) = Fallback(Invalid);
    Invalid = ~isfinite(Scale) | Scale < ScaleEps;
    Scale(Invalid) = 1;
end

function Report = EvaluateCalibratorRun(Metric,ProblemName,CalibratorVariant,RunIndex)
    SectionB = FieldOrDefault(Metric,'sectionB',struct());
    CalTrace = FieldOrDefault(SectionB,'calibrationTrace',repmat(InitCalibrationTraceRowLocal(),0,1));
    UseRows = CalTrace([CalTrace.audit_ready]);
    if isempty(UseRows)
        UseRows = repmat(InitCalibrationTraceRowLocal(),0,1);
    end

    Report = InitCalibratorRunReport();
    Report.problem = ProblemName;
    Report.calibratorVariant = CalibratorVariant;
    Report.run = RunIndex;
    Report.updateCount = numel(CalTrace);
    Report.auditReadyUpdateCount = numel(UseRows);
    Report.boundaryStarted = any([CalTrace.boundary_started]);
    Report.meanECE = MeanStructField(UseRows,'ece');
    Report.meanCoreNearGap = MeanStructField(UseRows,'core_near_gap');
    Report.TWS = MeanStructField(UseRows,'trust_weight');
    Report.TGP = MeanLogicalField(UseRows,'trust_gate');
    Report.finalCalibrator = ResolveFinalCalibrator(CalTrace);
    Report.dominantCalibrator = ResolveDominantCalibrator(UseRows,CalTrace);
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

function [Reports,UpdateReports] = EvaluateQueryRuns(Metric,ProblemName,CalibratorVariant,RunIndex,OracleContext,Params)
    SectionB = FieldOrDefault(Metric,'sectionB',struct());
    CandidateAudit = FieldOrDefault(SectionB,'candidateAudit',repmat(InitCandidateAuditRowLocal(),0,1));
    SelectionTrace = FieldOrDefault(SectionB,'selectionTrace',repmat(InitSelectionTraceRowLocal(),0,1));
    CalTrace = FieldOrDefault(SectionB,'calibrationTrace',repmat(InitCalibrationTraceRowLocal(),0,1));
    VariantRuntimeOptions = ResolveVariantRuntimeOptions(Params);
    Reports = repmat(InitQueryRunReport(),numel(Params.QueryVariants),1);
    UpdateReports = repmat(InitQueryUpdateReport(),0,1);
    for v = 1 : numel(Params.QueryVariants)
        Reports(v).problem = ProblemName;
        Reports(v).calibratorVariant = CalibratorVariant;
        Reports(v).queryVariant = Params.QueryVariants{v};
        Reports(v).run = RunIndex;
    end
    if isempty(CandidateAudit) || isempty(SelectionTrace)
        return;
    end

    CandidateDec = cat(1,CandidateAudit.candidateDec);
    OracleDB = EvaluateOracleDistance(OracleContext,CandidateDec);
    Generations = [SelectionTrace.generation]';
    FEs = [SelectionTrace.FE]';
    Budgets = [SelectionTrace.budget]';

    VariantData = repmat(InitVariantAccumulator(),numel(Params.QueryVariants),1);
    ParetoVariantIdx = find(strcmp(Params.QueryVariants,'ParetoOnly'),1,'first');
    for i = 1 : numel(SelectionTrace)
        Budget = Budgets(i);
        if ~isfinite(Budget) || Budget <= 0
            continue;
        end
        Mask = [CandidateAudit.generation]' == Generations(i) & [CandidateAudit.FE]' == FEs(i);
        if ~any(Mask)
            continue;
        end
        GroupIdx = find(Mask);
        Rows = CandidateAudit(GroupIdx);
        TrustContext = ResolveTrustContext(CalTrace,SelectionTrace(i));
        PoolCandidateCount = numel(GroupIdx);
        BoundaryDispersion = EvaluateBoundaryScoreDispersion(Rows);
        PickIdx = cell(numel(Params.QueryVariants),1);
        PoolReports = repmat(InitQueryUpdateReport(),numel(Params.QueryVariants),1);
        for v = 1 : numel(Params.QueryVariants)
            [SelectedDB,SelectedScore,SelectedCount,EligibleCount,PickIdx{v}] = SelectVariantRows( ...
                CandidateAudit,OracleDB,GroupIdx,Budget,Params.QueryVariants{v}, ...
                TrustContext,VariantRuntimeOptions{v},ProblemName,CalibratorVariant,RunIndex,Generations(i),FEs(i));
            VariantData(v).poolCount = VariantData(v).poolCount + 1;
            VariantData(v).candidateCount = VariantData(v).candidateCount + PoolCandidateCount;
            VariantData(v).eligibleCount = VariantData(v).eligibleCount + EligibleCount;
            VariantData(v).selectedCount = VariantData(v).selectedCount + SelectedCount;
            if isfinite(BoundaryDispersion)
                VariantData(v).boundaryDispersion(end+1,1) = BoundaryDispersion; %#ok<AGROW>
            end
            if ~isempty(SelectedDB)
                VariantData(v).oracleDB = [VariantData(v).oracleDB;SelectedDB]; %#ok<AGROW>
                VariantData(v).score = [VariantData(v).score;SelectedScore]; %#ok<AGROW>
            end
            PoolReports(v).problem = ProblemName;
            PoolReports(v).calibratorVariant = CalibratorVariant;
            PoolReports(v).queryVariant = Params.QueryVariants{v};
            PoolReports(v).run = RunIndex;
            PoolReports(v).generation = Generations(i);
            PoolReports(v).FE = FEs(i);
            PoolReports(v).budget = max(0,round(Budget));
            PoolReports(v).candidateCount = PoolCandidateCount;
            PoolReports(v).eligibleCount = EligibleCount;
            PoolReports(v).selectedCount = SelectedCount;
            PoolReports(v).boundaryScoreDispersion = BoundaryDispersion;
            FiniteMask = isfinite(SelectedDB) & isfinite(SelectedScore);
            PoolReports(v).oracleCount = sum(FiniteMask);
            if any(FiniteMask)
                OracleValues = SelectedDB(FiniteMask);
                ScoreValues = SelectedScore(FiniteMask);
                PoolReports(v).meanDB = mean(OracleValues,'omitnan');
                PoolReports(v).medianDB = median(OracleValues,'omitnan');
                PoolReports(v).QPTau = mean(OracleValues <= Params.OracleTau,'omitnan');
                PoolReports(v).rhoUtilityNegDB = SafeSpearman(ScoreValues,-OracleValues);
            end
        end
        if ~isempty(ParetoVariantIdx)
            BasePick = PickIdx{ParetoVariantIdx};
            for v = 1 : numel(Params.QueryVariants)
                Overlap = ComputeSelectionOverlap(BasePick,PickIdx{v},Budget);
                Differs = SelectionsDiffer(BasePick,PickIdx{v});
                if isfinite(Overlap)
                    VariantData(v).selectionOverlap(end+1,1) = Overlap; %#ok<AGROW>
                end
                VariantData(v).selectionDiffers(end+1,1) = double(Differs); %#ok<AGROW>
                PoolReports(v).selectionOverlapPareto = Overlap;
                PoolReports(v).selectionDiffersPareto = Differs;
            end
        end
        UpdateReports(end + (1:numel(PoolReports)),1) = PoolReports; %#ok<AGROW>
    end

    Gate = ResolveOracleAuditGate(VariantData,Params);
    for i = 1 : numel(UpdateReports)
        UpdateReports(i).oracleAuditEnabled = Gate.oracleAuditEnabled;
        UpdateReports(i).stopGoReason = Gate.stopGoReason;
        UpdateReports(i).stopGoTargetVariant = Gate.targetVariant;
        UpdateReports(i).stopGoBaseVariant = Gate.baseVariant;
        UpdateReports(i).stopGoMedianSelectionOverlapPareto = Gate.medianSelectionOverlapPareto;
        UpdateReports(i).stopGoSelectionDiffPoolRate = Gate.selectionDiffPoolRate;
        UpdateReports(i).stopGoMedianBoundaryScoreDispersion = Gate.medianBoundaryScoreDispersion;
        if ~Gate.oracleAuditEnabled
            UpdateReports(i).oracleCount = 0;
            UpdateReports(i).meanDB = NaN;
            UpdateReports(i).medianDB = NaN;
            UpdateReports(i).QPTau = NaN;
            UpdateReports(i).rhoUtilityNegDB = NaN;
        end
    end

    for v = 1 : numel(Params.QueryVariants)
        Reports(v).poolCount = VariantData(v).poolCount;
        Reports(v).candidateCount = VariantData(v).candidateCount;
        Reports(v).eligibleCount = VariantData(v).eligibleCount;
        Reports(v).selectedCount = VariantData(v).selectedCount;
        Reports(v).oracleAuditEnabled = Gate.oracleAuditEnabled;
        Reports(v).stopGoReason = Gate.stopGoReason;
        Reports(v).stopGoTargetVariant = Gate.targetVariant;
        Reports(v).stopGoBaseVariant = Gate.baseVariant;
        Reports(v).stopGoMedianSelectionOverlapPareto = Gate.medianSelectionOverlapPareto;
        Reports(v).stopGoSelectionDiffPoolRate = Gate.selectionDiffPoolRate;
        Reports(v).stopGoMedianBoundaryScoreDispersion = Gate.medianBoundaryScoreDispersion;
        Reports(v).overlapPoolCount = numel(VariantData(v).selectionOverlap);
        Reports(v).meanSelectionOverlapPareto = mean(VariantData(v).selectionOverlap,'omitnan');
        Reports(v).medianSelectionOverlapPareto = median(VariantData(v).selectionOverlap,'omitnan');
        Reports(v).selectionDiffPoolCount = sum(VariantData(v).selectionDiffers,'omitnan');
        Reports(v).selectionDiffPoolRate = mean(VariantData(v).selectionDiffers,'omitnan');
        Reports(v).dispersionPoolCount = numel(VariantData(v).boundaryDispersion);
        Reports(v).meanBoundaryScoreDispersion = mean(VariantData(v).boundaryDispersion,'omitnan');
        Reports(v).medianBoundaryScoreDispersion = median(VariantData(v).boundaryDispersion,'omitnan');
        if ~Gate.oracleAuditEnabled
            continue;
        end
        FiniteMask = isfinite(VariantData(v).oracleDB) & isfinite(VariantData(v).score);
        Reports(v).oracleCount = sum(FiniteMask);
        if ~any(FiniteMask)
            continue;
        end
        OracleValues = VariantData(v).oracleDB(FiniteMask);
        ScoreValues = VariantData(v).score(FiniteMask);
        Reports(v).meanDB = mean(OracleValues,'omitnan');
        Reports(v).medianDB = median(OracleValues,'omitnan');
        Reports(v).QPTau = mean(OracleValues <= Params.OracleTau,'omitnan');
        Reports(v).rhoUtilityNegDB = SafeSpearman(ScoreValues,-OracleValues);
    end
end

function [SelectedDB,SelectedScore,SelectedCount,EligibleCount,Pick] = SelectVariantRows( ...
    CandidateAudit,OracleDB,GroupIdx,Budget,VariantName,TrustContext,RuntimeOptions, ...
    ProblemName,CalibratorVariant,RunIndex,Generation,FE)

    Rows = CandidateAudit(GroupIdx);
    Eligible = ExtractRowField(Rows,'eligible',false) ~= 0;
    EligibleCount = sum(Eligible);
    Score = ResolveVariantScore( ...
        Rows,VariantName,TrustContext,RuntimeOptions,Budget, ...
        ProblemName,CalibratorVariant,RunIndex,Generation,FE);
    Valid = Eligible & isfinite(Score);
    ValidIdx = find(Valid);
    if isempty(ValidIdx)
        SelectedDB = zeros(0,1);
        SelectedScore = zeros(0,1);
        SelectedCount = 0;
        Pick = zeros(0,1);
        return;
    end

    Keep = min(max(0,round(Budget)),numel(ValidIdx));
    if Keep <= 0
        SelectedDB = zeros(0,1);
        SelectedScore = zeros(0,1);
        SelectedCount = 0;
        Pick = zeros(0,1);
        return;
    end

    RankTable = [-Score(ValidIdx),ValidIdx];
    RankTable = sortrows(RankTable,[1 2]);
    Pick = RankTable(1:Keep,2);
    SelectedDB = OracleDB(GroupIdx(Pick));
    SelectedScore = Score(Pick);
    SelectedCount = numel(Pick);
end

function Score = ResolveVariantScore( ...
    Rows,VariantName,TrustContext,RuntimeOptions,Budget, ...
    ProblemName,CalibratorVariant,RunIndex,Generation,FE)

    switch ResolveQueryVariantFamily(VariantName)
        case {'full','fullv2'}
            Score = ComputeBoundarySelectorUtility( ...
                'FullV2',BuildSelectorDetailFromRows(Rows,TrustContext),TrustContext,Budget,RuntimeOptions);
        case {'fullv2current','full_v2_current','full-v2-current'}
            Score = ComputeBoundarySelectorUtility( ...
                'FullV2Current',BuildSelectorDetailFromRows(Rows,TrustContext),TrustContext,Budget,RuntimeOptions);
        case {'fullnotrust','full_no_trust','full-no-trust'}
            Score = ComputeBoundarySelectorUtility( ...
                'FullNoTrust',BuildSelectorDetailFromRows(Rows,TrustContext),TrustContext,Budget,RuntimeOptions);
        case {'fullsofttrust','full_soft_trust','full-soft-trust'}
            Score = ComputeBoundarySelectorUtility( ...
                'FullSoftTrust',BuildSelectorDetailFromRows(Rows,TrustContext),TrustContext,Budget,RuntimeOptions);
        case 'paretoonly'
            Score = ExtractRowField(Rows,'paretoValue',NaN);
        case 'uncertainonly'
            Score = ExtractRowField(Rows,'queryScore',NaN);
        case 'highprobboundary'
            Score = ExtractRowField(Rows,'prob',NaN);
        case 'randboundary'
            Seed = HashName(sprintf('%s|%s|%d|%d|%d', ...
                ProblemName,CalibratorVariant,RunIndex,Generation,FE));
            Stream = RandStream('mt19937ar','Seed',Seed);
            Score = rand(Stream,numel(Rows),1);
        otherwise
            error('PRBCCMO:Step2QueryVariant', ...
                'Unsupported query variant ''%s''.',char(VariantName));
    end
end

function Detail = BuildSelectorDetailFromRows(Rows,TrustContext)
    Count = numel(Rows);
    Detail = struct();
    Detail.eligible = ExtractRowField(Rows,'eligible',false);
    Detail.paretoValue = ExtractRowField(Rows,'paretoValue',NaN);
    Detail.boundaryTrust = ExtractRowField(Rows,'boundaryTrust',NaN);
    Detail.trustWeight = ExtractRowField(Rows,'trustWeight',NaN);
    Detail.trustGate = repmat(logical(FieldOrDefault(TrustContext,'trustGate',false)),Count,1);
    Detail.trustECE = repmat(FieldOrDefault(TrustContext,'ece',inf),Count,1);
    Detail.trustCoreNearGap = repmat(FieldOrDefault(TrustContext,'coreNearGap',inf),Count,1);
end

function Dispersion = EvaluateBoundaryScoreDispersion(Rows)
    Eligible = ExtractRowField(Rows,'eligible',false) ~= 0;
    BoundaryScore = ExtractRowField(Rows,'boundaryTrust',NaN);
    BoundaryScore = BoundaryScore(Eligible & isfinite(BoundaryScore));
    if isempty(BoundaryScore)
        Dispersion = NaN;
        return;
    end
    Dispersion = std(BoundaryScore,0,'omitnan');
end

function Overlap = ComputeSelectionOverlap(BasePick,ComparePick,Budget)
    Overlap = NaN;
    Budget = round(Budget);
    if Budget <= 0
        return;
    end
    Overlap = numel(intersect(BasePick(:),ComparePick(:))) / Budget;
end

function Differs = SelectionsDiffer(BasePick,ComparePick)
    Differs = false;
    BasePick = sort(BasePick(:));
    ComparePick = sort(ComparePick(:));
    if numel(BasePick) ~= numel(ComparePick)
        Differs = true;
        return;
    end
    if isempty(BasePick) && isempty(ComparePick)
        return;
    end
    Differs = any(BasePick ~= ComparePick);
end

function Gate = ResolveOracleAuditGate(VariantData,Params)
    Gate = struct( ...
        'oracleAuditEnabled',true, ...
        'stopGoReason','gate_disabled', ...
        'targetVariant',Params.StopGoTargetVariant, ...
        'baseVariant',Params.StopGoBaseVariant, ...
        'medianSelectionOverlapPareto',NaN, ...
        'selectionDiffPoolRate',NaN, ...
        'medianBoundaryScoreDispersion',NaN);
    if ~Params.EnableStopGoGate
        return;
    end

    TargetIdx = find(strcmp(Params.QueryVariants,Params.StopGoTargetVariant),1,'first');
    BaseIdx = find(strcmp(Params.QueryVariants,Params.StopGoBaseVariant),1,'first');
    if isempty(TargetIdx) || isempty(BaseIdx)
        Gate.stopGoReason = 'missing_target_or_base_variant';
        return;
    end

    Overlap = VariantData(TargetIdx).selectionOverlap;
    Differs = VariantData(TargetIdx).selectionDiffers;
    Dispersion = VariantData(TargetIdx).boundaryDispersion;
    Gate.medianSelectionOverlapPareto = median(Overlap,'omitnan');
    Gate.selectionDiffPoolRate = mean(Differs,'omitnan');
    Gate.medianBoundaryScoreDispersion = median(Dispersion,'omitnan');

    Reasons = cell(1,0);
    if ~isfinite(Gate.medianSelectionOverlapPareto)
        Reasons{end+1} = 'missing_overlap'; %#ok<AGROW>
    elseif Gate.medianSelectionOverlapPareto >= Params.StopGoOverlapMedianMax
        Reasons{end+1} = 'overlap_collapsed'; %#ok<AGROW>
    end
    if ~isfinite(Gate.selectionDiffPoolRate)
        Reasons{end+1} = 'missing_diff_rate'; %#ok<AGROW>
    elseif Gate.selectionDiffPoolRate < Params.StopGoSelectionDiffRateMin
        Reasons{end+1} = 'diff_rate_too_low'; %#ok<AGROW>
    end
    if ~isfinite(Gate.medianBoundaryScoreDispersion)
        Reasons{end+1} = 'missing_dispersion'; %#ok<AGROW>
    elseif Gate.medianBoundaryScoreDispersion <= Params.StopGoDispersionMedianMin
        Reasons{end+1} = 'dispersion_too_low'; %#ok<AGROW>
    end

    if isempty(Reasons)
        Gate.oracleAuditEnabled = true;
        Gate.stopGoReason = 'oracle_audit_enabled';
    else
        Gate.oracleAuditEnabled = false;
        Gate.stopGoReason = strjoin(Reasons,'|');
    end
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

function OracleDB = EvaluateOracleDistance(Context,Dec)
    OracleDB = NaN(size(Dec,1),1);
    if isempty(Dec) || ~isstruct(Context) || ~FieldOrDefault(Context,'available',false)
        return;
    end
    RawProblem = feval(Context.rawProblem,'N',1);
    RawCon = RawProblem.CalCon(Dec);
    Scale = repmat(Context.scale,size(RawCon,1),1);
    OracleDB = min(abs(RawCon)./(Scale + Context.eps),[],2);
end

function Summary = SummarizeQueryReports(QueryReports,QueryVariants,CalibratorVariants)
    Rows = repmat(InitQuerySummaryRow(),0,1);
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
            Rows(end+1,1) = Row; %#ok<AGROW>
        end
    end
    Summary = Rows;
end

function Summary = SummarizeCalibratorReports(CalibratorReports,CalibratorVariants)
    Rows = repmat(InitCalibratorSummaryRow(),0,1);
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
        Rows(end+1,1) = Row; %#ok<AGROW>
    end
    Summary = Rows;
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

function Value = MeanStructField(Rows,Field)
    Value = NaN;
    if isempty(Rows)
        return;
    end
    Data = [Rows.(Field)];
    Value = mean(Data,'omitnan');
end

function Value = MeanLogicalField(Rows,Field)
    Value = NaN;
    if isempty(Rows)
        return;
    end
    Data = double([Rows.(Field)]);
    Value = mean(Data,'omitnan');
end

function Value = ResolveFinalCalibrator(CalTrace)
    Value = '';
    if isempty(CalTrace)
        return;
    end
    Last = find(~cellfun(@isempty,{CalTrace.calibrator}),1,'last');
    if isempty(Last)
        return;
    end
    Value = CalTrace(Last).calibrator;
end

function Value = ResolveDominantCalibrator(UseRows,AllRows)
    Value = ResolveFinalCalibrator(AllRows);
    if isempty(UseRows)
        return;
    end
    Names = {UseRows.calibrator};
    Names = Names(~cellfun(@isempty,Names));
    if isempty(Names)
        return;
    end
    UniqueNames = unique(Names,'stable');
    Count = zeros(numel(UniqueNames),1);
    for i = 1 : numel(UniqueNames)
        Count(i) = sum(strcmp(Names,UniqueNames{i}));
    end
    [~,Best] = max(Count);
    Value = UniqueNames{Best};
end

function PrintRunSummary(ProblemName,RunIndex,CalibratorReport,QueryReports)
    TargetVariant = ResolveSummaryTargetVariant(QueryReports);
    TargetMask = strcmp({QueryReports.queryVariant},TargetVariant);
    ParetoMask = strcmp({QueryReports.queryVariant},'ParetoOnly');
    RandMask = strcmp({QueryReports.queryVariant},'RandBoundary');
    TargetRow = QueryReports(TargetMask);
    ParetoRow = QueryReports(ParetoMask);
    RandRow = QueryReports(RandMask);
    fprintf(['[Step2] %s run %d %s -> finalCal=%s audit=%d ECE=%.4f CoreGap=%.4f ', ...
        'TWS=%.4f TGP=%.4f gate=%d %s(mean=%.4f ov=%.4f diff=%.4f) Pareto(mean=%.4f) Rand(mean=%.4f) Disp=%.4f\n'], ...
        ProblemName,RunIndex,CalibratorReport.calibratorVariant, ...
        CalibratorReport.finalCalibrator,CalibratorReport.auditReadyUpdateCount, ...
        CalibratorReport.meanECE,CalibratorReport.meanCoreNearGap, ...
        CalibratorReport.TWS,CalibratorReport.TGP, ...
        SafeScalarField(TargetRow,'oracleAuditEnabled'),TargetVariant, ...
        SafeScalarField(TargetRow,'meanDB'),SafeScalarField(TargetRow,'medianSelectionOverlapPareto'), ...
        SafeScalarField(TargetRow,'selectionDiffPoolRate'), ...
        SafeScalarField(ParetoRow,'meanDB'),SafeScalarField(RandRow,'meanDB'), ...
        SafeScalarField(TargetRow,'medianBoundaryScoreDispersion'));
end

function Name = ResolveSummaryTargetVariant(QueryReports)
    Preferred = {'FullSoftTrust','FullV2','FullV2Current','FullNoTrust','UncertainOnly','HighProbBoundary'};
    Names = {QueryReports.queryVariant};
    Name = 'ParetoOnly';
    for i = 1 : numel(Preferred)
        if any(strcmp(Names,Preferred{i}))
            Name = Preferred{i};
            return;
        end
    end
    for i = 1 : numel(Names)
        if ~strcmp(Names{i},'ParetoOnly')
            Name = Names{i};
            return;
        end
    end
end

function Name = ResolveBaseQueryVariant(QueryVariants)
    Name = QueryVariants{1};
    if any(strcmp(QueryVariants,'ParetoOnly'))
        Name = 'ParetoOnly';
    end
end

function Name = ResolveStopGoTargetVariant(QueryVariants,PreferredName,BaseVariant)
    PreferredName = CanonicalQueryVariant(PreferredName);
    if any(strcmp(QueryVariants,PreferredName))
        Name = PreferredName;
        return;
    end

    FallbackOrder = {'FullSoftTrust','FullV2','FullV2Current', ...
        'FullNoTrust','UncertainOnly','HighProbBoundary','RandBoundary'};
    for i = 1 : numel(FallbackOrder)
        Candidate = FallbackOrder{i};
        if strcmp(Candidate,BaseVariant)
            continue;
        end
        if any(strcmp(QueryVariants,Candidate))
            Name = Candidate;
            return;
        end
    end

    Name = QueryVariants{1};
    for i = 1 : numel(QueryVariants)
        if ~strcmp(QueryVariants{i},BaseVariant)
            Name = QueryVariants{i};
            return;
        end
    end
end

function Value = SafeScalarField(Row,Field)
    Value = NaN;
    if isempty(Row)
        return;
    end
    Value = Row(1).(Field);
end

function Value = SafeScalarStructField(Row,Field,Default)
    if nargin < 3
        Default = NaN;
    end
    Value = Default;
    if isempty(Row) || ~isstruct(Row) || ~isfield(Row,Field) || isempty(Row.(Field))
        Value = Default;
        return;
    end
    Value = Row.(Field);
end

function Value = FieldOrDefault(S,Field,Default)
    if isstruct(S) && isfield(S,Field) && ~isempty(S.(Field))
        Value = S.(Field);
    else
        Value = Default;
    end
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('PRBCCMO:Step2Input', ...
            'Inputs must be provided as name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~isfield(Params,Name)
            error('PRBCCMO:Step2Input', ...
                'Unknown parameter ''%s''.',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function Variants = NormalizeQueryVariantList(Variants)
    if ischar(Variants) || (isstring(Variants) && isscalar(Variants))
        Variants = {char(Variants)};
    end
    Variants = cellfun(@CanonicalQueryVariant,Variants,'UniformOutput',false);
end

function Overrides = NormalizeQueryVariantRuntimeOverrides(Overrides,VariantCount)
    if isempty(Overrides)
        Overrides = cell(VariantCount,1);
        return;
    end
    if isstruct(Overrides)
        Overrides = num2cell(Overrides(:));
    elseif ~iscell(Overrides)
        error('PRBCCMO:Step2VariantRuntimeOverrides', ...
            'QueryVariantRuntimeOverrides must be empty, a struct array, or a cell array.');
    end
    if isrow(Overrides)
        Overrides = Overrides(:);
    end
    if numel(Overrides) ~= VariantCount
        error('PRBCCMO:Step2VariantRuntimeOverrides', ...
            'QueryVariantRuntimeOverrides must align with QueryVariants.');
    end
    for i = 1 : numel(Overrides)
        if isempty(Overrides{i})
            Overrides{i} = struct();
        elseif ~isstruct(Overrides{i})
            error('PRBCCMO:Step2VariantRuntimeOverrides', ...
                'Each QueryVariantRuntimeOverrides entry must be a struct.');
        end
    end
end

function Variants = NormalizeCalibratorVariantList(Variants)
    if ischar(Variants) || (isstring(Variants) && isscalar(Variants))
        Variants = {char(Variants)};
    end
    Variants = cellfun(@CanonicalCalibratorVariant,Variants,'UniformOutput',false);
end

function RunIndices = NormalizeRunIndices(RunIndices,Runs)
    if isempty(RunIndices)
        RunIndices = 1 : Runs;
        return;
    end
    RunIndices = unique(double(RunIndices(:))','stable');
    RunIndices = RunIndices(isfinite(RunIndices) & RunIndices >= 1);
    RunIndices = round(RunIndices);
end

function Value = HashName(Text)
    Bytes = double(char(Text));
    Value = mod(sum((1:numel(Bytes)) .* Bytes),2^31-1);
    Value = max(Value,1);
end

function Name = CanonicalQueryVariant(Name)
    RawName = strtrim(char(Name));
    Name = lower(RawName);
    switch Name
        case {'full','fullv2','full_v2','full-v2'}
            Name = 'FullV2';
        case {'fullv2current','full_v2_current','full-v2-current','current'}
            Name = 'FullV2Current';
        case {'fullnotrust','full_no_trust','full-no-trust','notrust'}
            Name = 'FullNoTrust';
        case {'fullsofttrust','full_soft_trust','full-soft-trust','softtrust'}
            Name = 'FullSoftTrust';
        case {'paretoonly','pareto_only','pareto-only'}
            Name = 'ParetoOnly';
        case {'uncertainonly','uncertain_only','uncertain-only'}
            Name = 'UncertainOnly';
        case {'highprobboundary','high_prob_boundary','highprob-boundary'}
            Name = 'HighProbBoundary';
        case {'randboundary','rand_boundary','rand-boundary','random'}
            Name = 'RandBoundary';
        otherwise
            if IsCustomQueryVariantAlias(Name)
                Name = RawName;
            else
                error('PRBCCMO:Step2QueryVariant', ...
                    'Unsupported query variant ''%s''.',char(RawName));
            end
    end
end

function Family = ResolveQueryVariantFamily(Name)
    RawName = lower(strtrim(char(Name)));
    if startsWith(RawName,'fullsofttrust__')
        Family = 'fullsofttrust';
        return;
    end
    Family = RawName;
end

function Flag = IsCustomQueryVariantAlias(Name)
    Name = lower(strtrim(char(Name)));
    Flag = startsWith(Name,'fullsofttrust__');
end

function Name = CanonicalCalibratorVariant(Name)
    Name = lower(strtrim(char(Name)));
    switch Name
        case 'raw'
            Name = 'raw';
        case {'temperature','temp'}
            Name = 'temperature';
        case 'beta'
            Name = 'beta';
        case {'auto','auto_trust','auto_selected'}
            Name = 'auto';
        otherwise
            error('PRBCCMO:Step2CalibratorVariant', ...
                'Unsupported calibrator variant ''%s''.',char(Name));
    end
end

function Row = InitQueryRunReport()
    Row = struct( ...
        'problem','', ...
        'calibratorVariant','', ...
        'queryVariant','', ...
        'run',0, ...
        'poolCount',0, ...
        'candidateCount',0, ...
        'eligibleCount',0, ...
        'selectedCount',0, ...
        'oracleAuditEnabled',false, ...
        'stopGoReason','', ...
        'stopGoTargetVariant','', ...
        'stopGoBaseVariant','', ...
        'stopGoMedianSelectionOverlapPareto',NaN, ...
        'stopGoSelectionDiffPoolRate',NaN, ...
        'stopGoMedianBoundaryScoreDispersion',NaN, ...
        'oracleCount',0, ...
        'meanDB',NaN, ...
        'medianDB',NaN, ...
        'QPTau',NaN, ...
        'rhoUtilityNegDB',NaN, ...
        'overlapPoolCount',0, ...
        'meanSelectionOverlapPareto',NaN, ...
        'medianSelectionOverlapPareto',NaN, ...
        'selectionDiffPoolCount',0, ...
        'selectionDiffPoolRate',NaN, ...
        'dispersionPoolCount',0, ...
        'meanBoundaryScoreDispersion',NaN, ...
        'medianBoundaryScoreDispersion',NaN);
end

function Row = InitQueryUpdateReport()
    Row = struct( ...
        'problem','', ...
        'calibratorVariant','', ...
        'queryVariant','', ...
        'run',0, ...
        'generation',NaN, ...
        'FE',NaN, ...
        'budget',0, ...
        'candidateCount',0, ...
        'eligibleCount',0, ...
        'selectedCount',0, ...
        'oracleCount',0, ...
        'selectionOverlapPareto',NaN, ...
        'selectionDiffersPareto',false, ...
        'boundaryScoreDispersion',NaN, ...
        'oracleAuditEnabled',false, ...
        'stopGoReason','', ...
        'stopGoTargetVariant','', ...
        'stopGoBaseVariant','', ...
        'stopGoMedianSelectionOverlapPareto',NaN, ...
        'stopGoSelectionDiffPoolRate',NaN, ...
        'stopGoMedianBoundaryScoreDispersion',NaN, ...
        'meanDB',NaN, ...
        'medianDB',NaN, ...
        'QPTau',NaN, ...
        'rhoUtilityNegDB',NaN);
end

function Row = InitCalibratorRunReport()
    Row = struct( ...
        'problem','', ...
        'calibratorVariant','', ...
        'run',0, ...
        'updateCount',0, ...
        'auditReadyUpdateCount',0, ...
        'boundaryStarted',false, ...
        'meanECE',NaN, ...
        'meanCoreNearGap',NaN, ...
        'TWS',NaN, ...
        'TGP',NaN, ...
        'finalCalibrator','', ...
        'dominantCalibrator','');
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

function Row = InitCandidateAuditRowLocal()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'eligible',false, ...
        'prob',NaN, ...
        'queryScore',NaN, ...
        'paretoValue',NaN, ...
        'utility',NaN, ...
        'boundaryTrust',NaN, ...
        'trustWeight',NaN, ...
        'fullV2Utility',NaN, ...
        'fullV2Shortlisted',false, ...
        'candidateDec',zeros(1,0));
end

function Row = InitSelectionTraceRowLocal()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'budget',0);
end

function Row = InitCalibrationTraceRowLocal()
    Row = struct( ...
        'audit_ready',false, ...
        'boundary_started',false, ...
        'ece',NaN, ...
        'core_near_gap',NaN, ...
        'trust_weight',NaN, ...
        'trust_gate',false, ...
        'calibrator','');
end

function Row = InitOracleMetaRow()
    Row = struct( ...
        'problem','', ...
        'rawProblem','', ...
        'available',false, ...
        'constraintCount',0, ...
        'scaleMode','', ...
        'sampleCount',0);
end

function Data = InitVariantAccumulator()
    Data = struct( ...
        'poolCount',0, ...
        'candidateCount',0, ...
        'eligibleCount',0, ...
        'selectedCount',0, ...
        'oracleDB',zeros(0,1), ...
        'score',zeros(0,1), ...
        'selectionOverlap',zeros(0,1), ...
        'selectionDiffers',zeros(0,1), ...
        'boundaryDispersion',zeros(0,1));
end

function NoOutput(~,~)
end
