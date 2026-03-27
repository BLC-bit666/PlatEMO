function Results = check_PRBCCMO_step0_engineering(varargin)
% Step 0 engineering legality check for PRBCCMO-Lite.
%
% Optional name-value pairs:
%   'Problems'    : cellstr problem list, default {'DASCMOP1_BC','LIRCMOP1_BC','MW1_BC'}
%   'Runs'        : independent runs per problem, default 1
%   'Population'  : population size, default 100
%   'MaxFE'       : maximum function evaluations, default 20000
%   'SavePath'    : MAT output path, default 'prbccmo_step0_engineering.mat'
%   'SummaryCsv'  : CSV output path, default 'prbccmo_step0_engineering_runs.csv'
%   'Verbose'     : print run summaries, default true

    Params = struct( ...
        'Problems',{{'DASCMOP1_BC','LIRCMOP1_BC','MW1_BC'}}, ...
        'Runs',1, ...
        'Population',100, ...
        'MaxFE',20000, ...
        'SavePath','prbccmo_step0_engineering.mat', ...
        'SummaryCsv','prbccmo_step0_engineering_runs.csv', ...
        'Verbose',true);
    Params = ParseInputs(Params,varargin{:});

    RunReports = repmat(InitRunReport(),0,1);
    RunDetails = repmat(InitRunDetail(),0,1);
    Row = 0;
    for p = 1 : numel(Params.Problems)
        ProblemName = Params.Problems{p};
        for r = 1 : Params.Runs
            rng(r,'twister');
            Problem = feval(ProblemName,'N',Params.Population,'maxFE',Params.MaxFE);
            Algorithm = PRBCCMO('save',0);
            Algorithm.Solve(Problem);

            Row = Row + 1;
            Report = EvaluateSingleRun(Algorithm.metric,ProblemName,r);
            RunReports(Row,1) = Report;
            RunDetails(Row,1) = BuildRunDetail(Algorithm.metric,ProblemName,r);
            if Params.Verbose
                fprintf('[Step0] %s run %d: pass=%d, updates=%d, candidates=%d, seeds=%d, added=%d\n', ...
                    ProblemName,r,Report.passed,Report.updateAuditCount, ...
                    Report.totalCandidateRows,Report.totalSeedRows,Report.totalArchiveEventRows);
            end
        end
    end

    Summary = SummarizeRunReports(RunReports);
    Results = struct();
    Results.params = Params;
    Results.runReport = RunReports;
    Results.summary = Summary;
    Results.detail = RunDetails;

    if ~isempty(Params.SavePath)
        save(Params.SavePath,'Results');
    end
    if ~isempty(Params.SummaryCsv)
        writetable(struct2table(RunReports,'AsArray',true),Params.SummaryCsv);
    end
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('PRBCCMO:Step0Input','Name-value inputs must appear in pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        Value = varargin{i+1};
        if ~ischar(Name) && ~isstring(Name)
            error('PRBCCMO:Step0Input','Parameter names must be strings.');
        end
        Name = char(Name);
        if ~isfield(Params,Name)
            error('PRBCCMO:Step0Input','Unknown parameter: %s',Name);
        end
        Params.(Name) = Value;
    end
    if ischar(Params.Problems) || isstring(Params.Problems)
        Params.Problems = cellstr(Params.Problems);
    end
end

function Report = EvaluateSingleRun(Metric,ProblemName,RunIndex)
    Report = InitRunReport();
    Report.problem = ProblemName;
    Report.run = RunIndex;

    if ~isstruct(Metric) || ~isfield(Metric,'sectionB')
        Report.failure_reason = 'missing_sectionB_metric';
        return;
    end
    SectionB = Metric.sectionB;

    UpdateAudit = FieldOrDefault(SectionB,'updateAudit',repmat(InitBoundaryUpdateAuditRowLocal(),0,1));
    SelectionTrace = FieldOrDefault(SectionB,'selectionTrace',repmat(InitSelectionTraceRowLocal(),0,1));
    CandidateAudit = FieldOrDefault(SectionB,'candidateAudit',repmat(InitCandidateAuditRowLocal(),0,1));
    SeedAudit = FieldOrDefault(SectionB,'seedAudit',repmat(InitSeedAuditRowLocal(),0,1));
    BoundaryLineage = FieldOrDefault(SectionB,'boundaryLineage',repmat(InitBoundaryLineageRowLocal(),0,1));
    BoundaryGainTrace = FieldOrDefault(SectionB,'boundaryGainTrace',repmat(InitBoundaryGainTraceRowLocal(),0,1));
    ArchiveEvent = FieldOrDefault(SectionB,'archiveEvent',repmat(InitArchiveEventRowLocal(),0,1));

    Report.updateAuditCount = numel(UpdateAudit);
    Report.totalCandidateRows = numel(CandidateAudit);
    Report.totalSeedRows = numel(SeedAudit);
    Report.totalLineageRows = numel(BoundaryLineage);
    Report.totalArchiveEventRows = numel(ArchiveEvent);
    Report.boundaryStarted = any([SelectionTrace.selectedCount] > 0) || ~isempty(ArchiveEvent);

    [Report.updateAuditComplete,Report.updateAuditMismatchCount, ...
        Report.maxTrainCalOverlap,Report.maxTrainTestOverlap,Report.maxCalTestOverlap] = ...
        CheckUpdateAudit(UpdateAudit);
    [Report.candidateAuditComplete,Report.candidateAuditMismatchCount] = ...
        CheckGroupedCount(SelectionTrace,'candidateCount',CandidateAudit);
    [Report.seedAuditComplete,Report.seedAuditMismatchCount] = ...
        CheckGroupedCount(SelectionTrace,'selectedCount',SeedAudit);
    [Report.boundaryLineageComplete,Report.boundaryLineageMismatchCount] = ...
        CheckGroupedCount(SelectionTrace,'selectedCount',BoundaryLineage);
    [Report.archiveEventComplete,Report.archiveEventMismatchCount] = ...
        CheckGroupedCount(BoundaryGainTrace,'boundaryAddedCount',ArchiveEvent);

    Report.bufferSeparationPass = Report.maxTrainCalOverlap == 0 && ...
        Report.maxTrainTestOverlap == 0 && Report.maxCalTestOverlap == 0;
    Report.runLevelSummaryPass = true;
    Report.passed = Report.bufferSeparationPass && Report.updateAuditComplete && ...
        Report.candidateAuditComplete && Report.seedAuditComplete && ...
        Report.boundaryLineageComplete && Report.archiveEventComplete && ...
        Report.runLevelSummaryPass;
    if ~Report.passed
        Report.failure_reason = BuildFailureReason(Report);
    end
end

function [Pass,MismatchCount,MaxTrainCal,MaxTrainTest,MaxCalTest] = CheckUpdateAudit(UpdateAudit)
    Pass = ~isempty(UpdateAudit);
    MismatchCount = 0;
    MaxTrainCal = 0;
    MaxTrainTest = 0;
    MaxCalTest = 0;
    if isempty(UpdateAudit)
        return;
    end
    for i = 1 : numel(UpdateAudit)
        Row = UpdateAudit(i);
        ProbCount = numel(FieldOrDefault(Row,'prob',zeros(0,1)));
        LabelCount = numel(FieldOrDefault(Row,'label',zeros(0,1)));
        Count = FieldOrDefault(Row,'count',0);
        StrictSeparation = logical(FieldOrDefault(Row,'strict_separation',false));
        TrainCal = FieldOrDefault(Row,'train_cal_overlap',inf);
        TrainTest = FieldOrDefault(Row,'train_test_overlap',inf);
        CalTest = FieldOrDefault(Row,'cal_test_overlap',inf);
        MaxTrainCal = max(MaxTrainCal,TrainCal);
        MaxTrainTest = max(MaxTrainTest,TrainTest);
        MaxCalTest = max(MaxCalTest,CalTest);
        RowPass = StrictSeparation && ProbCount == LabelCount && Count == LabelCount;
        if ~RowPass
            Pass = false;
            MismatchCount = MismatchCount + 1;
        end
    end
end

function [Pass,MismatchCount] = CheckGroupedCount(TraceRows,CountField,AuditRows)
    Pass = true;
    MismatchCount = 0;
    if isempty(TraceRows)
        Pass = isempty(AuditRows);
        return;
    end
    AuditFE = ExtractFEVector(AuditRows);
    for i = 1 : numel(TraceRows)
        FE = FieldOrDefault(TraceRows(i),'FE',NaN);
        Expected = FieldOrDefault(TraceRows(i),CountField,0);
        Actual = sum(AuditFE == FE);
        if Actual ~= Expected
            Pass = false;
            MismatchCount = MismatchCount + 1;
        end
    end
end

function FE = ExtractFEVector(Rows)
    if isempty(Rows)
        FE = zeros(0,1);
        return;
    end
    FE = zeros(numel(Rows),1);
    for i = 1 : numel(Rows)
        FE(i) = FieldOrDefault(Rows(i),'FE',NaN);
    end
end

function Reason = BuildFailureReason(Report)
    Parts = cell(1,0);
    if ~Report.bufferSeparationPass
        Parts{end+1} = 'buffer_overlap'; %#ok<AGROW>
    end
    if ~Report.updateAuditComplete
        Parts{end+1} = 'update_audit_hole'; %#ok<AGROW>
    end
    if ~Report.candidateAuditComplete
        Parts{end+1} = 'candidate_audit_hole'; %#ok<AGROW>
    end
    if ~Report.seedAuditComplete
        Parts{end+1} = 'seed_audit_hole'; %#ok<AGROW>
    end
    if ~Report.boundaryLineageComplete
        Parts{end+1} = 'boundary_lineage_hole'; %#ok<AGROW>
    end
    if ~Report.archiveEventComplete
        Parts{end+1} = 'archive_event_hole'; %#ok<AGROW>
    end
    if isempty(Parts)
        Reason = '';
    else
        Reason = strjoin(Parts,'|');
    end
end

function Summary = SummarizeRunReports(RunReports)
    Summary = struct();
    Summary.runCount = numel(RunReports);
    Summary.passCount = sum([RunReports.passed]);
    Summary.failCount = Summary.runCount - Summary.passCount;
    Summary.passRate = Summary.passCount/max(Summary.runCount,1);
    Summary.maxTrainCalOverlap = max([RunReports.maxTrainCalOverlap]);
    Summary.maxTrainTestOverlap = max([RunReports.maxTrainTestOverlap]);
    Summary.maxCalTestOverlap = max([RunReports.maxCalTestOverlap]);
    Summary.totalUpdateAuditMismatch = sum([RunReports.updateAuditMismatchCount]);
    Summary.totalCandidateAuditMismatch = sum([RunReports.candidateAuditMismatchCount]);
    Summary.totalSeedAuditMismatch = sum([RunReports.seedAuditMismatchCount]);
    Summary.totalBoundaryLineageMismatch = sum([RunReports.boundaryLineageMismatchCount]);
    Summary.totalArchiveEventMismatch = sum([RunReports.archiveEventMismatchCount]);
end

function Detail = BuildRunDetail(Metric,ProblemName,RunIndex)
    Detail = InitRunDetail();
    Detail.problem = ProblemName;
    Detail.run = RunIndex;
    Detail.metric = Metric;
end

function Report = InitRunReport()
    Report = struct( ...
        'problem','', ...
        'run',0, ...
        'passed',false, ...
        'failure_reason','', ...
        'bufferSeparationPass',false, ...
        'updateAuditComplete',false, ...
        'candidateAuditComplete',false, ...
        'seedAuditComplete',false, ...
        'boundaryLineageComplete',false, ...
        'archiveEventComplete',false, ...
        'runLevelSummaryPass',false, ...
        'updateAuditCount',0, ...
        'updateAuditMismatchCount',0, ...
        'candidateAuditMismatchCount',0, ...
        'seedAuditMismatchCount',0, ...
        'boundaryLineageMismatchCount',0, ...
        'archiveEventMismatchCount',0, ...
        'totalCandidateRows',0, ...
        'totalSeedRows',0, ...
        'totalLineageRows',0, ...
        'totalArchiveEventRows',0, ...
        'maxTrainCalOverlap',0, ...
        'maxTrainTestOverlap',0, ...
        'maxCalTestOverlap',0, ...
        'boundaryStarted',false);
end

function Detail = InitRunDetail()
    Detail = struct( ...
        'problem','', ...
        'run',0, ...
        'metric',struct());
end

function Value = FieldOrDefault(S,Field,Default)
    if isstruct(S) && isfield(S,Field) && ~isempty(S.(Field))
        Value = S.(Field);
    else
        Value = Default;
    end
end

function Row = InitBoundaryUpdateAuditRowLocal()
    Row = struct('FE',NaN,'prob',zeros(0,1),'label',zeros(0,1), ...
        'count',0,'strict_separation',false, ...
        'train_cal_overlap',0,'train_test_overlap',0,'cal_test_overlap',0);
end

function Row = InitSelectionTraceRowLocal()
    Row = struct('FE',NaN,'candidateCount',0,'selectedCount',0);
end

function Row = InitCandidateAuditRowLocal()
    Row = struct('FE',NaN);
end

function Row = InitSeedAuditRowLocal()
    Row = struct('FE',NaN);
end

function Row = InitBoundaryLineageRowLocal()
    Row = struct('FE',NaN);
end

function Row = InitBoundaryGainTraceRowLocal()
    Row = struct('FE',NaN,'boundaryAddedCount',0);
end

function Row = InitArchiveEventRowLocal()
    Row = struct('FE',NaN);
end
