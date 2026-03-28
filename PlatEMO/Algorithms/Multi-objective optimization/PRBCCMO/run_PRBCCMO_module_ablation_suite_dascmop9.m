function Suite = run_PRBCCMO_module_ablation_suite_dascmop9(varargin)
% Run the current BoundaryCore ablation suite on the BC benchmark.
% The legacy function name is kept for compatibility; defaults now cover
% the complete 37-problem BC suite described in fix.md.
%
% The suite covers the ablations that are currently implemented in the
% mainline codebase:
%   bridge  : BridgeTopK {1,3} and BridgeActivationGap {0,0.02}
%   finder  : MidpointOnly / Probe3Only / BoundaryCore-noTrust / mainline
%   selector: ParetoOnly / BoundaryOnly / RandomWithinShortlist / mainline
%   refine  : FeasibleForwardOff / InfeasibleShrinkOff
%   model   : beta vs raw calibrator
%   final   : NoBoundary (bRho = 0)
%
% Optional name-value pairs:
%   'Runs'        : independent runs per problem, default 5
%   'RunSeeds'    : explicit run seeds, default 1001:1005
%   'Population'  : population size, default 100
%   'MaxFE'       : max function evaluations, default 200000
%   'ProblemNames': explicit problem list, default all 37 BC problems
%   'Workers'     : parallel workers used inside each ablation run, default 9
%   'UseParallel' : whether to enable parfor inside each run, default true
%   'OutputDir'   : output directory for CSV summaries, default timestamped dir
%   'BaseVariant' : paired-test reference variant, default 'boundarycore_beta'
%   'Verbose'     : print per-variant progress, default true
%
% Output CSV files:
%   suite_manifest.csv
%   suite_run_summary.csv
%   suite_problem_summary.csv
%   suite_family_summary.csv
%   suite_pooled_summary.csv
%   suite_paired_summary.csv
%   variant_*_summary.csv for each ablation

    Params = struct( ...
        'Runs',5, ...
        'RunSeeds',[], ...
        'Population',100, ...
        'MaxFE',200000, ...
        'ProblemNames',{{}}, ...
        'Workers',9, ...
        'UseParallel',true, ...
        'OutputDir','', ...
        'BaseVariant','boundarycore_beta', ...
        'Verbose',true);
    Params = ParseInputs(Params,varargin{:});
    Params.ProblemNames = NormalizeProblemNames(Params.ProblemNames);
    Params.RunSeeds = NormalizeRunSeeds(Params.RunSeeds,Params.Runs);
    Params.OutputDir = ResolveOutputDir(Params.OutputDir,numel(Params.RunSeeds));
    EnsureDirectory(Params.OutputDir);

    Variants = ResolveSuiteVariants();
    Manifest = BuildVariantManifest(Variants);
    writetable(Manifest,fullfile(Params.OutputDir,'suite_manifest.csv'));

    RunSummaryTable = table();
    ProblemSummaryTable = table();
    FamilySummaryTable = table();
    PooledSummaryTable = table();

    if logical(Params.Verbose)
        fprintf('[AblationSuite] outputDir=%s\n',Params.OutputDir);
        fprintf('[AblationSuite] problems=%d runs=%d workers=%d variants=%d\n', ...
            numel(Params.ProblemNames),numel(Params.RunSeeds),Params.Workers,numel(Variants));
    end

    Suite = struct();
    Suite.params = Params;
    Suite.variantManifest = Manifest;
    Suite.variantResults = repmat(struct( ...
        'variant','', ...
        'description','', ...
        'calibrator','', ...
        'elapsedSeconds',NaN, ...
        'runSummary',table(), ...
        'problemSummary',table(), ...
        'familySummary',table(), ...
        'pooledSummary',table()),numel(Variants),1);

    for i = 1 : numel(Variants)
        Variant = Variants(i);
        VariantDir = fullfile(Params.OutputDir,Variant.id);
        EnsureDirectory(VariantDir);
        if logical(Params.Verbose)
            fprintf('[AblationSuite] (%d/%d) start %s\n',i,numel(Variants),Variant.id);
        end

        ticVariant = tic;
        Results = benchmark_PRBCCMO_module_validation( ...
            'Runs',Params.Runs, ...
            'RunSeeds',Params.RunSeeds, ...
            'Population',Params.Population, ...
            'MaxFE',Params.MaxFE, ...
            'ProblemNames',Params.ProblemNames, ...
            'UseParallel',Params.UseParallel, ...
            'Workers',Params.Workers, ...
            'Calibrator',Variant.calibrator, ...
            'AlgorithmOverride',Variant.algorithmOverride, ...
            'RuntimeOverride',Variant.runtimeOverride, ...
            'SavePrefix','', ...
            'SaveMat',false, ...
            'Verbose',false);
        Elapsed = toc(ticVariant);

        RunTable = AddVariantColumns( ...
            struct2table(Results.runSummary,'AsArray',true),Variant,Elapsed);
        ProblemTable = AddVariantColumns( ...
            struct2table(Results.problemSummary,'AsArray',true),Variant,Elapsed);
        FamilyTable = AddVariantColumns( ...
            struct2table(Results.familySummary,'AsArray',true),Variant,Elapsed);
        PooledTable = AddVariantColumns( ...
            struct2table(Results.pooledSummary,'AsArray',true),Variant,Elapsed);

        writetable(RunTable,fullfile(VariantDir,'variant_run_summary.csv'));
        writetable(ProblemTable,fullfile(VariantDir,'variant_problem_summary.csv'));
        writetable(FamilyTable,fullfile(VariantDir,'variant_family_summary.csv'));
        writetable(PooledTable,fullfile(VariantDir,'variant_pooled_summary.csv'));

        RunSummaryTable = AppendTables(RunSummaryTable,RunTable);
        ProblemSummaryTable = AppendTables(ProblemSummaryTable,ProblemTable);
        FamilySummaryTable = AppendTables(FamilySummaryTable,FamilyTable);
        PooledSummaryTable = AppendTables(PooledSummaryTable,PooledTable);

        Suite.variantResults(i).variant = Variant.id;
        Suite.variantResults(i).description = Variant.description;
        Suite.variantResults(i).calibrator = Variant.calibrator;
        Suite.variantResults(i).elapsedSeconds = Elapsed;
        Suite.variantResults(i).runSummary = RunTable;
        Suite.variantResults(i).problemSummary = ProblemTable;
        Suite.variantResults(i).familySummary = FamilyTable;
        Suite.variantResults(i).pooledSummary = PooledTable;

        if logical(Params.Verbose)
            fprintf('[AblationSuite] done %s elapsed=%.1fs pooledM1=%.4f pooledM2=%.4f pooledM5valid=%d\n', ...
                Variant.id,Elapsed,PooledTable.M1_mean(1),PooledTable.M2_mean(1),PooledTable.M5_validRunCount(1));
        end

        clear Results RunTable ProblemTable FamilyTable PooledTable;
    end

    writetable(RunSummaryTable,fullfile(Params.OutputDir,'suite_run_summary.csv'));
    writetable(ProblemSummaryTable,fullfile(Params.OutputDir,'suite_problem_summary.csv'));
    writetable(FamilySummaryTable,fullfile(Params.OutputDir,'suite_family_summary.csv'));
    writetable(PooledSummaryTable,fullfile(Params.OutputDir,'suite_pooled_summary.csv'));
    PairedSummaryTable = BuildPairedSummary(RunSummaryTable,Variants,Params.BaseVariant);
    writetable(PairedSummaryTable,fullfile(Params.OutputDir,'suite_paired_summary.csv'));

    Suite.runSummary = RunSummaryTable;
    Suite.problemSummary = ProblemSummaryTable;
    Suite.familySummary = FamilySummaryTable;
    Suite.pooledSummary = PooledSummaryTable;
    Suite.pairedSummary = PairedSummaryTable;
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('run_PRBCCMO_module_ablation_suite_dascmop9:InvalidInput', ...
            'Inputs must be name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~(ischar(Name) || (isstring(Name) && isscalar(Name)))
            error('run_PRBCCMO_module_ablation_suite_dascmop9:InvalidInputName', ...
                'Input names must be character vectors or scalar strings.');
        end
        Name = char(Name);
        if ~isfield(Params,Name)
            error('run_PRBCCMO_module_ablation_suite_dascmop9:UnknownOption', ...
                'Unknown option ''%s''.',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function ProblemNames = NormalizeProblemNames(ProblemNames)
    if isempty(ProblemNames)
        ProblemNames = ResolveAllBCProblems();
        return;
    end
    if ischar(ProblemNames) || (isstring(ProblemNames) && isscalar(ProblemNames))
        ProblemNames = {char(ProblemNames)};
    end
    ProblemNames = cellfun(@char,ProblemNames(:)','UniformOutput',false);
end

function ProblemNames = ResolveAllBCProblems()
    ProblemNames = [ ...
        arrayfun(@(i)sprintf('DASCMOP%d_BC',i),1:9,'UniformOutput',false), ...
        arrayfun(@(i)sprintf('LIRCMOP%d_BC',i),1:14,'UniformOutput',false), ...
        arrayfun(@(i)sprintf('MW%d_BC',i),1:14,'UniformOutput',false)];
end

function RunSeeds = NormalizeRunSeeds(RunSeeds,Runs)
    if isempty(RunSeeds)
        RunSeeds = 1000 + (1:Runs);
        return;
    end
    RunSeeds = round(double(RunSeeds(:)'));
    RunSeeds = RunSeeds(isfinite(RunSeeds));
    if isempty(RunSeeds)
        error('run_PRBCCMO_module_ablation_suite_dascmop9:InvalidRunSeeds', ...
            'RunSeeds must contain at least one finite integer.');
    end
end

function OutputDir = ResolveOutputDir(OutputDir,RunCount)
    if nargin >= 1 && ~isempty(OutputDir)
        OutputDir = char(OutputDir);
        return;
    end
    if nargin < 2 || isempty(RunCount)
        RunCount = 5;
    end
    Stamp = char(string(datetime('now','Format','yyyyMMdd_HHmmss')));
    OutputDir = fullfile(pwd,sprintf('results_prbccmo_module_ablation_bc_r%d_%s',RunCount,Stamp));
end

function EnsureDirectory(PathStr)
    if exist(PathStr,'dir') ~= 7
        mkdir(PathStr);
    end
end

function Variants = ResolveSuiteVariants()
    Variants = repmat(struct( ...
        'id','', ...
        'description','', ...
        'calibrator','beta', ...
        'algorithmOverride',struct(), ...
        'runtimeOverride',struct()),15,1);

    Variants(1) = MakeVariant( ...
        'boundarycore_beta', ...
        'BoundaryCore mainline with beta calibrator', ...
        'beta',struct(),struct());
    Variants(2) = MakeVariant( ...
        'boundarycore_raw', ...
        'BoundaryCore mainline with raw calibrator', ...
        'raw',struct(),struct());
    Variants(3) = MakeVariant( ...
        'bridge_topk_1', ...
        'Bridge ablation with BridgeTopK = 1', ...
        'beta',struct(),struct('BridgeTopK',1));
    Variants(4) = MakeVariant( ...
        'bridge_topk_3', ...
        'Bridge ablation with BridgeTopK = 3', ...
        'beta',struct(),struct('BridgeTopK',3));
    Variants(5) = MakeVariant( ...
        'bridge_gap_0', ...
        'Bridge ablation with BridgeActivationGap = 0', ...
        'beta',struct(),struct('BridgeActivationGap',0));
    Variants(6) = MakeVariant( ...
        'bridge_gap_0p02', ...
        'Bridge ablation with BridgeActivationGap = 0.02', ...
        'beta',struct(),struct('BridgeActivationGap',0.02));
    Variants(7) = MakeVariant( ...
        'midpoint_only', ...
        'Finder ablation: midpoint only, no trust refine', ...
        'beta',struct(),struct( ...
            'BridgeScanLambda',0.5, ...
            'BridgeRefineStep',0, ...
            'DisableTrust',true));
    Variants(8) = MakeVariant( ...
        'probe3_only', ...
        'Finder ablation: three probes only, no local refine', ...
        'beta',struct(),struct( ...
            'BridgeScanLambda',[0.25 0.5 0.75], ...
            'BridgeRefineStep',0, ...
            'DisableTrust',true));
    Variants(9) = MakeVariant( ...
        'boundarycore_no_trust', ...
        'Trust ablation: disable trust but force placement refine', ...
        'beta',struct(),struct( ...
            'BridgeScanLambda',[0.25 0.5 0.75], ...
            'BridgeRefineStep',0.125, ...
            'DisableTrust',true, ...
            'ForcePlacementRefine',true));
    Variants(10) = MakeVariant( ...
        'selector_pareto_only', ...
        'Selector ablation: Pareto-only ordering', ...
        'beta',struct(),struct('SelectorMode','pareto_only'));
    Variants(11) = MakeVariant( ...
        'selector_boundary_only', ...
        'Selector ablation: boundary-only ordering', ...
        'beta',struct(),struct('SelectorMode','boundary_only'));
    Variants(12) = MakeVariant( ...
        'selector_random_shortlist', ...
        'Selector ablation: random selection within Pareto shortlist', ...
        'beta',struct(),struct('SelectorMode','random_within_shortlist'));
    Variants(13) = MakeVariant( ...
        'feasible_forward_off', ...
        'Refinement ablation: disable feasible forward exploit', ...
        'beta',struct(),struct('DisableFeasibleForward',true));
    Variants(14) = MakeVariant( ...
        'infeasible_shrink_off', ...
        'Refinement ablation: disable infeasible bracket shrink', ...
        'beta',struct(),struct('DisableInfeasibleShrink',true));
    Variants(15) = MakeVariant( ...
        'no_boundary', ...
        'Final-effect ablation: boundary budget disabled via bRho = 0', ...
        'beta',struct('bRho',0),struct());
end

function Variant = MakeVariant(Id,Description,Calibrator,AlgorithmOverride,RuntimeOverride)
    Variant = struct( ...
        'id',Id, ...
        'description',Description, ...
        'calibrator',Calibrator, ...
        'algorithmOverride',AlgorithmOverride, ...
        'runtimeOverride',RuntimeOverride);
end

function Manifest = BuildVariantManifest(Variants)
    Count = numel(Variants);
    Manifest = table('Size',[Count 5], ...
        'VariableTypes',{'string','string','string','string','string'}, ...
        'VariableNames',{'variant','description','calibrator','algorithmOverride','runtimeOverride'});
    for i = 1 : Count
        Manifest.variant(i) = string(Variants(i).id);
        Manifest.description(i) = string(Variants(i).description);
        Manifest.calibrator(i) = string(Variants(i).calibrator);
        Manifest.algorithmOverride(i) = string(StructToJson(Variants(i).algorithmOverride));
        Manifest.runtimeOverride(i) = string(StructToJson(Variants(i).runtimeOverride));
    end
end

function Text = StructToJson(Value)
    if isempty(Value)
        Text = '{}';
        return;
    end
    Text = jsonencode(Value);
end

function TableOut = AddVariantColumns(TableIn,Variant,Elapsed)
    TableOut = TableIn;
    Rows = height(TableOut);
    TableOut.variant = repmat(string(Variant.id),Rows,1);
    TableOut.variantDescription = repmat(string(Variant.description),Rows,1);
    TableOut.requestedCalibrator = repmat(string(Variant.calibrator),Rows,1);
    TableOut.variantElapsedSeconds = repmat(Elapsed,Rows,1);
end

function PairedTable = BuildPairedSummary(RunSummaryTable,Variants,BaseVariant)
    PairedRows = repmat(InitPairedSummaryRow(),0,1);
    if isempty(RunSummaryTable) || ~any(strcmp(RunSummaryTable.variant,BaseVariant))
        PairedTable = struct2table(PairedRows,'AsArray',true);
        return;
    end

    ScopeSpecs = [ ...
        struct('scope','problem','names',{unique(RunSummaryTable.problem,'stable')}), ...
        struct('scope','pooled','names',{{ResolvePooledSummaryName(RunSummaryTable)}})];
    Metrics = { ...
        'M1_selected','lower_better'; ...
        'M2','higher_better'; ...
        'UBY','higher_better'; ...
        'ABS','higher_better'; ...
        'AGS','higher_better'; ...
        'gateOpenRate','higher_better'; ...
        'refineUseCount','higher_better'; ...
        'refineGainPerSeed','higher_better'; ...
        'FFC','higher_better'; ...
        'FGY','higher_better'; ...
        'tightBracketAbsCount','higher_better'; ...
        'recoverAbsCount','higher_better'};

    for s = 1 : numel(ScopeSpecs)
        Scope = ScopeSpecs(s).scope;
        Names = ScopeSpecs(s).names;
        for g = 1 : numel(Names)
            GroupName = ResolveGroupName(Names,g);
            for m = 1 : size(Metrics,1)
                MetricName = Metrics{m,1};
                BetterDirection = Metrics{m,2};
                GroupRows = repmat(InitPairedSummaryRow(),0,1);
                for v = 1 : numel(Variants)
                    CompareVariant = Variants(v).id;
                    if strcmp(CompareVariant,BaseVariant)
                        continue;
                    end
                    [BaseData,CompareData,FamilyName] = ResolvePairedSamples( ...
                        RunSummaryTable,Scope,GroupName,BaseVariant,CompareVariant,MetricName);
                    if isempty(BaseData)
                        continue;
                    end
                    Row = InitPairedSummaryRow();
                    Row.scope = Scope;
                    Row.name = GroupName;
                    Row.family = FamilyName;
                    Row.metric = MetricName;
                    Row.baseVariant = BaseVariant;
                    Row.compareVariant = CompareVariant;
                    Row.baseDescription = ResolveVariantDescription(Variants,BaseVariant);
                    Row.compareDescription = ResolveVariantDescription(Variants,CompareVariant);
                    Row.betterDirection = BetterDirection;
                    Row.pairCount = numel(BaseData);
                    Row.baseMean = MeanOrNaN(BaseData);
                    Row.compareMean = MeanOrNaN(CompareData);
                    Row.meanDiff = MeanOrNaN(CompareData - BaseData);
                    Row.baseMedian = MedianOrNaN(BaseData);
                    Row.compareMedian = MedianOrNaN(CompareData);
                    Row.medianDiff = MedianOrNaN(CompareData - BaseData);
                    Row.signrankP = SafeSignrank(BaseData,CompareData);
                    GroupRows(end+1,1) = Row; %#ok<AGROW>
                end
                if isempty(GroupRows)
                    continue;
                end
                Adjusted = HolmAdjust([GroupRows.signrankP]);
                for i = 1 : numel(GroupRows)
                    GroupRows(i).holmAdjustedP = Adjusted(i);
                end
                PairedRows = [PairedRows; GroupRows]; %#ok<AGROW>
            end
        end
    end

    PairedTable = struct2table(PairedRows,'AsArray',true);
end

function [BaseData,CompareData,FamilyName] = ResolvePairedSamples( ...
    RunSummaryTable,Scope,GroupName,BaseVariant,CompareVariant,MetricName)
    FamilyName = 'ALL_BC';
    switch Scope
        case 'problem'
            ScopeMask = strcmp(RunSummaryTable.problem,GroupName);
        case 'pooled'
            ScopeMask = true(height(RunSummaryTable),1);
        otherwise
            ScopeMask = false(height(RunSummaryTable),1);
    end

    BaseMask = ScopeMask & strcmp(RunSummaryTable.variant,BaseVariant);
    CompareMask = ScopeMask & strcmp(RunSummaryTable.variant,CompareVariant);
    BaseRows = RunSummaryTable(BaseMask,:);
    CompareRows = RunSummaryTable(CompareMask,:);
    if strcmp(Scope,'problem') && height(BaseRows) > 0
        FamilyName = char(string(BaseRows.family(1)));
    end

    BaseData = zeros(0,1);
    CompareData = zeros(0,1);
    if isempty(BaseRows) || isempty(CompareRows)
        return;
    end

    BaseKeys = BuildPairKeys(BaseRows);
    CompareKeys = BuildPairKeys(CompareRows);
    [~,BaseIdx,CompareIdx] = intersect(BaseKeys,CompareKeys,'stable');
    if isempty(BaseIdx)
        return;
    end

    BaseValues = double(BaseRows.(MetricName)(BaseIdx));
    CompareValues = double(CompareRows.(MetricName)(CompareIdx));
    Valid = isfinite(BaseValues) & isfinite(CompareValues);
    BaseData = BaseValues(Valid);
    CompareData = CompareValues(Valid);
end

function Keys = BuildPairKeys(Rows)
    RowCount = height(Rows);
    Keys = cell(RowCount,1);
    for i = 1 : RowCount
        Keys{i} = sprintf('%s|%d|%d',char(string(Rows.problem(i))),Rows.run(i),Rows.seed(i));
    end
end

function Name = ResolveGroupName(Names,Index)
    if iscell(Names)
        Name = char(Names{Index});
    else
        Name = char(string(Names(Index)));
    end
end

function Description = ResolveVariantDescription(Variants,VariantId)
    Description = '';
    Match = find(strcmp({Variants.id},VariantId),1,'first');
    if ~isempty(Match)
        Description = Variants(Match).description;
    end
end

function Name = ResolvePooledSummaryName(RunSummaryTable)
    Name = sprintf('ALL_%d_BC',numel(unique(RunSummaryTable.problem)));
end

function Row = InitPairedSummaryRow()
    Row = struct( ...
        'scope','', ...
        'name','', ...
        'family','', ...
        'metric','', ...
        'baseVariant','', ...
        'compareVariant','', ...
        'baseDescription','', ...
        'compareDescription','', ...
        'betterDirection','', ...
        'pairCount',0, ...
        'baseMean',NaN, ...
        'compareMean',NaN, ...
        'meanDiff',NaN, ...
        'baseMedian',NaN, ...
        'compareMedian',NaN, ...
        'medianDiff',NaN, ...
        'signrankP',NaN, ...
        'holmAdjustedP',NaN);
end

function Value = SafeSignrank(A,B)
    Value = NaN;
    if isempty(A) || isempty(B) || numel(A) ~= numel(B)
        return;
    end
    if exist('signrank','file') ~= 2
        return;
    end
    try
        Value = signrank(A,B);
    catch
        Value = NaN;
    end
end

function Adjusted = HolmAdjust(PValues)
    Adjusted = nan(size(PValues));
    if isempty(PValues)
        return;
    end
    Valid = isfinite(PValues);
    if ~any(Valid)
        return;
    end
    ValidP = PValues(Valid);
    [SortedP,Order] = sort(ValidP(:));
    Count = numel(SortedP);
    AdjustedSorted = zeros(Count,1);
    RunningMax = 0;
    for i = 1 : Count
        Candidate = (Count - i + 1) * SortedP(i);
        RunningMax = max(RunningMax,Candidate);
        AdjustedSorted(i) = min(RunningMax,1);
    end
    Restored = zeros(Count,1);
    Restored(Order) = AdjustedSorted;
    Adjusted(Valid) = Restored;
end

function Result = AppendTables(Result,NewRows)
    if isempty(NewRows)
        return;
    end
    if isempty(Result)
        Result = NewRows;
        return;
    end
    Result = [Result;NewRows];
end

function Value = MeanOrNaN(Data)
    Data = double(Data(:));
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
        return;
    end
    Value = mean(Data);
end

function Value = MedianOrNaN(Data)
    Data = double(Data(:));
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
        return;
    end
    Value = median(Data);
end
