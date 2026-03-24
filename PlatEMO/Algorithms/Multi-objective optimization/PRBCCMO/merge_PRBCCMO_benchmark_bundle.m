function Bundle = merge_PRBCCMO_benchmark_bundle(varargin)
%MERGE_PRBCCMO_BENCHMARK_BUNDLE Merge PRBCCMO benchmark artifacts into one bundle.
%
% Optional name-value pairs:
%   'ProjectRoot'  : project root directory, default pwd
%   'OutputPrefix' : output prefix without suffix,
%                    default benchmark_PRBCCMO_20260324_bundle
%
% Output files:
%   <OutputPrefix>.mat : structured bundle with all source tables/results
%   <OutputPrefix>.csv : merged flat summary table with scope/level metadata

    Params = struct( ...
        'ProjectRoot',pwd, ...
        'OutputPrefix','benchmark_PRBCCMO_20260324_bundle');
    Params = ParseInputs(Params,varargin{:});

    ProjectRoot = char(string(Params.ProjectRoot));
    OutputPrefix = char(string(Params.OutputPrefix));
    OutputMat = fullfile(ProjectRoot,[OutputPrefix,'.mat']);
    OutputCsv = fullfile(ProjectRoot,[OutputPrefix,'.csv']);

    Sources = struct( ...
        'baselinePooled','benchmark_PRBCCMO_experiment0_20260323_15runs_p6_summaryonly_pooled.csv', ...
        'baselineProblem','benchmark_PRBCCMO_experiment0_20260323_15runs_p6_summaryonly_problem.csv', ...
        'activationFamily','benchmark_PRBCCMO_activation_200k_all37_r1_activation_family.csv', ...
        'activationProblem','benchmark_PRBCCMO_activation_200k_all37_r1_activation_problem.csv', ...
        'calibrationMat','benchmark_PRBCCMO_crossset_200k_all37_r1_calibration.mat', ...
        'calibrationFamily','benchmark_PRBCCMO_crossset_200k_all37_r1_calibration_family.csv', ...
        'calibrationPooled','benchmark_PRBCCMO_crossset_200k_all37_r1_calibration_pooled.csv', ...
        'calibrationProblem','benchmark_PRBCCMO_crossset_200k_all37_r1_calibration_problem.csv');

    SourcePaths = structfun(@(f)fullfile(ProjectRoot,f),Sources,'UniformOutput',false);
    RequireFiles(SourcePaths);

    BaselinePooled = NormalizeTextColumns(readtable(SourcePaths.baselinePooled));
    BaselineProblem = NormalizeTextColumns(readtable(SourcePaths.baselineProblem));
    ActivationFamily = NormalizeTextColumns(readtable(SourcePaths.activationFamily));
    ActivationProblem = NormalizeTextColumns(readtable(SourcePaths.activationProblem));
    CalibrationFamily = NormalizeTextColumns(readtable(SourcePaths.calibrationFamily));
    CalibrationPooled = NormalizeTextColumns(readtable(SourcePaths.calibrationPooled));
    CalibrationProblem = NormalizeTextColumns(readtable(SourcePaths.calibrationProblem));

    CalibrationMatData = load(SourcePaths.calibrationMat);

    MergedTables = { ...
        AttachMeta(BaselinePooled,'baseline','pooled',Sources.baselinePooled), ...
        AttachMeta(BaselineProblem,'baseline','problem',Sources.baselineProblem), ...
        AttachMeta(ActivationFamily,'activation','family',Sources.activationFamily), ...
        AttachMeta(ActivationProblem,'activation','problem',Sources.activationProblem), ...
        AttachMeta(CalibrationFamily,'calibration','family',Sources.calibrationFamily), ...
        AttachMeta(CalibrationPooled,'calibration','pooled',Sources.calibrationPooled), ...
        AttachMeta(CalibrationProblem,'calibration','problem',Sources.calibrationProblem)};

    Merged = MergeTablesWithUnion(MergedTables);
    writetable(Merged,OutputCsv);

    Bundle = struct();
    Bundle.generatedAt = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    Bundle.projectRoot = ProjectRoot;
    Bundle.output = struct('mat',OutputMat,'csv',OutputCsv);
    Bundle.sources = Sources;
    Bundle.tables = struct();
    Bundle.tables.baseline = struct('pooled',BaselinePooled,'problem',BaselineProblem);
    Bundle.tables.activation = struct('family',ActivationFamily,'problem',ActivationProblem);
    Bundle.tables.calibration = struct( ...
        'family',CalibrationFamily, ...
        'pooled',CalibrationPooled, ...
        'problem',CalibrationProblem);
    if isfield(CalibrationMatData,'Results')
        Bundle.savedCalibrationResults = CalibrationMatData.Results;
    else
        Bundle.savedCalibrationResults = CalibrationMatData;
    end
    Bundle.mergedTable = Merged;
    Bundle.highlights = BuildHighlights(ActivationFamily,CalibrationFamily,CalibrationPooled);

    save(OutputMat,'Bundle','-v7.3');
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('merge_PRBCCMO_benchmark_bundle:InvalidInput', ...
            'Inputs must be name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if isstring(Name)
            Name = char(Name);
        end
        if ~isfield(Params,Name)
            error('merge_PRBCCMO_benchmark_bundle:UnknownOption', ...
                'Unknown option: %s',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function RequireFiles(SourcePaths)
    Names = fieldnames(SourcePaths);
    Missing = strings(0,1);
    for i = 1 : numel(Names)
        Path = SourcePaths.(Names{i});
        if ~isfile(Path)
            Missing(end+1,1) = string(Path); %#ok<AGROW>
        end
    end
    if ~isempty(Missing)
        error('merge_PRBCCMO_benchmark_bundle:MissingSource', ...
            'Missing source files:\n%s',strjoin(cellstr(Missing),newline));
    end
end

function T = NormalizeTextColumns(T)
    Names = T.Properties.VariableNames;
    for i = 1 : numel(Names)
        Name = Names{i};
        Value = T.(Name);
        if iscellstr(Value) || ischar(Value) || iscategorical(Value)
            T.(Name) = string(Value);
        elseif iscell(Value)
            if isempty(Value)
                T.(Name) = strings(height(T),1);
            elseif all(cellfun(@(x)ischar(x) || (isstring(x) && isscalar(x)),Value))
                T.(Name) = string(Value);
            end
        end
    end
end

function T = AttachMeta(T,Scope,Level,SourceFile)
    Rows = height(T);
    T = addvars(T, ...
        repmat(string(Scope),Rows,1), ...
        repmat(string(Level),Rows,1), ...
        repmat(string(SourceFile),Rows,1), ...
        'Before',1, ...
        'NewVariableNames',{'scope','level','sourceFile'});
end

function Merged = MergeTablesWithUnion(Tables)
    Schema = struct();
    Order = strings(0,1);
    for i = 1 : numel(Tables)
        T = Tables{i};
        Names = string(T.Properties.VariableNames);
        for j = 1 : numel(Names)
            Name = char(Names(j));
            if ~isfield(Schema,Name)
                Schema.(Name) = class(T.(Name));
                Order(end+1,1) = Names(j); %#ok<AGROW>
            end
        end
    end

    AllNames = cellstr(Order)';
    Normalized = cell(size(Tables));
    for i = 1 : numel(Tables)
        T = Tables{i};
        Rows = height(T);
        for j = 1 : numel(AllNames)
            Name = AllNames{j};
            if ~ismember(Name,T.Properties.VariableNames)
                T.(Name) = MissingColumn(Schema.(Name),Rows);
            end
        end
        T = T(:,AllNames);
        Normalized{i} = T;
    end
    Merged = vertcat(Normalized{:});
end

function Value = MissingColumn(ClassName,Rows)
    switch ClassName
        case 'double'
            Value = NaN(Rows,1);
        case 'single'
            Value = nan(Rows,1,'single');
        case 'logical'
            Value = false(Rows,1);
        case {'string'}
            Value = repmat("",Rows,1);
        otherwise
            error('merge_PRBCCMO_benchmark_bundle:UnsupportedClass', ...
                'Unsupported variable class: %s',ClassName);
    end
end

function Highlights = BuildHighlights(ActivationFamily,CalibrationFamily,CalibrationPooled)
    Highlights = struct();
    Highlights.activationFamiliesFullyStarted = ActivationFamily.family( ...
        ActivationFamily.boundaryOffspringProblems == ActivationFamily.problems);

    BetaMask = CalibrationFamily.variant == "beta";
    AutoMask = CalibrationFamily.variant == "auto_trust";

    Highlights.bestFamilyECE_beta = CalibrationFamily(:,{'family','pooledECE','pooledCoreNearGap'});
    Highlights.bestFamilyECE_beta = Highlights.bestFamilyECE_beta(BetaMask,:);

    Highlights.bestFamilyTrust_auto = CalibrationFamily(:,{'family','meanTrustGatePassRate'});
    Highlights.bestFamilyTrust_auto = Highlights.bestFamilyTrust_auto(AutoMask,:);

    [~,BestECEIdx] = min(CalibrationPooled.pooledECE);
    [~,BestTrustIdx] = max(CalibrationPooled.meanTrustGatePassRate);
    Highlights.bestOverallECEVariant = CalibrationPooled.variant(BestECEIdx);
    Highlights.bestOverallTrustVariant = CalibrationPooled.variant(BestTrustIdx);
end
