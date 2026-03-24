function Results = benchmark_PRBCCMO_crossset_suite(varargin)
% Cross-set PRBCCMO audit suite for DAS-CMOP_BC, LIR-CMOP_BC, and MW_BC.
%
% Optional name-value pairs:
%   'Runs'             : number of independent runs, default 1
%   'RunSeeds'         : explicit seed list, default []
%   'Population'       : population size N, default 100
%   'MaxFE'            : maximum function evaluations, default 200000
%   'ProblemSets'      : problem-family list, default {'DASCMOP_BC','LIRCMOP_BC','MW_BC'}
%   'ProblemNames'     : explicit problem list, overrides ProblemSets when not empty
%   'UseParallel'      : whether to enable parfor in calibration audit, default false
%   'Workers'          : parallel workers for calibration audit, default 0
%   'RunActivation'    : whether to run activation diagnostic, default true
%   'RunCalibration'   : whether to run calibration/trust audit, default true
%   'ActivationVariant': activation diagnostic variant, default 'auto_trust'
%   'SavePrefix'       : output prefix without suffix, default 'benchmark_PRBCCMO_crossset_suite'
%   'SaveSummaryOnly'  : save compact MAT result for calibration audit, default true

    Params = struct( ...
        'Runs',1, ...
        'RunSeeds',[], ...
        'Population',100, ...
        'MaxFE',200000, ...
        'ProblemSets',{{'DASCMOP_BC','LIRCMOP_BC','MW_BC'}}, ...
        'ProblemNames',{{}}, ...
        'UseParallel',false, ...
        'Workers',0, ...
        'RunActivation',true, ...
        'RunCalibration',true, ...
        'ActivationVariant','auto_trust', ...
        'SavePrefix','benchmark_PRBCCMO_crossset_suite', ...
        'SaveSummaryOnly',true);
    Params = ParseInputs(Params,varargin{:});

    [ProblemNames,ProblemMeta] = ResolveProblemSelection(Params.ProblemNames,Params.ProblemSets);

    Results = struct();
    Results.params = Params;
    Results.problemNames = ProblemNames;
    Results.problemMeta = ProblemMeta;

    if Params.RunActivation
        ActivationArgs = { ...
            'Runs',Params.Runs, ...
            'Population',Params.Population, ...
            'MaxFE',Params.MaxFE, ...
            'ProblemNames',ProblemNames, ...
            'Variant',Params.ActivationVariant, ...
            'Verbose',false};
        if ~isempty(Params.RunSeeds)
            ActivationArgs = [ActivationArgs,{'RunSeeds',Params.RunSeeds}]; %#ok<AGROW>
        end
        Activation = diagnose_PRBCCMO_boundary_activation(ActivationArgs{:});
        Activation.problemSummary = AttachFamilyField(Activation.problemSummary,ProblemMeta);
        Activation.familySummary = SummarizeActivationFamilies(Activation.problemSummary);
        Results.activation = Activation;

        if ~isempty(Params.SavePrefix)
            ActivationMatPath = [Params.SavePrefix,'_activation.mat'];
            save(ActivationMatPath,'Activation','-v7.3');
            WriteStructArrayCsv(Activation.problemSummary,[Params.SavePrefix,'_activation_problem.csv']);
            WriteStructArrayCsv(Activation.familySummary,[Params.SavePrefix,'_activation_family.csv']);
        end
    end

    if Params.RunCalibration
        CalibrationSavePath = '';
        if ~isempty(Params.SavePrefix)
            CalibrationSavePath = [Params.SavePrefix,'_calibration.mat'];
        end
        CalibrationArgs = { ...
            'Runs',Params.Runs, ...
            'Population',Params.Population, ...
            'MaxFE',Params.MaxFE, ...
            'ProblemNames',ProblemNames, ...
            'UseParallel',Params.UseParallel, ...
            'Workers',Params.Workers, ...
            'SavePath',CalibrationSavePath, ...
            'SaveSummaryOnly',Params.SaveSummaryOnly};
        if ~isempty(Params.RunSeeds)
            CalibrationArgs = [CalibrationArgs,{'RunSeeds',Params.RunSeeds}]; %#ok<AGROW>
        end
        Calibration = benchmark_PRBCCMO_experiment0(CalibrationArgs{:});
        Calibration.problemSummary = AttachFamilyField(Calibration.problemSummary,ProblemMeta);
        Calibration.runSummary = AttachFamilyField(Calibration.runSummary,ProblemMeta);
        Calibration.familySummary = SummarizeCalibrationFamilies( ...
            Calibration.updateRows,Calibration.runSummary,ProblemMeta);
        Results.calibration = Calibration;

        if ~isempty(Params.SavePrefix)
            WriteStructArrayCsv(Calibration.problemSummary,[Params.SavePrefix,'_calibration_problem.csv']);
            WriteStructArrayCsv(Calibration.pooledSummary,[Params.SavePrefix,'_calibration_pooled.csv']);
            WriteStructArrayCsv(Calibration.familySummary,[Params.SavePrefix,'_calibration_family.csv']);
        end
    end
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('benchmark_PRBCCMO_crossset_suite:InvalidInput', ...
            'Inputs must be name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if isstring(Name)
            Name = char(Name);
        end
        if ~isfield(Params,Name)
            error('benchmark_PRBCCMO_crossset_suite:UnknownOption', ...
                'Unknown option: %s',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function [ProblemNames,ProblemMeta] = ResolveProblemSelection(ExplicitProblemNames,ProblemSets)
    if nargin >= 1 && ~isempty(ExplicitProblemNames)
        ProblemNames = NormalizeCellStr(ExplicitProblemNames);
    else
        Families = NormalizeCellStr(ProblemSets);
        ProblemNames = cell(0,1);
        for i = 1 : numel(Families)
            ProblemNames = [ProblemNames;ResolveFamilyProblemNames(Families{i})]; %#ok<AGROW>
        end
    end
    ProblemNames = unique(ProblemNames,'stable');
    ProblemMeta = BuildProblemMeta(ProblemNames);
end

function Names = ResolveFamilyProblemNames(FamilyName)
    Key = lower(strrep(strrep(strtrim(FamilyName),'-',''),'_',''));
    switch Key
        case {'dascmopbc','dascmop'}
            Names = arrayfun(@(i)sprintf('DASCMOP%d_BC',i),1:9,'UniformOutput',false)';
        case {'lircmopbc','lircmop'}
            Names = arrayfun(@(i)sprintf('LIRCMOP%d_BC',i),1:14,'UniformOutput',false)';
        case {'mwbc','mw'}
            Names = arrayfun(@(i)sprintf('MW%d_BC',i),1:14,'UniformOutput',false)';
        otherwise
            error('benchmark_PRBCCMO_crossset_suite:UnknownProblemSet', ...
                'Unknown problem set: %s',FamilyName);
    end
end

function ProblemMeta = BuildProblemMeta(ProblemNames)
    ProblemMeta = repmat(struct('problem','','family',''),numel(ProblemNames),1);
    for i = 1 : numel(ProblemNames)
        ProblemMeta(i).problem = ProblemNames{i};
        ProblemMeta(i).family = ResolveProblemFamily(ProblemNames{i});
    end
end

function Family = ResolveProblemFamily(ProblemName)
    if startsWith(ProblemName,'DASCMOP','IgnoreCase',true)
        Family = 'DASCMOP_BC';
    elseif startsWith(ProblemName,'LIRCMOP','IgnoreCase',true)
        Family = 'LIRCMOP_BC';
    elseif startsWith(ProblemName,'MW','IgnoreCase',true)
        Family = 'MW_BC';
    else
        Family = 'UNKNOWN';
    end
end

function Rows = AttachFamilyField(Rows,ProblemMeta)
    if isempty(Rows)
        return;
    end
    FamilyMap = containers.Map();
    for i = 1 : numel(ProblemMeta)
        FamilyMap(ProblemMeta(i).problem) = ProblemMeta(i).family;
    end
    for i = 1 : numel(Rows)
        ProblemName = '';
        if isfield(Rows,'problem')
            ProblemName = char(Rows(i).problem);
        elseif isfield(Rows,'Problem')
            ProblemName = char(Rows(i).Problem);
        end
        if isKey(FamilyMap,ProblemName)
            Rows(i).family = FamilyMap(ProblemName);
        else
            Rows(i).family = ResolveProblemFamily(ProblemName);
        end
    end
end

function Summary = SummarizeActivationFamilies(ProblemSummary)
    Families = unique({ProblemSummary.family},'stable');
    Summary = repmat(InitActivationFamilyRow(),numel(Families),1);
    for i = 1 : numel(Families)
        Family = Families{i};
        Mask = strcmp({ProblemSummary.family},Family);
        Summary(i,1) = SummarizeActivationFamilySubset(ProblemSummary(Mask),Family);
    end
end

function Row = InitActivationFamilyRow()
    Row = struct( ...
        'family','', ...
        'problems',0, ...
        'runs',0, ...
        'regularFeasibleProblems',0, ...
        'sharedSectorProblems',0, ...
        'activeSectorProblems',0, ...
        'candidatePoolProblems',0, ...
        'boundarySeedProblems',0, ...
        'boundaryOffspringProblems',0, ...
        'meanSharedSectorRunRatio',NaN, ...
        'meanActiveSectorRunRatio',NaN, ...
        'meanBoundaryOffspringRunRatio',NaN, ...
        'medianFirstSharedSectorGeneration',NaN, ...
        'medianFirstActiveSectorGeneration',NaN, ...
        'medianFirstBoundaryOffspringGeneration',NaN, ...
        'dominantBridgeBlocker','', ...
        'dominantSelectionBlocker','', ...
        'dominantBlocker','');
end

function Row = SummarizeActivationFamilySubset(ProblemSummary,Family)
    Row = InitActivationFamilyRow();
    Row.family = Family;
    Row.problems = numel(ProblemSummary);
    if isempty(ProblemSummary)
        return;
    end

    Runs = max(1,[ProblemSummary.runs]);
    Row.runs = sum([ProblemSummary.runs]);
    Row.regularFeasibleProblems = sum([ProblemSummary.regularFeasibleRuns] > 0);
    Row.sharedSectorProblems = sum([ProblemSummary.sharedSectorRuns] > 0);
    Row.activeSectorProblems = sum([ProblemSummary.activeSectorRuns] > 0);
    Row.candidatePoolProblems = sum([ProblemSummary.candidatePoolRuns] > 0);
    Row.boundarySeedProblems = sum([ProblemSummary.boundarySeedRuns] > 0);
    Row.boundaryOffspringProblems = sum([ProblemSummary.boundaryOffspringRuns] > 0);
    Row.meanSharedSectorRunRatio = mean([ProblemSummary.sharedSectorRuns]./Runs);
    Row.meanActiveSectorRunRatio = mean([ProblemSummary.activeSectorRuns]./Runs);
    Row.meanBoundaryOffspringRunRatio = mean([ProblemSummary.boundaryOffspringRuns]./Runs);
    Row.medianFirstSharedSectorGeneration = MedianFinite([ProblemSummary.medianFirstSharedSectorGeneration]);
    Row.medianFirstActiveSectorGeneration = MedianFinite([ProblemSummary.medianFirstActiveSectorGeneration]);
    Row.medianFirstBoundaryOffspringGeneration = MedianFinite([ProblemSummary.medianFirstBoundaryOffspringGeneration]);
    Row.dominantBridgeBlocker = ResolveDominantLabel({ProblemSummary.dominantBridgeBlocker});
    Row.dominantSelectionBlocker = ResolveDominantLabel({ProblemSummary.dominantSelectionBlocker});
    Row.dominantBlocker = ResolveDominantLabel({ProblemSummary.dominantBlocker});
end

function Summary = SummarizeCalibrationFamilies(UpdateRows,RunSummary,ProblemMeta)
    Families = unique({ProblemMeta.family},'stable');
    Variants = unique({RunSummary.variant},'stable');
    Summary = repmat(InitCalibrationFamilyRow(),numel(Families)*numel(Variants),1);
    Row = 0;
    for i = 1 : numel(Families)
        Family = Families{i};
        FamilyProblems = {ProblemMeta(strcmp({ProblemMeta.family},Family)).problem};
        for j = 1 : numel(Variants)
            Variant = Variants{j};
            Row = Row + 1;
            MaskRun = strcmp({RunSummary.family},Family) & strcmp({RunSummary.variant},Variant);
            MaskUpdate = ismember({UpdateRows.problem},FamilyProblems) & strcmp({UpdateRows.variant},Variant);
            Summary(Row,1) = SummarizeCalibrationFamilySubset( ...
                UpdateRows(MaskUpdate),RunSummary(MaskRun),Family,Variant,numel(FamilyProblems));
        end
    end
end

function Row = InitCalibrationFamilyRow()
    Row = struct( ...
        'family','', ...
        'variant','', ...
        'problems',0, ...
        'runs',0, ...
        'updates',0, ...
        'validUpdates',0, ...
        'invalidUpdates',0, ...
        'auditReadyUpdates',0, ...
        'notYetAuditableUpdates',0, ...
        'coldStartUpdates',0, ...
        'singleClassUpdates',0, ...
        'invalidUpdateRatio',NaN, ...
        'auditReadyUpdateRatio',NaN, ...
        'notYetAuditableUpdateRatio',NaN, ...
        'coldStartUpdateRatio',NaN, ...
        'singleClassUpdateRatio',NaN, ...
        'pooledCount',0, ...
        'pooledBrier',NaN, ...
        'pooledECE',NaN, ...
        'pooledCoreNearCount',0, ...
        'pooledCoreNearGap',NaN, ...
        'pooledCoreNearPositiveCount',0, ...
        'pooledRelaxedNearCount',0, ...
        'pooledRelaxedNearGap',NaN, ...
        'pooledRelaxedNearPositiveCount',0, ...
        'meanTrustGatePassRate',NaN, ...
        'medianTrustGatePassRate',NaN, ...
        'meanRunPooledECE',NaN, ...
        'meanRunPooledCoreNearGap',NaN);
end

function Row = SummarizeCalibrationFamilySubset(UpdateRows,RunSummary,Family,Variant,ProblemCount)
    Row = InitCalibrationFamilyRow();
    Row.family = Family;
    Row.variant = Variant;
    Row.problems = ProblemCount;
    Row.runs = numel(RunSummary);
    Row.updates = numel(UpdateRows);
    if isempty(RunSummary)
        return;
    end

    Row.validUpdates = sum([RunSummary.validUpdateCount]);
    Row.invalidUpdates = sum([RunSummary.invalidUpdateCount]);
    Row.auditReadyUpdates = sum([RunSummary.auditReadyUpdateCount]);
    Row.notYetAuditableUpdates = sum([RunSummary.notYetAuditableUpdateCount]);
    Row.coldStartUpdates = sum([RunSummary.coldStartUpdateCount]);
    Row.singleClassUpdates = sum([RunSummary.singleClassUpdateCount]);
    Row.invalidUpdateRatio = SafeRatio(Row.invalidUpdates,Row.updates);
    Row.auditReadyUpdateRatio = SafeRatio(Row.auditReadyUpdates,Row.updates);
    Row.notYetAuditableUpdateRatio = SafeRatio(Row.notYetAuditableUpdates,Row.updates);
    Row.coldStartUpdateRatio = SafeRatio(Row.coldStartUpdates,Row.updates);
    Row.singleClassUpdateRatio = SafeRatio(Row.singleClassUpdates,Row.updates);
    Row.meanTrustGatePassRate = MeanScalarStructField(RunSummary,'trustGatePassRate');
    Row.medianTrustGatePassRate = MedianScalarStructField(RunSummary,'trustGatePassRate');
    Row.meanRunPooledECE = MeanScalarStructField(RunSummary,'pooledECE');
    Row.meanRunPooledCoreNearGap = MeanScalarStructField(RunSummary,'pooledCoreNearGap');
    Row.pooledCount = sum([RunSummary.pooledCount]);
    Row.pooledBrier = WeightedMeanScalarStructField(RunSummary,'pooledBrier','pooledCount');
    Row.pooledCoreNearCount = sum([RunSummary.pooledCoreNearCount]);
    Row.pooledCoreNearPositiveCount = sum([RunSummary.pooledCoreNearPositiveCount]);
    Row.pooledCoreNearGap = ResolveNearGap(Row.pooledCoreNearPositiveCount,Row.pooledCoreNearCount);
    Row.pooledRelaxedNearCount = sum([RunSummary.pooledRelaxedNearCount]);
    Row.pooledRelaxedNearPositiveCount = sum([RunSummary.pooledRelaxedNearPositiveCount]);
    Row.pooledRelaxedNearGap = ResolveNearGap(Row.pooledRelaxedNearPositiveCount,Row.pooledRelaxedNearCount);
    [Row.pooledECE,~] = ResolveAggregatedECE(RunSummary);
end

function WriteStructArrayCsv(Data,Path)
    if isempty(Data)
        writetable(table(),Path);
        return;
    end
    T = struct2table(Data,'AsArray',true);
    writetable(T,Path);
end

function Names = NormalizeCellStr(Value)
    if ischar(Value) || isstring(Value)
        Names = cellstr(Value);
        return;
    end
    if ~iscell(Value)
        error('benchmark_PRBCCMO_crossset_suite:InvalidTextList', ...
            'Expected a char, string, or cell array of strings.');
    end
    Names = Value(:);
    for i = 1 : numel(Names)
        if isstring(Names{i})
            Names{i} = char(Names{i});
        end
        if ~ischar(Names{i})
            error('benchmark_PRBCCMO_crossset_suite:InvalidTextValue', ...
                'List elements must be char or string.');
        end
    end
end

function Value = MeanScalarStructField(S,Field)
    if isempty(S)
        Value = NaN;
        return;
    end
    Data = arrayfun(@(Row)FieldOrDefault(Row,Field,NaN),S);
    Value = MeanFinite(Data);
end

function Value = MedianScalarStructField(S,Field)
    if isempty(S)
        Value = NaN;
        return;
    end
    Data = arrayfun(@(Row)FieldOrDefault(Row,Field,NaN),S);
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
    else
        Value = median(Data);
    end
end

function Value = WeightedMeanScalarStructField(S,ValueField,WeightField)
    if isempty(S)
        Value = NaN;
        return;
    end
    Values = arrayfun(@(Row)FieldOrDefault(Row,ValueField,NaN),S);
    Weights = arrayfun(@(Row)FieldOrDefault(Row,WeightField,0),S);
    Value = WeightedMeanFinite(Values,Weights);
end

function [ECE,BinStats] = ResolveAggregatedECE(Source)
    BinStats = struct('count',zeros(1,10),'probSum',zeros(1,10),'labelSum',zeros(1,10));
    if isempty(Source)
        ECE = NaN;
        return;
    end

    Width = ResolveBinWidth(Source);
    BinStats = struct('count',zeros(1,Width),'probSum',zeros(1,Width),'labelSum',zeros(1,Width));
    for i = 1 : numel(Source)
        BinStats.count = BinStats.count + NormalizeBinRow(FieldOrDefault(Source(i),'eceBinCount',[]),Width);
        BinStats.probSum = BinStats.probSum + NormalizeBinRow(FieldOrDefault(Source(i),'eceBinProbSum',[]),Width);
        BinStats.labelSum = BinStats.labelSum + NormalizeBinRow(FieldOrDefault(Source(i),'eceBinLabelSum',[]),Width);
    end

    Total = sum(BinStats.count);
    if Total <= 0
        ECE = NaN;
        return;
    end

    Valid = BinStats.count > 0;
    MeanProb = zeros(size(BinStats.count));
    FeasibleRate = zeros(size(BinStats.count));
    MeanProb(Valid) = BinStats.probSum(Valid)./BinStats.count(Valid);
    FeasibleRate(Valid) = BinStats.labelSum(Valid)./BinStats.count(Valid);
    ECE = sum((BinStats.count(Valid)./Total).*abs(FeasibleRate(Valid)-MeanProb(Valid)));
end

function Width = ResolveBinWidth(Source)
    Width = 10;
    if isempty(Source)
        return;
    end
    Sample = FieldOrDefault(Source(1),'eceBinCount',zeros(1,10));
    if ~isempty(Sample)
        Width = numel(Sample);
    end
end

function Row = NormalizeBinRow(Data,Width)
    Row = zeros(1,Width);
    if isempty(Data)
        return;
    end
    Data = double(Data(:))';
    Take = min(numel(Data),Width);
    Row(1:Take) = Data(1:Take);
end

function Value = ResolveNearGap(PositiveCount,TotalCount)
    if TotalCount <= 0
        Value = NaN;
        return;
    end
    Value = abs(PositiveCount/TotalCount - 0.5);
end

function Value = SafeRatio(Numerator,Denominator)
    if Denominator <= 0
        Value = NaN;
    else
        Value = Numerator/Denominator;
    end
end

function Value = MeanFinite(Data)
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
    else
        Value = mean(Data);
    end
end

function Value = MedianFinite(Data)
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
    else
        Value = median(Data);
    end
end

function Value = WeightedMeanFinite(Data,Weights)
    Valid = isfinite(Data) & isfinite(Weights) & Weights > 0;
    if ~any(Valid)
        Value = NaN;
        return;
    end
    Value = sum(Data(Valid).*Weights(Valid))/sum(Weights(Valid));
end

function Label = ResolveDominantLabel(Labels)
    if isempty(Labels)
        Label = '';
        return;
    end
    [Names,~,Index] = unique(Labels,'stable');
    Count = accumarray(Index(:),1);
    [~,Best] = max(Count);
    Label = Names{Best};
end

function Value = FieldOrDefault(Row,Field,Default)
    if isstruct(Row) && isfield(Row,Field) && ~isempty(Row.(Field))
        Value = Row.(Field);
    else
        Value = Default;
    end
end
