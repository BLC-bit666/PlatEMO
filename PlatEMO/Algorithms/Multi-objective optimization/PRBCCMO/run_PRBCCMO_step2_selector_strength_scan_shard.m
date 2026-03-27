function Results = run_PRBCCMO_step2_selector_strength_scan_shard(RunIndex,varargin)
% Run one selector-strength scan shard for a single paired seed without MAT outputs.

    if nargin < 1 || isempty(RunIndex)
        error('PRBCCMO:SelectorStrengthScanShardInput', ...
            'RunIndex must be provided.');
    end

    Params = struct( ...
        'BaseDir',DefaultBaseDir(), ...
        'Problems',{{}}, ...
        'Runs',8, ...
        'Population',100, ...
        'MaxFE',200000, ...
        'Verbose',true);
    Params = ParseInputs(Params,varargin{:});

    RunIndex = round(double(RunIndex));
    if ~isfinite(RunIndex) || RunIndex < 1
        error('PRBCCMO:SelectorStrengthScanShardInput', ...
            'RunIndex must be a positive integer.');
    end

    [QueryVariants,QueryVariantRuntimeOverrides,ManifestRows] = BuildQueryVariantGrid();
    BaseDir = char(Params.BaseDir);
    ShardDir = fullfile(BaseDir,sprintf('shard_%02d',RunIndex));
    if exist(ShardDir,'dir') ~= 7
        mkdir(ShardDir);
    end

    CheckArgs = { ...
        'Runs',Params.Runs, ...
        'RunIndices',RunIndex, ...
        'Population',Params.Population, ...
        'MaxFE',Params.MaxFE, ...
        'QueryVariants',QueryVariants, ...
        'QueryVariantRuntimeOverrides',QueryVariantRuntimeOverrides, ...
        'CalibratorVariants',{'raw','beta'}, ...
        'SavePath','', ...
        'QueryCsv',fullfile(ShardDir,'step2_query_runs.csv'), ...
        'QueryUpdateCsv','', ...
        'CalibratorCsv',fullfile(ShardDir,'step2_calibrator_runs.csv'), ...
        'QueryPairedCsv','', ...
        'CalibratorPairedCsv','', ...
        'EnableStopGoGate',false, ...
        'Verbose',logical(Params.Verbose)};
    if ~isempty(Params.Problems)
        CheckArgs = [CheckArgs, {'Problems',Params.Problems}]; %#ok<AGROW>
    end
    Results = check_PRBCCMO_step2_query_trust(CheckArgs{:});

    writetable(ManifestRows,fullfile(ShardDir,'selector_scan_variants.csv'));
    WriteDone(fullfile(ShardDir,'DONE.txt'),RunIndex);
end

function BaseDir = DefaultBaseDir()
    Here = fileparts(mfilename('fullpath'));
    BaseDir = fullfile(Here,'results_step2_selector_strength_scan_r8_20260326');
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('PRBCCMO:SelectorStrengthScanShardInput', ...
            'Inputs must be provided as name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~isfield(Params,Name)
            error('PRBCCMO:SelectorStrengthScanShardInput', ...
                'Unknown parameter ''%s''.',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function [QueryVariants,Overrides,ManifestRows] = BuildQueryVariantGrid()
    AlphaLowSet = [0.10,0.15,0.20,0.25];
    ShortlistSet = [2,3,4];
    FallbackCapSet = [0.25,0.35,0.50];

    QueryVariants = {'ParetoOnly';'FullSoftTrust'};
    Overrides = {struct();struct()};
    Manifest = repmat(InitManifestRow(),0,1);

    Manifest(end+1,1) = MakeManifestRow('ParetoOnly','pareto',NaN,NaN,NaN,false,false); %#ok<AGROW>
    CurrentOptions = BuildBoundaryRuntimeOptions();
    Manifest(end+1,1) = MakeManifestRow( ...
        'FullSoftTrust','current',CurrentOptions.FullAlphaLowMax, ...
        CurrentOptions.FullShortlistFactor,CurrentOptions.TrustFallbackCap, ...
        true,true); %#ok<AGROW>

    for a = 1 : numel(AlphaLowSet)
        for k = 1 : numel(ShortlistSet)
            for c = 1 : numel(FallbackCapSet)
                VariantName = sprintf('FullSoftTrust__a%03d__k%d__cap%03d', ...
                    round(100*AlphaLowSet(a)),ShortlistSet(k),round(100*FallbackCapSet(c)));
                QueryVariants{end+1,1} = VariantName; %#ok<AGROW>
                Overrides{end+1,1} = struct( ... %#ok<AGROW>
                    'FullAlphaLowMax',AlphaLowSet(a), ...
                    'FullShortlistFactor',ShortlistSet(k), ...
                    'TrustFallbackCap',FallbackCapSet(c));
                Manifest(end+1,1) = MakeManifestRow( ... %#ok<AGROW>
                    VariantName,'scan',AlphaLowSet(a),ShortlistSet(k),FallbackCapSet(c), ...
                    false,IsCurrentConfig(AlphaLowSet(a),ShortlistSet(k),FallbackCapSet(c),CurrentOptions));
            end
        end
    end

    ManifestRows = struct2table(Manifest,'AsArray',true);
end

function Flag = IsCurrentConfig(AlphaLow,ShortlistFactor,FallbackCap,CurrentOptions)
    Flag = abs(AlphaLow - CurrentOptions.FullAlphaLowMax) <= 1e-12 && ...
        abs(ShortlistFactor - CurrentOptions.FullShortlistFactor) <= 1e-12 && ...
        abs(FallbackCap - CurrentOptions.TrustFallbackCap) <= 1e-12;
end

function Row = InitManifestRow()
    Row = struct( ...
        'queryVariant','', ...
        'variantGroup','', ...
        'alphaLowMax',NaN, ...
        'shortlistFactor',NaN, ...
        'trustFallbackCap',NaN, ...
        'isCurrentVariant',false, ...
        'matchesCurrentConfig',false);
end

function Row = MakeManifestRow(QueryVariant,VariantGroup,AlphaLow,ShortlistFactor,FallbackCap,IsCurrentVariant,MatchesCurrentConfig)
    Row = InitManifestRow();
    Row.queryVariant = QueryVariant;
    Row.variantGroup = VariantGroup;
    Row.alphaLowMax = AlphaLow;
    Row.shortlistFactor = ShortlistFactor;
    Row.trustFallbackCap = FallbackCap;
    Row.isCurrentVariant = logical(IsCurrentVariant);
    Row.matchesCurrentConfig = logical(MatchesCurrentConfig);
end

function WriteDone(FilePath,RunIndex)
    FID = fopen(FilePath,'w');
    if FID < 0
        error('PRBCCMO:SelectorStrengthScanShardIO', ...
            'Unable to open DONE file: %s',FilePath);
    end
    Cleanup = onCleanup(@() fclose(FID));
    fprintf(FID,'run=%d\ncompleted=%s\n',RunIndex,datestr(now,31));
    clear Cleanup;
end
