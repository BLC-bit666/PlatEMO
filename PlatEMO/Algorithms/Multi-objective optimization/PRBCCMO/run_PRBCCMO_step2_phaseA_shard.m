function Results = run_PRBCCMO_step2_phaseA_shard(RunIndex,varargin)
% Run one Phase A Step-2 shard for a single paired seed.

    if nargin < 1 || isempty(RunIndex)
        error('PRBCCMO:PhaseAShardInput', ...
            'RunIndex must be provided.');
    end

    Params = struct( ...
        'BaseDir',DefaultBaseDir(), ...
        'Runs',10, ...
        'Population',100, ...
        'MaxFE',200000, ...
        'Verbose',true);
    Params = ParseInputs(Params,varargin{:});

    RunIndex = round(double(RunIndex));
    if ~isfinite(RunIndex) || RunIndex < 1
        error('PRBCCMO:PhaseAShardInput', ...
            'RunIndex must be a positive integer.');
    end

    BaseDir = char(Params.BaseDir);
    ShardDir = fullfile(BaseDir,sprintf('shard_%02d',RunIndex));
    if exist(ShardDir,'dir') ~= 7
        mkdir(ShardDir);
    end
    TmpDir = fullfile(ShardDir,'tmp');
    if exist(TmpDir,'dir') ~= 7
        mkdir(TmpDir);
    end

    Results = check_PRBCCMO_step2_query_trust( ...
        'Runs',Params.Runs, ...
        'RunIndices',RunIndex, ...
        'Population',Params.Population, ...
        'MaxFE',Params.MaxFE, ...
        'QueryVariants',{'ParetoOnly','FullV2'}, ...
        'CalibratorVariants',{'raw','beta'}, ...
        'SavePath',fullfile(TmpDir,'step2_results.mat'), ...
        'QueryCsv',fullfile(ShardDir,'step2_query_runs.csv'), ...
        'QueryUpdateCsv',fullfile(ShardDir,'step2_query_updates.csv'), ...
        'CalibratorCsv',fullfile(ShardDir,'step2_calibrator_runs.csv'), ...
        'QueryPairedCsv',fullfile(ShardDir,'step2_query_paired.csv'), ...
        'CalibratorPairedCsv',fullfile(ShardDir,'step2_calibrator_paired.csv'), ...
        'Verbose',logical(Params.Verbose));

    writetable(struct2table(Results.querySummary,'AsArray',true), ...
        fullfile(ShardDir,'step2_query_summary.csv'));
    writetable(struct2table(Results.calibratorSummary,'AsArray',true), ...
        fullfile(ShardDir,'step2_calibrator_summary.csv'));
    WriteDone(fullfile(ShardDir,'DONE.txt'),RunIndex);
end

function BaseDir = DefaultBaseDir()
    Here = fileparts(mfilename('fullpath'));
    BaseDir = fullfile(Here,'results_step2_dascmop9_r10_phaseA_parallel_20260326');
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('PRBCCMO:PhaseAShardInput', ...
            'Inputs must be provided as name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~isfield(Params,Name)
            error('PRBCCMO:PhaseAShardInput', ...
                'Unknown parameter ''%s''.',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function WriteDone(FilePath,RunIndex)
    FID = fopen(FilePath,'w');
    if FID < 0
        error('PRBCCMO:PhaseAShardIO', ...
            'Unable to open DONE file: %s',FilePath);
    end
    Cleanup = onCleanup(@() fclose(FID));
    fprintf(FID,'run=%d\ncompleted=%s\n',RunIndex,datestr(now,31));
    clear Cleanup;
end
