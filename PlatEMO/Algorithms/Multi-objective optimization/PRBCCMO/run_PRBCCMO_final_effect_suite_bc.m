function Suite = run_PRBCCMO_final_effect_suite_bc(varargin)
% Run the fix.md final-effect validation on the complete BC suite.
%
% Default comparison:
%   boundarycore_beta vs no_boundary vs CCMO vs NAEMT2025
%
% Optional name-value pairs:
%   'Runs'             : independent runs per problem, default 30
%   'RunSeeds'         : explicit run seeds, default 1001:1000+Runs
%   'Population'       : population size, default 100
%   'MaxFE'            : max function evaluations, default 200000
%   'ProblemNames'     : explicit BC problem list, default all 37 BC problems
%   'Workers'          : parallel workers, default 6
%   'UseParallel'      : whether to enable parfor, default true
%   'SaveSlots'        : saved checkpoints per run, default 100
%   'PRBCCMOCalibrator': {'beta','raw'}, default 'beta'
%   'BaselineAlgs'     : external baselines, default {'CCMO','NAEMT2025'}
%   'VariantSpecs'     : explicit variant list, default auto-resolved
%   'BaseVariant'      : paired-test reference, default 'boundarycore_beta'
%   'OutputDir'        : output directory, default timestamped dir in pwd
%   'SaveMat'          : whether to save MAT, default true
%   'Verbose'          : print progress, default true

    Params = struct( ...
        'Runs',30, ...
        'RunSeeds',[], ...
        'Population',100, ...
        'MaxFE',200000, ...
        'ProblemNames',{{}}, ...
        'Workers',6, ...
        'UseParallel',true, ...
        'SaveSlots',100, ...
        'PRBCCMOCalibrator','beta', ...
        'BaselineAlgs',{{'CCMO','NAEMT2025'}}, ...
        'VariantSpecs',repmat(struct('id','','label','','algorithm','', ...
            'parameter',[]),0,1), ...
        'BaseVariant','boundarycore_beta', ...
        'OutputDir','', ...
        'SaveMat',true, ...
        'Verbose',true);
    Params = ParseInputs(Params,varargin{:});
    Params.ProblemNames = NormalizeProblemNames(Params.ProblemNames);
    Params.RunSeeds = NormalizeRunSeeds(Params.RunSeeds,Params.Runs);
    Params.OutputDir = ResolveOutputDir(Params.OutputDir);
    EnsureDirectory(Params.OutputDir);

    SavePrefix = fullfile(Params.OutputDir,sprintf( ...
        'benchmark_PRBCCMO_final_effect_%dbc_r%d', ...
        numel(Params.ProblemNames),numel(Params.RunSeeds)));

    if logical(Params.Verbose)
        fprintf('[FinalEffectSuite] outputDir=%s\n',Params.OutputDir);
        fprintf('[FinalEffectSuite] problems=%d runs=%d workers=%d\n', ...
            numel(Params.ProblemNames),numel(Params.RunSeeds),Params.Workers);
    end

    Results = benchmark_PRBCCMO_final_effect_validation( ...
        'Runs',Params.Runs, ...
        'RunSeeds',Params.RunSeeds, ...
        'Population',Params.Population, ...
        'MaxFE',Params.MaxFE, ...
        'ProblemNames',Params.ProblemNames, ...
        'UseParallel',Params.UseParallel, ...
        'Workers',Params.Workers, ...
        'SaveSlots',Params.SaveSlots, ...
        'PRBCCMOCalibrator',Params.PRBCCMOCalibrator, ...
        'BaselineAlgs',Params.BaselineAlgs, ...
        'VariantSpecs',Params.VariantSpecs, ...
        'BaseVariant',Params.BaseVariant, ...
        'SavePrefix',SavePrefix, ...
        'SaveMat',Params.SaveMat, ...
        'Verbose',Params.Verbose);

    Suite = struct();
    Suite.params = Params;
    Suite.outputDir = Params.OutputDir;
    Suite.savePrefix = SavePrefix;
    Suite.results = Results;
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('run_PRBCCMO_final_effect_suite_bc:InvalidInput', ...
            'Inputs must be name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~(ischar(Name) || (isstring(Name) && isscalar(Name)))
            error('run_PRBCCMO_final_effect_suite_bc:InvalidInputName', ...
                'Input names must be character vectors or scalar strings.');
        end
        Name = char(Name);
        if ~isfield(Params,Name)
            error('run_PRBCCMO_final_effect_suite_bc:UnknownOption', ...
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
        error('run_PRBCCMO_final_effect_suite_bc:InvalidRunSeeds', ...
            'RunSeeds must contain at least one finite integer.');
    end
end

function OutputDir = ResolveOutputDir(OutputDir)
    if nargin >= 1 && ~isempty(OutputDir)
        OutputDir = char(OutputDir);
        return;
    end
    Stamp = char(string(datetime('now','Format','yyyyMMdd_HHmmss')));
    OutputDir = fullfile(pwd,['results_prbccmo_final_effect_bc_r30_',Stamp]);
end

function EnsureDirectory(PathStr)
    if exist(PathStr,'dir') ~= 7
        mkdir(PathStr);
    end
end
