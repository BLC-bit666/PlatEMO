function Results = run_CCMO_GAN_BDG_fixmd_four_experiments( ...
        workerCount,problemNames,N,D,maxFE,runIds)
%run_CCMO_GAN_BDG_fixmd_four_experiments Compatibility wrapper.
%   Old C4-C7/ref-vector branches have been retired. New runs use the
%   FixMD global nondominated mainline.

    rootDir = fileparts(which('platemo'));
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(workerCount)
        workerCount = 6;
    end
    if nargin < 2 || isempty(problemNames)
        problemNames = defaultProblemList_fixmd_BDG();
    end
    if nargin < 3 || isempty(N)
        N = 100;
    end
    if nargin < 4
        D = [];
    end
    if nargin < 5 || isempty(maxFE)
        maxFE = 100000;
    end
    if nargin < 6 || isempty(runIds)
        runIds = 1:3;
    end

    stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
    baseDir = fullfile(rootDir,'Data','CCMO_GAN_BDG');
    outRoot = fullfile(baseDir,sprintf( ...
        'FixMD_global_runs%d_n%d_fe%d_%dw_%s', ...
        numel(runIds),round(double(N)),round(double(maxFE)), ...
        workerCount,stamp));
    try
        diagOptions = struct( ...
            'variant',"FixMD_GNDk60_nearseg_huber_z0", ...
            'algorithmParams',{fixmdGlobalAlgorithmParams_BDG()});
        [Summary,outDir] = run_CCMO_GAN_BDG_boundary_diagnostics( ...
            outRoot,workerCount,problemNames,N,D,maxFE,runIds, ...
            maxFE,diagOptions);
        Row = emptyFixmdRunRow_BDG();
        Row.variant = "FixMD_GNDk60_nearseg_huber_z0";
        Row.outDir = string(outDir);
        Row.status = string(all(Summary.status == "ok"));
        if Row.status == "true"
            Row.status = "ok";
        else
            Row.status = "has_error";
        end
    catch err
        Row = emptyFixmdRunRow_BDG();
        Row.variant = "FixMD_GNDk60_nearseg_huber_z0";
        Row.outDir = string(outRoot);
        Row.status = "failed";
        Row.error_message = string(getReport(err,'extended', ...
            'hyperlinks','off'));
    end
    Results = struct2table(Row);
    writetable(Results,fullfile(baseDir,sprintf( ...
        'fixmd_global_experiment_dir_%s.csv',stamp)));
end

function Params = fixmdGlobalAlgorithmParams_BDG()
    Params = {1,20,0,5,0.20,50,0,200,0,1,1,1, ...
        "g2sl",64,0.9,1e-4,1e-4,"epoch"};
end

function Row = emptyFixmdRunRow_BDG()
    Row = struct('variant',"",'outDir',"",'status',"not_started", ...
        'error_message',"");
end

function names = defaultProblemList_fixmd_BDG()
    names = { ...
        'DASCMOP1_BC'; ...
        'DASCMOP2_BC'; ...
        'DASCMOP4_BC'; ...
        'DASCMOP5_BC'; ...
        'LIRCMOP5_BC'; ...
        'LIRCMOP6_BC'; ...
        'LIRCMOP7_BC'; ...
        'LIRCMOP8_BC'; ...
        'LIRCMOP9_BC'; ...
        'LIRCMOP10_BC'};
end
