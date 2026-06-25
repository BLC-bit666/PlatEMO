function Results = run_CCMO_GAN_BDG_gnd_covgate_experiments( ...
        workerCount,problemNames,N,D,maxFE,runIds)
%run_CCMO_GAN_BDG_gnd_covgate_experiments Run global-ND controls.
%   The two variants isolate:
%   1) ref-local ND -> global ND only;
%   2) global ND plus a post-filter CGAN training coverage gate.

    rootDir = fileparts(which('platemo'));
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(workerCount)
        workerCount = 7;
    end
    if nargin < 2 || isempty(problemNames)
        problemNames = defaultProblemList_gndcovgate_BDG();
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

    workerCount = max(1,round(double(workerCount)));
    variants = ["GND_keep80"; ...
        "GND_keep80_covGate"];
    variantSets = ["gnd_keep80"; ...
        "gnd_keep80_covgate"];
    minRefCov = [0.00;0.50];
    minTargetTriples = [0;50];
    algorithmParams = gndCovGateAlgorithmParams_BDG();
    stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
    baseDir = fullfile(rootDir,'Data','CCMO_GAN_BDG');
    if ~isfolder(baseDir)
        mkdir(baseDir);
    end
    Rows = repmat(emptyGNDCovGateRunRow_BDG(),numel(variants),1);
    for i = 1 : numel(variants)
        outDir = fullfile(baseDir,sprintf( ...
            '%s_runs%d_n%d_fe%d_%dw_%s', ...
            char(variants(i)),numel(runIds),round(double(N)), ...
            round(double(maxFE)),workerCount,stamp));
        Rows(i).variant = variants(i);
        Rows(i).variantSet = variantSets(i);
        Rows(i).cganTrainMinRefCov = minRefCov(i);
        Rows(i).cganTrainMinTargetTriples = minTargetTriples(i);
        Rows(i).outDir = string(outDir);
        try
            run_CCMO_GAN_BDG_archive_pareto_filters_fullscope( ...
                outDir,workerCount,problemNames,N,D,maxFE,runIds, ...
                variantSets(i),algorithmParams,false);
            Rows(i).status = "ok";
        catch err
            Rows(i).status = "failed";
            Rows(i).error_message = string(getReport(err,'extended', ...
                'hyperlinks','off'));
        end
    end
    Results = struct2table(Rows);
    writetable(Results,fullfile(baseDir,sprintf( ...
        'gnd_covgate_experiment_dirs_%s.csv',stamp)));
end

function Params = gndCovGateAlgorithmParams_BDG()
    Params = {1,50,4,5,0.20,50,0,200,0,1,1,1, ...
        "g2sl",64,0.9,1e-4,1e-4,"epoch"};
end

function Row = emptyGNDCovGateRunRow_BDG()
    Row = struct('variant',"",'variantSet',"", ...
        'cganTrainMinRefCov',NaN, ...
        'cganTrainMinTargetTriples',NaN, ...
        'outDir',"",'status',"not_started",'error_message',"");
end

function names = defaultProblemList_gndcovgate_BDG()
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
