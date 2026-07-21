function run_CBS_branch100k_pilot()
%RUN_CBS_BRANCH100K_PILOT Paired second-half branch test for BLS.
%   For every mainline campaign run of the seven decomposition problems,
%   restart from the saved ~100k-FE snapshot (both populations seeded
%   with the snapshot decision vectors via initDecs) and spend the
%   remaining budget twice under identical seeds:
%     bls_on  - boundary calibration every generation (the mainline
%               second-half dynamics: empty guide buffer, 40/60 fallback
%               composition, no training) - also the fidelity anchor
%               against the true mainline finals;
%     bls_off - identical except calibration off (pure 40/60 backbone).
%   The paired per-seed difference isolates the second-half BLS effect.
%   The branch budget is 200000 - snapshotFE + 2N, compensating the two
%   injected initial evaluations that the true continuation never pays.
%   Results table: <root>/branch_pilot_v1/branch100k_results.mat

    rootDir = fileparts(which('platemo'));
    assert(~isempty(rootDir),'platemo must be on the path.');
    problems = ["DASCMOP2_BC","DASCMOP4_BC","LIRCMOP5_BC","LIRCMOP9_BC", ...
        "LIRCMOP10_BC","LIRCMOP12_BC","LIRCMOP13_BC"];
    seeds = 1:10;
    arms = ["bls_on","bls_off"];
    mainDir = fullfile(rootDir,'Data','CBS_RegionWGAN_GP');
    outDir = fullfile(rootDir,'branch_pilot_v1');
    if ~isfolder(outDir); mkdir(outDir); end

    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    zeroPar = {0,Defaults.zDim,Defaults.ganIter,Defaults.ganMiniBatch, ...
        Defaults.nCritic,Defaults.minGANTrainCount,Defaults.sampleSigma};

    [P,S,A] = ndgrid(1:numel(problems),seeds,1:numel(arms));
    tasks = [P(:),S(:),A(:)];
    total = size(tasks,1);
    rows = cell(total,1);
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= 10
        if ~isempty(pool); delete(pool); end
        parpool('local',10);
    end
    queue = parallel.pool.DataQueue;
    done = 0;
    afterEach(queue,@report);
    parfor t = 1:total
        Row = runBranch(problems(tasks(t,1)),tasks(t,2), ...
            arms(tasks(t,3)),mainDir,zeroPar);
        rows{t} = Row;
        send(queue,Row);
    end
    Results = struct2table(vertcat(rows{:}));
    save(fullfile(outDir,'branch100k_results.mat'),'Results');
    disp('branch100k pilot complete.');

    function report(Row)
        done = done + 1;
        fprintf('[%d/%d] %s seed=%d %s IGD=%.6g (main final %.6g)\n', ...
            done,total,Row.problem,Row.seed,Row.arm,Row.igd_branch, ...
            Row.igd_main_final);
    end
end

function Row = runBranch(problemName,seed,arm,mainDir,zeroPar)
    d = dir(fullfile(mainDir,sprintf('CBS_RegionWGAN_GP_%s_M*_%d.mat', ...
        char(problemName),seed)));
    assert(numel(d) == 1,'expected one mainline file for %s run %d.', ...
        problemName,seed);
    Saved = load(fullfile(d(1).folder,d(1).name),'result','metric');
    snapFE = double(Saved.result{1,1});
    SnapDecs = Saved.result{1,2}.decs;
    N = size(SnapDecs,1);
    branchFE = 200000 - snapFE + 2*N;
    try
        maxNumCompThreads(1);
    catch
    end
    rng(seed,'twister');
    Constructor = str2func(char(problemName));
    Problem = Constructor('N',N,'maxFE',branchFE);
    switches = {'save',0,'outputFcn',@quietOutput, ...
        'guideMode','on','scoutMode','off','generatorMode','wgan', ...
        'blsFeed','off','parameter',{zeroPar{:}}, ...
        'initDecs',SnapDecs}; %#ok<CCAT1>
    if arm == "bls_on"
        switches = [switches,{'boundarySearch','on','blsWindow','full'}];
    else
        switches = [switches,{'boundarySearch','off','blsWindow','late'}];
    end
    Algorithm = CBS_RegionWGAN_GP(switches{:});
    Algorithm.Solve(Problem);
    Final = Algorithm.result{end,2};
    Row = struct('problem',string(problemName),'seed',double(seed), ...
        'arm',string(arm),'snapFE',snapFE,'branchFE',branchFE, ...
        'igd_branch',IGD(Final,Problem.optimum), ...
        'igd_main_100k',double(Saved.metric.IGD(1)), ...
        'igd_main_final',double(Saved.metric.IGD(end)));
end

function quietOutput(~,~)
end
