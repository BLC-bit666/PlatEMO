function test_CBS_cgan_ablation()
%TEST_CBS_CGAN_ABLATION Verify the two CGAN ablation branches.

    repoRoot = fileparts(fileparts(fileparts(fileparts( ...
        mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    testGeneratorModeValidation();
    testBranchesRunAndDiffer();
    testAblationRunnerArms();
    fprintf('CBS CGAN-ablation regressions passed.\n');
end

function testGeneratorModeValidation()
    Default = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
    assert(Default.effectiveGeneratorMode() == "wgan");
    Simple = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
        'generatorMode',"copynoise");
    assert(Simple.effectiveGeneratorMode() == "copynoise");
    didThrow = false;
    try
        CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
            'generatorMode',"vae");
    catch Error
        didThrow = strcmp(Error.identifier, ...
            'CBSRegionGAN:BadGeneratorMode');
    end
    assert(didThrow);
end

function testBranchesRunAndDiffer()
    mainline = shortRun({});
    noGAN = shortRun({'parameter',{0,6,10,8,2,8,0.3}});
    simple = shortRun({'generatorMode','copynoise'});
    assert(~isequal(mainline,noGAN), ...
        'nGen=0 did not change the trajectory.');
    assert(~isequal(mainline,simple), ...
        'copynoise did not change the trajectory.');
end

function value = shortRun(extraArgs)
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',20,'D',10,'maxFE',3000);
    baseArgs = {'save',0,'outputFcn',@(varargin)[],'guideMode','off'};
    if ~isempty(extraArgs) && ~strcmp(extraArgs{1},'parameter')
        args = [baseArgs,extraArgs];
    elseif ~isempty(extraArgs)
        args = [extraArgs,baseArgs];
    else
        args = baseArgs;
    end
    if isempty(extraArgs) || strcmp(extraArgs{1},'parameter')
        % Mainline and no-GAN runs share the reduced GAN parameters so the
        % only difference in the no-GAN arm is nGen = 0.
        if isempty(extraArgs)
            args = [{'parameter',{10,6,10,8,2,8,0.3}},baseArgs];
        end
    end
    Algorithm = CBS_RegionWGAN_GP(args{:});
    Algorithm.Solve(Problem);
    assert(Problem.FE == 3000);
    Population = Algorithm.result{end,2};
    decs = Population.decs;
    value = sum(decs(:));
end

function testAblationRunnerArms()
    outDir = string(tempname);
    cleanup = onCleanup(@()removeTree(outDir));
    Options = struct('problems',"LIRCMOP6_BC",'seeds',1, ...
        'maxFE',600,'N',10,'arms',["A0","GN"]);
    Summary = run_CBS_operator_triage(outDir,1,Options);
    assert(height(Summary) == 2);
    assert(all(Summary.status == "ok"));
    assert(all(Summary.finalFE == 600));
    assert(isequal(string(Summary.gen),["wgan";"copynoise"]));
    assert(isequal(Summary.nGenZero,[1;0]));
    Reused = run_CBS_operator_triage(outDir,1,Options);
    assert(all(Reused.reused == 1));
end

function removeTree(pathValue)
    if isfolder(pathValue)
        rmdir(pathValue,'s');
    end
end
