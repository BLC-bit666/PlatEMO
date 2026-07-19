function test_CBS_boundary_search()
%TEST_CBS_BOUNDARY_SEARCH Verify the post-stop boundary line search.

    repoRoot = fileparts(fileparts(fileparts(fileparts( ...
        mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    testSwitchValidation();
    testSearchChangesTrajectoryAndKeepsBudget();
    testGracefulWithoutFeasibleSolutions();
    testTriageRunnerS2BArm();
    fprintf('CBS boundary-search regressions passed.\n');
end

function testSwitchValidation()
    Default = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
    assert(Default.effectiveBoundarySearch() == "on");
    Off = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
        'boundarySearch',"off");
    assert(Off.effectiveBoundarySearch() == "off");
    didThrow = false;
    try
        CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
            'boundarySearch',"maybe");
    catch Error
        didThrow = strcmp(Error.identifier, ...
            'CBSRegionGAN:BadBoundarySearch');
    end
    assert(didThrow);
end

function testSearchChangesTrajectoryAndKeepsBudget()
    off = runShort("off");
    on = runShort("on");
    assert(~isequal(off,on), ...
        'Boundary search did not change the trajectory.');
end

function value = runShort(state)
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',20,'D',10,'maxFE',2000);
    Algorithm = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
        'operatorMode','ga_de_half','boundarySearch',char(state));
    Algorithm.Solve(Problem);
    assert(Problem.FE == 2000);
    Population = Algorithm.result{end,2};
    decs = Population.decs;
    value = sum(decs(:));
end

function testGracefulWithoutFeasibleSolutions()
    rng(7,'twister');
    Problem = DASCMOP1_BC('N',10,'D',5,'maxFE',400);
    Algorithm = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
        'boundarySearch',"on");
    Algorithm.Solve(Problem);
    assert(Problem.FE == 400);
end

function testTriageRunnerS2BArm()
    outDir = string(tempname);
    cleanup = onCleanup(@()removeTree(outDir));
    Options = struct('problems',"LIRCMOP6_BC",'seeds',1, ...
        'maxFE',600,'N',10,'arms',["S2","S2B"]);
    Summary = run_CBS_operator_triage(outDir,1,Options);
    assert(height(Summary) == 2);
    assert(all(Summary.status == "ok"));
    assert(all(Summary.finalFE == 600));
    assert(isequal(string(Summary.bls),["off";"on"]));
    Reused = run_CBS_operator_triage(outDir,1,Options);
    assert(all(Reused.reused == 1));
end

function removeTree(pathValue)
    if isfolder(pathValue)
        rmdir(pathValue,'s');
    end
end
