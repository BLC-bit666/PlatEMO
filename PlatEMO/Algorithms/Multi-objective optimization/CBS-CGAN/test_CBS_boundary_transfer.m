function test_CBS_boundary_transfer()
%TEST_CBS_BOUNDARY_TRANSFER Certified boundary-target DE mainline.

    repoRoot = fileparts(fileparts(fileparts(fileparts( ...
        mfilename('fullpath')))));
    addCBSPaths(repoRoot);

    expected = [0.25,0.55,0.20,0.25,0.75];
    assert(isequal(CBS_RegionWGAN_GP.operatorSharesAtFE(99,100), ...
        expected));
    assert(isequal(CBS_RegionWGAN_GP.operatorSharesAtFE(100,100), ...
        expected));

    % Disable GAN training only to keep this mechanism regression short.
    % Reproduction, calibration, the 50% cutoff, and BT remain active.
    Parameters = {0,6,0,32,4,32,0.3};
    rng(4242,'twister');
    MainProblem = LIRCMOP5_BC('N',30,'D',10,'maxFE',2400);
    Mainline = CBS_RegionWGAN_GP('parameter',Parameters,'save',2, ...
        'outputFcn',@(varargin)[]);
    Mainline.Solve(MainProblem);
    FirstResult = Mainline.result;
    FirstSnapshot = Mainline.boundaryTransferSnapshot();

    assert(MainProblem.FE == 2400);
    assert(FirstSnapshot.bracketRows > 0);
    assert(FirstSnapshot.lateRequested > 0 && FirstSnapshot.lateUsed > 0);
    assert(FirstSnapshot.lateRequested == ...
        FirstSnapshot.lateUsed+FirstSnapshot.lateFallback);
    assert(all(isfinite(FirstResult{end,2}.objs),'all'));

    % Reusing one ALGORITHM object must not leak archived targets or counters.
    rng(4242,'twister');
    RepeatProblem = LIRCMOP5_BC('N',30,'D',10,'maxFE',2400);
    Mainline.Solve(RepeatProblem);
    assert(RepeatProblem.FE == 2400);
    assertResultEqual(FirstResult,Mainline.result);
    assert(isequal(FirstSnapshot,Mainline.boundaryTransferSnapshot()));
    fprintf(['CBS boundary-transfer test passed ' ...
        '(late guide use: %d/%d).\n'], ...
        FirstSnapshot.lateUsed,FirstSnapshot.lateRequested);
end

function assertResultEqual(A,B)
    assert(isequal(size(A),size(B)));
    for row = 1 : size(A,1)
        assert(isequal(A{row,1},B{row,1}));
        assert(isequal(A{row,2}.decs,B{row,2}.decs));
        assert(isequal(A{row,2}.objs,B{row,2}.objs));
        assert(isequal(A{row,2}.cons,B{row,2}.cons));
    end
end
