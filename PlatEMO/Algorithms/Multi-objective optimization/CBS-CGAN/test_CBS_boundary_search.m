function test_CBS_boundary_search()
%TEST_CBS_BOUNDARY_SEARCH Verify the always-on calibration primitive.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addCBSPaths(repoRoot);
    testBudgetAndEvaluation();
    testRangeNormalizedNearestEndpoint();
    testNoFeasibleInput();
    fprintf('CBS boundary calibration regressions passed.\n');
end

function testRangeNormalizedNearestEndpoint()
    Problem = CF4_BC('N',3,'D',2,'maxFE',20);
    feasible = [0.5,0];
    rawNearest = [0.3,0];
    normalizedNearest = [0.5,-0.3];
    Pop = Problem.Evaluation([feasible;rawNearest;normalizedNearest]);
    assert(Pop(1).cons <= 0 && Pop(2).cons > 0 && Pop(3).cons > 0);
    span = Problem.upper-Problem.lower;
    assert(norm(rawNearest-feasible) < norm(normalizedNearest-feasible));
    assert(norm((rawNearest-feasible)./span) > ...
        norm((normalizedNearest-feasible)./span));

    beforeFE = Problem.FE;
    [Candidates,Brackets] = RefineBoundaryObservations_RC( ...
        Problem,Pop,Pop([]),1);
    expected = (feasible+normalizedNearest)/2;
    assert(isscalar(Candidates) && Problem.FE == beforeFE+1);
    assert(max(abs(Candidates.decs-expected),[],'all') < 1e-12);
    assert(max(abs(Brackets.xf-feasible),[],'all') < 1e-12);
    assert(max(abs(Brackets.xi-expected),[],'all') < 1e-12);
    assert(all(isfinite(Brackets.ell)) && all(Brackets.ell > 0));
end

function testBudgetAndEvaluation()
    Problem = LIRCMOP6_BC('N',3,'D',5,'maxFE',20);
    Decs = [0.2*ones(1,5);0.8*ones(1,5);0.5*ones(1,5)];
    Pop = SOLUTION(Decs,[1 2;2 1;3 3],[0;0;1]);
    rng(17,'twister');
    Candidates = RefineBoundaryObservations_RC( ...
        Problem,Pop(1:2),Pop(3),5);
    assert(numel(Candidates) == 5 && Problem.FE == 5);
end

function testNoFeasibleInput()
    Problem = LIRCMOP6_BC('N',2,'D',5,'maxFE',20);
    Pop = SOLUTION([zeros(1,5);ones(1,5)],[1 2;2 1],[1;2]);
    Candidates = RefineBoundaryObservations_RC(Problem,Pop,Pop([]),5);
    assert(isempty(Candidates) && Problem.FE == 0);
end
