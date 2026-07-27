function test_CBS_boundary_search()
%TEST_CBS_BOUNDARY_SEARCH Verify the always-on calibration primitive.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    testBudgetAndEvaluation();
    testNoFeasibleInput();
    fprintf('CBS boundary calibration regressions passed.\n');
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
