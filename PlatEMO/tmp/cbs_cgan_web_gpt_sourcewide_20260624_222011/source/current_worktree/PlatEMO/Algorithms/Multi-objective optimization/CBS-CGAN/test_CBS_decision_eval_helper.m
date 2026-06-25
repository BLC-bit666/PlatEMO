function test_CBS_decision_eval_helper()
%TEST_CBS_DECISION_EVAL_HELPER Verify objective evaluation helper fallback.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));

    Problem = LIRCMOP5_BC('N',8,'D',6,'maxFE',100);
    Dec = Problem.Initialization(5).decs;
    originalFE = Problem.FE;
    [Obj,Con] = EvaluateDecisions_CBS(Problem,Dec);

    assert(size(Obj,1) == 5 && size(Obj,2) == Problem.M, ...
        'EvaluateDecisions_CBS must return the problem objective dimension.');
    assert(size(Con,1) == 5, ...
        'EvaluateDecisions_CBS must return one constraint row per decision.');
    assert(Problem.FE == originalFE, ...
        'EvaluateDecisions_CBS must not consume FE on the active Problem.');
    assert(any(abs(Obj(:)) > 1e-12), ...
        'LIRCMOP_BC fallback must not use the base PROBLEM.CalObj zeros.');

    fprintf('CBS decision evaluation helper regression passed.\n');
end
