function [Obj,Con] = EvaluateDecisions_CBS(Problem,Dec)
%EVALUATEDECISIONS_CBS Evaluate objectives without corrupting active FE.

    Obj = zeros(0,Problem.M);
    Con = zeros(size(Dec,1),0);
    if isempty(Dec)
        return;
    end

    Dec = Problem.CalDec(Dec);
    Obj = Problem.CalObj(Dec);
    Con = Problem.CalCon(Dec);
    if size(Obj,2) == Problem.M
        return;
    end

    EvalProblem = feval(class(Problem),'N',Problem.N,'D',Problem.D, ...
        'maxFE',max(Problem.maxFE,size(Dec,1)));
    Pop = EvalProblem.Evaluation(Dec);
    Obj = Pop.objs;
    Con = Pop.cons;
end
