function Offspring = OperatorDECurrentToRand1(Problem,Parent,Parameter)
% DE/current-to-rand/1 with polynomial mutation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if nargin > 2
        [CR,F,proM,disM] = deal(Parameter{:});
    else
        [CR,F,proM,disM] = deal(1,0.5,1,20);
    end

    if isa(Parent(1),'SOLUTION')
        evaluated = true;
        ParentDec = Parent.decs;
    else
        evaluated = false;
        ParentDec = Parent;
    end
    [N,D] = size(ParentDec);

    Parent1 = ParentDec(randperm(N),:);
    Parent2 = ParentDec(randperm(N),:);
    Parent3 = ParentDec(randperm(N),:);

    Trial = ParentDec;
    Site  = rand(N,D) < CR;
    Trial(Site) = ParentDec(Site) + F.*(Parent1(Site)-ParentDec(Site)) + F.*(Parent2(Site)-Parent3(Site));

    Trial = OperatorDE(Problem,Trial,Trial,Trial,{1,0,proM,disM});
    if evaluated
        Offspring = Problem.Evaluation(Trial);
    else
        Offspring = Trial;
    end
end
