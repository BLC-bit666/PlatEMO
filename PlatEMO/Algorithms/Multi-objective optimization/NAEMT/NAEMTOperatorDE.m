function Offspring = NAEMTOperatorDE(Problem,Parent,Parameter)
% DE/current-to-rand/1 followed by polynomial mutation.

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
    if N < 4
        error('NAEMT:PopulationTooSmall', ...
            'DE/current-to-rand/1 requires at least four individuals.');
    end

    Donor = zeros(N,D);
    for i = 1 : N
        candidates = [1:i-1,i+1:N];
        selected   = candidates(randperm(N-1,3));
        Donor(i,:) = ParentDec(i,:) + F.*(ParentDec(selected(1),:)-ParentDec(i,:)) ...
                   + F.*(ParentDec(selected(2),:)-ParentDec(selected(3),:));
    end

    Site = rand(N,D) < CR;
    Site(sub2ind([N,D],(1:N)',randi(D,N,1))) = true;
    Trial       = ParentDec;
    Trial(Site) = Donor(Site);

    % Reuse PlatEMO's polynomial mutation and bound repair. With identical
    % parents and F=0, its differential term is exactly zero.
    Trial = OperatorDE(Problem,Trial,Trial,Trial,{1,0,proM,disM});
    if evaluated
        Offspring = Problem.Evaluation(Trial);
    else
        Offspring = Trial;
    end
end
