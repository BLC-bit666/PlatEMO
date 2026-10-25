function Offspring = NAEMTOperatorDE(Problem,Parent,Parameter)
% DE/current-to-rand/1 followed by polynomial mutation; K=1 gives rand/1.

    if nargin > 2
        if numel(Parameter) < 4 || numel(Parameter) > 5
            error('NAEMT:InvalidDEParameters','Expected {CR,F,proM,disM[,K]}.');
        end
        [CR,F,proM,disM] = deal(Parameter{1:4});
        if numel(Parameter) == 5
            K = Parameter{5};
        else
            K = F;
        end
    else
        [CR,F,proM,disM,K] = deal(1,0.5,1,20,1);
    end
    validateattributes(K,{'numeric'},{'scalar','real','finite','>=',0,'<=',1});
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
        Donor(i,:) = ParentDec(i,:) + K.*(ParentDec(selected(1),:)-ParentDec(i,:)) ...
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
