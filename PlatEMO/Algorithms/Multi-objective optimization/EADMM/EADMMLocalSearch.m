function Checked = EADMMLocalSearch(Problem,Archive,Population)
% Solve EADMM's discrepancy subproblem with MATLAB's genetic algorithm.

    PopCon = Population.cons;
    ell    = size(PopCon,2);
    if ell < 1
        error('EADMM:MissingConstraints', ...
            'EADMM requires at least one separately queryable constraint.');
    end
    gamma  = sum(all(PopCon<=0,2));
    rho    = gamma/numel(Population)*ell;

    options = R2021aGAOptions(Problem.D);
    Checked = [];
    center  = [];

    for i = 1 : numel(Archive)
        center  = Archive(i).dec;
        bestDec = ga(@EvaluateCandidate,Problem.D,[],[],[],[], ...
                     Problem.lower,Problem.upper,[],options);
        solution = Problem.Evaluation(Problem.CalDec(reshape(bestDec,1,[])));
        if isempty(Checked)
            Checked = solution;
        else
            Checked(end+1) = solution; %#ok<AGROW>
        end
    end

    function value = EvaluateCandidate(x)
        x     = Problem.CalDec(reshape(x,1,[]));
        con   = Problem.CalCon(x);
        value = sum(~(con<=0),2) + rho.*sum((x-center).^2,2);
    end
end

function options = R2021aGAOptions(D)
% Explicit fallback settings inferred from MATLAB R2021a GA defaults.

    if D <= 5
        populationSize = 50;
    else
        populationSize = 200;
    end
    maxGenerations = 100*D;

    options = optimoptions('ga', ...
        'ConstraintTolerance',1e-3, ...
        'CreationFcn',@gacreationuniform, ...
        'CrossoverFcn',@crossoverscattered, ...
        'CrossoverFraction',0.8, ...
        'Display','off', ...
        'EliteCount',ceil(0.05*populationSize), ...
        'FitnessScalingFcn',@fitscalingrank, ...
        'FunctionTolerance',1e-6, ...
        'MaxGenerations',maxGenerations, ...
        'MaxStallGenerations',50, ...
        'MutationFcn',@mutationadaptfeasible, ...
        'PopulationSize',populationSize, ...
        'SelectionFcn',@selectionstochunif, ...
        'UseParallel',false, ...
        'UseVectorized',false);
end
