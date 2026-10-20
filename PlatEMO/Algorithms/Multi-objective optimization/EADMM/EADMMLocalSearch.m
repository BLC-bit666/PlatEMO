function Checked = EADMMLocalSearch(Problem,Archive,Population,localN)
% Solve EADMM's discrepancy subproblem with MATLAB's genetic algorithm.

    if nargin < 4; localN = EADMMValidateSetup(Problem); end

    PopCon = Population.cons;
    ell    = size(PopCon,2);
    if ell < 1
        error('EADMM:MissingConstraints', ...
            'EADMM requires at least one queryable binary constraint signal.');
    end
    gamma  = sum(all(PopCon<=0,2));
    rho    = gamma/numel(Population)*ell;

    options = R2021aGAOptions(Problem.D,localN);
    Checked = [];
    center  = [];

    for i = 1 : numel(Archive)
        center  = Archive(i).dec;
        bestDec = ga(@EvaluateCandidate,Problem.D,[],[],[],[], ...
                     Problem.lower,Problem.upper,[],options);
        solution = Problem.Evaluation(reshape(bestDec,1,[]));
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

function options = R2021aGAOptions(D,populationSize)
% Only the paper's local population size is exposed; other options stay fixed.
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
