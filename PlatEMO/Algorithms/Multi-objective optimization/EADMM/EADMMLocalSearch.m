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
    Checked = cell(1,numel(Archive));
    center  = [];
    metadata = metaclass(Problem);
    repairMethod = metadata.MethodList(strcmp({metadata.MethodList.Name},'CalDec'));
    defaultRepair = strcmp(repairMethod.DefiningClass.Name,'PROBLEM');

    for i = 1 : numel(Archive)
        center  = Archive(i).dec;
        bestDec = ga(@EvaluateCandidate,Problem.D,[],[],[],[], ...
                     Problem.lower,Problem.upper,[],options);
        Checked{i} = Problem.Evaluation(reshape(bestDec,1,[]));
    end
    Checked = [Checked{:}];

    function value = EvaluateCandidate(x)
        x = reshape(x,1,[]);
        if defaultRepair
            % The base CalDec repairs one row here. Preserve min/max order
            % and read current bounds/encoding on every constraint query.
            x = max(min(x,Problem.upper),Problem.lower);
            encoding = Problem.encoding;
            index = encoding==2 | encoding==3 | encoding==4;
            x(:,index) = round(x(:,index));
        else
            x = Problem.CalDec(x);
        end
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
