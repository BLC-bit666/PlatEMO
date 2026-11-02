function Checked = EADMMLocalSearch(Problem,Archive,Population,localN,maxGenerations,maxStallGenerations)
% Solve EADMM's discrepancy subproblem with MATLAB's genetic algorithm.

    if nargin < 4; localN = EADMMValidateSetup(Problem); end
    if nargin < 5; maxGenerations = 100*Problem.D; end
    if nargin < 6; maxStallGenerations = 50; end

    PopCon = Population.cons;
    ell    = size(PopCon,2);
    if ell < 1
        error('EADMM:MissingConstraints', ...
            'EADMM requires at least one queryable binary constraint signal.');
    end
    gamma  = sum(all(PopCon<=0,2));
    rho    = gamma/numel(Population)*ell;

    options = LocalGAOptions(localN,maxGenerations,maxStallGenerations);
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

function options = LocalGAOptions(populationSize,maxGenerations,maxStallGenerations)
% Preserve the GA operators while allowing each backbone to set its budget.
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
        'MaxStallGenerations',maxStallGenerations, ...
        'MutationFcn',@mutationadaptfeasible, ...
        'PopulationSize',populationSize, ...
        'SelectionFcn',@selectionstochunif, ...
        'UseParallel',false, ...
        'UseVectorized',false);
end
