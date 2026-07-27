function Population = EADMMIBEASelection(Population,N,kappa)
% Vanilla IBEA environmental selection with unique EADMM-local naming.

    Next          = 1 : numel(Population);
    [Fitness,I,C] = EADMMIBEACalFitness(Population.objs,kappa);
    while numel(Next) > N
        [~,x]   = min(Fitness(Next));
        Fitness = Fitness + exp(-I(Next(x),:)./C(Next(x))./kappa);
        Next(x) = [];
    end
    Population = Population(Next);
end
