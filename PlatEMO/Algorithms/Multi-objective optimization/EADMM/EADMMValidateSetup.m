function EADMMValidateSetup(Problem)
% Validate dependencies and the minimum two-population budget of EADMM.

    if isempty(which('ga')) || isempty(which('optimoptions'))
        error('EADMM:MissingGA', ...
            ['EADMM requires ga from Global Optimization Toolbox, as ' ...
             'specified in the paper.']);
    end
    if Problem.maxFE < 2*Problem.N
        error('EADMM:InsufficientBudget', ...
            'maxFE must cover both initial populations of size N.');
    end
end
