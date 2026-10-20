function localN = EADMMValidateSetup(Problem,localN)
% Validate dependencies and the minimum two-population budget of EADMM.

    if nargin < 2; localN = 0; end
    validateattributes(localN,{'numeric'}, ...
        {'scalar','real','finite','integer','nonnegative'});
    if localN == 0
        if Problem.D <= 5; localN = 50; else; localN = 200; end
    elseif localN < 2
        error('EADMM:LocalPopulationTooSmall', ...
            'The local GA population must contain at least two individuals.');
    end

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
