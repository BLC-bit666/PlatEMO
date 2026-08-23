function Model = NAEMTTrainModel(DataX,DataY,Model)
% Train or continue training the single-hidden-layer MLP used by NA-EMT.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group.
%--------------------------------------------------------------------------

    if isempty(DataX) || isempty(DataY) || size(DataX,1) ~= numel(DataY)
        error('NAEMT:InvalidTrainingData', ...
            'The MLP training decisions and labels must be nonempty and aligned.');
    end
    DataY = double(DataY(:)');
    if any(~ismember(DataY,[0,1]))
        error('NAEMT:InvalidTrainingLabels', ...
            'The MLP labels must be binary, with feasible=1 and infeasible=0.');
    end

    if nargin < 3 || isempty(Model)
        if isempty(which('patternnet'))
            error('NAEMT:MissingPatternNet', ...
                'NAEMT requires patternnet from Deep Learning Toolbox.');
        end
        net = patternnet(10,'trainscg');
        net.layers{1}.transferFcn  = 'poslin';
        net.layers{2}.transferFcn  = 'logsig';
        net.performFcn             = 'crossentropy';
        net.divideFcn              = 'dividerand';
        net.divideMode             = 'sample';
        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio   = 0;
        net.divideParam.testRatio  = 0.2;
        net.trainParam.showWindow  = false;
        if isfield(net.trainParam,'showCommandLine')
            net.trainParam.showCommandLine = false;
        end
        Model = struct('Net',[]);
    else
        if ~isstruct(Model) || ~isscalar(Model) || ~isfield(Model,'Net') || ...
                ~isa(Model.Net,'network')
            error('NAEMT:InvalidModel', ...
                'Warm-start retraining requires a previously trained NA-EMT network.');
        end
        net = Model.Net;
        if net.inputs{1}.size ~= size(DataX,2)
            error('NAEMT:InconsistentInputDimension', ...
                'The updated training data must match the existing network input size.');
        end
    end
    % MATLAB train() continues from the weights stored in an existing net.
    Model.Net = train(net,DataX',DataY);
end
