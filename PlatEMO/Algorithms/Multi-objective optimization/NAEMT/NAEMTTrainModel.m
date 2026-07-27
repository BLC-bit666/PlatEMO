function Model = NAEMTTrainModel(DataX,DataY)
% Train the single-hidden-layer MLP used by NA-EMT.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group.
%--------------------------------------------------------------------------

    if isempty(DataX) || isempty(DataY) || size(DataX,1) ~= numel(DataY)
        error('NAEMT:InvalidTrainingData', ...
            'The MLP training decisions and labels must be nonempty and aligned.');
    end
    DataY = double(DataY(:)');
    Model = struct('Net',[]);
    if isempty(which('patternnet'))
        error('NAEMT:MissingPatternNet', ...
            'NAEMT requires patternnet from Deep Learning Toolbox.');
    end

    net = patternnet(10,'trainscg');
    net.layers{1}.transferFcn              = 'poslin';
    net.layers{2}.transferFcn              = 'logsig';
    net.performFcn                         = 'crossentropy';
    net.divideFcn                          = 'dividerand';
    net.divideMode                         = 'sample';
    net.divideParam.trainRatio             = 0.8;
    net.divideParam.valRatio               = 0;
    net.divideParam.testRatio              = 0.2;
    net.inputs{1}.processFcns              = {};
    net.outputs{net.numLayers}.processFcns = {};
    net.trainParam.showWindow              = false;
    if isfield(net.trainParam,'showCommandLine')
        net.trainParam.showCommandLine = false;
    end
    Model.Net = train(net,DataX',DataY);
end
