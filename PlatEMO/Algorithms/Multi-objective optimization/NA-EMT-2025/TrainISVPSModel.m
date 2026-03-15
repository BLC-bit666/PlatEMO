function Model = TrainISVPSModel(DataX,DataY)
% Train the MLP model used in ISVPS

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    DataY = double(DataY(:)');
    if isempty(which('patternnet'))
        error('NAEMT2025:MissingPatternNet', ...
            'patternnet is unavailable in the current MATLAB path.');
    end
    if isempty(DataX) || numel(unique(DataY)) < 2
        error('NAEMT2025:InvalidTrainingData', ...
            'Paper-faithful MLP training requires both feasible and infeasible samples.');
    end

    Model = struct();
    Model.InputSize  = size(DataX,2);
    Model.HiddenSize = 10;

    net = patternnet(10,'trainscg');
    net.layers{1}.transferFcn                  = 'poslin';
    net.layers{2}.transferFcn                  = 'logsig';
    net.performFcn                             = 'crossentropy';
    net.divideFcn                              = 'dividerand';
    net.divideMode                             = 'sample';
    net.divideParam.trainRatio                 = 0.8;
    net.divideParam.valRatio                   = 0;
    net.divideParam.testRatio                  = 0.2;
    net.inputs{1}.processFcns                  = {};
    net.outputs{net.numLayers}.processFcns     = {};
    net.trainParam.showWindow                  = false;
    if isfield(net.trainParam,'showCommandLine')
        net.trainParam.showCommandLine = false;
    end

    Model.Net = train(net,DataX',DataY);
end
