function Value = PredictISVPS(Model,Dec)
% Predict the constraint satisfaction value of candidate solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if isempty(Dec)
        Value = zeros(0,1);
        return;
    end

    Value = Model.Net(Dec')';
    Value = min(max(Value,0),1);
    Value = Value(:);
end
