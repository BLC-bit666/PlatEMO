function Value = NAEMTPredict(Model,Dec)
% Predict the probability-like value assigned to an infeasible solution.

    if isempty(Dec)
        Value = zeros(0,1);
    else
        Value = Model.Net(Dec')';
        Value = min(max(Value(:),0),1);
    end
end
