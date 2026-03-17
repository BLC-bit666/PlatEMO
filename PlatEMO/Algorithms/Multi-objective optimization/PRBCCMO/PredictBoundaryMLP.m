function P = PredictBoundaryMLP(Model,X)
% Predict boundary probabilities with the lightweight MLP.

    if isempty(X)
        P = zeros(0,1);
        return;
    end
    if isempty(Model)
        P = 0.5*ones(size(X,1),1);
        return;
    end

    Xn = (X-Model.Mu)./Model.Sigma;
    H  = tanh(Xn*Model.W1 + repmat(Model.b1,size(Xn,1),1));
    Temp = 1;
    if isfield(Model,'Temp') && ~isempty(Model.Temp)
        Temp = max(Model.Temp,1e-6);
    end
    Z  = (H*Model.W2 + Model.b2)./Temp;
    P  = 1./(1+exp(-Z));
    P  = min(max(P,1e-6),1-1e-6);
end
