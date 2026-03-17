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
    Z  = H*Model.W2 + Model.b2;
    P  = 1./(1+exp(-Z));
end
