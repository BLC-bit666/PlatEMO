function Model = TrainBoundaryMLP(X,Y,Hidden,Epoch,LR)
% Train a lightweight one-hidden-layer MLP for boundary probability.

    Model = [];
    if isempty(X) || size(X,1) < 4
        return;
    end

    Y = double(Y(:) > 0);
    if numel(unique(Y)) < 2
        return;
    end

    [N,D] = size(X);
    Hidden = max(2,round(Hidden));
    Epoch  = max(1,round(Epoch));
    LR     = max(LR,1e-4);
    Lambda = 1e-4;

    Mu    = mean(X,1);
    Sigma = std(X,0,1);
    Sigma(Sigma<1e-12) = 1;
    Xn    = (X-Mu)./Sigma;

    W1 = 0.1*randn(D,Hidden);
    b1 = zeros(1,Hidden);
    W2 = 0.1*randn(Hidden,1);
    b2 = 0;

    for e = 1 : Epoch
        H = tanh(Xn*W1 + repmat(b1,N,1));
        Z = H*W2 + b2;
        P = 1./(1+exp(-Z));
        D2 = P - Y;

        dW2 = (H'*D2)./N + Lambda*W2;
        db2 = sum(D2)./N;
        D1  = (D2*W2').*(1-H.^2);
        dW1 = (Xn'*D1)./N + Lambda*W1;
        db1 = sum(D1,1)./N;

        Step = LR/sqrt(e);
        W1 = W1 - Step*dW1;
        b1 = b1 - Step*db1;
        W2 = W2 - Step*dW2;
        b2 = b2 - Step*db2;
    end

    Model.Mu    = Mu;
    Model.Sigma = Sigma;
    Model.W1    = W1;
    Model.b1    = b1;
    Model.W2    = W2;
    Model.b2    = b2;
end
