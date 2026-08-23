function [Fitness,I,C] = EADMMIBEACalFitness(PopObj,kappa)
% Additive-epsilon indicator fitness used by the PlatEMO IBEA backbone.

    N      = size(PopObj,1);
    minimum = min(PopObj,[],1);
    range   = max(PopObj,[],1) - minimum;
    PopObj = (PopObj-repmat(minimum,N,1))./repmat(range,N,1);
    I      = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(PopObj(i,:)-PopObj(j,:));
        end
    end
    C = max(abs(I),[],1);
    Fitness = sum(-exp(-I./repmat(C,N,1)./kappa),1) + 1;
end
