function value = EADMMMOEADAggregation(PopObj,W,Z)
% Modified Tchebycheff function specified in the EADMM paper.

    value = max(abs(PopObj-repmat(Z,size(PopObj,1),1))./W,[],2);
end
