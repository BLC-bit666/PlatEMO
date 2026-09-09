function [TrainX,TrainC,QueryRefs] = BuildBoundaryDataset_RC(BMem,W,Problem)
%BUILDBOUNDARYDATASET_RC Build the unique pairflag training data set.
%   Every finite boundary-memory pair contributes a feasible anchor with
%   condition [W(ref),1]. Its finite infeasible partner contributes a row
%   with condition [W(ref),0]. Repeated memory rows remain repeated so the
%   memory weights the training distribution exactly as observed.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if isempty(BMem) || ~isstruct(BMem) || ...
            ~isfield(BMem,'x_b') || isempty(BMem.x_b)
        TrainX = zeros(0,Problem.D);
        TrainC = zeros(0,size(W,2)+1);
        QueryRefs = zeros(0,1);
        return;
    end

    valid = all(isfinite(BMem.x_b),2) & all(isfinite(BMem.y_b),2);
    refs = round(double(BMem.ref(valid)));
    validRef = isfinite(refs) & refs >= 1 & refs <= size(W,1);
    Xb = double(BMem.x_b(valid,:));
    Xb = Xb(validRef,:);
    refs = refs(validRef);

    if isfield(BMem,'x_i') && isequal(size(BMem.x_i),size(BMem.x_b))
        Xi = double(BMem.x_i(valid,:));
    else
        Xi = nan(numel(validRef),size(Xb,2));
    end
    Xi = Xi(validRef,:);
    keepI = all(isfinite(Xi),2);

    TrainX = [Xb;Xi(keepI,:)];
    TrainC = [double(W(refs,:)),ones(numel(refs),1); ...
        double(W(refs(keepI),:)),zeros(sum(keepI),1)];
    QueryRefs = unique(refs,'stable');
end
