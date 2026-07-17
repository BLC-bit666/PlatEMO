function [TrainX,TrainC,QueryRefs] = BuildBoundaryDataset_RC(BMem,W,Problem)
%BUILDBOUNDARYDATASET_RC Build the fixed reference-conditioned data set.
%   Repeated BMem rows are kept exactly as weighted training observations.
%   TrainX contains feasible-anchor decisions, TrainC contains their reference
%   vectors, and QueryRefs lists the populated reference-vector indices.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Handle an empty boundary memory
    if isempty(BMem) || ~isstruct(BMem) || ...
            ~isfield(BMem,'x_b') || isempty(BMem.x_b)
        TrainX = zeros(0,Problem.D);
        TrainC = zeros(0,size(W,2));
        QueryRefs = zeros(0,1);
        return;
    end

    %% Build aligned decision-condition training rows
    valid = all(isfinite(BMem.x_b),2) & all(isfinite(BMem.y_b),2);
    refs = round(double(BMem.ref(valid)));
    validRef = isfinite(refs) & refs >= 1 & refs <= size(W,1);
    TrainX = double(BMem.x_b(valid,:));
    TrainX = TrainX(validRef,:);
    refs = refs(validRef);
    TrainC = double(W(refs,:));
    QueryRefs = unique(refs,'stable');
end
