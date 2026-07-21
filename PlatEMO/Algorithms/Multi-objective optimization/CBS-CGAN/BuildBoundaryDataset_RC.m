function [TrainX,TrainC,QueryRefs] = BuildBoundaryDataset_RC(BMem,W, ...
        Problem,datasetMode)
%BUILDBOUNDARYDATASET_RC Build the fixed reference-conditioned data set.
%   Repeated BMem rows are kept exactly as weighted training observations.
%   In "anchor" mode (default) TrainX contains feasible-anchor decisions.
%   In "landing" mode each pair row yields exactly ONE landing target
%   x_b + lambda*(x_i - x_b) with lambda cycling deterministically over
%   {1.5, 2, 3} by row index, clamped to the box constraints; the one-row-
%   per-pair rule keeps the row count and the training-eligibility gate
%   byte-comparable with the anchor mode. TrainC contains the anchors'
%   reference vectors and QueryRefs lists the reference indices that own
%   at least one training row.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Handle an empty boundary memory
    if nargin < 4 || isempty(datasetMode)
        datasetMode = "anchor";
    end
    datasetMode = lower(strtrim(string(datasetMode)));
    if ~(isscalar(datasetMode) && ...
            ismember(datasetMode,["anchor","landing"]))
        error('CBSRegionGAN:BadDatasetMode', ...
            'datasetMode must be anchor or landing.');
    end
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
    if datasetMode == "landing"
        Xi = double(BMem.x_i(valid,:));
        Xi = Xi(validRef,:);
        keep = all(isfinite(Xi),2);
        TrainX = TrainX(keep,:);
        Xi = Xi(keep,:);
        refs = refs(keep);
        rowCount = size(TrainX,1);
        ladder = [1.5;2;3];
        lambda = ladder(mod((1:rowCount)'-1,3)+1);
        TrainX = TrainX + lambda.*(Xi-TrainX);
        lowerB = repmat(double(Problem.lower),rowCount,1);
        upperB = repmat(double(Problem.upper),rowCount,1);
        TrainX = min(max(TrainX,lowerB),upperB);
    end
    TrainC = double(W(refs,:));
    QueryRefs = unique(refs,'stable');
end
