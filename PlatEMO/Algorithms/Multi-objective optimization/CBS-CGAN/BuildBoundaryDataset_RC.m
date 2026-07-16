function [TrainX,TrainC,QueryRefs] = BuildBoundaryDataset_RC(BMem,W,Problem)
%BUILDBOUNDARYDATASET_RC Build the fixed reference-conditioned data set.
%   Legacy BMem rows are kept exactly as weighted training observations.

    if isempty(BMem) || ~isstruct(BMem) || ...
            ~isfield(BMem,'x_b') || isempty(BMem.x_b)
        TrainX = zeros(0,Problem.D);
        TrainC = zeros(0,size(W,2));
        QueryRefs = zeros(0,1);
        return;
    end

    valid = all(isfinite(BMem.x_b),2) & all(isfinite(BMem.y_b),2);
    refs = round(double(BMem.ref(valid)));
    validRef = isfinite(refs) & refs >= 1 & refs <= size(W,1);
    TrainX = double(BMem.x_b(valid,:));
    TrainX = TrainX(validRef,:);
    refs = refs(validRef);
    TrainC = double(W(refs,:));
    QueryRefs = unique(refs,'stable');
end
