function [Ref,Scale,Yn] = AssignReferenceVectors_CBS(Y,W,Scale)
%ASSIGNREFERENCEVECTORS_CBS Assign objectives under one reusable scale.
%   REF = ASSIGNREFERENCEVECTORS_CBS(Y,W) min-max normalizes Y and assigns
%   each row to the reference vector with maximum cosine similarity.
%   REF = ASSIGNREFERENCEVECTORS_CBS(Y,W,SCALE) reuses SCALE.minimum and
%   SCALE.span so rows produced in adjacent generations retain the same
%   reference-frame definition.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    Y = double(Y);
    W = double(W);
    objectiveCount = size(Y,2);
    if size(W,2) ~= objectiveCount
        error('CBSRegionGAN:ReferenceDimensionMismatch', ...
            'Objective rows and reference vectors must have equal width.');
    end

    if nargin < 3 || isempty(Scale)
        if isempty(Y)
            minimum = zeros(1,objectiveCount);
            span = ones(1,objectiveCount);
        else
            minimum = min(Y,[],1);
            span = max(Y,[],1)-minimum;
            invalid = ~isfinite(minimum) | ~isfinite(span);
            minimum(invalid) = 0;
            span(invalid) = 1;
            span(span <= eps) = 1;
        end
    else
        if ~isstruct(Scale) || ~isfield(Scale,'minimum') || ...
                ~isfield(Scale,'span')
            error('CBSRegionGAN:InvalidReferenceScale', ...
                'Reference scale must contain minimum and span.');
        end
        minimum = reshape(double(Scale.minimum),1,[]);
        span = reshape(double(Scale.span),1,[]);
        if numel(minimum) ~= objectiveCount || ...
                numel(span) ~= objectiveCount || ...
                any(~isfinite(minimum)) || any(~isfinite(span)) || ...
                any(span <= 0)
            error('CBSRegionGAN:InvalidReferenceScale', ...
                'Reference scale must be finite, positive, and match Y.');
        end
    end
    Scale = struct('minimum',minimum,'span',span);
    Yn = (Y-minimum)./span;
    Yn(~isfinite(Yn)) = 0;

    if isempty(Y)
        Ref = zeros(0,1);
        return;
    elseif isempty(W)
        error('CBSRegionGAN:EmptyReferenceVectors', ...
            'At least one reference vector is required.');
    end
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    rowNorm = sqrt(sum(Yn.^2,2));
    Yu = Yn./max(rowNorm,eps);
    [~,Ref] = max(Yu*Wn',[],2);
    zeroRows = rowNorm <= eps;
    if any(zeroRows)
        distance2 = max(0,sum(Yn(zeroRows,:).^2,2) + ...
            sum(W.^2,2)' - 2*(Yn(zeroRows,:)*W'));
        [~,Ref(zeroRows)] = min(distance2,[],2);
    end
    Ref = reshape(Ref,size(Y,1),1);
end
