function [Ref,Scale,Yn] = GANCMOReference(Y,W,Scale)
%GANCMOREFERENCE Assign objectives under the supplied reference frame.
%   Mainline SCALE.minimum is the historical ideal point; SCALE.span is one.
%   Reuse this snapshot for all endpoints in an archive update/training query.
%   W stays fixed while a new origin can change an old endpoint's direction.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    Y = double(Y);
    W = double(W);
    objectiveCount = size(Y,2);
    if size(W,2) ~= objectiveCount
        error('GANCMO:ReferenceDimensionMismatch', ...
            'Objective rows and reference vectors must have equal width.');
    end

    if ~isstruct(Scale) || ~isfield(Scale,'minimum') || ...
            ~isfield(Scale,'span')
        error('GANCMO:InvalidReferenceScale', ...
            'Reference scale must contain minimum and span.');
    end
    minimum = reshape(double(Scale.minimum),1,[]);
    span = reshape(double(Scale.span),1,[]);
    if numel(minimum) ~= objectiveCount || ...
            numel(span) ~= objectiveCount || ...
            any(~isfinite(minimum)) || any(~isfinite(span)) || ...
            any(span <= 0)
        error('GANCMO:InvalidReferenceScale', ...
            'Reference scale must be finite, positive, and match Y.');
    end
    Scale = struct('minimum',minimum,'span',span);
    Yn = (Y-minimum)./span;
    Yn(~isfinite(Yn)) = 0;

    if isempty(Y)
        Ref = zeros(0,1);
        return;
    elseif isempty(W)
        error('GANCMO:EmptyReferenceVectors', ...
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
