function [State,triggered,delta,threshold] = GANCMODetection(State,Data,Previous,referenceCount)
% Formula-2 change from the last successful training; threshold never rises.
% Learn the distance scale from local archive motion, without a D-based clock.
    triggered = false; delta = NaN; threshold = NaN;
    if isempty(State)
        if ~validPoints(Previous); return; end
        State = initialize(Previous,referenceCount);
    end
    threshold = State.theta;
    if ~validPoints(Data) || ~validPoints(Previous); return; end
    current = (Data.xF+Data.xI)/2;
    previous = (Previous.xF+Previous.xI)/2;
    delta = mean(sqrt(min(pdist2(current,previous,'squaredeuclidean'),[],2)));
    if ~isfinite(delta); return; end
    motion = mean(sqrt(min(pdist2(current,State.previous,'squaredeuclidean'),[],2)));
    State.previous = current;
    % Zero scores are observed; zero local motion does not lower the scale.
    if isfinite(motion) && motion>0
        State.samples = State.samples+1;
        resolution = spacing(current);
        if isfinite(resolution)
            candidate = max(sqrt(resolution*motion),sqrt(State.referenceCount)*motion);
            State.smoothed = State.smoothed+(candidate-State.smoothed)/sqrt(State.samples);
            State.theta = min(State.theta,State.smoothed);
        end
    end
    threshold = State.theta;
    triggered = abs(delta)>threshold;
end

function State = initialize(Data,referenceCount)
    points = (Data.xF+Data.xI)/2;
    [radius,count] = spacing(points);
    if ~isfinite(radius)
        [radius,count] = spacing([Data.xF;Data.xI]);
    end
    assert(isfinite(radius) && radius>0 && count>=2,'GANCMO:DegenerateInitialScale', ...
        'Legal initial training data must contain distinct finite endpoints.');
    assert(referenceCount>=numel(unique(Data.ref)) && referenceCount==floor(referenceCount));
    theta = sqrt(referenceCount)*radius;
    State = struct('theta',theta,'samples',count,'previous',points, ...
        'smoothed',theta,'referenceCount',referenceCount);
end

function [radius,count] = spacing(points)
% RMS nearest distinct neighbor distance, not distance from the centroid.
    points = unique(points,'rows'); count = size(points,1); radius = NaN;
    if count<2; return; end
    distances = pdist2(points,points,'squaredeuclidean'); distances(1:count+1:end) = Inf;
    % Retain the original sqrt-then-square rounding of each nearest distance.
    radius = sqrt(mean(sqrt(min(distances,[],2)).^2));
end

function valid = validPoints(Data)
    valid = ~isempty(Data.xF) && ~isempty(Data.xI) && ...
        all(isfinite([Data.xF;Data.xI]),'all');
end
