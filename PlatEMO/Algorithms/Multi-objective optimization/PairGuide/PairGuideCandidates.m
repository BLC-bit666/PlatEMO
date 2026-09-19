function Dec = PairGuideCandidates(Model,Archive,W,Scale,P1,P2,Problem,quota)
% Generate 500 unevaluated queries and retain unique in-box/exploration points.
    values = (1:size(W,1))'; total = 500;
    refs = repelem(values,floor(total/numel(values)),1);
    remainder = mod(total,numel(values));
    if remainder>0; refs = [refs;values(randperm(numel(values),remainder))]; end
    refs = refs(randperm(numel(refs)));
    sides = zeros(total,1); phase = randi(2)-1;
    for ref = 1:size(W,1)
        local = find(refs==ref);
        sides(local) = mod((0:numel(local)-1)'+phase,2);
        phase = mod(phase+numel(local),2);
    end
    X = PairGuideModel.sample(Model,[double(W(refs,:)),sides],Problem);
    span = double(Problem.upper)-double(Problem.lower); span(span<=eps) = 1;
    Z = (double(X)-Problem.lower)./span;
    valid = all(isfinite(Z),2) & all(Z>=0 & Z<=1,2);
    knownY = [Archive.yf;Archive.yi;double(P1.objs)];
    knownY = knownY(all(isfinite(knownY),2),:);
    knownRefs = unique(PairGuideReference(knownY,W,Scale));
    uncovered = valid & ~ismember(refs,knownRefs);
    endpoints = [Archive.xf;Archive.xi];
    lo = (min(double(endpoints),[],1)-Problem.lower)./span;
    hi = (max(double(endpoints),[],1)-Problem.lower)./span;
    inside = valid & all(Z>=lo-1e-12 & Z<=hi+1e-12,2);
    F = (double(Archive.xf)-Problem.lower)./span;
    distanceF = inf(total,1);
    for j = 1:size(F,1)
        distanceF = min(distanceF,sqrt(sum((Z-F(j,:)).^2,2)));
    end
    outside = uncovered & ~inside;
    outsideQuota = floor(quota*.2); insideQuota = quota-outsideQuota;
    Current = [P1,P2]; base = (double(Current.decs)-Problem.lower)./span;
    chosen = zeros(quota,1); count = 0;
    local = find(inside); [~,order] = sortrows([distanceF(local),local]); local = local(order);
    for k = reshape(local,1,[])
        if count>=insideQuota; break; end
        if any(vecnorm(base-Z(k,:),2,2)<=1e-6); continue; end
        count = count+1; chosen(count) = k; base(end+1,:) = Z(k,:); %#ok<AGROW>
    end
    outsideCount = 0;
    if outsideQuota>0
        groups = unique(refs(outside)); groups = groups(randperm(numel(groups)));
        for ref = reshape(groups,1,[])
            for k = reshape(find(outside & refs==ref),1,[])
                if any(vecnorm(base-Z(k,:),2,2)<=1e-6); continue; end
                count = count+1; chosen(count) = k; base(end+1,:) = Z(k,:); %#ok<AGROW>
                outsideCount = outsideCount+1; break;
            end
            if outsideCount>=outsideQuota; break; end
        end
    end
    % Return unused exploration slots to the in-box pool; shortages use DE.
    for k = reshape(local,1,[])
        if count>=quota; break; end
        if any(vecnorm(base-Z(k,:),2,2)<=1e-6); continue; end
        count = count+1; chosen(count) = k; base(end+1,:) = Z(k,:); %#ok<AGROW>
    end
    Dec = double(X(chosen(1:count),:));
end
