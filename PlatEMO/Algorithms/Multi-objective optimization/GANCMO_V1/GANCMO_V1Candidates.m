function Dec = GANCMO_V1Candidates(Model,Archive,W,P1,P2,Problem,quota,S,GuideData,allowUse)
% Direction-local all-pair boxes; cover represented directions first.
    if nargin<10; allowUse=true; end
    Dec = zeros(0,Problem.D);
    if quota<=0 || isempty(Model); return; end
    directionCount = size(W,1);
    total = max(S.candidateCount,quota);
    % Randomize the remainder allocation as well as the candidate order.
    refs = repmat(randperm(directionCount)',ceil(total/directionCount),1);
    refs = refs(1:total); refs = refs(randperm(total));
    % Reserve the same noise and final permutation as the former full query.
    % Skipping network work must not shift this or later query random streams.
    noise = Model.settings.noiseSigma*randn(total,Problem.D);
    directionOrder = randperm(directionCount);
    % Keep subsequent query streams paired while the training window is closed.
    if ~allowUse; return; end
    span = Problem.upper-Problem.lower; span(span<=eps) = 1;
    Current = [P1,P2];
    base = ([double(Current.decs);Archive.xf;Archive.xi]-Problem.lower)./span;
    % Repeated endpoints cannot change the nearest exclusion distance.
    base = unique(base,'rows');
    pairedDirection = false(directionCount,1); pairedDirection(GuideData.ref) = true;
    % Stable partition preserves random order within each group and its RNG.
    directionOrder = [directionOrder(pairedDirection(directionOrder)), ...
        directionOrder(~pairedDirection(directionOrder))];
    if GuideData.count>0
        % Bound all retained F/I endpoints belonging to this direction.
        lower = nan(directionCount,Problem.D); upper = lower;
        for ref = reshape(GuideData.ref,1,[])
            rows = Archive.ref==ref;
            endpoints = (double([Archive.xf(rows,:);Archive.xi(rows,:)])-Problem.lower)./span;
            lower(ref,:) = min(endpoints,[],1);
            upper(ref,:) = max(endpoints,[],1);
        end
    end
    Dec = zeros(quota,Problem.D); selected = zeros(quota,Problem.D);
    count = 0; next = 1; fullBatch = false;
    while count<quota && next<=directionCount
        % Keep a small batch when only a few slots remain, instead of repeated
        % tiny network calls. Unused directions retain their original priority.
        last = min(directionCount,next+max(8,quota-count)-1);
        if fullBatch; last = directionCount; end
        directions = directionOrder(next:last); next = last+1;
        slots = find(ismember(refs,directions));
        if isempty(slots); continue; end
        batchRefs = refs(slots);
        % Tiny matrix products can use a different floating-point reduction.
        % Repeat a reserved row to retain the full-query matrix kernel; padded
        % outputs are discarded and consume neither new noise nor extra slots.
        sampleSlots = [slots;repmat(slots(end),max(0,min(16,total)-numel(slots)),1)];
        X = GANCMO_V1Model.sample(Model,double(W(refs(sampleSlots),:)),Problem,noise(sampleSlots,:));
        X = X(1:numel(slots),:);
        valid = all(isfinite(X),2);
        X = min(Problem.upper,max(Problem.lower,X));
        Z = (double(X)-Problem.lower)./span;
        hasPair = pairedDirection(batchRefs);
        if GuideData.count>0
            localRefs = batchRefs(hasPair);
            valid(hasPair) = valid(hasPair) & all(Z(hasPair,:)>=lower(localRefs,:)-1e-12 & ...
                Z(hasPair,:)<=upper(localRefs,:)+1e-12,2);
        end
        if ~isempty(base)
            % Rejected box/nonfinite rows cannot become valid again.
            rows = find(valid);
            valid(rows) = min(pdist2(Z(rows,:),base),[],2)>1e-6;
        end
        criticScore = nan(numel(slots),1); unpaired = find(valid & ~hasPair);
        if ~isempty(unpaired)
            criticScore(unpaired) = GANCMO_V1Model.score(Model,X(unpaired,:), ...
                double(W(batchRefs(unpaired),:)),Problem);
            valid(unpaired) = isfinite(criticScore(unpaired));
        end
        for j = directions
            rows = find(batchRefs==j & valid);
            if isempty(rows); continue; end
            if pairedDirection(j)
                F = GuideData.xF(GuideData.ref==j,:);
                rank = min(pdist2(Z(rows,:),F),[],2);
            else
                rank = -criticScore(rows);
            end
            % rows preserve ascending slot IDs; stable sort retains that tie order.
            [~,order] = sort(rank);
            best = 0; second = 0;
            for k = reshape(rows(order),1,[])
                if count>0 && any(vecnorm(selected(1:count,:)-Z(k,:),2,2)<=1e-6)
                    continue;
                end
                if best==0
                    best = k;
                    if pairedDirection(j) || fullBatch; break; end
                else
                    second = k; break;
                end
            end
            if best==0; continue; end
            if second>0
                % Near-tied critic scores can change rounding with batch size.
                % Resolve them in the original full-query numerical context.
                tolerance = 8*double(eps(single(max(1,max(abs(criticScore([best second])))))));
                if criticScore(best)-criticScore(second)<=tolerance
                    fullBatch = true; count = 0; next = 1; break;
                end
            end
            count = count+1; Dec(count,:) = double(X(best,:)); selected(count,:) = Z(best,:);
            if count>=quota; break; end
        end
    end
    Dec = Dec(1:count,:);
end
