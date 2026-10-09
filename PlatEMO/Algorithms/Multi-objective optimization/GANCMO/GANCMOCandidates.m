function [Dec,Info] = GANCMOCandidates(Model,Archive,W,P1,P2,Problem,quota,S,Data)
% Uniform queries; active directions use F distance, others use critic rank.
    Dec = zeros(0,Problem.D);
    Info = struct('ref',zeros(0,1),'targetRef',zeros(0,1),'distanceF',zeros(0,1), ...
        'criticScore',zeros(0,1),'hasActivePair',false(0,1), ...
        'available',0,'diverse',0,'queried',0,'queryCounts',zeros(size(W,1),1));
    if quota<=0 || isempty(Model) || ~Model.ready; return; end
    ready = find(Archive.active);
    assert(isequal(Archive.ref(ready),Data.ref));
    directionCount = size(W,1);
    total = max(S.candidateCount,quota);
    Info.queried = total;
    % Randomize the remainder allocation as well as the candidate order.
    refs = repmat(randperm(directionCount)',ceil(total/directionCount),1);
    refs = refs(1:total); refs = refs(randperm(total));
    Info.queryCounts = accumarray(refs,1,[directionCount,1]);
    X = GANCMOModel.sample(Model,[double(W(refs,:)),ones(total,1)],Problem);
    finite = all(isfinite(X),2);
    X = min(Problem.upper,max(Problem.lower,X));
    span = Problem.upper-Problem.lower; span(span<=eps) = 1;
    Z = (double(X)-Problem.lower)./span;
    Current = [P1,P2];
    base = ([double(Current.decs);Archive.xf;Archive.xi]-Problem.lower)./span;
    activeDirection = false(directionCount,1); activeDirection(Data.ref) = true;
    hasPair = activeDirection(refs); valid = finite;
    if ~isempty(ready)
        % Only paired request directions are constrained by the common box.
        endpoints = ([Archive.xf(ready,:);Archive.xi(ready,:)]-Problem.lower)./span;
        lower = min(endpoints,[],1); upper = max(endpoints,[],1);
        valid(hasPair) = valid(hasPair) & all(Z(hasPair,:)>=lower-1e-12 & Z(hasPair,:)<=upper+1e-12,2);
    end
    if ~isempty(base); valid = valid & min(pdist2(Z,base),[],2)>1e-6; end
    queues = cell(directionCount,1); distanceF = nan(total,1); criticScore = nan(total,1);
    targetRef = zeros(total,1);
    unpaired = find(valid & ~hasPair);
    if ~isempty(unpaired)
        criticScore(unpaired) = GANCMOModel.score(Model,X(unpaired,:), ...
            [double(W(refs(unpaired),:)),ones(numel(unpaired),1)],Problem);
        valid(unpaired) = isfinite(criticScore(unpaired));
    end
    % Retain F's target direction for diagnostics only, never for quotas.
    dataRef = Data.ref;
    for j = 1:directionCount
        rows = find(refs==j & valid);
        if isempty(rows); continue; end
        if activeDirection(j)
            F = Data.xF(Data.ref==j,:);
            [distanceF(rows),nearest] = min(pdist2(Z(rows,:),F),[],2);
            targets = dataRef(Data.ref==j); targetRef(rows) = targets(nearest);
            rank = distanceF(rows);
        else
            rank = -criticScore(rows);
        end
        [~,order] = sortrows([rank,rows]); queues{j} = rows(order);
    end
    Info.available = nnz(valid);
    Info.diverse = nnz(~cellfun(@isempty,queues));
    chosen = zeros(quota,1); count = 0;
    % One representative per request direction, with equal access to G slots.
    % If its best point duplicates an earlier choice, try its next candidate.
    for j = randperm(directionCount)
        for k = reshape(queues{j},1,[])
            if count>0 && any(vecnorm(Z(chosen(1:count),:)-Z(k,:),2,2)<=1e-6)
                continue;
            end
            count = count+1; chosen(count) = k; break;
        end
        if count>=quota; break; end
    end
    chosen = chosen(1:count); Dec = double(X(chosen,:));
    Info.ref = refs(chosen); Info.targetRef = targetRef(chosen); Info.distanceF = distanceF(chosen);
    Info.criticScore = criticScore(chosen); Info.hasActivePair = hasPair(chosen);
end
