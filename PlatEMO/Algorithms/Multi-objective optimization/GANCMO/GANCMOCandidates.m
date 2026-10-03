function [Dec,Info] = GANCMOCandidates(Model,Archive,W,Scale,P1,P2,Problem,quota,S)
% Feasible-condition queries; direction-balanced selection; no evaluations.
    Dec = zeros(0,Problem.D);
    Info = struct('ref',zeros(0,1),'targetRef',zeros(0,1),'distanceF',zeros(0,1), ...
        'available',0,'diverse',0,'queried',0);
    targetCap = Inf;
    if isfield(S,'candidateTargetCap'); targetCap = S.candidateTargetCap; end
    if quota<=0 || isempty(Model) || ~Model.ready; return; end
    Data = GANCMOArchive.training(Archive,W,Scale,Problem,S.maxTrainingObjectiveGap);
    supported = intersect(unique(Data.ref),unique(Model.lastData.ref));
    if isempty(supported); return; end
    total = max(S.candidateCount,quota);
    Info.queried = total;
    refs = repmat(supported,ceil(total/numel(supported)),1);
    refs = refs(1:total); refs = refs(randperm(total));
    X = GANCMOModel.sample(Model,[double(W(refs,:)),ones(total,1)],Problem);
    finite = all(isfinite(X),2);
    X = min(Problem.upper,max(Problem.lower,X));
    span = Problem.upper-Problem.lower; span(span<=eps) = 1;
    Z = (double(X)-Problem.lower)./span;
    Current = [P1,P2];
    base = ([double(Current.decs);Archive.xf;Archive.xi]-Problem.lower)./span;
    valid = finite;
    if ~isempty(base); valid = valid & min(pdist2(Z,base),[],2)>1e-6; end
    queues = cell(numel(supported),1); distanceF = inf(total,1);
    targetRef = zeros(total,1);
    if isfinite(targetCap)
        ready = find(Archive.active & vecnorm(Archive.yf-Archive.yi,2,2)<=S.maxTrainingObjectiveGap);
        assert(isequal(Archive.ref(ready),Data.ref));
        fixed = struct('minimum',zeros(1,Problem.M),'span',ones(1,Problem.M));
        dataRef = GANCMOReference(Archive.yf(ready,:),W,fixed);
    end
    for j = 1:numel(supported)
        rows = find(refs==supported(j) & valid);
        if isempty(rows); continue; end
        F = Data.xF(Data.ref==supported(j),:);
        [distanceF(rows),nearest] = min(pdist2(Z(rows,:),F),[],2);
        if isfinite(targetCap)
            targets = dataRef(Data.ref==supported(j)); targetRef(rows) = targets(nearest);
        end
        fitted = distanceF(rows)<=S.maxCandidateDistanceF;
        valid(rows(~fitted)) = false; rows = rows(fitted);
        [~,order] = sortrows([distanceF(rows),rows]); queues{j} = rows(order);
    end
    Info.available = nnz(valid); Info.diverse = Info.available;
    if isfinite(targetCap)
        % Multiple query IDs can map to the same physical objective direction.
        % Keep its closest fitted raw proposals, using only already evaluated F.
        keep = false(total,1);
        for ref = reshape(unique(targetRef(valid)),1,[])
            local = find(valid & targetRef==ref);
            [~,rank] = sortrows([distanceF(local),local]);
            keep(local(rank(1:min(targetCap,numel(local))))) = true;
        end
        for j = 1:numel(queues); queues{j} = queues{j}(keep(queues{j})); end
        Info.diverse = nnz(keep);
    end
    order = randperm(numel(supported));
    positions = ones(numel(supported),1); chosen = zeros(quota,1); count = 0;
    while count<quota
        advanced = false;
        for j = order
            rows = queues{j};
            while positions(j)<=numel(rows)
                k = rows(positions(j)); positions(j) = positions(j)+1;
                if count>0 && any(vecnorm(Z(chosen(1:count),:)-Z(k,:),2,2)<=1e-6)
                    continue;
                end
                count = count+1; chosen(count) = k;
                advanced = true; break;
            end
            if count>=quota; break; end
        end
        if ~advanced; break; end
    end
    chosen = chosen(1:count); Dec = double(X(chosen,:));
    Info.ref = refs(chosen); Info.targetRef = targetRef(chosen); Info.distanceF = distanceF(chosen);
end
