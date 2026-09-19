function [O1,O2,Guided] = PairGuideOffspring(P,P1,P2,F1,F2,n1,n2,q,Archive,seed,generation)
% Preserve the independent GA/DE streams, parent order and evaluation order.
    saved = RandStream.getGlobalStream;
    cleanup = onCleanup(@()RandStream.setGlobalStream(saved));
    ga = round(.25*n1); reserve = round(.20*n1); blocks = [n1-ga-reserve,reserve];
    setStream(seed+1009*generation+1);
    GA = gaChildren(P,P1,F1,ga);
    Index = zeros(n1-ga,3); states = cell(1,2); offsets = [2,5]; first = 1;
    for k = 1:2
        stream = setStream(seed+1009*generation+offsets(k));
        rows = first:first+blocks(k)-1; first = first+blocks(k);
        Index(rows,:) = parentIndices(P1,F1,blocks(k)); states{k} = stream.State;
    end
    X = double(P1.decs); span = P.upper-P.lower; span(span<=eps) = 1;
    U = (X-P.lower)./span; n = size(Index,1); m = size(q,1);
    assert(m<=n && all(isfinite(q),'all') && all(q>=P.lower-1e-12 & q<=P.upper+1e-12,'all'));
    slots = (n-m+1:n)'; selected = zeros(n,1); bias = zeros(n,P.D);
    Q = (q-P.lower)./span; distance = pdist2(Q,U(Index(slots,1),:));
    for j = 1:m
        [~,at] = min(distance(:)); [qi,si] = ind2sub(size(distance),at);
        selected(slots(si)) = qi; distance(qi,:) = Inf; distance(:,si) = Inf;
    end
    used = reshape(find(selected),[],1);
    bias(used,:) = .5*(Q(selected(used),:)-U(Index(used,1),:));
    DE = cell(1,2); first = 1;
    for k = 1:2
        ix = first:first+blocks(k)-1; first = first+blocks(k);
        DE{k} = P1([]);
        if blocks(k)==0; continue; end
        stream = setStream(1); stream.State = states{k};
        proposal = OperatorDE(P,X(Index(ix,1),:), ...
            X(Index(ix,2),:)+2*bias(ix,:).*span,X(Index(ix,3),:));
        local = find(selected(ix)>0);
        if ~isempty(local)
            Y = PairGuideBoundaryStep((proposal(local,:)-P.lower)./span, ...
                Q(selected(ix(local)),:),(Archive.xf-P.lower)./span,(Archive.xi-P.lower)./span);
            proposal(local,:) = P.lower+Y.*span;
        end
        if m>0
            exact = ismember(proposal,q,'rows');
            if any(exact)
                % Recreate the ordinary child only when the direct-q guard needs it.
                after = stream.State; stream.State = states{k};
                ordinary = OperatorDE(P,X(Index(ix,1),:),X(Index(ix,2),:),X(Index(ix,3),:));
                stream.State = after; proposal(exact,:) = ordinary(exact,:);
            end
            assert(~any(ismember(proposal,q,'rows')),'PairGuide:DirectQuery','A query must never be evaluated directly.');
        end
        DE{k} = P.Evaluation(proposal);
    end
    O1 = [GA,DE{:}]; Guided = O1(ga+used);
    ga = round(.25*n2);
    setStream(seed+1009*generation+3); GA = gaChildren(P,P2,F2,ga);
    DE = P2([]);
    if n2-ga>0
        setStream(seed+1009*generation+4); Index = parentIndices(P2,F2,n2-ga);
        DE = OperatorDE(P,P2(Index(:,1)),P2(Index(:,2)),P2(Index(:,3)));
    end
    O2 = [GA,DE];
end

function stream = setStream(seed)
    stream = RandStream('mt19937ar','Seed',mod(seed,2^32-1));
    RandStream.setGlobalStream(stream);
end

function Offspring = gaChildren(P,Pop,Fitness,count)
    Offspring = Pop([]);
    if count==0; return; end
    [~,order] = sortrows(Fitness(:)); rank = zeros(numel(order),1); rank(order) = 1:numel(order);
    parents = TournamentSelection(2,2*count,rank);
    Offspring = OperatorGAhalf(P,Pop(parents));
end

function idx = parentIndices(Pop,Fitness,count)
    n = numel(Pop);
    assert(count==0 || n>=3,'PairGuide:DEPopulationTooSmall','DE needs at least three parents.');
    if count==n; base = (1:n)'; else; base = randperm(n,count)'; end
    [~,order] = sort(Fitness,'ascend'); rank = zeros(n,1); rank(order) = 1:n;
    idx = zeros(count,3); idx(:,1) = base; allRows = 1:n;
    for k = 1:count
        candidates = allRows(allRows~=base(k));
        a = candidates(TournamentSelection(2,1,rank(candidates)));
        candidates = candidates(candidates~=a);
        b = candidates(TournamentSelection(2,1,rank(candidates)));
        idx(k,2:3) = [a,b];
    end
end
