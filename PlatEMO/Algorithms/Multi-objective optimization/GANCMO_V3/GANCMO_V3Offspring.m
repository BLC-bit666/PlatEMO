function [O1,O2] = GANCMO_V3Offspring(P,P1,P2,F1,F2,n1,n2,seed,generation)
% Both populations use the unchanged DE parent selection and random streams.
    saved = RandStream.getGlobalStream;
    cleanup = onCleanup(@()RandStream.setGlobalStream(saved));
    nDE = n1;
    XDE = zeros(0,P.D);
    if nDE>0
        setStream(seed+1009*generation+2);
        idx = parentIndices(P1,F1,nDE); X = double(P1.decs);
        XDE = OperatorDE(P,X(idx(:,1),:),X(idx(:,2),:),X(idx(:,3),:));
    end
    O1 = P1([]);
    if n1>0; O1 = P.Evaluation(XDE); end
    O2 = P2([]);
    if n2>0
        setStream(seed+1009*generation+4);
        idx = parentIndices(P2,F2,n2);
        O2 = OperatorDE(P,P2(idx(:,1)),P2(idx(:,2)),P2(idx(:,3)));
    end
end

function setStream(seed)
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',mod(seed,2^32-1)));
end

function idx = parentIndices(Pop,Fitness,count)
    n = numel(Pop);
    assert(count==0 || n>=3,'GANCMO_V3:DEPopulationTooSmall','DE needs three parents.');
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
