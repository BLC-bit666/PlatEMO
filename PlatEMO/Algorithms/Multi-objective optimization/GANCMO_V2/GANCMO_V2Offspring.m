function [O1,O2,Generated] = GANCMO_V2Offspring(P,P1,P2,F1,F2,n1,n2,q,seed,generation,type)
% Native GA/DE fills P1 slots around direct CGAN children; P2 uses only GA/DE.
    saved = RandStream.getGlobalStream;
    cleanup = onCleanup(@()RandStream.setGlobalStream(saved));
    m = size(q,1); nVariation = n1-m;
    assert(nVariation>=0 && all(isfinite(q),'all') && ...
        all(q>=P.lower & q<=P.upper,'all'),'GANCMO_V2:BadDirectOffspring');
    XVariation = zeros(0,P.D);
    if nVariation>0
        setStream(seed+1009*generation+2);
        if type==1
            XVariation = gaDecisions(P,P1,F1,nVariation);
        else
            idx = parentIndices(P1,F1,nVariation); X = double(P1.decs);
            XVariation = OperatorDE(P,X(idx(:,1),:),X(idx(:,2),:),X(idx(:,3),:));
        end
    end
    O1 = P1([]);
    if n1>0; O1 = P.Evaluation([XVariation;q]); end
    Generated = O1(nVariation+1:n1);
    O2 = P2([]);
    if n2>0
        setStream(seed+1009*generation+4);
        if type==1
            O2 = P.Evaluation(gaDecisions(P,P2,F2,n2));
        else
            idx = parentIndices(P2,F2,n2);
            O2 = OperatorDE(P,P2(idx(:,1)),P2(idx(:,2)),P2(idx(:,3)));
        end
    end
end

function X = gaDecisions(P,Population,Fitness,count)
% Native OperatorGA returns pairs. Trim an odd surplus before any evaluation.
    parents = TournamentSelection(2,2*ceil(count/2),Fitness);
    decisions = double(Population.decs);
    X = OperatorGA(P,decisions(parents,:));
    X = X(1:count,:);
end

function setStream(seed)
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',mod(seed,2^32-1)));
end

function idx = parentIndices(Pop,Fitness,count)
    n = numel(Pop);
    assert(count==0 || n>=3,'GANCMO_V2:DEPopulationTooSmall','DE needs three parents.');
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
