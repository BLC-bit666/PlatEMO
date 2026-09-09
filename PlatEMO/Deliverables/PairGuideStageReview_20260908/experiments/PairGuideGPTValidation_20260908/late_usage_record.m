function Row=late_usage_record(E,Record,arm)
% Immediate retention by exact decision-vector matching; not lineage credit.
selected=0;feasible=0;inP1=0;inP2=0;inEither=0;fullFeasibleP1=0;generations=0;
for k=1:numel(E.generations)
 G=E.generations{k};if G.startFE<50000;continue;end
 P=E.population{k+1};X=G.use.childDecs;generations=generations+1;
 fullFeasibleP1=fullFeasibleP1+double(all(P.p1Cons<=0,'all'));
 a=ismember(X,P.p1Decs,'rows');b=ismember(X,P.p2Decs,'rows');
 assert(nnz(a)==nnz(G.use.survivedP1));
 selected=selected+size(X,1);feasible=feasible+nnz(all(G.use.childCons<=0,2));
 inP1=inP1+nnz(a);inP2=inP2+nnz(b);inEither=inEither+nnz(a|b);
end
Row=struct('problem',Record.problem,'seed',Record.seed,'arm',string(arm),'lateSelected',selected, ...
 'lateFeasible',feasible,'lateRetainedP1',inP1,'lateRetainedP2',inP2,'lateRetainedEither',inEither, ...
 'lateRetainedNeither',selected-inEither,'lateGenerations',generations,'lateFullyFeasibleP1Generations',fullFeasibleP1);
end
