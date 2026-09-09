function Table=boundary_proxy_rows(E,problem,seed,arm)
% True short opposite-side pairs are witnesses, not a global boundary oracle.
% Coordinates for LIRCMOP5..8_BC already occupy the unit decision box.
fe=cellfun(@(p)p.observationFE,E.population);
base=E.population{find(fe==50000,1)}.archive;
rows=struct([]);
for threshold=[.01 .02 .04]
    origin=midpoints(base,threshold);
    for k=find(fe>=50000 & mod(fe,1000)==0)
        P=E.population{k}; current=midpoints(P.archive,threshold);
        if isempty(current); novel=current;
        elseif isempty(origin); novel=current;
        else; novel=current(min(pdist2(current,origin)/sqrt(size(current,2)),[],2)>threshold,:); end
        novel=thin(novel,threshold);
        covered=0;
        if ~isempty(novel)
            covered=nnz(min(pdist2(novel,P.p1Decs)/sqrt(size(novel,2)),[],2)<=threshold);
        end
        row=struct('problem',string(problem),'seed',seed,'arm',string(arm),'FE',fe(k), ...
            'threshold',threshold,'baselineShortMidpoints',size(origin,1), ...
            'currentShortMidpoints',size(current,1),'novelSeparatedWitnesses',size(novel,1), ...
            'witnessesCoveredByP1',covered);
        rows=[rows;row]; %#ok<AGROW>
    end
end
Table=struct2table(rows);
end
function X=midpoints(A,t)
if isempty(A.xf); X=A.xf; return; end
short=vecnorm(A.xf-A.xi,2,2)/sqrt(size(A.xf,2))<=t;
X=unique((A.xf(short,:)+A.xi(short,:))/2,'rows');
end
function Y=thin(X,t)
Y=X([],:);
for k=1:size(X,1)
    if isempty(Y) || all(vecnorm(Y-X(k,:),2,2)/sqrt(size(X,2))>t); Y(end+1,:)=X(k,:); end %#ok<AGROW>
end
end
