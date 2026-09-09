function Results = audit_PairGuide_boundary_validation(folder,coverageTarget,valuableRefs)
%AUDIT_PAIRGUIDE_BOUNDARY_VALIDATION Independent, explicitly charged offline audit.
%   Uses initial real infeasible points, never the algorithm's learned archive.
%   Reports a verified crossing-distance UPPER BOUND, not nearest-boundary distance.
%   coverageTarget and valuableRefs must be predeclared for a confirmatory study.
    if nargin<2; coverageTarget=NaN; end
    if nargin<3; valuableRefs=[]; end
    files=dir(fullfile(folder,'*_seed*_*.mat'));
    Results=cell(numel(files),1);
    for k=1:numel(files)
        R=load(fullfile(files(k).folder,files(k).name),'Record','Audit');
        E=R.Audit.evidence;
        constructor=str2func(char(R.Record.problem));
        P=constructor('N',R.Record.N,'maxFE',1); % Default dimension, no search.
        PairGuideCost_RC('start',P);
        cleanup=onCleanup(@()PairGuideCost_RC('stop'));
        Initial=E.evaluations{1};
        initialLabels=P.CalCon(Initial.decisions);
        assert(isequal(all(initialLabels<=0,2),all(Initial.constraints<=0,2)));
        infeasible=Initial.decisions(any(initialLabels>0,2),:);
        span=double(P.upper)-double(P.lower); span(span<=eps)=1;
        [~,Scale]=AssignReferenceVectors_CBS(Initial.objectives,E.W);
        mask=valuableRefs;
        if isempty(mask); mask=(1:size(E.W,1))'; end
        assert(all(mask>=1 & mask<=size(E.W,1)));
        mask=unique(mask);
        targets=1000:1000:R.Record.finalFE;
        times=cellfun(@(s)s.observationFE,E.population);
        Checkpoints=cell(numel(targets),1);
        coverage=nan(numel(targets),1); actual=nan(numel(targets),1);
        timer=tic;
        for j=1:numel(targets)
            row=find(times<=targets(j),1,'last');
            if isempty(row); continue; end
            S=E.population{row};
            valid=all(S.p1Cons<=0,2);
            X=S.p1Decs(valid,:); Y=S.p1Objs(valid,:);
            upperBound=inf(size(X,1),1);
            if ~isempty(X) && ~isempty(infeasible)
                assert(all(P.CalCon(X)<=0,'all'));
                xn=(X-P.lower)./span;
                in=(infeasible-P.lower)./span;
                dist=max(0,sum(xn.^2,2)+sum(in.^2,2)'-2*xn*in');
                [~,nearest]=min(dist,[],2);
                low=X; high=infeasible(nearest,:);
                for step=1:30
                    mid=(low+high)/2;
                    feasible=all(P.CalCon(mid)<=0,2);
                    low(feasible,:)=mid(feasible,:);
                    high(~feasible,:)=mid(~feasible,:);
                end
                middle=(low+high)/2;
                upperBound=(vecnorm((middle-X)./span,2,2)+ ...
                    0.5*vecnorm((high-low)./span,2,2))/sqrt(P.D);
            end
            refs=AssignReferenceVectors_CBS(Y,E.W,Scale);
            covered=unique(refs(upperBound<=0.003));
            coverage(j)=nnz(ismember(mask,covered))/numel(mask);
            actual(j)=S.observationFE;
            Checkpoints{j}=struct('targetFE',targets(j),'actualFE',actual(j), ...
                'p1Decs',X,'reference',refs,'boundaryDistanceUpperBound',upperBound, ...
                'coverage',coverage(j),'epsilonRMS',0.003);
        end
        first=NaN; sustained=NaN;
        if isfinite(coverageTarget)
            hit=find(coverage>=coverageTarget,1);
            if ~isempty(hit); first=actual(hit); end
            for j=1:max(0,numel(coverage)-4)
                if all(coverage(j:j+4)>=coverageTarget)
                    sustained=actual(j); break;
                end
            end
        end
        valid=isfinite(coverage)&isfinite(actual);
        auc=NaN;
        if nnz(valid)>1 && max(actual(valid))>min(actual(valid))
            auc=trapz(actual(valid),coverage(valid))/(max(actual(valid))-min(actual(valid)));
        end
        Results{k}=struct('problem',R.Record.problem,'seed',R.Record.seed, ...
            'mode',R.Record.mode,'protocol',"initial-opposite-label-segment-upper-bound", ...
            'valuableRefs',mask,'coverageTarget',coverageTarget, ...
            'firstCoverageFE',first,'sustainedFiveCheckpointFE',sustained, ...
            'coverageAUC',auc,'checkpoints',{Checkpoints}, ...
            'oracleCalls',PairGuideCost_RC('snapshot'),'seconds',toc(timer), ...
            'searchFeedback',false);
        clear cleanup;
    end
    save(fullfile(folder,'boundary_audit.mat'),'Results','-v7.3');
end
