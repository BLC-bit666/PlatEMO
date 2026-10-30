classdef GANCMOArchive
% Rebuild nearest legal pairs from endpoint pools; at most five per direction.
    methods(Static)
        function [Archive,Scale,Guide] = update(Archive,Evaluated,W,Problem,ReferenceScale)
            if isempty(Archive); Archive=emptyArchive(Problem.D,Problem.M); end
            span=decisionSpan(Problem); neighbors=referenceNeighborhoods(W);
            [fx,fy,ix,iy]=evaluatedRows(Evaluated,Problem);
            [~,Scale]=GANCMOReference(zeros(0,Problem.M),W,ReferenceScale);
            % Historical endpoints remain available; old pair identities do not.
            [fx,fy]=uniqueEndpoints([fx;Archive.xf],[fy;Archive.yf],Problem);
            [ix,iy]=uniqueEndpoints([ix;Archive.xi],[iy;Archive.yi],Problem);
            fronts=zeros(0,1);
            if ~isempty(fy); fronts=reshape(NDSort(fy,inf),[],1); end
            validF=fronts==1;
            validI=~dominatedByAny(iy,fy(fronts==1,:),Scale,1e-12);
            frefs=GANCMOReference(fy,W,Scale); irefs=GANCMOReference(iy,W,Scale);
            % Each F receives one nearest legal I under the current origin.
            % Sorted endpoints resolve exact ties without old-pair preference.
            eligibleF=find(validF); count=numel(eligibleF); Candidate=emptyArchive(Problem.D,Problem.M);
            Candidate.ref=zeros(count,1);
            Candidate.xf=zeros(count,Problem.D); Candidate.xi=zeros(count,Problem.D);
            Candidate.yf=zeros(count,Problem.M); Candidate.yi=zeros(count,Problem.M);
            used=false(count,1);
            % Legal I membership is shared by all F endpoints on this ray.
            % Rebuild this lookup on every archive update under the current Z.
            localByRef=cell(size(W,1),1);
            for ref=reshape(unique(frefs(eligibleF)),1,[])
                localByRef{ref}=find(neighbors(ref,irefs)' & validI);
            end
            for k=1:count
                f=eligibleF(k); ref=frefs(f);
                local=localByRef{ref};
                decisionGap2=sum(((ix(local,:)-fx(f,:))./span).^2,2);
                local=local(decisionGap2>1e-24);
                if isempty(local); continue; end
                objectiveGap2=sum((iy(local,:)-fy(f,:)).^2,2);
                [~,best]=min(objectiveGap2); i=local(best);
                Candidate.ref(k)=ref; Candidate.xf(k,:)=fx(f,:); Candidate.yf(k,:)=fy(f,:);
                Candidate.xi(k,:)=ix(i,:); Candidate.yi(k,:)=iy(i,:);
                used(k)=true;
            end
            Candidate=subsetArchive(Candidate,find(used));
            [Archive,Guide]=retainCapacity(Candidate,W,Scale,Problem);
        end

        function Data = training(Archive,W,Scale,Problem)
            rows=(1:numel(Archive.ref))';
            legal=all(isfinite(Archive.xf(rows,:)),2) & all(isfinite(Archive.xi(rows,:)),2) & ...
                Archive.ref(rows,:)>=1 & Archive.ref(rows,:)<=size(W,1);
            rows=rows(legal);
            % All valid stored pairs participate in training.
            Data = struct('xF',zeros(0,Problem.D),'xI',zeros(0,Problem.D), ...
                'cF',zeros(0,Problem.M+1),'cI',zeros(0,Problem.M+1), ...
                'ref',Archive.ref(rows,:),'count',numel(rows));
            if ~isempty(rows)
                lower = double(Problem.lower); span = decisionSpan(Problem);
                Data.xF = (Archive.xf(rows,:)-lower)./span;
                Data.xI = (Archive.xi(rows,:)-lower)./span;
                refF = GANCMOReference(Archive.yf(rows,:),W,Scale);
                refI = GANCMOReference(Archive.yi(rows,:),W,Scale);
                Data.cF = [double(W(refF,:)),ones(Data.count,1)];
                Data.cI = [double(W(refI,:)),zeros(Data.count,1)];
            end
        end
    end
end

function [Archive,Guide]=retainCapacity(Archive,W,Scale,Problem)
% Each direction keeps its best five; empty slots are never transferred.
    S=GANCMOSettings; limit=S.pairsPerDirection;
    count=numel(Archive.ref); rows=(1:count)'; rank=zeros(count,1);
    gaps2=sum((Archive.yf-Archive.yi).^2,2);
    decisionGap=sqrt(sum(((Archive.xf-Archive.xi)./decisionSpan(Problem)).^2,2));
    for ref=reshape(unique(Archive.ref),1,[])
        local=find(Archive.ref==ref);
        quality=max(normalizeObjectives(Archive.yf(local,:),Scale)./(W(ref,:)+1e-12),[],2);
        [~,order]=sortrows([quality,gaps2(local),decisionGap(local),local]);
        rank(local(order))=(1:numel(local))';
    end
    rows=rows(rank<=limit);
    Archive=subsetArchive(Archive,rows);
    % Query guides use the shortest normalized decision gap among retained
    % pairs; archive capacity ranking and all-pair training stay independent.
    decisionGap=decisionGap(rows); refs=unique(Archive.ref);
    Guide=zeros(numel(refs),1);
    for k=1:numel(refs)
        local=find(Archive.ref==refs(k));
        [~,nearest]=min(decisionGap(local));
        Guide(k)=local(nearest);
    end
end

function [FeasX,FeasY,InfX,InfY] = evaluatedRows( ...
        Population,Problem)
%EVALUATEDROWS Split all ordinary Union rows by true feasibility.

    FeasX = zeros(0,Problem.D);
    FeasY = zeros(0,Problem.M);
    InfX = zeros(0,Problem.D);
    InfY = zeros(0,Problem.M);
    if isempty(Population)
        return;
    end
    X = double(Population.decs);
    Y = double(Population.objs);
    C = double(Population.cons);
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    valid = all(isfinite(X),2) & all(isfinite(Y),2) & ...
        all(isfinite(C),2) & all(X >= lower-1e-12,2) & ...
        all(X <= upper+1e-12,2);
    X = X(valid,:);
    Y = Y(valid,:);
    C = C(valid,:);
    if isempty(X)
        return;
    end
    [X,rows] = unique(X,'rows','stable');
    Y = Y(rows,:);
    C = C(rows,:);
    feasible = constraintViolation(C) <= 0;
    FeasX = X(feasible,:);
    FeasY = Y(feasible,:);
    InfX = X(~feasible,:);
    InfY = Y(~feasible,:);
end

function [X,Y] = uniqueEndpoints(X,Y,Problem)
% Keep valid evaluated endpoints in a canonical, pair-independent order.
    lower = double(Problem.lower); upper = double(Problem.upper);
    valid = all(isfinite(X),2) & all(isfinite(Y),2) & ...
        all(X >= lower-1e-12 & X <= upper+1e-12,2);
    X=X(valid,:); Y=Y(valid,:);
    [X,rows]=unique(X,'rows'); Y=Y(rows,:);
end

function Archive = emptyArchive(D,M)
%EMPTYARCHIVE Only direction and the two evaluated endpoints are stored.
    Archive = struct('ref',zeros(0,1),'xf',zeros(0,D),'yf',zeros(0,M), ...
        'xi',zeros(0,D),'yi',zeros(0,M));
end

function Archive = subsetArchive(Archive,rows)
%SUBSETARCHIVE Preserve the canonical endpoint order.
    fields = {'ref','xf','yf','xi','yi'};
    for k = 1:numel(fields)
        Archive.(fields{k}) = Archive.(fields{k})(rows,:);
    end
end

function span = decisionSpan(Problem)
%DECISIONSPAN Positive decision-box scale.

    span = double(Problem.upper)-double(Problem.lower);
    span(span <= eps) = 1;
end

function value = constraintViolation(C)
%CONSTRAINTVIOLATION Sum positive constraints rowwise.

    if isempty(C)
        value = zeros(size(C,1),1);
    else
        value = sum(max(0,double(C)),2);
    end
end

function refs = neighborRefs(W,ref,totalCount)
%NEIGHBORREFS Angular neighborhood whose total includes its own direction.

    order = angularOrder(W,ref);
    total = min(size(W,1),max(1,round(double(totalCount))));
    refs = order(1:total);
end

function neighbors = referenceNeighborhoods(W)
%REFERENCENEIGHBORHOODS Cache the same scalar angular ordering for fixed W.
    persistent lastW lastNeighbors
    if isempty(lastW) || ~isequal(W,lastW)
        lastW = W;
        lastNeighbors = false(size(W,1));
        for ref = 1:size(W,1)
            lastNeighbors(ref,neighborRefs(W,ref,5)) = true;
        end
    end
    neighbors = lastNeighbors;
end

function order = angularOrder(W,ref)
%ANGULARORDER Sort unit reference vectors by cosine angle.

    W = double(W);
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    angular = 1-Wn*Wn(ref,:)';
    [~,order] = sortrows([angular,(1:size(W,1))'],[1 2]);
end

function dominated = dominatedByAny(Y,Candidates,Scale,tolerance)
%DOMINATEDBYANY True when any candidate Pareto-dominates each Y row.

    dominated = false(size(Y,1),1);
    if isempty(Y) || isempty(Candidates)
        return;
    end
    Yn = normalizeObjectives(Y,Scale);
    Cn = normalizeObjectives(Candidates,Scale);
    for row = 1 : size(Yn,1)
        dominated(row) = any(all(Cn <= Yn(row,:)+tolerance,2) & ...
            any(Cn < Yn(row,:)-tolerance,2));
    end
end

function Yn = normalizeObjectives(Y,Scale)
%NORMALIZEOBJECTIVES Apply the shared reference-vector objective scale.

    Yn = (double(Y)-reshape(double(Scale.minimum),1,[]))./ ...
        reshape(double(Scale.span),1,[]);
end
