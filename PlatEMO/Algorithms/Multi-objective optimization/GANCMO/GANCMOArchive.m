classdef GANCMOArchive
% Unified endpoint eligibility, pair fronts and quality-first direction guides.
    methods(Static)
        function [Archive,Scale,Guide] = update(Archive,Evaluated,W,Problem,ReferenceScale,SelectedP1,Guide)
            if isempty(Archive); Archive=emptyArchive(Problem.D,Problem.M); end
            span=decisionSpan(Problem); neighbors=referenceNeighborhoods(W);
            [fx,fy,ix,iy]=evaluatedRows(Evaluated,Problem);
            [~,Scale]=GANCMOReference(zeros(0,Problem.M),W,ReferenceScale);
            Archive.ref=GANCMOReference(Archive.yf,W,Scale);
            previous=[Archive.ref(Guide,:),Archive.xf(Guide,:),Archive.xi(Guide,:)];
            fx=[fx;Archive.xf]; fy=[fy;Archive.yf];
            ix=[ix;Archive.xi]; iy=[iy;Archive.yi];
            [fx,rows]=unique(fx,'rows','stable'); fy=fy(rows,:);
            [ix,rows]=unique(ix,'rows','stable'); iy=iy(rows,:);
            fronts=zeros(0,1);
            if ~isempty(fy); fronts=reshape(NDSort(fy,inf),[],1); end
            validF=fronts==1;
            validI=~dominatedByAny(iy,fy(fronts==1,:),Scale,1e-12);
            % P1 membership affects pair-front filtering, never endpoint eligibility.
            [pfx,~,~,~]=evaluatedRows(SelectedP1,Problem);
            assert(all(ismember(pfx,fx,'rows')),'GANCMO:P1Pool');
            frefs=GANCMOReference(fy,W,Scale); irefs=GANCMOReference(iy,W,Scale);
            % A moving origin can separate an old pair's directions. Keep
            % its endpoints in the common pool even when the pair is removed.
            Archive=subsetArchive(Archive,find(validArchiveRows(Archive,size(W,1),Problem)));
            [foundF,f]=ismember(Archive.xf,fx,'rows');
            [foundI,i]=ismember(Archive.xi,ix,'rows');
            assert(all(foundF & foundI),'GANCMO:FrontPool');
            adjacent=neighbors(sub2ind(size(neighbors),frefs(f),irefs(i)));
            Archive=subsetArchive(Archive,find(validF(f) & validI(i) & adjacent));
            % For each F, keep its nearest legal I as a new candidate pair.
            % New/old F and I share this pool: no sequential endpoint mutations.
            eligibleF=find(validF); count=numel(eligibleF); Candidate=emptyArchive(Problem.D,Problem.M);
            Candidate.ref=zeros(count,1);
            Candidate.xf=zeros(count,Problem.D); Candidate.xi=zeros(count,Problem.D);
            Candidate.yf=zeros(count,Problem.M); Candidate.yi=zeros(count,Problem.M);
            used=false(count,1);
            for k=1:count
                f=eligibleF(k); ref=frefs(f);
                local=find(neighbors(ref,irefs)' & validI);
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
            Archive=appendArchive(Archive,Candidate);
            % Existing rows stay first, preserving the age order for exact ties.
            [~,uniqueRows]=unique([Archive.ref,Archive.xf,Archive.xi],'rows','stable');
            Archive=subsetArchive(Archive,uniqueRows);
            Archive=retainPairFront(Archive,W,Scale,pfx);
            [Archive,Guide]=retainCapacity(Archive,previous,W,Scale,Problem);
        end

        function Data = training(Archive,W,Scale,Problem,rows)
            if nargin<5; rows=(1:numel(Archive.ref))'; end
            legal=all(isfinite(Archive.xf(rows,:)),2) & all(isfinite(Archive.xi(rows,:)),2) & ...
                Archive.ref(rows,:)>=1 & Archive.ref(rows,:)<=size(W,1);
            rows=rows(legal);
            % All stored pairs train; an explicit subset supplies direction guides.
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
            assert(size(unique([Data.ref,Data.xF,Data.xI],'rows'),1)==Data.count, ...
                'GANCMO:DuplicateTrainingPair','Training pairs must be distinct.');
        end
    end
end

function Archive = appendArchive(Archive,Other)
    fields = {'ref','xf','yf','xi','yi'};
    for k = 1:numel(fields)
        Archive.(fields{k}) = [Archive.(fields{k});Other.(fields{k})];
    end
end

function Archive = retainPairFront(Archive,W,Scale,currentF)
% Retain the current-P1 membership, objective-quality and proximity tradeoff.
% Historical close pairs remain alternatives when current P1 lacks a close pair.
    keep = false(numel(Archive.ref),1);
    for ref = reshape(unique(Archive.ref),1,[])
        rows = find(Archive.ref==ref);
        quality = max(normalizeObjectives(Archive.yf(rows,:),Scale)./(double(W(ref,:))+1e-12),[],2);
        objectiveGap = vecnorm(Archive.yf(rows,:)-Archive.yi(rows,:),2,2);
        current = ismember(Archive.xf(rows,:),currentF,'rows');
        front = NDSort([double(~current),quality,objectiveGap],1);
        keep(rows) = reshape(front,[],1)==1;
    end
    Archive = subsetArchive(Archive,find(keep));
end

function [Archive,Guide]=retainCapacity(Archive,previous,W,Scale,Problem)
% F quality first, then objective gap; exact ties keep the previous guide.
    count=numel(Archive.ref); rows=(1:count)'; rank=zeros(count,1);
    gaps2=sum((Archive.yf-Archive.yi).^2,2);
    decisionGap=sqrt(sum(((Archive.xf-Archive.xi)./decisionSpan(Problem)).^2,2));
    incumbent=ismember([Archive.ref,Archive.xf,Archive.xi],previous,'rows');
    for ref=reshape(unique(Archive.ref),1,[])
        local=find(Archive.ref==ref);
        % Several previous guides may migrate into the same current direction.
        quality=max(normalizeObjectives(Archive.yf(local,:),Scale)./(W(ref,:)+1e-12),[],2);
        [~,order]=sortrows([quality,gaps2(local),~incumbent(local),decisionGap(local),local]);
        rank(local(order))=(1:numel(local))';
    end
    if count>500
        [~,order]=sortrows([rank~=1,rank,Archive.ref,rows]);
        rows=sort(order(1:500)); Archive=subsetArchive(Archive,rows);
    end
    Guide=find(rank(rows)==1);
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

function valid = validArchiveRows(Archive,refCount,Problem)
%VALIDARCHIVEROWS Geometry validity; endpoint roles certify opposite real labels.
    lower = double(Problem.lower); upper = double(Problem.upper);
    gap=sqrt(sum(((double(Archive.xf)-double(Archive.xi))./decisionSpan(Problem)).^2,2));
    valid = Archive.ref >= 1 & Archive.ref <= refCount & isfinite(gap) & gap > 0 & ...
        all(isfinite([Archive.xf,Archive.xi,Archive.yf,Archive.yi]),2) & ...
        all(Archive.xf >= lower-1e-12 & Archive.xf <= upper+1e-12,2) & ...
        all(Archive.xi >= lower-1e-12 & Archive.xi <= upper+1e-12,2);
end

function Archive = emptyArchive(D,M)
%EMPTYARCHIVE Only direction and the two evaluated endpoints are stored.
    Archive = struct('ref',zeros(0,1),'xf',zeros(0,D),'yf',zeros(0,M), ...
        'xi',zeros(0,D),'yi',zeros(0,M));
end

function Archive = subsetArchive(Archive,rows)
%SUBSETARCHIVE Preserve row order, including older rows before newer rows.
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
