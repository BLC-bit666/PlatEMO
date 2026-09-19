classdef PairGuideArchive
% Real feasible/infeasible endpoint pairs, with one active pair per direction.
    methods(Static)
        function [Archive,Scale] = update(Archive,Evaluated,Guided,W,Problem)
            span = decisionSpan(Problem); neighbors = referenceNeighborhoods(W);
            if isempty(Archive); Archive = emptyArchive(Problem.D,Problem.M); end
            Historical = Archive;
            [fx,fy,ix,iy] = evaluatedRows(Evaluated,zeros(0,Problem.D),Problem);
            [gx,gy,gf] = feedbackRows(Guided,Problem);
            if isempty(gx)
                ofx = fx; ofy = fy; oix = ix; oiy = iy;
            else
                [ofx,ofy,oix,oiy] = evaluatedRows(Evaluated,gx,Problem);
            end
            [~,Scale] = PairGuideReference([fy;iy;Archive.yf;Archive.yi],W);
            Archive = refresh(Archive,W,Scale,Problem);
            ordinaryX = [ofx;oix;Historical.xf;Historical.xi];
            ordinaryY = [ofy;oiy;Historical.yf;Historical.yi];
            ordinaryFeasible = [true(size(ofx,1),1);false(size(oix,1),1); ...
                true(size(Historical.xf,1),1);false(size(Historical.xi,1),1)];
            Archive = tightenRows(Archive,gx,gy,gf,W,Scale,span,neighbors);
            Archive = tightenRows(Archive,ordinaryX,ordinaryY,ordinaryFeasible,W,Scale,span,neighbors);
            Archive = refresh(Archive,W,Scale,Problem);
            fx = [fx;Historical.xf]; fy = [fy;Historical.yf];
            ix = [ix;Historical.xi]; iy = [iy;Historical.yi];
            [fx,uniqueRows] = unique(fx,'rows','stable'); fy = fy(uniqueRows,:);
            [ix,uniqueRows] = unique(ix,'rows','stable'); iy = iy(uniqueRows,:);
            fronts = zeros(0,1);
            if ~isempty(fy); fronts = reshape(NDSort(fy,inf),[],1); end
            feasibleEligible = fronts==1;
            infeasibleEligible = ~dominatedByAny(iy,fy(fronts==1,:),Scale,1e-12);
            frefs = PairGuideReference(fy,W,Scale); irefs = PairGuideReference(iy,W,Scale);
            Candidate = emptyArchive(Problem.D,Problem.M);
            Candidate.ref = zeros(size(fx,1),1);
            Candidate.xf = zeros(size(fx)); Candidate.yf = zeros(size(fy));
            Candidate.xi = zeros(size(fx)); Candidate.yi = zeros(size(fy));
            Candidate.gap = zeros(size(fx,1),1); count = 0;
            for ref = reshape(unique(frefs(feasibleEligible)),1,[])
                candidates = find(frefs==ref & feasibleEligible);
                local = find(neighbors(ref,irefs)' & infeasibleEligible);
                for f = reshape(candidates,1,[])
                    gaps = sqrt(sum(((ix(local,:)-fx(f,:))./span).^2,2));
                    admissible = gaps>1e-12; eligible = local(admissible); gaps = gaps(admissible);
                    if isempty(eligible); continue; end
                    [gap,j] = min(gaps); count = count+1;
                    Candidate.ref(count) = ref;
                    Candidate.xf(count,:) = fx(f,:); Candidate.yf(count,:) = fy(f,:);
                    Candidate.xi(count,:) = ix(eligible(j),:); Candidate.yi(count,:) = iy(eligible(j),:);
                    Candidate.gap(count) = gap;
                end
            end
            firstNewId = Archive.nextId;
            Candidate.id = firstNewId+(0:size(fx,1)-1)';
            Candidate.active = false(size(fx,1),1);
            Candidate = subsetArchive(Candidate,(1:count)');
            Archive = appendArchive(Archive,Candidate);
            Archive = refresh(Archive,W,Scale,Problem);
            Archive = subsetArchive(Archive,find(validArchiveRows(Archive,size(W,1),Problem)));
            [~,byId] = sort(Archive.id);
            [~,distinct] = unique([Archive.ref(byId),Archive.xf(byId,:),Archive.xi(byId,:)],'rows','stable');
            Archive = subsetArchive(Archive,sort(byId(distinct)));
            % Tightened lineages can merge. Retain legal historical alternatives
            % for capacity competition; never fill a slot with an invalid pair.
            if numel(Archive.id)<min(500,numel(Historical.id))
                Reserve = refresh(Historical,W,Scale,Problem);
                Reserve = subsetArchive(Reserve,find(validArchiveRows(Reserve,size(W,1),Problem)));
                [~,distinct] = unique([Reserve.ref,Reserve.xf,Reserve.xi],'rows','stable');
                Reserve = subsetArchive(Reserve,distinct);
                duplicate = ismember([Reserve.ref,Reserve.xf,Reserve.xi], ...
                    [Archive.ref,Archive.xf,Archive.xi],'rows');
                Reserve = subsetArchive(Reserve,find(~duplicate));
                Reserve.id = firstNewId+numel(Candidate.id)+(0:numel(Reserve.id)-1)';
                Reserve.active(:) = false; Archive = appendArchive(Archive,Reserve);
            end
            [foundF,whereF] = ismember(Archive.xf,fx,'rows');
            [foundI,whereI] = ismember(Archive.xi,ix,'rows');
            assert(all(foundF & foundI),'PairGuide:FrontPool','An archive endpoint is missing from the evaluated pool.');
            Archive = subsetArchive(Archive,find(feasibleEligible(whereF) & infeasibleEligible(whereI)));
            Archive = retainArchive(Archive,W,Scale);
            new = Archive.id>=firstNewId; added = nnz(new);
            Archive.id(new) = firstNewId+(0:added-1)'; Archive.nextId = firstNewId+added;
        end

        function Data = training(Archive,W,Scale,Problem)
            rows = find(Archive.active & all(isfinite(Archive.xf),2) & ...
                all(isfinite(Archive.xi),2) & Archive.ref>=1 & Archive.ref<=size(W,1));
            Data = struct('xF',zeros(0,Problem.D),'xI',zeros(0,Problem.D), ...
                'cF',zeros(0,Problem.M+1),'cI',zeros(0,Problem.M+1), ...
                'ref',Archive.ref(rows),'count',numel(rows));
            if ~isempty(rows)
                lower = double(Problem.lower); span = decisionSpan(Problem);
                Data.xF = (Archive.xf(rows,:)-lower)./span;
                Data.xI = (Archive.xi(rows,:)-lower)./span;
                refF = PairGuideReference(Archive.yf(rows,:),W,Scale);
                refI = PairGuideReference(Archive.yi(rows,:),W,Scale);
                Data.cF = [double(W(refF,:)),ones(Data.count,1)];
                Data.cI = [double(W(refI,:)),zeros(Data.count,1)];
            end
            assert(numel(unique(Data.ref))==Data.count,'PairGuide:NonUniqueTrainingReference', ...
                'Every active reference must own one training pair.');
        end
    end
end

function Archive = appendArchive(Archive,Other)
    fields = {'id','ref','xf','yf','xi','yi','gap','active'};
    for k = 1:numel(fields)
        Archive.(fields{k}) = [Archive.(fields{k});Other.(fields{k})];
    end
end

function Archive = retainArchive(Archive,W,Scale)
    Archive.active(:) = false; rank = zeros(numel(Archive.id),1);
    for ref = reshape(unique(Archive.ref),1,[])
        rows = find(Archive.ref==ref);
        quality = max(normalizeObjectives(Archive.yf(rows,:),Scale)./(double(W(ref,:))+1e-12),[],2);
        [~,order] = sortrows([quality,Archive.gap(rows),Archive.id(rows)]);
        rank(rows(order)) = (1:numel(rows))'; Archive.active(rows(order(1))) = true;
    end
    if numel(Archive.id)>500
        [~,order] = sortrows([~Archive.active,rank,Archive.ref,Archive.id]);
        Archive = subsetArchive(Archive,sort(order(1:500)));
    end
end

function Archive = tightenRows(Archive,X,Y,feasible,W,Scale,span,neighbors)
% Preserve sequential tightening: later endpoints see earlier improvements.
    if isempty(X) || isempty(Archive.id); return; end
    refs = PairGuideReference(Y,W,Scale); localRows = cell(size(W,1),1);
    for ref = reshape(unique(refs),1,[])
        localRows{ref} = find(neighbors(ref,Archive.ref)');
    end
    for k = 1:size(X,1)
        local = localRows{refs(k)};
        if feasible(k)
            gap = sqrt(sum(((X(k,:)-Archive.xi(local,:))./span).^2,2));
        else
            gap = sqrt(sum(((Archive.xf(local,:)-X(k,:))./span).^2,2));
        end
        accepted = ~(gap<=0 | gap>=Archive.gap(local)-1e-12);
        rows = local(accepted);
        if feasible(k)
            Archive.xf(rows,:) = repmat(X(k,:),numel(rows),1);
            Archive.yf(rows,:) = repmat(Y(k,:),numel(rows),1);
        else
            Archive.xi(rows,:) = repmat(X(k,:),numel(rows),1);
            Archive.yi(rows,:) = repmat(Y(k,:),numel(rows),1);
        end
        Archive.gap(rows) = gap(accepted);
    end
end

function [X,Y,feasible] = feedbackRows(Population,Problem)
    X = zeros(0,Problem.D); Y = zeros(0,Problem.M); feasible = false(0,1);
    if isempty(Population); return; end
    X = double(Population.decs); Y = double(Population.objs); C = double(Population.cons);
    valid = all(isfinite(X),2) & all(isfinite(Y),2) & all(isfinite(C),2) & ...
        all(X>=double(Problem.lower)-1e-12,2) & all(X<=double(Problem.upper)+1e-12,2);
    X = X(valid,:); Y = Y(valid,:); feasible = constraintViolation(C(valid,:))<=0;
end

function Archive = refresh(Archive,W,Scale,Problem)
    if isempty(Archive.id); return; end
    Archive.ref = PairGuideReference(Archive.yf,W,Scale);
    Archive.gap = sqrt(sum(((double(Archive.xf)-double(Archive.xi))./decisionSpan(Problem)).^2,2));
end

function [FeasX,FeasY,InfX,InfY] = evaluatedRows( ...
        Population,Excluded,Problem)
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
    if ~isempty(Excluded)
        valid = valid & ~ismember(X,double(Excluded),'rows');
    end
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
    valid = isfinite(Archive.id) & Archive.id > 0 & Archive.id == fix(Archive.id) & ...
        Archive.ref >= 1 & Archive.ref <= refCount & ...
        isfinite(Archive.gap) & Archive.gap > 0 & ...
        all(isfinite([Archive.xf,Archive.xi,Archive.yf,Archive.yi]),2) & ...
        all(Archive.xf >= lower-1e-12 & Archive.xf <= upper+1e-12,2) & ...
        all(Archive.xi >= lower-1e-12 & Archive.xi <= upper+1e-12,2);
end

function Archive = emptyArchive(D,M)
%EMPTYARCHIVE Six semantic fields plus derived gap/active caches and ID allocator.
    Archive = struct('id',zeros(0,1),'ref',zeros(0,1), ...
        'xf',zeros(0,D),'yf',zeros(0,M),'xi',zeros(0,D),'yi',zeros(0,M), ...
        'gap',zeros(0,1),'active',false(0,1),'nextId',1);
end

function Archive = subsetArchive(Archive,rows)
%SUBSETARCHIVE Preserve stable IDs and their allocator.
    fields = {'id','ref','xf','yf','xi','yi','gap','active'};
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

