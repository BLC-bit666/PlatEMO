function [AF,AI,Diag] = UpdateBoundaryArchive_BDG(Problem,AF,AI,...
    Population1,Offspring1,Population2,Offspring2,...
    W,bisectK,perRef,nearTau,xTau)

    maxPair = max(10,ceil(0.2*Problem.N));
    Diag = EmptyArchiveDiag_BDG();

    Fdecs = zeros(0,Problem.D);
    Fobjs = zeros(0,Problem.M);
    Idecs = zeros(0,Problem.D);
    Iobjs = zeros(0,Problem.M);

    % ===== 1) parent-child feasibility flip pairs =====
    [fd,fo,id,io,evalCount] = CollectFlipPairs_BDG(Problem,Population1,Offspring1,bisectK,maxPair);
    Diag.pair_count_flip = Diag.pair_count_flip + size(fd,1);
    Diag.pair_eval_count = Diag.pair_eval_count + evalCount;
    Fdecs = [Fdecs;fd]; Fobjs = [Fobjs;fo]; Idecs = [Idecs;id]; Iobjs = [Iobjs;io];

    [fd,fo,id,io,evalCount] = CollectFlipPairs_BDG(Problem,Population2,Offspring2,bisectK,maxPair);
    Diag.pair_count_flip = Diag.pair_count_flip + size(fd,1);
    Diag.pair_eval_count = Diag.pair_eval_count + evalCount;
    Fdecs = [Fdecs;fd]; Fobjs = [Fobjs;fo]; Idecs = [Idecs;id]; Iobjs = [Iobjs;io];

    % ===== 2) cross-pop heterogeneous nearest pairs =====
    [fd,fo,id,io,evalCount] = CollectCrossPairs_BDG(Problem,...
        [Population1,Offspring1],[Population2,Offspring2],...
        W,bisectK,maxPair,nearTau);
    Diag.pair_count_cross = Diag.pair_count_cross + size(fd,1);
    Diag.pair_eval_count = Diag.pair_eval_count + evalCount;
    Fdecs = [Fdecs;fd]; Fobjs = [Fobjs;fo]; Idecs = [Idecs;id]; Iobjs = [Iobjs;io];

    [fd,fo,id,io,evalCount] = CollectCrossPairs_BDG(Problem,...
        [Population2,Offspring2],[Population1,Offspring1],...
        W,bisectK,maxPair,nearTau);
    Diag.pair_count_cross = Diag.pair_count_cross + size(fd,1);
    Diag.pair_eval_count = Diag.pair_eval_count + evalCount;
    Fdecs = [Fdecs;fd]; Fobjs = [Fobjs;fo]; Idecs = [Idecs;id]; Iobjs = [Iobjs;io];

    [fd,fo,id,io,evalCount] = CollectCrossPairs_BDG(Problem,...
        [Population1,Offspring1],[Population1,Offspring1],...
        W,bisectK,maxPair,nearTau);
    Diag.pair_count_cross = Diag.pair_count_cross + size(fd,1);
    Diag.pair_eval_count = Diag.pair_eval_count + evalCount;
    Fdecs = [Fdecs;fd]; Fobjs = [Fobjs;fo]; Idecs = [Idecs;id]; Iobjs = [Iobjs;io];

    [fd,fo,id,io,evalCount] = CollectCrossPairs_BDG(Problem,...
        [Population2,Offspring2],[Population2,Offspring2],...
        W,bisectK,maxPair,nearTau);
    Diag.pair_count_cross = Diag.pair_count_cross + size(fd,1);
    Diag.pair_eval_count = Diag.pair_eval_count + evalCount;
    Fdecs = [Fdecs;fd]; Fobjs = [Fobjs;fo]; Idecs = [Idecs;id]; Iobjs = [Iobjs;io];

    % ===== 3) archive self-enhancement =====
    kEnh = max(1,ceil(bisectK/2));
    [fd,fo,id,io,evalCount] = SelfEnhancePairs_BDG(Problem,AF,AI,kEnh);
    Diag.pair_count_self = Diag.pair_count_self + size(fd,1);
    Diag.pair_eval_count = Diag.pair_eval_count + evalCount;
    Fdecs = [Fdecs;fd]; Fobjs = [Fobjs;fo]; Idecs = [Idecs;id]; Iobjs = [Iobjs;io];
    Diag.pair_eval_FE = Diag.pair_eval_count;

    if isempty(Fdecs)
        Diag.AF_candidate_count = size(AF.decs,1);
        if Diag.AF_candidate_count > 0
            Diag.AF_retention_ratio = 1;
        end
        return;
    end

    % merge old archive and new pairs, then rebuild
    if ~isempty(AF.decs)
        Fdecs = [AF.decs;Fdecs];
        Fobjs = [AF.objs;Fobjs];
        Idecs = [AI.decs;Idecs];
        Iobjs = [AI.objs;Iobjs];
    end

    Diag.AF_candidate_count = size(Fdecs,1);
    [AF,AI,RebuildDiag] = RebuildArchive_BDG(Problem,Fdecs,Fobjs,Idecs,Iobjs,W,perRef,xTau);
    if Diag.AF_candidate_count > 0
        Diag.AF_retention_ratio = size(AF.decs,1) / Diag.AF_candidate_count;
    end
    Diag = MergeRebuildDiag_BDG(Diag,RebuildDiag);
end

function Diag = EmptyArchiveDiag_BDG()
    Diag = struct( ...
        'pair_count_flip',0, ...
        'pair_count_cross',0, ...
        'pair_count_self',0, ...
        'pair_eval_count',0, ...
        'pair_eval_FE',0, ...
        'AF_candidate_count',0, ...
        'AF_retention_ratio',NaN, ...
        'AF_front1_count',0, ...
        'AF_front1_ratio',NaN, ...
        'AF_kept_front1_ratio',NaN, ...
        'score_gap_obj_med',NaN, ...
        'score_gap_dec_med',NaN, ...
        'score_conv_med',NaN, ...
        'score_rank_med',NaN);
end

function Diag = MergeRebuildDiag_BDG(Diag,RebuildDiag)
    names = fieldnames(RebuildDiag);
    for i = 1 : numel(names)
        Diag.(names{i}) = RebuildDiag.(names{i});
    end
end

function [Fdecs,Fobjs,Idecs,Iobjs,evalCount] = CollectFlipPairs_BDG(Problem,P,O,bisectK,maxPair)
    Fdecs = zeros(0,Problem.D);
    Fobjs = zeros(0,Problem.M);
    Idecs = zeros(0,Problem.D);
    Iobjs = zeros(0,Problem.M);
    evalCount = 0;

    if isempty(P) || isempty(O)
        return;
    end

    flagP = IsFeasibleSet_BDG(P);
    flagO = IsFeasibleSet_BDG(O);
    idx   = find(flagP ~= flagO);
    if isempty(idx)
        return;
    end

    ObjPair = [P(idx).objs;O(idx).objs];
    Y = NormalizeObjMat_BDG(ObjPair);
    n = numel(idx);
    d = sqrt(sum((Y(1:n,:) - Y(n+1:end,:)).^2,2));
    [~,ord] = sort(d,'ascend');
    idx = idx(ord(1:min(maxPair,numel(ord))));

    for k = 1 : numel(idx)
        i = idx(k);
        if flagP(i)
            sF = P(i); sI = O(i);
        else
            sF = O(i); sI = P(i);
        end
        [sF,sI,cost] = RefinePair_BDG(Problem,sF,sI,bisectK);
        evalCount = evalCount + cost;

        Fdecs(end+1,:) = sF.decs; %#ok<AGROW>
        Fobjs(end+1,:) = sF.objs; %#ok<AGROW>
        Idecs(end+1,:) = sI.decs; %#ok<AGROW>
        Iobjs(end+1,:) = sI.objs; %#ok<AGROW>
    end
end

function [Fdecs,Fobjs,Idecs,Iobjs,evalCount] = CollectCrossPairs_BDG(Problem,PA,PB,W,bisectK,maxPair,nearTau)
    Fdecs = zeros(0,Problem.D);
    Fobjs = zeros(0,Problem.M);
    Idecs = zeros(0,Problem.D);
    Iobjs = zeros(0,Problem.M);
    evalCount = 0;

    if isempty(PA) || isempty(PB)
        return;
    end

    flagA = IsFeasibleSet_BDG(PA);
    flagB = IsFeasibleSet_BDG(PB);

    FA = PA(flagA);
    IB = PB(~flagB);

    if isempty(FA) || isempty(IB)
        return;
    end

    FF = FirstFront_BDG(FA.objs);
    FA = FA(FF);
    if isempty(FA)
        return;
    end

    ObjAll = [FA.objs;IB.objs];
    [Y,zmin,zmax] = NormalizeObjMat_BDG(ObjAll);
    nF  = numel(FA);
    YF  = Y(1:nF,:);
    YI  = Y(nF+1:end,:);
    refF = AssignRefFromObj_BDG(FA.objs,W,zmin,zmax);
    refI = AssignRefFromObj_BDG(IB.objs,W,zmin,zmax);

    cnt = 0;
    for j = 1 : size(W,1)
        idf = find(refF == j);
        idi = find(refI == j);
        if isempty(idf) || isempty(idi)
            continue;
        end

        [~,ord] = sort(sum(YF(idf,:),2),'ascend');
        idf = idf(ord(1:min(2,numel(ord))));

        for t = 1 : numel(idf)
            d = pdist2(YF(idf(t),:),YI(idi,:));
            [mind,pos] = min(d);
            if mind <= nearTau
                sF = FA(idf(t));
                sI = IB(idi(pos));
                [sF,sI,cost] = RefinePair_BDG(Problem,sF,sI,bisectK);
                evalCount = evalCount + cost;

                Fdecs(end+1,:) = sF.decs; %#ok<AGROW>
                Fobjs(end+1,:) = sF.objs; %#ok<AGROW>
                Idecs(end+1,:) = sI.decs; %#ok<AGROW>
                Iobjs(end+1,:) = sI.objs; %#ok<AGROW>

                cnt = cnt + 1;
                if cnt >= maxPair
                    return;
                end
            end
        end
    end
end

function [Fdecs,Fobjs,Idecs,Iobjs,evalCount] = SelfEnhancePairs_BDG(Problem,AF,AI,kEnh)
    Fdecs = zeros(0,Problem.D);
    Fobjs = zeros(0,Problem.M);
    Idecs = zeros(0,Problem.D);
    Iobjs = zeros(0,Problem.M);
    evalCount = 0;

    if isempty(AF.decs) || isempty(AI.decs)
        return;
    end

    numE = min([10,size(AF.decs,1),size(AI.decs,1)]);
    [~,ord] = sort(AF.score,'ascend');
    ord = ord(1:numE);

    for k = 1 : numel(ord)
        i  = ord(k);
        sF = Problem.Evaluation(AF.decs(i,:));
        sI = Problem.Evaluation(AI.decs(i,:));
        [sF,sI,cost] = RefinePair_BDG(Problem,sF,sI,kEnh);
        evalCount = evalCount + 2 + cost;

        Fdecs(end+1,:) = sF.decs; %#ok<AGROW>
        Fobjs(end+1,:) = sF.objs; %#ok<AGROW>
        Idecs(end+1,:) = sI.decs; %#ok<AGROW>
        Iobjs(end+1,:) = sI.objs; %#ok<AGROW>
    end
end

function [AF,AI,Diag] = RebuildArchive_BDG(Problem,Fdecs,Fobjs,Idecs,Iobjs,W,perRef,xTau)
    Diag = EmptyRebuildDiag_BDG();
    rank = ParetoRank_BDG(Fobjs);
    front1 = rank == 1;
    Diag.AF_front1_count = sum(front1);
    Diag.AF_front1_ratio = SafeRatio_BDG(Diag.AF_front1_count,numel(front1));
    ref = AssignRefFromObj_BDG(Fobjs,W);
    [Fn,zmin,zmax] = NormalizeObjMat_BDG(Fobjs);
    In = NormalizeObjMat_BDG(Iobjs,zmin,zmax);
    Xn = NormalizeDecMat_BDG(Fdecs,Problem.lower,Problem.upper);
    Xi = NormalizeDecMat_BDG(Idecs,Problem.lower,Problem.upper);

    gapObj = sqrt(sum((Fn - In).^2,2));
    gapDec = sqrt(sum((Xn - Xi).^2,2)) / sqrt(size(Xn,2));
    conv = sqrt(sum(Fn.^2,2));
    rankPenalty = double(rank(:) - 1);

    score = 0.35*NormalizeScoreVector_BDG(gapObj) + ...
        0.35*NormalizeScoreVector_BDG(gapDec) + ...
        0.20*NormalizeScoreVector_BDG(conv) + ...
        0.10*NormalizeScoreVector_BDG(rankPenalty);

    keep = false(size(score));
    for j = 1 : size(W,1)
        idx = find(ref == j);
        if isempty(idx)
            continue;
        end
        keepLocal = GreedyCluster_BDG(Xn(idx,:),score(idx),perRef,xTau);
        keep(idx(keepLocal)) = true;
    end

    if ~any(keep)
        [~,ord] = sort(score,'ascend');
        keep(ord(1:min(perRef,numel(ord)))) = true;
    end

    AF.decs  = Fdecs(keep,:);
    AF.objs  = Fobjs(keep,:);
    AF.ref   = ref(keep);
    AF.score = score(keep);

    AI.decs  = Idecs(keep,:);
    AI.objs  = Iobjs(keep,:);
    AI.ref   = ref(keep);
    AI.score = score(keep);

    Diag.AF_kept_front1_ratio = SafeRatio_BDG(sum(front1(keep)),sum(keep));
    Diag.score_gap_obj_med = MedianFiniteArchive_BDG(gapObj(keep));
    Diag.score_gap_dec_med = MedianFiniteArchive_BDG(gapDec(keep));
    Diag.score_conv_med = MedianFiniteArchive_BDG(conv(keep));
    Diag.score_rank_med = MedianFiniteArchive_BDG(rank(keep));
end

function Diag = EmptyRebuildDiag_BDG()
    Diag = struct( ...
        'AF_front1_count',0, ...
        'AF_front1_ratio',NaN, ...
        'AF_kept_front1_ratio',NaN, ...
        'score_gap_obj_med',NaN, ...
        'score_gap_dec_med',NaN, ...
        'score_conv_med',NaN, ...
        'score_rank_med',NaN);
end

function [sF,sI,evalCount] = RefinePair_BDG(Problem,sF,sI,K)
    evalCount = 0;
    for t = 1 : K
        sM = Problem.Evaluation((sF.decs + sI.decs)/2);
        evalCount = evalCount + 1;
        if IsFeasibleSet_BDG(sM)
            sF = sM;
        else
            sI = sM;
        end
    end
end

function flag = IsFeasibleSet_BDG(P)
    if isempty(P)
        flag = false(0,1);
        return;
    end
    if isempty(P.cons)
        flag = true(numel(P),1);
    else
        flag = all(P.cons <= 0,2);
    end
end

function front = FirstFront_BDG(PopObj)
    N = size(PopObj,1);
    domCount = zeros(N,1);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                domCount(j) = domCount(j) + 1;
            elseif k == -1
                domCount(i) = domCount(i) + 1;
            end
        end
    end
    front = domCount == 0;
end

function rank = ParetoRank_BDG(PopObj)
    N = size(PopObj,1);
    rank = zeros(N,1);
    remain = true(N,1);
    frontNo = 1;
    while any(remain)
        ids = find(remain);
        front = FirstFront_BDG(PopObj(ids,:));
        rank(ids(front)) = frontNo;
        remain(ids(front)) = false;
        frontNo = frontNo + 1;
    end
end

function ref = AssignRefFromObj_BDG(PopObj,W,zmin,zmax)
    if nargin < 3
        Y = NormalizeObjMat_BDG(PopObj);
    else
        Y = NormalizeObjMat_BDG(PopObj,zmin,zmax);
    end
    Wn = W ./ sqrt(sum(W.^2,2) + 1e-12);
    Yn = Y ./ sqrt(sum(Y.^2,2) + 1e-12);
    [~,ref] = max(Yn*Wn',[],2);
end

function [Y,zmin,zmax] = NormalizeObjMat_BDG(Obj,zmin,zmax)
    if isempty(Obj)
        Y = Obj;
        return;
    end
    if nargin < 2
        zmin = min(Obj,[],1);
        zmax = max(Obj,[],1);
    end
    Y = (Obj - zmin) ./ (zmax - zmin + 1e-12);
    Y = min(max(Y,0),1);
end

function Xn = NormalizeDecMat_BDG(X,lower,upper)
    Xn = (X - lower) ./ (upper - lower + 1e-12);
    Xn = min(max(Xn,0),1);
end

function score = NormalizeScoreVector_BDG(x)
    x = double(x(:));
    if isempty(x)
        score = zeros(size(x));
        return;
    end
    xmin = min(x);
    xmax = max(x);
    if xmax <= xmin
        score = zeros(size(x));
    else
        score = (x - xmin) ./ (xmax - xmin + 1e-12);
    end
end

function value = MedianFiniteArchive_BDG(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = median(x);
    end
end

function value = SafeRatio_BDG(num,den)
    if den <= 0
        value = NaN;
    else
        value = double(num) / double(den);
    end
end

function keep = GreedyCluster_BDG(X,score,maxNum,tau)
    keep = false(size(score));
    if isempty(X)
        return;
    end

    [~,ord] = sort(score,'ascend');
    sel = [];

    for ii = 1 : numel(ord)
        i = ord(ii);
        if isempty(sel)
            keep(i) = true;
            sel = i;
        else
            d = sqrt(sum((X(i,:) - X(sel,:)).^2,2)) / sqrt(size(X,2));
            if min(d) > tau
                keep(i) = true;
                sel = [sel;i]; %#ok<AGROW>
            end
        end
        if sum(keep) >= maxNum
            break;
        end
    end

    if ~any(keep)
        keep(ord(1)) = true;
    end
end
