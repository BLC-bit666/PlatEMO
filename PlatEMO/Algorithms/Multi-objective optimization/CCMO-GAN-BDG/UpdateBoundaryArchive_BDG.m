function [AF,AI,Diag] = UpdateBoundaryArchive_BDG(Problem,AF,AI,...
    Population1,Offspring1,Population2,Offspring2,...
    W,perRef,nearTau,Options)

    baseMaxPair = max(10,ceil(0.2*Problem.N));
    if nargin < 11 || isempty(Options)
        Options = struct();
    end
    Options = NormalizeArchiveOptions_BDG(Options);
    maxPair = ArchiveSourceMaxPair_BDG(baseMaxPair,Options.sourceCapMode);
    Diag = EmptyArchiveDiag_BDG();
    Diag.score_gap_weight = Options.scoreGapWeight;
    Diag.score_rank_weight = Options.scoreRankWeight;
    Diag.archive_source_cap_mode = Options.sourceCapMode;
    Diag.archive_pareto_filter_mode = Options.paretoFilterMode;
    Diag.archive_pair_direction_mode = Options.pairDirectionMode;
    Diag.archive_pair_ref_mode = Options.pairRefMode;
    PreviousAF = AF;
    PreviousAI = AI;
    hadPreviousArchive = HasBoundaryArchivePairs_BDG(PreviousAF,PreviousAI);

    Fdecs = zeros(0,Problem.D);
    Fobjs = zeros(0,Problem.M);
    Idecs = zeros(0,Problem.D);
    Iobjs = zeros(0,Problem.M);

    if UseUnifiedRefNeighborPairing_BDG(Options)
        [Fdecs,Fobjs,Idecs,Iobjs,SourceDiag] = ...
            CollectUnifiedRefNeighborPairs_BDG(Problem,AF,AI, ...
            Population1,Offspring1,Population2,Offspring2, ...
            W,maxPair,nearTau,Options.pairRefMode);
        Diag.pair_count_flip = SourceDiag.pair_count_flip;
        Diag.pair_count_cross = SourceDiag.pair_count_cross;
        Diag.pair_eval_count = SourceDiag.pair_eval_count;
    else
        % ===== 1) parent-child feasibility flip pairs =====
        [fd,fo,id,io,evalCount] = CollectFlipPairs_BDG(Problem,Population1,Offspring1,maxPair);
        Diag.pair_count_flip = Diag.pair_count_flip + size(fd,1);
        Diag.pair_eval_count = Diag.pair_eval_count + evalCount;
        Fdecs = [Fdecs;fd]; Fobjs = [Fobjs;fo]; Idecs = [Idecs;id]; Iobjs = [Iobjs;io];

        [fd,fo,id,io,evalCount] = CollectFlipPairs_BDG(Problem,Population2,Offspring2,maxPair);
        Diag.pair_count_flip = Diag.pair_count_flip + size(fd,1);
        Diag.pair_eval_count = Diag.pair_eval_count + evalCount;
        Fdecs = [Fdecs;fd]; Fobjs = [Fobjs;fo]; Idecs = [Idecs;id]; Iobjs = [Iobjs;io];

        % ===== 2) cross-pop heterogeneous nearest pairs =====
        CandidatePool = [Population1,Offspring1,Population2,Offspring2];
        [fd,fo,id,io,evalCount] = CollectNearestPairs_BDG(Problem, ...
            CandidatePool,maxPair,nearTau);
        Diag.pair_count_cross = Diag.pair_count_cross + size(fd,1);
        Diag.pair_eval_count = Diag.pair_eval_count + evalCount;
        Fdecs = [Fdecs;fd]; Fobjs = [Fobjs;fo]; Idecs = [Idecs;id]; Iobjs = [Iobjs;io];
    end
    Diag.pair_eval_FE = Diag.pair_eval_count;

    if isempty(Fdecs)
        Diag.AF_candidate_count = size(AF.decs,1);
        if Diag.AF_candidate_count > 0
            Diag.AF_retention_ratio = 1;
        end
        return;
    end

    % merge old archive and new pairs, then rebuild. Unified reference
    % pairing already pools the previous archive before pairing.
    if ~UseUnifiedRefNeighborPairing_BDG(Options) && ~isempty(AF.decs)
        Fdecs = [AF.decs;Fdecs];
        Fobjs = [AF.objs;Fobjs];
        Idecs = [AI.decs;Idecs];
        Iobjs = [AI.objs;Iobjs];
    end

    Diag.AF_candidate_count = size(Fdecs,1);
    [NewAF,NewAI,RebuildDiag] = RebuildArchive_BDG(Problem,Fdecs,Fobjs,Idecs,Iobjs,W,perRef,Options);
    if ShouldRetainPreviousArchive_BDG(NewAF,NewAI,RebuildDiag, ...
            PreviousAF,PreviousAI,hadPreviousArchive)
        AF = PreviousAF;
        AI = PreviousAI;
    else
        AF = NewAF;
        AI = NewAI;
    end
    if Diag.AF_candidate_count > 0
        Diag.AF_retention_ratio = size(AF.decs,1) / Diag.AF_candidate_count;
    end
    Diag = MergeRebuildDiag_BDG(Diag,RebuildDiag);
end

function flag = HasBoundaryArchivePairs_BDG(AF,AI)
    flag = isstruct(AF) && isstruct(AI) && ...
        isfield(AF,'decs') && isfield(AI,'decs') && ...
        ~isempty(AF.decs) && ~isempty(AI.decs) && ...
        size(AF.decs,1) == size(AI.decs,1);
end

function flag = ShouldRetainPreviousArchive_BDG( ...
        AF,AI,Diag,PreviousAF,PreviousAI,hadPreviousArchive)
    flag = false;
    if ~hadPreviousArchive || Diag.archive_pair_direction_candidate_count <= 0
        return;
    end
    newCount = ArchivePairCount_BDG(AF,AI);
    previousCount = ArchivePairCount_BDG(PreviousAF,PreviousAI);
    if newCount == 0 && Diag.archive_pair_direction_keep_count == 0
        flag = true;
        return;
    end
    flag = IsSevereArchiveShrink_BDG(newCount,previousCount);
end

function count = ArchivePairCount_BDG(AF,AI)
    count = 0;
    if ~HasBoundaryArchivePairs_BDG(AF,AI)
        return;
    end
    count = min(size(AF.decs,1),size(AI.decs,1));
end

function flag = IsSevereArchiveShrink_BDG(newCount,previousCount)
    minStablePairs = 6;
    flag = newCount > 0 && previousCount > 0 && ...
        newCount < previousCount && ...
        newCount < minStablePairs;
end

function Diag = EmptyArchiveDiag_BDG()
    Diag = struct( ...
        'pair_count_flip',0, ...
        'pair_count_cross',0, ...
        'pair_eval_count',0, ...
        'pair_eval_FE',0, ...
        'AF_candidate_count',0, ...
        'AF_retention_ratio',NaN, ...
        'AF_front1_count',0, ...
        'AF_front1_ratio',NaN, ...
        'AF_kept_front1_ratio',NaN, ...
        'score_gap_obj_med',NaN, ...
        'score_rank_med',NaN, ...
        'score_gap_weight',0.80, ...
        'score_rank_weight',0.20, ...
        'archive_source_cap_mode',"limited", ...
        'archive_pareto_filter_mode',"none", ...
        'archive_pareto_filter_candidate_count',0, ...
        'archive_pareto_filter_keep_count',0, ...
        'archive_pareto_filter_keep_ratio',NaN, ...
        'archive_pair_direction_mode',"none", ...
        'archive_pair_direction_candidate_count',0, ...
        'archive_pair_direction_keep_count',0, ...
        'archive_pair_direction_keep_ratio',NaN, ...
        'archive_pair_ref_mode',"none", ...
        'archive_pair_ref_candidate_count',0, ...
        'archive_pair_ref_keep_count',0, ...
        'archive_pair_ref_keep_ratio',NaN);
end

function Diag = MergeRebuildDiag_BDG(Diag,RebuildDiag)
    names = fieldnames(RebuildDiag);
    for i = 1 : numel(names)
        Diag.(names{i}) = RebuildDiag.(names{i});
    end
end

function [Fdecs,Fobjs,Idecs,Iobjs,evalCount] = CollectFlipPairs_BDG(Problem,P,O,maxPair)
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
        Fdecs(end+1,:) = sF.decs; %#ok<AGROW>
        Fobjs(end+1,:) = sF.objs; %#ok<AGROW>
        Idecs(end+1,:) = sI.decs; %#ok<AGROW>
        Iobjs(end+1,:) = sI.objs; %#ok<AGROW>
    end
end

function [Fdecs,Fobjs,Idecs,Iobjs,evalCount] = CollectNearestPairs_BDG(Problem,P,maxPair,nearTau)
    Fdecs = zeros(0,Problem.D);
    Fobjs = zeros(0,Problem.M);
    Idecs = zeros(0,Problem.D);
    Iobjs = zeros(0,Problem.M);
    evalCount = 0;

    if isempty(P)
        return;
    end

    flag = IsFeasibleSet_BDG(P);
    FA = P(flag);
    IB = P(~flag);

    if isempty(FA) || isempty(IB)
        return;
    end

    FF = FirstFront_BDG(FA.objs);
    FA = FA(FF);
    if isempty(FA)
        return;
    end

    ObjAll = [FA.objs;IB.objs];
    Y = NormalizeObjMat_BDG(ObjAll);
    nF  = numel(FA);
    YF  = Y(1:nF,:);
    YI  = Y(nF+1:end,:);

    [~,ord] = sort(sum(YF,2),'ascend');
    ord = ord(1:min(maxPair,numel(ord)));
    cnt = 0;
    for t = 1 : numel(ord)
        d = sqrt(sum((YI - YF(ord(t),:)).^2,2));
        [mind,pos] = min(d);
        if mind <= nearTau
            sF = FA(ord(t));
            sI = IB(pos);

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

function flag = UseUnifiedRefNeighborPairing_BDG(Options)
    flag = isstruct(Options) && isfield(Options,'pairRefMode') && ...
        NormalizeArchivePairRefMode_BDG(Options.pairRefMode) ~= "none";
end

function [Fdecs,Fobjs,Idecs,Iobjs,Diag] = ...
        CollectUnifiedRefNeighborPairs_BDG(Problem,AF,AI, ...
        Population1,Offspring1,Population2,Offspring2, ...
        W,maxPair,nearTau,pairRefMode)
    Diag = struct('pair_count_flip',0,'pair_count_cross',0, ...
        'pair_eval_count',0);
    FPoolDecs = zeros(0,Problem.D);
    FPoolObjs = zeros(0,Problem.M);
    IPoolDecs = zeros(0,Problem.D);
    IPoolObjs = zeros(0,Problem.M);

    [fd,fo,id,io,cnt] = CollectFlipCandidatePools_BDG( ...
        Problem,Population1,Offspring1,maxPair);
    Diag.pair_count_flip = Diag.pair_count_flip + cnt;
    FPoolDecs = [FPoolDecs;fd]; FPoolObjs = [FPoolObjs;fo];
    IPoolDecs = [IPoolDecs;id]; IPoolObjs = [IPoolObjs;io];

    [fd,fo,id,io,cnt] = CollectFlipCandidatePools_BDG( ...
        Problem,Population2,Offspring2,maxPair);
    Diag.pair_count_flip = Diag.pair_count_flip + cnt;
    FPoolDecs = [FPoolDecs;fd]; FPoolObjs = [FPoolObjs;fo];
    IPoolDecs = [IPoolDecs;id]; IPoolObjs = [IPoolObjs;io];

    CandidatePool = [Population1,Offspring1,Population2,Offspring2];
    [fd,fo,id,io] = CollectCrossCandidatePools_BDG( ...
        Problem,CandidatePool,maxPair);
    FPoolDecs = [FPoolDecs;fd]; FPoolObjs = [FPoolObjs;fo];
    IPoolDecs = [IPoolDecs;id]; IPoolObjs = [IPoolObjs;io];

    if HasBoundaryArchivePairs_BDG(AF,AI)
        FPoolDecs = [FPoolDecs;AF.decs];
        FPoolObjs = [FPoolObjs;AF.objs];
        IPoolDecs = [IPoolDecs;AI.decs];
        IPoolObjs = [IPoolObjs;AI.objs];
    end

    [FPoolDecs,FPoolObjs] = UniqueArchiveCandidates_BDG( ...
        FPoolDecs,FPoolObjs,Problem);
    [IPoolDecs,IPoolObjs] = UniqueArchiveCandidates_BDG( ...
        IPoolDecs,IPoolObjs,Problem);

    [Fdecs,Fobjs,Idecs,Iobjs] = PairUnifiedRefNeighborCandidates_BDG( ...
        Problem,FPoolDecs,FPoolObjs,IPoolDecs,IPoolObjs, ...
        W,nearTau,maxPair,pairRefMode);
    Diag.pair_count_cross = size(Fdecs,1);
end

function [Fdecs,Fobjs,Idecs,Iobjs,count] = ...
        CollectFlipCandidatePools_BDG(Problem,P,O,maxPair)
    Fdecs = zeros(0,Problem.D);
    Fobjs = zeros(0,Problem.M);
    Idecs = zeros(0,Problem.D);
    Iobjs = zeros(0,Problem.M);
    count = 0;
    if isempty(P) || isempty(O)
        return;
    end
    flagP = IsFeasibleSet_BDG(P);
    flagO = IsFeasibleSet_BDG(O);
    idx = find(flagP ~= flagO);
    if isempty(idx)
        return;
    end
    ObjPair = [P(idx).objs;O(idx).objs];
    Y = NormalizeObjMat_BDG(ObjPair);
    n = numel(idx);
    d = sqrt(sum((Y(1:n,:) - Y(n+1:end,:)).^2,2));
    [~,ord] = sort(d,'ascend');
    idx = idx(ord(1:min(maxPair,numel(ord))));
    count = numel(idx);
    for k = 1 : numel(idx)
        i = idx(k);
        if flagP(i)
            sF = P(i); sI = O(i);
        else
            sF = O(i); sI = P(i);
        end
        Fdecs(end+1,:) = sF.decs; %#ok<AGROW>
        Fobjs(end+1,:) = sF.objs; %#ok<AGROW>
        Idecs(end+1,:) = sI.decs; %#ok<AGROW>
        Iobjs(end+1,:) = sI.objs; %#ok<AGROW>
    end
end

function [Fdecs,Fobjs,Idecs,Iobjs] = ...
        CollectCrossCandidatePools_BDG(Problem,P,maxPair)
    Fdecs = zeros(0,Problem.D);
    Fobjs = zeros(0,Problem.M);
    Idecs = zeros(0,Problem.D);
    Iobjs = zeros(0,Problem.M);
    if isempty(P)
        return;
    end
    flag = IsFeasibleSet_BDG(P);
    FA = P(flag);
    IB = P(~flag);
    if isempty(FA) || isempty(IB)
        return;
    end
    FF = FirstFront_BDG(FA.objs);
    FA = FA(FF);
    if isempty(FA)
        return;
    end
    YF = NormalizeObjMat_BDG(FA.objs);
    [~,ord] = sort(sum(YF,2),'ascend');
    ord = ord(1:min(maxPair,numel(ord)));
    FA = FA(ord);
    Fdecs = FA.decs;
    Fobjs = FA.objs;
    Idecs = IB.decs;
    Iobjs = IB.objs;
end

function [Decs,Objs] = UniqueArchiveCandidates_BDG(Decs,Objs,Problem)
    if isempty(Decs)
        Decs = zeros(0,Problem.D);
        Objs = zeros(0,Problem.M);
        return;
    end
    [~,ia] = unique([Decs,Objs],'rows','stable');
    Decs = Decs(ia,:);
    Objs = Objs(ia,:);
end

function [Fdecs,Fobjs,Idecs,Iobjs] = PairUnifiedRefNeighborCandidates_BDG( ...
        Problem,FPoolDecs,FPoolObjs,IPoolDecs,IPoolObjs, ...
        W,nearTau,maxPair,pairRefMode)
    Fdecs = zeros(0,Problem.D);
    Fobjs = zeros(0,Problem.M);
    Idecs = zeros(0,Problem.D);
    Iobjs = zeros(0,Problem.M);
    if isempty(FPoolDecs) || isempty(IPoolDecs) || isempty(W)
        return;
    end

    ObjAll = [FPoolObjs;IPoolObjs];
    [~,zmin,zmax] = NormalizeObjMat_BDG(ObjAll);
    Fn = NormalizeObjMat_BDG(FPoolObjs,zmin,zmax);
    In = NormalizeObjMat_BDG(IPoolObjs,zmin,zmax);
    refF = AssignRefFromObj_BDG(FPoolObjs,W,zmin,zmax);
    refI = AssignRefFromObj_BDG(IPoolObjs,W,zmin,zmax);
    neighborCount = ArchivePairRefNeighborCount_BDG(pairRefMode);
    pairDist = zeros(0,1);

    for i = 1 : size(FPoolDecs,1)
        local = RefNeighborAllowed_BDG(refF(i),refI,W,neighborCount);
        if ~any(local)
            continue;
        end
        d = sqrt(sum((In - Fn(i,:)).^2,2));
        local = local & d <= nearTau;
        if ~any(local)
            continue;
        end
        idx = find(local);
        [bestDist,pos] = min(d(idx));
        j = idx(pos);
        Fdecs(end+1,:) = FPoolDecs(i,:); %#ok<AGROW>
        Fobjs(end+1,:) = FPoolObjs(i,:); %#ok<AGROW>
        Idecs(end+1,:) = IPoolDecs(j,:); %#ok<AGROW>
        Iobjs(end+1,:) = IPoolObjs(j,:); %#ok<AGROW>
        pairDist(end+1,1) = bestDist; %#ok<AGROW>
    end

    if isfinite(maxPair) && size(Fdecs,1) > maxPair
        [~,ord] = sort(pairDist,'ascend');
        ord = ord(1:maxPair);
        Fdecs = Fdecs(ord,:);
        Fobjs = Fobjs(ord,:);
        Idecs = Idecs(ord,:);
        Iobjs = Iobjs(ord,:);
    end
end

function [AF,AI,Diag] = RebuildArchive_BDG(Problem,Fdecs,Fobjs,Idecs,Iobjs,W,perRef,Options)
    Diag = EmptyRebuildDiag_BDG();
    if nargin < 8 || isempty(Options)
        Options = struct();
    end
    Options = NormalizeArchiveOptions_BDG(Options);
    Diag.archive_source_cap_mode = Options.sourceCapMode;
    rank = ParetoRank_BDG(Fobjs);
    front1 = rank == 1;
    Diag.AF_front1_count = sum(front1);
    Diag.AF_front1_ratio = SafeRatio_BDG(Diag.AF_front1_count,numel(front1));
    ref = AssignRefFromObj_BDG(Fobjs,W);
    paretoKeep = ArchiveParetoFilterMask_BDG(Fobjs,ref, ...
        Options.paretoFilterMode);
    Diag.archive_pareto_filter_mode = Options.paretoFilterMode;
    Diag.archive_pareto_filter_candidate_count = numel(paretoKeep);
    Diag.archive_pareto_filter_keep_count = sum(paretoKeep);
    Diag.archive_pareto_filter_keep_ratio = SafeRatio_BDG( ...
        Diag.archive_pareto_filter_keep_count, ...
        Diag.archive_pareto_filter_candidate_count);
    if ~any(paretoKeep)
        paretoKeep = true(size(paretoKeep));
    end
    Fdecs = Fdecs(paretoKeep,:);
    Fobjs = Fobjs(paretoKeep,:);
    Idecs = Idecs(paretoKeep,:);
    Iobjs = Iobjs(paretoKeep,:);
    rank  = rank(paretoKeep);
    front1 = front1(paretoKeep);
    ref = ref(paretoKeep);

    directionKeep = ArchivePairDirectionMask_BDG( ...
        Fobjs,Iobjs,Options.pairDirectionMode);
    Diag.archive_pair_direction_mode = Options.pairDirectionMode;
    Diag.archive_pair_direction_candidate_count = numel(directionKeep);
    Diag.archive_pair_direction_keep_count = sum(directionKeep);
    Diag.archive_pair_direction_keep_ratio = SafeRatio_BDG( ...
        Diag.archive_pair_direction_keep_count, ...
        Diag.archive_pair_direction_candidate_count);
    Fdecs = Fdecs(directionKeep,:);
    Fobjs = Fobjs(directionKeep,:);
    Idecs = Idecs(directionKeep,:);
    Iobjs = Iobjs(directionKeep,:);
    rank  = rank(directionKeep);
    front1 = front1(directionKeep);
    ref = ref(directionKeep);

    pairRefKeep = ArchivePairRefMask_BDG(Fobjs,Iobjs,W,Options.pairRefMode);
    Diag.archive_pair_ref_mode = Options.pairRefMode;
    Diag.archive_pair_ref_candidate_count = numel(pairRefKeep);
    Diag.archive_pair_ref_keep_count = sum(pairRefKeep);
    Diag.archive_pair_ref_keep_ratio = SafeRatio_BDG( ...
        Diag.archive_pair_ref_keep_count, ...
        Diag.archive_pair_ref_candidate_count);
    Fdecs = Fdecs(pairRefKeep,:);
    Fobjs = Fobjs(pairRefKeep,:);
    Idecs = Idecs(pairRefKeep,:);
    Iobjs = Iobjs(pairRefKeep,:);
    rank  = rank(pairRefKeep);
    front1 = front1(pairRefKeep);
    ref = ref(pairRefKeep);
    if isempty(Fdecs)
        AF = InitArchiveLike_BDG(Problem);
        AI = InitArchiveLike_BDG(Problem);
        Diag.AF_kept_front1_ratio = NaN;
        return;
    end

    [Fn,zmin,zmax] = NormalizeObjMat_BDG(Fobjs);
    In = NormalizeObjMat_BDG(Iobjs,zmin,zmax);

    gapObj = sqrt(sum((Fn - In).^2,2));
    rankPenalty = double(rank(:) - 1);

    score = Options.scoreGapWeight*NormalizeScoreVector_BDG(gapObj) + ...
        Options.scoreRankWeight*NormalizeScoreVector_BDG(rankPenalty);

    keep = false(size(score));
    refs = unique(ref(isfinite(ref) & ref > 0),'stable');
    for j = 1 : numel(refs)
        idx = find(ref == refs(j));
        if isempty(idx)
            continue;
        end
        [~,ord] = sort(score(idx),'ascend');
        idx = idx(ord(1:min(perRef,numel(ord))));
        keep(idx) = true;
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
    Diag.score_rank_med = MedianFiniteArchive_BDG(rank(keep));
    Diag.score_gap_weight = Options.scoreGapWeight;
    Diag.score_rank_weight = Options.scoreRankWeight;
end

function Diag = EmptyRebuildDiag_BDG()
    Diag = struct( ...
        'AF_front1_count',0, ...
        'AF_front1_ratio',NaN, ...
        'AF_kept_front1_ratio',NaN, ...
        'score_gap_obj_med',NaN, ...
        'score_rank_med',NaN, ...
        'score_gap_weight',0.80, ...
        'score_rank_weight',0.20, ...
        'archive_source_cap_mode',"limited", ...
        'archive_pareto_filter_mode',"none", ...
        'archive_pareto_filter_candidate_count',0, ...
        'archive_pareto_filter_keep_count',0, ...
        'archive_pareto_filter_keep_ratio',NaN, ...
        'archive_pair_direction_mode',"none", ...
        'archive_pair_direction_candidate_count',0, ...
        'archive_pair_direction_keep_count',0, ...
        'archive_pair_direction_keep_ratio',NaN, ...
        'archive_pair_ref_mode',"none", ...
        'archive_pair_ref_candidate_count',0, ...
        'archive_pair_ref_keep_count',0, ...
        'archive_pair_ref_keep_ratio',NaN);
end

function Options = NormalizeArchiveOptions_BDG(Options)
    if ~isstruct(Options)
        Options = struct();
    end
    if ~isfield(Options,'scoreGapWeight') || isempty(Options.scoreGapWeight)
        Options.scoreGapWeight = 1.00;
    end
    if ~isfield(Options,'scoreRankWeight') || isempty(Options.scoreRankWeight)
        Options.scoreRankWeight = 0.00;
    end
    if ~isfield(Options,'sourceCapMode') || isempty(Options.sourceCapMode)
        Options.sourceCapMode = "limited";
    end
    if ~isfield(Options,'paretoFilterMode') || isempty(Options.paretoFilterMode)
        Options.paretoFilterMode = "global_af_nd";
    end
    if ~isfield(Options,'pairDirectionMode') || isempty(Options.pairDirectionMode)
        Options.pairDirectionMode = "ai_dominates_af";
    end
    if ~isfield(Options,'pairRefMode') || isempty(Options.pairRefMode)
        Options.pairRefMode = "none";
    end
    gapWeight = double(Options.scoreGapWeight);
    rankWeight = double(Options.scoreRankWeight);
    assert(isfinite(gapWeight) && isfinite(rankWeight) && ...
        gapWeight >= 0 && rankWeight >= 0, ...
        'UpdateBoundaryArchive_BDG:BadScoreWeights', ...
        'Archive score weights must be finite nonnegative scalars.');
    total = gapWeight + rankWeight;
    assert(total > 0, ...
        'UpdateBoundaryArchive_BDG:BadScoreWeights', ...
        'At least one archive score weight must be positive.');
    Options.scoreGapWeight = gapWeight / total;
    Options.scoreRankWeight = rankWeight / total;
    Options.sourceCapMode = NormalizeArchiveSourceCapMode_BDG( ...
        Options.sourceCapMode);
    Options.paretoFilterMode = NormalizeArchiveParetoFilterMode_BDG( ...
        Options.paretoFilterMode);
    Options.pairDirectionMode = NormalizeArchivePairDirectionMode_BDG( ...
        Options.pairDirectionMode);
    Options.pairRefMode = NormalizeArchivePairRefMode_BDG( ...
        Options.pairRefMode);
end

function maxPair = ArchiveSourceMaxPair_BDG(baseMaxPair,mode)
    mode = NormalizeArchiveSourceCapMode_BDG(mode);
    switch mode
        case "limited"
            maxPair = baseMaxPair;
        case "none"
            maxPair = Inf;
    end
end

function mode = NormalizeArchiveSourceCapMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["limited","none"];
    assert(ismember(mode,valid), ...
        'UpdateBoundaryArchive_BDG:BadSourceCapMode', ...
        'Archive sourceCapMode must be one of: %s.', ...
        strjoin(valid,", "));
end

function mode = NormalizeArchiveParetoFilterMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["global_af_nd","ref_af_nd"];
    assert(ismember(mode,valid), ...
        'UpdateBoundaryArchive_BDG:BadParetoFilterMode', ...
        'Archive paretoFilterMode must be one of: %s.', ...
        strjoin(valid,", "));
end

function keep = ArchiveParetoFilterMask_BDG(PopObj,ref,mode)
    mode = NormalizeArchiveParetoFilterMode_BDG(mode);
    n = size(PopObj,1);
    keep = true(n,1);
    switch mode
        case "global_af_nd"
            keep = FirstFront_BDG(PopObj);
        case "ref_af_nd"
            keep = false(n,1);
            ref = double(ref(:));
            refs = unique(ref(isfinite(ref) & ref > 0),'stable');
            for i = 1 : numel(refs)
                idx = find(ref == refs(i));
                if isempty(idx)
                    continue;
                end
                localFront = FirstFront_BDG(PopObj(idx,:));
                keep(idx(localFront)) = true;
            end
    end
end

function mode = NormalizeArchivePairDirectionMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["ai_dominates_af","af_not_dominates_ai"];
    assert(ismember(mode,valid), ...
        'UpdateBoundaryArchive_BDG:BadPairDirectionMode', ...
        'Archive pairDirectionMode must be one of: %s.', ...
        strjoin(valid,", "));
end

function mode = NormalizeArchivePairRefMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["none","neighbor4"];
    assert(ismember(mode,valid), ...
        'UpdateBoundaryArchive_BDG:BadPairRefMode', ...
        'Archive pairRefMode must be one of: %s.', ...
        strjoin(valid,", "));
end

function keep = ArchivePairDirectionMask_BDG(Fobjs,Iobjs,mode)
    mode = NormalizeArchivePairDirectionMode_BDG(mode);
    n = size(Fobjs,1);
    keep = true(n,1);
    if isempty(Fobjs) || isempty(Iobjs)
        keep = false(n,1);
        return;
    end
    epsTol = 1e-12;
    switch mode
        case "ai_dominates_af"
            keep = all(Iobjs <= Fobjs + epsTol,2) & ...
                any(Iobjs < Fobjs - epsTol,2);
        case "af_not_dominates_ai"
            afDominatesAI = all(Fobjs <= Iobjs + epsTol,2) & ...
                any(Fobjs < Iobjs - epsTol,2);
            keep = ~afDominatesAI;
    end
end

function keep = ArchivePairRefMask_BDG(Fobjs,Iobjs,W,mode)
    mode = NormalizeArchivePairRefMode_BDG(mode);
    n = size(Fobjs,1);
    keep = true(n,1);
    if mode == "none"
        return;
    end
    if isempty(Fobjs) || isempty(Iobjs) || isempty(W)
        keep = false(n,1);
        return;
    end
    ObjAll = [Fobjs;Iobjs];
    [~,zmin,zmax] = NormalizeObjMat_BDG(ObjAll);
    refF = AssignRefFromObj_BDG(Fobjs,W,zmin,zmax);
    refI = AssignRefFromObj_BDG(Iobjs,W,zmin,zmax);
    neighborCount = ArchivePairRefNeighborCount_BDG(mode);
    keep = RefNeighborAllowed_BDG(refF,refI,W,neighborCount);
end

function neighborCount = ArchivePairRefNeighborCount_BDG(mode)
    mode = NormalizeArchivePairRefMode_BDG(mode);
    if mode == "neighbor4"
        neighborCount = 4;
    else
        neighborCount = 0;
    end
end

function keep = RefNeighborAllowed_BDG(refF,refI,W,neighborCount)
    if size(W,2) == 2
        keep = abs(double(refF(:)) - double(refI(:))) <= neighborCount;
    else
        keep = ReferenceNeighborMask_BDG(refF,refI,W,neighborCount);
    end
end

function keep = ReferenceNeighborMask_BDG(refF,refI,W,neighborCount)
    refF = double(refF(:));
    refI = double(refI(:));
    keep = false(size(refF));
    Wn = W ./ sqrt(sum(W.^2,2) + 1e-12);
    neighborCount = max(0,round(double(neighborCount)));
    for i = 1 : numel(refF)
        if ~isfinite(refF(i)) || ~isfinite(refI(i)) || ...
                refF(i) < 1 || refF(i) > size(W,1) || ...
                refI(i) < 1 || refI(i) > size(W,1)
            continue;
        end
        score = Wn(round(refF(i)),:) * Wn';
        [~,ord] = sort(score,'descend');
        allowed = ord(1:min(neighborCount+1,numel(ord)));
        keep(i) = ismember(round(refI(i)),allowed);
    end
end

function A = InitArchiveLike_BDG(Problem)
    A.decs  = zeros(0,Problem.D);
    A.objs  = zeros(0,Problem.M);
    A.ref   = zeros(0,1);
    A.score = zeros(0,1);
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
        if nargin < 2
            zmin = zeros(1,size(Obj,2));
            zmax = zeros(1,size(Obj,2));
        end
        return;
    end
    if nargin < 2
        zmin = min(Obj,[],1);
        zmax = max(Obj,[],1);
    end
    Y = (Obj - zmin) ./ (zmax - zmin + 1e-12);
    Y = min(max(Y,0),1);
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
