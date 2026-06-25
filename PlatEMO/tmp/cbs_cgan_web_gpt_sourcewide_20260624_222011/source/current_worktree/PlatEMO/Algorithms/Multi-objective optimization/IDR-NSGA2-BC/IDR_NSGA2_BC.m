classdef IDR_NSGA2_BC < ALGORITHM
% <2026> <multi> <real/integer/label/binary/permutation> <constrained/none>
% IDR-NSGA2-BC: LHS bootstrapping + adaptive sentinel + direction-guided mating
%
% Target problem class (per ChatGPT-多目标进化优化算法探讨.md):
%   - Binary constraints (CV ∈ {0,1}), unknown/binary feasibility feedback
%   - Fragmented/disconnected feasible regions (multiple feasible components)
%   - 2~3 objectives (e.g., DASCMOP*_BC with M=2/3)
%
% Core innovations aligned with the doc:
%   1) MLP learns PoF as a continuous surrogate of binary CV (for ranking + boundary preference)
%   2) KNN recommends an individual-level direction in objective space for infeasible individuals
%   3) Direction is enforced ONLY via mating selection + environmental selection (no global ref vectors)
%   4) Extra diversity module (answering the doc's Open Question #2):
%      Feasible non-dominated archive + adaptive grid truncation + candidate injection
%
% H        --- 10     --- Hidden units of the MLP (ReLU activation)
% Epoch    --- 50     --- Training epochs per generation (online update)
% LR0      --- 0.01   --- Initial learning rate (with decay)
% Lambda   --- 1e-4   --- L2 regularization coefficient
% MaxTrain --- 2000   --- Maximum balanced FIFO archive size (1:1)
% K        --- 10     --- KNN neighbor count for direction computation
% T        --- 5      --- Mating candidate pool size for direction-guided selection
% AmaxF    --- 1000   --- Max size of feasible non-dominated archive (diversity memory)
% AinF     --- 200    --- Max injected archive candidates per generation (<=2N recommended)
%
% Current branch refinements:
%   1. LHS bootstrapping: when initial population has 0 feasible, spend up
%      to 20% of maxFE on Latin Hypercube Sampling to discover feasible solutions
%   2. MLP cold start: graceful degradation to standard tournament when no model
%   3. Adaptive sentinel quota: NI_frac varies with feasRatio
%   4. Conditional sentinel-friendly FrontNo: only when feasRatio < 0.3
%   5. Keep original S4 mating selection (double-tournament, empirically superior)

%------------------------------- Reference --------------------------------
% K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. A fast and elitist
% multiobjective genetic algorithm: NSGA-II. IEEE Transactions on
% Evolutionary Computation, 2002, 6(2): 182-197.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [H, Epoch, LR0, Lambda, MaxTrain, K, T, AmaxF, AinF] = ...
                Algorithm.ParameterSet(10, 50, 0.01, 1e-4, 2000, 10, 5, 1000, 200);

            %% Initialization
            Population = Problem.Initialization();
            CV0    = CalCV(Population.cons);
            ArcDec = Population.decs;
            ArcLab = (CV0 == 0);
            model  = [];
            needRetrain = true;
            FeasArchive = [];

            if size(ArcDec, 1) > MaxTrain
                keepIdx = TrimArchiveIdx(ArcLab, MaxTrain, size(ArcLab, 1));
                ArcDec  = ArcDec(keepIdx, :);
                ArcLab  = ArcLab(keepIdx);
            end

            %% Phase I: LHS bootstrapping (ONLY when initial population has 0 feasible)
            if ~any(CV0 == 0)
                bootBudget = floor(0.20 * Problem.maxFE);
                bootMaxFE  = min(Problem.maxFE, Problem.FE + bootBudget);
                batchSize  = 1000;

                % Pool holds all evaluated SOLUTION objects (no re-evaluation)
                Pool = Population;
                nFeasFound = 0;

                while Problem.FE < bootMaxFE && nFeasFound < 5
                    remainFE = bootMaxFE - Problem.FE;
                    nSample  = min(batchSize, remainFE);
                    if nSample <= 0
                        break;
                    end

                    % LHS in decision space
                    lhsDec = LatinHypercubeSample(nSample, Problem.D, ...
                        Problem.lower, Problem.upper);
                    lhsPop = Problem.Evaluation(lhsDec);

                    Pool = [Pool, lhsPop];

                    % Add to MLP archive
                    lhsCV  = CalCV(lhsPop.cons);
                    lhsLab = (lhsCV == 0);
                    nFeasFound = nFeasFound + sum(lhsLab);

                    ArcDec = [ArcDec; lhsPop.decs];
                    ArcLab = [ArcLab; lhsLab];
                    if size(ArcDec, 1) > MaxTrain
                        keepIdx = TrimArchiveIdx(ArcLab, MaxTrain, size(ArcLab, 1));
                        ArcDec  = ArcDec(keepIdx, :);
                        ArcLab  = ArcLab(keepIdx);
                    end
                end

                % Rebuild initial population from Pool (no re-evaluation)
                CVpool = CalCV(Pool.cons);
                if any(CVpool == 0)
                    % Have feasible: NSGA-II ranking with constraint dominance
                    Population = SelectFromPool(Pool, Problem.N, true);
                else
                    % No feasible: diversity-based selection (ignore constraints)
                    Population = SelectFromPool(Pool, Problem.N, false);
                end

                CV0 = CalCV(Population.cons);
            end

            % Initial mating ranks (conditional sentinel-friendly)
            [FrontNo, CrowdDis] = ComputeMatingRankV3(Population);
            feasRatio = sum(CV0 == 0) / max(1, numel(Population));

            %% Phase II: Main evolution loop
            while Algorithm.NotTerminated(Population)
                N = Problem.N;
                M = size(Population.objs, 2);

                % --- (1) Online MLP update ---
                if needRetrain
                    model = TryTrainMLP(model, ArcDec, ArcLab, H, Epoch, LR0, Lambda);
                    needRetrain = isempty(model);
                end

                % --- (2) Compute PoF for current population ---
                CVpop = CalCV(Population.cons);
                if ~isempty(model)
                    PoFpop = PredictMLP(model, Population.decs);
                else
                    PoFpop = zeros(numel(Population), 1);
                end
                PoFpop(CVpop == 0) = 1;

                % --- (3) Mating selection ---
                if ~isempty(model)
                    % Full direction-guided mating (original S4 style)
                    [DirVec, Conf, fMin, fRange] = ...
                        ComputeKNNDirections(Population.objs, CVpop, PoFpop, K);
                    [MatingPool, PairP1, PairP2] = DirectionMatingSelection(...
                        Population.objs, CVpop, PoFpop, DirVec, Conf, ...
                        fMin, fRange, FrontNo, CrowdDis, T);
                else
                    % Cold start: standard NSGA-II tournament
                    MatingPool = TournamentSelection(2, N, FrontNo, -CrowdDis);
                    DirVec = zeros(numel(Population), M);
                    fMin   = min(Population.objs, [], 1);
                    fRange = max(Population.objs, [], 1) - fMin;
                    fRange(fRange < 1e-12) = 1e-12;
                    PairP1 = [];
                    PairP2 = [];
                end

                % --- (4) Generate offspring ---
                Offspring = OperatorGA(Problem, Population(MatingPool));

                % --- (5) Compute DirReward for offspring ---
                nPop = numel(Population);
                nOff = numel(Offspring);
                if ~isempty(model)
                    DirRewardOff = ComputeDirReward(...
                        Offspring.objs, Population.objs, ...
                        DirVec, CVpop, fMin, fRange, PairP1, PairP2);
                else
                    DirRewardOff = zeros(nOff, 1);
                end

                % --- (5b) PoF gradient correction toward boundary ---
                % Only apply when feasible solutions are scarce and model is available.
                % Limits FE overhead to ~5% of offspring per generation.
                if ~isempty(model) && feasRatio < 0.3
                    offDecs = Offspring.decs;
                    pOff_gc = PredictMLP(model, offDecs);
                    % Only correct offspring from infeasible originators, far from boundary
                    CVoff_gc = CalCV(Offspring.cons);
                    needCorrect = find(CVoff_gc > 0 & (pOff_gc < 0.3 | pOff_gc > 0.8));
                    maxCorrect = max(1, ceil(0.05 * nOff));
                    if numel(needCorrect) > maxCorrect
                        [~, si] = sort(abs(pOff_gc(needCorrect) - 0.5), 'descend');
                        needCorrect = needCorrect(si(1:maxCorrect));
                    end
                    if ~isempty(needCorrect)
                        D_dec = size(offDecs, 2);
                        h_gc = 1e-4;
                        corrDecs = offDecs(needCorrect, :);
                        for ii = 1:numel(needCorrect)
                            x0 = corrDecs(ii, :);
                            p0 = pOff_gc(needCorrect(ii));
                            grad = zeros(1, D_dec);
                            for dd = 1:D_dec
                                xp = x0; xp(dd) = xp(dd) + h_gc;
                                grad(dd) = (PredictMLP(model, xp) - p0) / h_gc;
                            end
                            gn2 = dot(grad, grad);
                            if gn2 > 1e-12
                                stepVec = 0.1 * (0.5 - p0) * grad / gn2;
                                corrDecs(ii, :) = max(Problem.lower, ...
                                    min(Problem.upper, x0 + stepVec));
                            end
                        end
                        corrPop = Problem.Evaluation(corrDecs);
                        mask = true(1, nOff);
                        mask(needCorrect) = false;
                        Offspring = [Offspring(mask), corrPop];
                        DirRewardOff = [DirRewardOff(mask); DirRewardOff(needCorrect)];
                        nOff = numel(Offspring);
                    end
                end

                % --- (6) Update balanced FIFO archive ---
                CVoff  = CalCV(Offspring.cons);
                labOff = (CVoff == 0);
                ArcDec = [ArcDec; Offspring.decs];
                ArcLab = [ArcLab; labOff];
                if size(ArcDec, 1) > MaxTrain
                    keepIdx = TrimArchiveIdx(ArcLab, MaxTrain, size(ArcLab, 1));
                    ArcDec  = ArcDec(keepIdx, :);
                    ArcLab  = ArcLab(keepIdx);
                end

                % --- (7) Compute PoF for union ---
                Union = [Population, Offspring];
                CVu   = CalCV(Union.cons);
                if ~isempty(model)
                    PoF = PredictMLP(model, Union.decs);
                else
                    PoF = zeros(numel(Union), 1);
                end
                PoF(CVu == 0) = 1;

                % --- (8) Build DirReward for union ---
                DirReward = [zeros(nPop, 1); DirRewardOff];

                % --- (9) MLP accuracy check ---
                if ~isempty(model)
                    pOff    = PredictMLP(model, Offspring.decs);
                    predLab = pOff >= 0.5;
                    acc     = mean(double(predLab(:) == labOff(:)));
                    needRetrain = acc < 0.7;
                else
                    needRetrain = true;
                end

                % --- (10) Adaptive sentinel quota ---
                feasRatio = sum(CVu == 0) / numel(Union);
                if feasRatio < 0.05
                    NI_frac = 0.4;
                elseif feasRatio < 0.2
                    NI_frac = 0.3;
                elseif feasRatio < 0.5
                    NI_frac = 0.2;
                else
                    NI_frac = 0.1;
                end

                % --- (11) Generation counters ---
                gen = ceil(Problem.FE / Problem.N);
                maxGen = ceil(Problem.maxFE / Problem.N);

                % --- (12) Update feasible non-dominated archive (diversity memory) ---
                FeasArchive = UpdateFeasibleArchive(FeasArchive, Union, AmaxF);

                % Inject archive candidates only after early convergence stage
                if gen >= ceil(0.20 * max(1, maxGen))
                    FeasInject = SelectArchiveForInjection(FeasArchive, AinF, N);
                else
                    FeasInject = [];
                end

                % --- (13) Environmental selection (with archive injection) ---
                [Population, FrontNo, CrowdDis] = ...
                    EnvironmentalSelectionIDR_BC(...
                        [Union, FeasInject], ...
                        N, ...
                        [PoF; ones(numel(FeasInject), 1)], ...
                        [DirReward; zeros(numel(FeasInject), 1)], ...
                        NI_frac, feasRatio, gen, maxGen);

            end
        end
    end
end

%% ======================== Helper functions =========================

function CV = CalCV(Cons)
    if isempty(Cons)
        CV = zeros(size(Cons, 1), 1);
        return;
    end
    CV = double(any(Cons > 0, 2));
end

function [FrontNo, CrowdDis] = ComputeMatingRankV3(Population)
% Compute mating ranks: sentinel-friendly when feasRatio < 0.3, else standard.
    CV = CalCV(Population.cons);
    feasRatio = sum(CV == 0) / max(1, numel(Population));
    if ~any(CV == 0)
        % No feasible at all: keep objective selection pressure (ignore constraints)
        FrontNo  = NDSort(Population.objs, zeros(numel(Population), 1), numel(Population));
        CrowdDis = CrowdingDistance(Population.objs, FrontNo);
    elseif feasRatio < 0.3
        [FrontNo, CrowdDis] = ComputeSentinelFriendlyRank(Population.objs, CV);
    else
        FrontNo  = NDSort(Population.objs, Population.cons, numel(Population));
        CrowdDis = CrowdingDistance(Population.objs, FrontNo);
    end
end

function Population = SelectFromPool(Pool, N, useConstraintDominance)
% Select N individuals from a SOLUTION pool without re-evaluation.
    if numel(Pool) <= N
        Population = Pool;
        return;
    end

    Objs = Pool.objs;
    if useConstraintDominance
        Cons = Pool.cons;
    else
        Cons = zeros(numel(Pool), 1);
    end

    [FrontNo, MaxFNo] = NDSort(Objs, Cons, N);
    Next = FrontNo < MaxFNo;

    CrowdDis = CrowdingDistance(Objs, FrontNo);
    Last = find(FrontNo == MaxFNo);
    [~, Rank] = sort(CrowdDis(Last), 'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;

    Population = Pool(Next);
end

function [FrontNo, CrowdDis] = ComputeSentinelFriendlyRank(Objs, CV)
% Sentinel-friendly ranking: feasible get NDSort ranks, infeasible get maxFeasRank+1
    N = size(Objs, 1);
    feasIdx = find(CV == 0);
    infIdx  = find(CV > 0);
    FrontNo = zeros(1, N);

    if ~isempty(feasIdx)
        [fNo, ~] = NDSort(Objs(feasIdx, :), zeros(numel(feasIdx), 1), numel(feasIdx));
        maxFeasRank = max(fNo);
        FrontNo(feasIdx) = fNo;
    else
        maxFeasRank = 0;
    end
    FrontNo(infIdx) = maxFeasRank + 1;

    CrowdDis = CrowdingDistance(Objs, FrontNo);
end

function X = LatinHypercubeSample(n, D, lower, upper)
% Latin Hypercube Sampling without toolbox dependency
    X = zeros(n, D);
    for j = 1:D
        perm = randperm(n);
        X(:, j) = (perm' - rand(n, 1)) / n;
    end
    X = X .* (upper - lower) + lower;
end

function DirRewardOff = ComputeDirReward(OffObjs, PopObjs, DirVec, CVpop, fMin, fRange, PairP1, PairP2)
% Compute direction reward: offspring alignment with the direction(s) of its
% paired parents. This matches the doc: direction affects evolution via
% mating selection + environmental selection, not by explicit decision-space mapping.
    nOff = size(OffObjs, 1);
    DirRewardOff = zeros(nOff, 1);

    OffObjsN = (OffObjs - fMin) ./ fRange;
    PopObjsN = (PopObjs - fMin) ./ fRange;

    hasDir = any(abs(DirVec) > 1e-12, 2);
    half = numel(PairP1);
    if half < 1 || numel(PairP2) ~= half
        return;
    end

    % OperatorGA generates 2 offspring per pair: k and k+half (when N even)
    maxK = min(half, floor(nOff / 2));
    for k = 1:maxK
        p1 = PairP1(k);
        p2 = PairP2(k);
        parents = [p1, p2];

        for child = [k, k + maxK]
            bestReward = 0;
            for pi = 1:2
                p = parents(pi);
                if p >= 1 && p <= size(PopObjsN, 1) && CVpop(p) > 0 && hasDir(p)
                    diff = OffObjsN(child, :) - PopObjsN(p, :);
                    d_norm = norm(diff);
                    if d_norm > 1e-12
                        cosVal = dot(diff, DirVec(p, :)) / d_norm;
                        bestReward = max(bestReward, max(0, cosVal));
                    end
                end
            end
            DirRewardOff(child) = bestReward;
        end
    end
end

function [MatingPool, PairP1, PairP2] = DirectionMatingSelection(Objs, CV, PoF, DirVec, Conf, fMin, fRange, FrontNo, CrowdDis, T)
% Direction-guided mating selection with explicit parent pairing.
% PairP1(k) is the originator selected by tournament, PairP2(k) is its mate
% chosen by direction alignment + PoF (when applicable).
    N = size(Objs, 1);
    half = floor(N / 2);
    PairP1 = [];
    PairP2 = [];
    MatingPool = zeros(1, 2 * half);

    if half < 1
        return;
    end

    ObjsN = (Objs - fMin) ./ fRange;

    isInfeas     = (CV > 0);
    hasDirection = any(abs(DirVec) > 1e-12, 2);

    % Parent1: standard tournament (sentinel-friendly FrontNo already provided)
    PairP1 = TournamentSelection(2, half, FrontNo, -CrowdDis);
    PairP2 = zeros(1, half);

    for k = 1:half
        i = PairP1(k);
        if isInfeas(i) && hasDirection(i)
            % Sample candidates without replacement (avoid duplicates / self-mating)
            pool = [1:i-1, i+1:N];
            tEff = min(T, numel(pool));
            candidates = pool(randperm(numel(pool), tEff));
            fi = ObjsN(i, :);
            di = DirVec(i, :);
            ci = Conf(i);

            scores = zeros(tEff, 1);
            for t = 1:tEff
                j = candidates(t);
                diff_j = ObjsN(j, :) - fi;
                d_norm = norm(diff_j);
                if d_norm > 1e-12
                    cosVal = dot(diff_j, di) / d_norm;
                else
                    cosVal = 0;
                end
                align_j   = max(0, cosVal);
                scores(t) = ci * align_j + (1 - ci) * PoF(j);
            end

            [~, best] = max(scores);
            PairP2(k) = candidates(best);
        else
            % Mate by tournament (objective pressure / diversity)
            k1 = randi(N);
            k2 = randi(N);
            if FrontNo(k1) < FrontNo(k2) || ...
               (FrontNo(k1) == FrontNo(k2) && CrowdDis(k1) > CrowdDis(k2))
                PairP2(k) = k1;
            else
                PairP2(k) = k2;
            end
        end
    end

    MatingPool = [PairP1, PairP2];

    % If N is odd, append one more parent (fallback tournament)
    if numel(MatingPool) < N
        extra = TournamentSelection(2, N - numel(MatingPool), FrontNo, -CrowdDis);
        MatingPool = [MatingPool, extra];
    end
end

function Archive = UpdateFeasibleArchive(Archive, Pool, Amax)
% Maintain a feasible non-dominated archive with grid-based truncation.
% Purpose: preserve disconnected PF components and improve coverage.
    if Amax <= 0
        Archive = [];
        return;
    end
    if isempty(Pool)
        return;
    end

    CV = CalCV(Pool.cons);
    feas = Pool(CV == 0);
    if isempty(feas)
        return;
    end

    if isempty(Archive)
        Cand = feas;
    else
        Cand = [Archive, feas];
    end

    % Remove NaN objective rows
    Objs = Cand.objs;
    valid = ~any(isnan(Objs), 2);
    Cand = Cand(valid);
    if isempty(Cand)
        Archive = [];
        return;
    end

    % Keep only the non-dominated set
    nC = numel(Cand);
    FrontNo = NDSort(Cand.objs, zeros(nC, 1), nC);
    Cand = Cand(FrontNo == 1);
    if numel(Cand) <= Amax
        Archive = Cand;
        return;
    end

    keep = GridDiversityPrune(Cand.objs, Amax);
    Archive = Cand(keep);
end

function Inject = SelectArchiveForInjection(Archive, AinF, N)
% Select a diverse subset of the feasible archive to inject as extra candidates.
    if isempty(Archive) || AinF <= 0
        Inject = [];
        return;
    end
    cap = min([AinF, 2 * N, numel(Archive)]);
    if numel(Archive) <= cap
        Inject = Archive;
        return;
    end
    keep = GridDiversityPrune(Archive.objs, cap);
    Inject = Archive(keep);
end

function keep = GridDiversityPrune(Objs, target)
% Prune a set of objective vectors to 'target' points with adaptive grid diversity.
% - Minimize removal of rare grid cells
% - Remove most crowded points inside the most crowded cell
% - Protect extreme points (min of each objective)
    n = size(Objs, 1);
    if target >= n
        keep = true(n, 1);
        return;
    end

    M = size(Objs, 2);
    keep = true(n, 1);

    % Protect per-objective minima (extremes)
    protect = false(n, 1);
    [~, idxMin] = min(Objs, [], 1);
    protect(unique(idxMin(:))) = true;

    % Normalize once (stable grid assignment)
    fMin = min(Objs, [], 1);
    fMax = max(Objs, [], 1);
    fRange = fMax - fMin;
    fRange(fRange < 1e-12) = 1e-12;
    ObjsN = (Objs - fMin) ./ fRange;

    % Grid resolution: more cells than target to preserve gaps/components
    div = max(3, ceil((2 * target)^(1 / max(1, M))));
    cell = floor(ObjsN * div);
    cell(cell >= div) = div - 1;
    cell(cell < 0) = 0;

    % Linearize cell indices (base-div)
    lin = cell(:, 1);
    base = 1;
    for m = 2:M
        base = base * div;
        lin = lin + cell(:, m) * base;
    end
    lin = lin + 1;

    while sum(keep) > target
        rem = find(keep);
        linRem = lin(rem);
        [u, ~, ic] = unique(linRem);
        counts = accumarray(ic, 1);
        [~, order] = sort(counts, 'descend');

        removed = false;
        for oi = 1:numel(order)
            cellId = u(order(oi));
            members = rem(linRem == cellId);
            if numel(members) <= 1
                continue;
            end
            cand = members(~protect(members));
            if isempty(cand)
                continue;
            end

            X = ObjsN(members, :);
            D2 = sum(X.^2, 2) + sum(X.^2, 2)' - 2 * (X * X');
            D = sqrt(max(D2, 0));
            D(1:size(D, 1) + 1:end) = inf;
            nn = min(D, [], 2);

            % Prefer removing a non-protected member with smallest nn-distance
            [~, localOrder] = sort(nn, 'ascend');
            for li = 1:numel(localOrder)
                toRemove = members(localOrder(li));
                if ~protect(toRemove)
                    keep(toRemove) = false;
                    removed = true;
                    break;
                end
            end

            if removed
                break;
            end
        end

        if ~removed
            % Fallback: distance-based truncation among removable points
            rem2 = find(keep & ~protect);
            if isempty(rem2)
                break;
            end
            X = ObjsN(rem2, :);
            D2 = sum(X.^2, 2) + sum(X.^2, 2)' - 2 * (X * X');
            D = sqrt(max(D2, 0));
            D(1:size(D, 1) + 1:end) = inf;
            nn = min(D, [], 2);
            [~, pos] = min(nn);
            keep(rem2(pos)) = false;
        end
    end
end

function [DirVec, Conf, fMin, fRange] = ComputeKNNDirections(Objs, CV, PoF, K)
% KNN-based direction recommendation for infeasible individuals.
    [N, M] = size(Objs);
    DirVec = zeros(N, M);
    Conf   = zeros(N, 1);

    fMin   = min(Objs, [], 1);
    fMax   = max(Objs, [], 1);
    fRange = fMax - fMin;
    fRange(fRange < 1e-12) = 1e-12;
    ObjsN  = (Objs - fMin) ./ fRange;

    infeasIdx = find(CV > 0);
    if isempty(infeasIdx) || N < 2
        return;
    end

    Keff = min(K, N - 1);
    if Keff < 1
        return;
    end

    D2   = sum(ObjsN.^2, 2) + sum(ObjsN.^2, 2)' - 2 * (ObjsN * ObjsN');
    Dist = sqrt(max(D2, 0));

    isFeas = (CV == 0);
    confThreshold = 2 / Keff;

    for t = 1:numel(infeasIdx)
        i = infeasIdx(t);

        dists_i    = Dist(i, :);
        dists_i(i) = Inf;
        [~, sIdx]  = sort(dists_i);
        neighbors  = sIdx(1:Keff);

        feasNeigh   = neighbors(isFeas(neighbors));
        infeasNeigh = neighbors(~isFeas(neighbors));
        conf_i = numel(feasNeigh) / Keff;

        fi = ObjsN(i, :);

        if ~isempty(feasNeigh)
            % Attraction-repulsion direction (doc-aligned):
            %   d ≈ mean(feasible - current) - mean(infeasible - current)
            % Use distance-weighting for robustness under sparse feasibility.
            dists_feas = dists_i(feasNeigh);
            wF = 1 ./ max(dists_feas(:), 1e-12);
            wF = wF / max(1e-12, sum(wF));
            DeltaF = ObjsN(feasNeigh, :) - fi;
            attr = wF' * DeltaF;

            if ~isempty(infeasNeigh)
                dists_inf = dists_i(infeasNeigh);
                wI = 1 ./ max(dists_inf(:), 1e-12);
                wI = wI / max(1e-12, sum(wI));
                DeltaI = ObjsN(infeasNeigh, :) - fi;
                rep = wI' * DeltaI;
            else
                rep = zeros(1, M);
            end

            d_i = attr - rep;

            % Degenerate case: fall back to nearest feasible neighbor direction
            if norm(d_i) <= 1e-12
                [~, nearestIdx] = min(dists_feas);
                d_i = ObjsN(feasNeigh(nearestIdx), :) - fi;
            end

            % With too few feasible neighbors, reduce confidence (direction is noisy)
            if conf_i < confThreshold
                conf_i = 0.5 * conf_i;
            end
        else
            % No feasible neighbors: infer a direction that increases PoF locally
            DeltaAll = ObjsN(neighbors, :) - fi;
            DeltaP   = PoF(neighbors) - PoF(i);
            if rank(DeltaAll) >= 1
                beta = DeltaAll \ DeltaP;
                d_i  = beta';
            else
                d_i = zeros(1, M);
            end
            conf_i = 0.5 * conf_i;
        end

        d_norm = norm(d_i);
        if d_norm > 1e-12
            d_i = d_i / d_norm;
        end

        DirVec(i, :) = d_i;
        Conf(i)      = conf_i;
    end
end

function model = TryTrainMLP(model, Dec, Lab, H, Epoch, LR0, Lambda)
    if numel(Lab) < 20 || numel(unique(Lab)) < 2
        return;
    end
    model = TrainMLP(Dec, Lab, H, Epoch, LR0, Lambda);
end

function model = TrainMLP(X, Lab, H, Epoch, LR0, Lambda)
    X = double(X);
    y = double(Lab(:));
    [n, D] = size(X);

    mu  = mean(X, 1);
    sig = std(X, 0, 1) + 1e-12;
    Xn  = (X - mu) ./ sig;

    n1 = sum(y == 1);
    n0 = n - n1;
    w1 = n / (2 * max(1, n1));
    w0 = n / (2 * max(1, n0));
    w  = w0 + (w1 - w0) .* y;
    wn = w / n;

    W1 = randn(D, H) * sqrt(2 / D);
    b1 = zeros(1, H);
    W2 = randn(H, 1) * sqrt(2 / H);
    b2 = 0;

    for ep = 1:Epoch
        lr = LR0 / (1 + 0.01 * ep);
        Z1   = Xn * W1 + b1;
        A1   = max(0, Z1);
        Z2   = A1 * W2 + b2;
        yhat = 1 ./ (1 + exp(-Z2));

        dZ2 = wn .* (yhat - y);
        dW2 = A1' * dZ2 + Lambda * W2;
        db2 = sum(dZ2);
        dA1 = dZ2 * W2';
        dZ1 = dA1 .* (Z1 > 0);
        dW1 = Xn' * dZ1 + Lambda * W1;
        db1 = sum(dZ1, 1);

        W1 = W1 - lr * dW1;
        b1 = b1 - lr * db1;
        W2 = W2 - lr * dW2;
        b2 = b2 - lr * db2;
    end

    model.W1  = W1;  model.b1  = b1;
    model.W2  = W2;  model.b2  = b2;
    model.mu  = mu;  model.sig = sig;
end

function p = PredictMLP(model, X)
    X  = double(X);
    Xn = (X - model.mu) ./ model.sig;
    A1 = max(0, Xn * model.W1 + model.b1);
    Z2 = A1 * model.W2 + model.b2;
    p  = 1 ./ (1 + exp(-Z2));
    p  = min(max(p, 0), 1);
end

function keepIdx = TrimArchiveIdx(ArcLab, MaxTrain, nTotal)
    y   = ArcLab(:) > 0.5;
    id1 = find(y);
    id0 = find(~y);

    nKeep1 = min(numel(id1), floor(MaxTrain / 2));
    nKeep0 = min(numel(id0), MaxTrain - nKeep1);

    id1 = id1(max(1, end-nKeep1+1):end);
    id0 = id0(max(1, end-nKeep0+1):end);

    keepIdx = [id0; id1];
    if numel(keepIdx) < MaxTrain
        rest = setdiff((1:nTotal)', keepIdx, 'stable');
        rest = rest(max(1, end-(MaxTrain-numel(keepIdx))+1):end);
        keepIdx = [keepIdx; rest];
    end
end

