function [Population, FrontNo, CrowdDis] = EnvironmentalSelectionIDR_BC(Union, N, PoF, DirReward, NI_frac, feasRatio, gen, maxGen)
% Environmental selection for IDR-NSGA2-BC (CDPPV + Boundary Preference):
%   - 3-tier CDPPV classification: true-feasible / pseudo-feasible / pseudo-infeasible
%   - Adaptive epsilon threshold based on generation progress
%   - Elite pool (tier -1 + tier 0): NSGA-II with virtual constraints
%   - Sentinel pool (tier 1): composite score with boundary preference + greedy diversity
%   - Conditional sentinel-friendly FrontNo for mating (when feasRatio < 0.3)
%
% Inputs:
%   Union     : combined parent + offspring population
%   N         : target population size
%   PoF       : numel(Union) x 1 probability of feasibility
%   DirReward : numel(Union) x 1 direction reward
%   NI_frac   : sentinel quota fraction (dynamic)
%   feasRatio : fraction of feasible in Union
%   gen       : current generation number
%   maxGen    : estimated total generations

    PoF       = PoF(:);
    DirReward = DirReward(:);

    %% 3-tier CDPPV classification
    CV = CalCVLocal(Union.cons);

    % Adaptive epsilon: starts high (lenient), decreases over time
    % Keep above 0.55 so pseudo-infeasible pool still contains boundary-like points
    epsilon_t = max(0.55, 0.95 - 0.40 * gen / max(1, maxGen));

    idxTF = find(CV == 0);                          % Tier -1: true feasible
    idxPF = find(CV > 0 & PoF >= epsilon_t);        % Tier  0: pseudo-feasible
    idxPI = find(CV > 0 & PoF <  epsilon_t);        % Tier  1: pseudo-infeasible (sentinels)

    % Elite pool = true feasible + pseudo-feasible
    idxElite = [idxTF(:); idxPF(:)];
    idxSent  = idxPI(:);

    %% Dynamic quota
    NI = ceil(NI_frac * N);
    if numel(idxElite) < N - NI
        NE = numel(idxElite);
        NI = N - NE;
    else
        NE = N - NI;
    end
    NI = min(NI, numel(idxSent));
    NE = min(N - NI, numel(idxElite));

    Choose = zeros(0, 1);

    %% Select from elite pool (NSGA-II with virtual constraints)
    if NE > 0
        if numel(idxElite) <= NE
            ChooseE = idxElite(:);
        else
            UE = Union(idxElite);
            % Virtual constraints: 0 for true feasible, (1-PoF) for pseudo-feasible
            nTF = numel(idxTF);
            ConsE = zeros(numel(idxElite), 1);
            if numel(idxPF) > 0
                ConsE(nTF+1:end) = 1 - PoF(idxPF);
            end

            [eNo, eMax] = NDSort(UE.objs, ConsE, NE);
            next = eNo < eMax;
            cd = CrowdingDistance(UE.objs, eNo);
            last = find(eNo == eMax);
            [~, rk] = sort(cd(last), 'descend');
            next(last(rk(1:NE-sum(next)))) = true;
            ChooseE = idxElite(next);
        end
        Choose = [Choose; ChooseE(:)];
    end

    %% Select sentinels from pseudo-infeasible pool (boundary preference + diversity)
    if NI > 0 && ~isempty(idxSent)
        Objs_all = Union.objs;

        fMinU   = min(Objs_all, [], 1);
        fMaxU   = max(Objs_all, [], 1);
        fRangeU = fMaxU - fMinU;
        fRangeU(fRangeU < 1e-12) = 1e-12;
        ObjsN   = (Objs_all - fMinU) ./ fRangeU;

        ObjsN_S = ObjsN(idxSent, :);

        % dist_elite: distance to nearest elite solution (true feasible + pseudo-feasible)
        if ~isempty(idxElite)
            ObjsN_F = ObjsN(idxElite, :);
            D2 = sum(ObjsN_S.^2, 2) + sum(ObjsN_F.^2, 2)' - 2 * (ObjsN_S * ObjsN_F');
            D  = sqrt(max(D2, 0));
            distElite = min(D, [], 2);
            maxDist   = max(distElite);
            if maxDist > 1e-12
                distElite = distElite / maxDist;
            end
        else
            distElite = ones(numel(idxSent), 1);
        end

        pS  = PoF(idxSent);
        drS = DirReward(idxSent);

        % Boundary preference: PoF close to 0.5 gets highest bonus
        boundaryBonus = 1 - abs(2 * pS - 1);

        % nearElite: prefer sentinels close to elite region (boundary proximity)
        nearElite = 1 - distElite;

        baseScores = 0.35 * pS + 0.15 * nearElite + 0.2 * drS + 0.3 * boundaryBonus;

        % Objective pressure:
        % - When PoF provides no discrimination, rely more on objectives.
        % - Otherwise, keep a mild objective component to help sentinels traverse infeasible
        %   barriers toward the Pareto region (often reduces IGD on large-infeasible-region cases).
        objQuality = 1 - mean(ObjsN_S, 2); % smaller objectives => higher quality
        oqMin = min(objQuality);
        oqMax = max(objQuality);
        if oqMax - oqMin > 1e-12
            objQuality = (objQuality - oqMin) / (oqMax - oqMin);
        else
            objQuality = zeros(size(objQuality));
        end

        pSpread = max(pS) - min(pS);
        if isempty(idxElite) || pSpread < 1e-12
            objW = 0.50;
        else
            prog  = gen / max(1, maxGen);
            feasW = min(1, feasRatio / 0.50);
            objW  = 0.05 + 0.25 * prog * feasW;  % in [0.05, 0.30]
            objW  = max(0.05, min(0.30, objW));
        end
        sentinelScores = (1 - objW) * baseScores + objW * objQuality;

        % Greedy diversity-aware selection
        % Diversity weight: stronger early, weaker later
        divW = 0.60 - 0.30 * gen / max(1, maxGen);
        divW = max(0.30, min(0.70, divW));

        nS = numel(idxSent);
        selectedLocal = zeros(0, 1);           % indices into idxSent
        remainingMask = true(nS, 1);

        % Min distance to already chosen (elite) in normalized objective space
        if ~isempty(Choose)
            ObjsN_C = ObjsN(Choose, :);
            D2 = sum(ObjsN_S.^2, 2) + sum(ObjsN_C.^2, 2)' - 2 * (ObjsN_S * ObjsN_C');
            D  = sqrt(max(D2, 0));
            minDist = min(D, [], 2);
        else
            minDist = inf(nS, 1);
        end

        for k = 1:NI
            rem = find(remainingMask);
            if isempty(rem)
                break;
            end

            md = minDist(rem);
            if all(isinf(md))
                divBonus = zeros(numel(rem), 1);
            else
                mdInf = isinf(md);
                maxMD = max(md(~mdInf));
                if maxMD > 1e-12
                    divBonus = md / maxMD;
                    divBonus(mdInf) = 1;
                else
                    divBonus = zeros(numel(rem), 1);
                end
            end

            effScores = sentinelScores(rem) + divW * divBonus;
            [~, bestPos] = max(effScores);
            bestLocal = rem(bestPos);

            selectedLocal = [selectedLocal; bestLocal];
            remainingMask(bestLocal) = false;

            % Update minDist w.r.t. the newly selected sentinel
            newPt = ObjsN_S(bestLocal, :);
            d = sqrt(sum((ObjsN_S - newPt).^2, 2));
            minDist = min(minDist, d);
        end

        ChooseS = idxSent(selectedLocal);
        Choose  = [Choose; ChooseS(:)];
    end

    %% Fallback
    if numel(Choose) < N
        remain = setdiff((1:numel(Union))', Choose, 'stable');
        need   = N - numel(Choose);
        if need > 0 && ~isempty(remain)
            UR = Union(remain);
            [rNo, rMax] = NDSort(UR.objs, UR.cons, need);
            next = rNo < rMax;
            cd   = CrowdingDistance(UR.objs, rNo);
            last = find(rNo == rMax);
            [~, rk] = sort(cd(last), 'descend');
            next(last(rk(1:need-sum(next)))) = true;
            Choose = [Choose; remain(next)];
        end
    end

    Population = Union(Choose);

    %% Compute FrontNo for mating selection
    %  Conditional: sentinel-friendly only when feasRatio < 0.3
    CVnew   = CalCVLocal(Population.cons);
    feasIdx = find(CVnew == 0);
    infIdx  = find(CVnew > 0);

    FrontNo  = zeros(1, numel(Population));
    CrowdDis = zeros(1, numel(Population));

    if feasRatio < 0.3
        % Sentinel-friendly: sentinels get maxFeasRank+1 (can participate in mating)
        if ~isempty(feasIdx)
            [fNo, ~] = NDSort(Population(feasIdx).objs, zeros(numel(feasIdx), 1), numel(feasIdx));
            maxFeasRank = max(fNo);
            FrontNo(feasIdx) = fNo;
        else
            maxFeasRank = 0;
        end
        FrontNo(infIdx) = maxFeasRank + 1;
    else
        % Standard constraint dominance
        [FrontNo, ~] = NDSort(Population.objs, Population.cons, numel(Population));
    end

    CrowdDis = CrowdingDistance(Population.objs, FrontNo);
end

function CV = CalCVLocal(Cons)
    if isempty(Cons)
        CV = zeros(size(Cons, 1), 1);
        return;
    end
    CV = double(any(Cons > 0, 2));
end

