function varargout=suffix_native_helpers(action,varargin)
% Exact copies of existing local backbone helpers for an isolated suffix test.
switch action
 case 'p1'; [varargout{1:nargout}]=pairOffspring(varargin{:});
 case 'p2'; [varargout{1:nargout}]=scopedBackbone(varargin{:});
 case 'pair_only'; [varargout{1:nargout}]=pairOnlySample(varargin{:});
end
end

function [Offspring,Trace,Log] = pairOffspring(Problem,Population,Fitness, ...
        count,Pending,Config,seed,generation)
%PAIROFFSPRING Consume exactly the saved q; missing slots use the same DE.
    startFE = Problem.FE;
    Trace = emptyUseTrace();
    Trace.childDecs = zeros(0,Problem.D);
    Trace.childObjs = zeros(0,Problem.M);
    Trace.childCons = zeros(0,1);
    ga = round(0.25*count); quota = round(0.20*count);
    de = count-ga-quota;
    A = scopedOperator("ga",Problem,Population,Fitness,ga,seed+1009*generation+1);
    B = scopedOperator("de",Problem,Population,Fitness,de,seed+1009*generation+2);
    Offspring = [A,B];
    Trace.active = quota > 0; Trace.requested = quota;
    selected = zeros(0,1);
    if isfield(Pending,'decs')
        span = double(Problem.upper)-double(Problem.lower); span(span <= eps) = 1;
        base = [double(Population.decs);double(Offspring.decs)];
        for k = 1:size(Pending.decs,1)
            q = Pending.decs(k,:);
            if any(~isfinite(q)) || any(q < Problem.lower-1e-12) || ...
                    any(q > Problem.upper+1e-12) || ...
                    (Pending.ids(k) > 0 && nnz(Pending.ids(selected) == Pending.ids(k)) >= 2)
                continue;
            end
            if ~isempty(base) && any(sqrt(sum(((base-q)./span).^2,2)) ...
                    <= Config.pairDuplicateTolerance)
                continue;
            end
            selected(end+1,1) = k; %#ok<AGROW>
            base(end+1,:) = q; %#ok<AGROW>
            if numel(selected) >= quota; break; end
        end
    end
    if quota == 0; selected = zeros(0,1); end
    fallback = quota-numel(selected);
    C = scopedOperator("de",Problem,Population,Fitness,fallback,seed+1009*generation+5);
    Offspring = [Offspring,C];
    Trace.fallback = fallback;
    Trace.fallbackNoPool = fallback*double(~isfield(Pending,'decs'));
    Trace.fallbackMapping = fallback-Trace.fallbackNoPool;
    Trace.selected = numel(selected); Trace.mappedValid = numel(selected);
    Trace.productionFE = NaN;
    Trace.productionGeneration = NaN;
    Trace.consumptionGeneration = generation;
    Trace.modelVersion = NaN;
    Trace.generatedXf = zeros(0,Problem.D); Trace.generatedXi = zeros(0,Problem.D);
    Trace.evalIDs = zeros(0,1);
    if ~isempty(selected)
        before = Problem.FE;
        Guided = Problem.Evaluation(Pending.decs(selected,:));
        if ~isequal(double(Guided.decs),Pending.decs(selected,:))
            error('CBSPairGuide:ChangedPending','Evaluation changed a pending native candidate.');
        end
        Offspring = [Offspring,Guided];
        Trace.childDecs = double(Guided.decs);
        Trace.childObjs = double(Guided.objs);
        Trace.childCons = double(Guided.cons);
        Trace.matchedPairIds = Pending.ids(selected);
        Trace.requestedRefs = Pending.refs(selected); Trace.requestedSides = Pending.sides(selected);
        Trace.generatedXf = Pending.xf(selected,:);
        Trace.generatedXi = Pending.xi(selected,:);
        Trace.parentObjs = Pending.yf(selected,:);
        Trace.productionFE = Pending.productionFE;
        Trace.productionGeneration = Pending.productionGeneration;
        Trace.modelVersion = Pending.modelVersion;
        Trace.selectedTargetDecs = Trace.childDecs;
        Trace.selectedCenterDecs = Trace.childDecs;
        Trace.evalIDs = before+(1:numel(selected))';
        Trace.fullGuideFE = numel(selected);
        Trace.constraintGuideFE = numel(selected);
        feasible = all(Guided.cons <= 0,2);
        dominates = all(Trace.childObjs <= Trace.parentObjs+1e-12,2) & ...
            any(Trace.childObjs < Trace.parentObjs-1e-12,2);
        Trace.feasibleChildren = nnz(feasible);
        Trace.dominatingChildren = nnz(dominates);
        Trace.childUseful = nnz(feasible & dominates);
        if all(Pending.ids(selected)==0)
            Trace.dominatingChildren=NaN; Trace.childUseful=NaN;
        end
        Trace.selectedConditions = numel(unique(Pending.refs(selected)));
    end
    Log = evaluationRecord(Offspring,startFE,"P1");
    Log.origin = [repmat("GA",ga,1);repmat("ordinary_DE",de,1); ...
        repmat("fallback_DE",fallback,1);repmat("PairGuide",numel(selected),1)];
end

function Offspring = scopedBackbone(Problem,Population,Fitness,count,seed,generation)
%SCOPEDBACKBONE P2 remains 25% GA and 75% ordinary DE.
    ga = round(0.25*count);
    A = scopedOperator("ga",Problem,Population,Fitness,ga,seed+1009*generation+3);
    B = scopedOperator("de",Problem,Population,Fitness,count-ga,seed+1009*generation+4);
    Offspring = [A,B];
end

function [Raw,Sample] = pairOnlySample(Archive,Info,Problem)
%PAIRONLYSAMPLE Causal control with the identical true interval geometry.
    [~,rows] = ismember(Info.pairIds,Archive.id);
    lower = double(Problem.lower);
    span = double(Problem.upper)-lower; span(span <= eps) = 1;
    f = (Archive.xf(rows,:)-lower)./span;
    i = (Archive.xi(rows,:)-lower)./span;
    alpha = 0.4+0.2*rand(numel(rows),1);
    Raw = lower+((1-alpha).*f+alpha.*i).*span;
    Sample = struct('generatedF',f,'generatedI',i,'alpha',alpha, ...
        'z',zeros(numel(rows),0),'endpointForwardRows',0,'forwardSeconds',0);
end

function Population = scopedOperator(kind,Problem,Parents,Fitness,count,seed)
%SCOPEDOPERATOR Ordinary operators have independent generation/slot streams.
    if count <= 0
        Population = Parents([]);
        return;
    end
    stream = RandStream('mt19937ar','Seed',mod(seed,2^32-1));
    cleanup = pairStreamScope(stream); %#ok<NASGU>
    if kind == "ga"
        Population = gaOffspring(Problem,Parents,Fitness,count);
    else
        Population = deOffspring(Problem,Parents,Fitness,count);
    end
    clear cleanup;
end

function cleanup = pairStreamScope(stream)
%PAIRSTREAMSCOPE Restore the caller's stream after training/query.
    previous = RandStream.getGlobalStream;
    RandStream.setGlobalStream(stream);
    cleanup = onCleanup(@()RandStream.setGlobalStream(previous));
end

function Offspring = gaOffspring(Problem,Population,Fitness,count)
%GAOFFSPRING Platform SBX and polynomial-mutation offspring.

    matingPool = platformTournamentSelection(2,2*count,Fitness);
    Offspring = OperatorGAhalf(Problem,Population(matingPool));
end

function Offspring = deOffspring(Problem,Population,Fitness,count)
%DEOFFSPRING Ordinary DE with mutually distinct parent rows.

    Offspring = OperatorDEDistinct_CBS(Problem,Population,Fitness,count);
end

function index = platformTournamentSelection(K,N,Fitness)
%PLATFORMTOURNAMENTSELECTION Preserve stable fitness-tie behavior.

    [~,order] = sortrows(reshape(Fitness,[],1));
    rank = zeros(numel(order),1);
    rank(order) = 1:numel(order);
    index = TournamentSelection(K,N,rank);
end

function Trace = emptyUseTrace()
%EMPTYUSETRACE Empty utilization-side mechanism record.

    Trace = struct('active',false,'requested',0,'selected',0, ...
        'fallback',0,'fallbackNoPool',0, ...
        'fallbackNoCurrentFeasible',0,'fallbackInvalidScale',0, ...
        'fallbackMapping',0,'memoryParentsAdded',0, ...
        'memoryParentsUsed',0,'selectedConditions',0,'mappedValid',0, ...
        'mapDropped',0,'alpha',zeros(0,1),'h',zeros(0,1), ...
        'd',zeros(0,1),'centerStep',zeros(0,1), ...
        'actualStep',zeros(0,1),'directionCosine',zeros(0,1), ...
        'feasibleChildren',0,'dominatingChildren',0, ...
        'selectedTargetCount',0,'selectedTargetFeasible',0, ...
        'selectedTargetUseful',0,'centerFeasible',0, ...
        'centerUseful',0,'childUseful',0, ...
        'targetFeasibleLostAtCenter',0, ...
        'targetInfeasibleRecoveredAtCenter',0, ...
        'centerFeasibleLostAtMutation',0, ...
        'centerInfeasibleRecoveredAtMutation',0, ...
        'childDecs',zeros(0,0),'childObjs',zeros(0,0), ...
        'childCons',zeros(0,0),'parentObjs',zeros(0,0), ...
        'selectedTargetDecs',zeros(0,0), ...
        'selectedCenterDecs',zeros(0,0), ...
        'matchedPairIds',zeros(0,1), ...
        'fullGuideFE',0,'constraintGuideFE',0, ...
        'pairDecisionRngBefore',struct(), ...
        'selectedBoundaryDistance',zeros(0,1), ...
        'selectedBoundaryBand',zeros(0,1), ...
        'selectedBoundarySupported',false(0,1), ...
        'centerBoundaryDistance',zeros(0,1), ...
        'centerBoundaryBand',zeros(0,1), ...
        'centerBoundarySupported',false(0,1), ...
        'childBoundaryDistance',zeros(0,1), ...
        'childBoundaryBand',zeros(0,1), ...
        'childBoundarySupported',false(0,1), ...
        'selectedDirectionCoverage',NaN, ...
        'selectedDirectionEntropy',NaN, ...
        'selectedNearDuplicateRate',NaN);
end

function Record = evaluationRecord(Population,startFE,origin)
%EVALUATIONRECORD Unique real evaluation IDs live only in the audit.
    Record = struct('firstEvalID',double(startFE)+1, ...
        'lastEvalID',double(startFE)+numel(Population),'origin',origin, ...
        'decisions',double(Population.decs),'objectives',double(Population.objs), ...
        'constraints',double(Population.cons));
end
