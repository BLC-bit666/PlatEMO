function Candidates = RefineBoundaryObservations_RC( ...
        Problem,Population1,Population2,budget)
%REFINEBOUNDARYOBSERVATIONS_RC Active boundary observation for the module.
%   The boundary-memory module observes the constraint boundary in two
%   ways: passively, through ordinary evolutionary offspring, and
%   actively, through the two calibration primitives implemented here.
%   Feasibility-feedback calibration contracts feasible-infeasible
%   decision intervals on the single-bit feasibility oracle so that the
%   feasible-side iterates land next to the constraint boundary; coverage
%   completion evaluates the decision midpoint of the sparsest adjacent
%   feasible nondominated pair, with one contraction repair step when the
%   midpoint lands infeasible. Every candidate is a real evaluation: it
%   competes in the ordinary environmental selection and joins the
%   same-generation boundary-memory harvest. The interval-contraction
%   core follows the classic boundary-operator principle (Michalewicz et
%   al.; repair by binary interpolation, GECCO 2007).
%BOUNDARYLINESEARCH Membership-oracle line search after the CGAN stops.
%   Primitive 1 bisects feasible-infeasible decision segments so that the
%   feasible-side iterates land next to the constraint boundary; primitive
%   2 evaluates the decision midpoint of the sparsest adjacent feasible
%   nondominated pair, with one bisection repair step when the midpoint is
%   infeasible. Both follow the classic boundary-operator idea
%   (Michalewicz et al.; repair by binary interpolation, GECCO 2007) and
%   only require the single-bit feasibility oracle. Engineering utility,
%   not an algorithmic contribution.

    Candidates = Population1([]);
    All = [Population1,Population2];
    cv = sum(max(0,All.cons),2);
    Feasible = All(cv <= 0);
    if isempty(Feasible) || budget <= 0
        return;
    end
    front = NDSort(Feasible.objs,1) == 1;
    Front = Feasible(front);
    FrontDecs = Front.decs;
    Infeasible = All(cv > 0);
    used = 0;

    %% Primitive 1: pin anchors onto the boundary by bisection
    if ~isempty(Infeasible)
        InfDecs = Infeasible.decs;
        for anchor = 1 : min(3,size(FrontDecs,1))
            if used >= budget
                break;
            end
            pick = randi(size(FrontDecs,1));
            feasiblePoint = FrontDecs(pick,:);
            distance2 = sum((InfDecs-feasiblePoint).^2,2);
            [~,nearest] = min(distance2);
            infeasiblePoint = InfDecs(nearest,:);
            steps = min(4,budget-used);
            for k = 1 : steps
                middle = (feasiblePoint+infeasiblePoint)/2;
                candidate = Problem.Evaluation(middle);
                used = used + 1;
                Candidates = [Candidates,candidate]; %#ok<AGROW>
                if sum(max(0,candidate.cons),2) <= 0
                    feasiblePoint = middle;
                else
                    infeasiblePoint = middle;
                end
            end
        end
    end

    %% Primitive 2: fill the sparsest objective-space gaps
    if used < budget && numel(Front) >= 2
        FrontObjs = Front.objs;
        distance = sqrt(max(0,sum(FrontObjs.^2,2) + ...
            sum(FrontObjs.^2,2)' - 2*(FrontObjs*FrontObjs')));
        distance(1:numel(Front)+1:end) = inf;
        [nearestDist,nearestIdx] = min(distance,[],2);
        [~,order] = sort(nearestDist,'descend');
        for t = 1 : numel(order)
            if used >= budget
                break;
            end
            i = order(t);
            j = nearestIdx(i);
            middle = (FrontDecs(i,:)+FrontDecs(j,:))/2;
            candidate = Problem.Evaluation(middle);
            used = used + 1;
            Candidates = [Candidates,candidate]; %#ok<AGROW>
            if used < budget && sum(max(0,candidate.cons),2) > 0
                repaired = (middle+FrontDecs(i,:))/2;
                candidate = Problem.Evaluation(repaired);
                used = used + 1;
                Candidates = [Candidates,candidate]; %#ok<AGROW>
            end
        end
    end
end

