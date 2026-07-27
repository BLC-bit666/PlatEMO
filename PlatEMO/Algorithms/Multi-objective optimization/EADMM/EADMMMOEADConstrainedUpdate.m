function Population = EADMMMOEADConstrainedUpdate(Population,Candidate,scope,W,Z)
% Neighbour replacement under EADMM's four constrained MOEA/D rules.

    candidateViolation = sum(~(Candidate.con<=0));
    parentViolation    = sum(~(Population(scope).cons<=0),2);
    replace            = false(numel(scope),1);
    replace(candidateViolation==0 & parentViolation>0) = true;
    replace(candidateViolation>0 & parentViolation>candidateViolation) = true;

    equal = parentViolation == candidateViolation;
    if any(equal)
        selected = find(equal);
        gOld = EADMMMOEADAggregation(Population(scope(selected)).objs,W(scope(selected),:),Z);
        gNew = EADMMMOEADAggregation(repmat(Candidate.obj,numel(selected),1),W(scope(selected),:),Z);
        replace(selected) = gNew <= gOld;
    end
    Population(scope(replace)) = Candidate;
end
