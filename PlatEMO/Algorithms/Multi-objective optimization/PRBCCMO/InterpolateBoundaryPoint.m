function Dec = InterpolateBoundaryPoint(Problem,FeasibleDec,InfeasibleDec,Lambda)
% Build an encoding-aware point on a feasible-infeasible segment.

    Dec = FeasibleDec;
    RealIdx = find(Problem.encoding<=2);
    if ~isempty(RealIdx)
        Dec(RealIdx) = FeasibleDec(RealIdx) + Lambda*(InfeasibleDec(RealIdx)-FeasibleDec(RealIdx));
    end
    OtherIdx = setdiff(1:Problem.D,RealIdx);
    if ~isempty(OtherIdx) && Lambda > 0.5
        Dec(OtherIdx) = InfeasibleDec(OtherIdx);
    end
    Dec = Problem.CalDec(Dec);
end
