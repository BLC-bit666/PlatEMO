function FrontNo = NAEMTCDPPVNDSort(PopObj,Value,epsilon,nSort)
% Nondominated sorting under the CDPPV relation.

    N       = size(PopObj,1);
    FrontNo = inf(1,N);
    Value   = Value(:);
    high    = Value >= epsilon;
    low     = Value < epsilon;

    % Rows dominate columns. NaN values belong to neither confidence group.
    dominates = (high & low') | (low & low' & Value > Value');
    index     = find(high);
    noWorse   = true(numel(index));
    better    = false(numel(index));
    for m = 1 : size(PopObj,2)
        objective = PopObj(index,m);
        noWorse = noWorse & objective <= objective';
        better  = better  | objective < objective';
    end
    dominates(index,index) = noWorse & better;

    dominatedCount = sum(dominates,1);
    current        = find(dominatedCount==0);
    front          = 1;
    assigned       = 0;
    while ~isempty(current) && assigned < min(nSort,N)
        FrontNo(current) = front;
        assigned = assigned + numel(current);
        dominatedCount = dominatedCount - sum(dominates(current,:),1);
        current = find(dominatedCount==0 & FrontNo==inf);
        front = front + 1;
    end
end
