function test_CBS_boundary_contract_review()
%TEST_CBS_BOUNDARY_CONTRACT_REVIEW Verify current boundary data contract.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));

    testDatasetBuildsReferenceObjectiveConditions();
    testDatasetBuildsReferenceOnlyConditions();
    testBoundaryMemorySchemaHasPairedEndpoints();
    fprintf('CBS boundary contract regressions passed.\n');
end

function testDatasetBuildsReferenceObjectiveConditions()
    [Problem,W,BMem,Samples] = fixture();
    Options = struct('conditionMode',"ref_y",'queryConditionBudget',1);

    [TrainX,TrainC,~,BMemOut,Info] = BuildBoundaryDataset_CBS( ...
        BMem,Samples,W,Problem,Options);

    expectedY = (BMem.y_b - Info.objMin)./Info.objSpan;
    assert(max(abs(TrainX - BMem.x_b),[],'all') <= 1e-12, ...
        'Dataset must train on boundary target decisions.');
    assert(isequal(size(TrainC),[2 4]), ...
        'ref_y conditions must contain reference and normalized objective columns.');
    assert(max(abs(TrainC - [W(BMem.ref,:),expectedY]),[],'all') <= 1e-12, ...
        'ref_y conditions must match reference plus normalized boundary objectives.');
    assert(isequal(sort(fieldnames(BMemOut)),sort(currentBoundaryFields())), ...
        'Boundary memory output must expose only the current paired endpoint schema.');
end

function testDatasetBuildsReferenceOnlyConditions()
    [Problem,W,BMem,Samples] = fixture();
    Options = struct('conditionMode',"ref_only",'queryConditionBudget',1);

    [~,TrainC,~,~,Info] = BuildBoundaryDataset_CBS( ...
        BMem,Samples,W,Problem,Options);

    assert(isequal(size(TrainC),[2 2]), ...
        'ref_only conditions must contain only reference coordinates.');
    assert(max(abs(TrainC - W(BMem.ref,:)),[],'all') <= 1e-12, ...
        'ref_only conditions must match the selected reference vectors.');
    assert(string(Info.condition_mode) == "ref_only", ...
        'Dataset diagnostics must record the selected condition mode.');
end

function testBoundaryMemorySchemaHasPairedEndpoints()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.51 0.49;0.49 0.51];
    X = makeDecisionRows(Problem,4);
    Y = [5.08 4.92;5.10 4.90;4.92 5.08;4.90 5.10];
    C = [0;1;0;1];
    Pop = SOLUTION(X,Y,C);
    Empty = Pop([]);
    Options = struct( ...
        'pairNeighborRefRadius',1, ...
        'maxCandidatePairsPerRef',1, ...
        'minBoundaryLength',2);

    [BMem,Diag] = UpdateBoundaryMemory_CBS([],Pop,Empty,Empty,Empty,W,Options);

    assert(Diag.bmem_count == 2 && Diag.finite_gap_count == 2, ...
        'Paired feasible/infeasible rows should form finite boundary memory.');
    assert(isequal(sort(fieldnames(BMem)),sort(currentBoundaryFields())), ...
        'Boundary memory must use the current paired endpoint schema.');
    assert(all(isfinite(BMem.gap)), ...
        'Boundary memory rows must have finite gap values.');
end

function [Problem,W,BMem,Samples] = fixture()
    rng(7,'twister');
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [1.0 0.0;0.5 0.5;0.0 1.0];
    Xb = makeDecisionRows(Problem,2);
    Xi = makeDecisionRows(Problem,2);
    BMem = struct( ...
        'ref',[1;2], ...
        'y_b',[1.0 2.0;2.0 1.0], ...
        'gap',[0.05;0.07], ...
        'x_b',Xb, ...
        'x_f',Xb, ...
        'y_f',[1.0 2.0;2.0 1.0], ...
        'x_i',Xi, ...
        'y_i',[1.1 1.9;1.9 1.1]);
    Samples = SOLUTION(Xb,BMem.y_b,zeros(2,1));
end

function X = makeDecisionRows(Problem,n)
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    X = repmat(lower,n,1) + rand(n,Problem.D).*repmat(upper-lower,n,1);
end

function fields = currentBoundaryFields()
    fields = {'ref';'y_b';'gap';'x_b';'x_f';'y_f';'x_i';'y_i'};
end
