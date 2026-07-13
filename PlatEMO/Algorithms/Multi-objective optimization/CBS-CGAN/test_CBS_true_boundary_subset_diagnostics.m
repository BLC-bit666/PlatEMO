function test_CBS_true_boundary_subset_diagnostics()
%TEST_CBS_TRUE_BOUNDARY_SUBSET_DIAGNOSTICS Verify shared projections exactly.

    PF = [linspace(0,1,101)',linspace(1,0,101)'];
    Obj = [0.10,0.90;0.25,0.77;0.60,0.43;0.85,0.20];
    Con = [0;1;0;0];
    Masks = [true,false;true,true;true,false;true,true];

    Batched = RunRegionGAN_RC('trueboundarysubsetdiagnostics', ...
        Obj,Con,PF,Masks,struct());
    for i = 1 : size(Masks,2)
        Expected = RunRegionGAN_RC('trueboundarydiagnostics', ...
            Obj(Masks(:,i),:),Con(Masks(:,i),:),PF,struct());
        assertMetricEqual(Batched(i),Expected);
    end

    Empty = RunRegionGAN_RC('trueboundarysubsetdiagnostics', ...
        Obj,Con,PF,false(size(Obj,1),1),struct());
    assert(all(structfun(@(x)isnan(x),Empty)));

    didThrow = false;
    try
        RunRegionGAN_RC('trueboundarysubsetdiagnostics', ...
            Obj,Con,PF,true(size(Obj,1)-1,1),struct());
    catch ME
        didThrow = strcmp(ME.identifier,'CBSRegionGAN:SubsetRowMismatch');
    end
    assert(didThrow,'Mismatched subset masks must be rejected.');
    fprintf('CBS true-boundary subset diagnostics test passed.\n');
end

function assertMetricEqual(Actual,Expected)
    names = fieldnames(Expected);
    for i = 1 : numel(names)
        a = Actual.(names{i});
        b = Expected.(names{i});
        assert(isequaln(a,b), ...
            'Subset metric %s differs from the standalone diagnostic.', ...
            names{i});
    end
end
