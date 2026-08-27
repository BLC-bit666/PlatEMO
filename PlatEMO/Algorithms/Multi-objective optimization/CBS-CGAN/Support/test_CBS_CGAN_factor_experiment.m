function test_CBS_CGAN_factor_experiment()
%TEST_CBS_CGAN_FACTOR_EXPERIMENT Verify paired sequential attribution.

    problems = ["DASCMOP1_BC";"DASCMOP9_BC";"LIRCMOP5_BC"];
    runs = (1:2)';
    count = 3*numel(problems)*numel(runs);
    armColumn = zeros(count,1);
    problemColumn = strings(count,1);
    runColumn = zeros(count,1);
    igd100 = zeros(count,1);
    igd200 = zeros(count,1);
    next = 0;
    for arm = 0 : 2
        for problem = reshape(problems,1,[])
            for run = reshape(runs,1,[])
                base = 1+0.01*run;
                if arm == 0
                    igd = base;
                elseif arm == 1
                    igd = 0.96*base;
                else
                    igd = 0.96^2*base;
                end
                next = next+1;
                armColumn(next) = arm;
                problemColumn(next) = problem;
                runColumn(next) = run;
                igd100(next) = 1.1*igd;
                igd200(next) = igd;
            end
        end
    end
    RunSummary = table(armColumn,problemColumn,runColumn,igd100,igd200, ...
        repmat("ok",count,1),'VariableNames', ...
        {'arm','problem','run','IGD100K','IGD200K','status'});
    [Effects,Pairwise] = analyze_CBS_CGAN_factor_experiment(RunSummary);
    assert(height(Effects) == 3 && all(Effects.pass) && ...
        all(abs(Effects.gmeanRatio200K-[0.96;0.96;0.96^2]) < 1e-12));
    assert(height(Pairwise) == 18 && ...
        isequal(unique(Pairwise.effect,'stable'), ...
        ["generation_screening";"utilization";"end_to_end"]));
    fprintf('CBS CGAN sequential attribution analysis passed.\n');
end
