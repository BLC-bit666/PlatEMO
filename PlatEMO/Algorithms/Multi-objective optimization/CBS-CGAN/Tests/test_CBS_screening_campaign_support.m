function test_CBS_screening_campaign_support()
%TEST_CBS_SCREENING_CAMPAIGN_SUPPORT Verify protocol and NaN-safe analysis.

    repoRoot = fileparts(which('platemo'));
    addCBSPaths(repoRoot);
    Protocol = CBS_CGAN_screening_protocol(repoRoot,10);
    assert(Protocol.nWorker == 10 && Protocol.popSize == 100 && ...
        Protocol.maxFE == 2e5 && Protocol.saveNum == 20 && ...
        isequal(Protocol.runs,1:5) && numel(Protocol.problems) == 5 && ...
        isequal(Protocol.baseParameters,[0 5 2 5 0 0 0 0 1]));

    runner = string(fileread(which('run_CBS_CGAN_screening_campaign')));
    stages = ["E1_keep_anchor","E2_pair_directions", ...
        "E3_all_feasible","E4_cap10","E5_split_gate", ...
        "E5b_keep_with_split","E6_balanced_batch", ...
        "E7_memory_parent","E8_global_critic"];
    positions = zeros(size(stages));
    for i = 1 : numel(stages)
        found = strfind(runner,stages(i));
        assert(~isempty(found));
        positions(i) = found(1);
    end
    assert(all(diff(positions) > 0) && ...
        contains(runner,"parforOptions") && ...
        contains(runner,"'SubrangeSize',1") && ...
        contains(runner,"platemo('algorithm',specification") && ...
        contains(runner,"resultComplete"));

    testNaNDeletionAndIndependentComparison();
    fprintf('CBS sequential-screening campaign support tests passed.\n');
end

function testNaNDeletionAndIndependentComparison()
    tempRoot = tempname;
    mkdir(tempRoot);
    cleanup = onCleanup(@()rmdir(tempRoot,'s'));
    algorithms = ["Control","Candidate"];
    problems = ["DASCMOP1_BC","LIRCMOP10_BC"];
    runs = 1:5;
    for algorithm = algorithms
        mkdir(fullfile(tempRoot,'Data',algorithm));
    end
    for problem = problems
        for run = runs
            writeSyntheticRun(tempRoot,"Control",problem,run,[1 1]);
            scores = [0.95 0.95];
            if problem == "LIRCMOP10_BC" && run == 5
                scores(2) = NaN;
            end
            writeSyntheticRun(tempRoot,"Candidate",problem,run,scores);
        end
    end
    [Candidates,Problems,Runs] = analyze_CBS_CGAN_screening_stage( ...
        tempRoot,"Control","Candidate",problems,runs);
    assert(height(Candidates) == 1 && Candidates.allTasksComplete && ...
        Candidates.pass && Candidates.wins == 2 && ...
        Candidates.losses == 0 && Candidates.gmeanRatio200K <= 0.98);
    lir = Problems.problem == "LIRCMOP10_BC";
    assert(Problems.controlN(lir) == 5 && ...
        Problems.candidateN(lir) == 4 && ...
        Problems.candidateNaN(lir) == 1 && ...
        abs(Problems.ratio200K(lir)-0.95) < 1e-12);
    assert(all(Runs.status == "ok") && ...
        all(cellfun(@isstruct,Runs.audit)));
    clear cleanup;
end

function writeSyntheticRun(rootPath,algorithm,problem,run,scores)
    result = {1e5,[];2e5,[]}; %#ok<NASGU>
    metric = struct('IGD',double(scores), ...
        'CBSAudit',struct('unsafeTrainingEventRate',0.25)); %#ok<NASGU>
    name = sprintf('%s_%s_M2_D5_%d.mat',algorithm,problem,run);
    save(fullfile(rootPath,'Data',algorithm,name),'result','metric');
end
