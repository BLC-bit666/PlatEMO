function test_CCMO_GAN_BDG_archive_shrink_guard()
% Regression tests for CCMO-GAN-BDG archive shrink protection.

    rootDir = fileparts(which('platemo'));
    addpath(genpath(rootDir));

    assertStableHalfShrinkAcceptsNewArchive();
    assertTinyShrinkRetainsPreviousArchive();
end

function assertStableHalfShrinkAcceptsNewArchive()
    Problem = struct('N',100,'D',2,'M',2);
    W = [1 0;0 1;0.5 0.5];
    perRef = 20;
    nearTau = -1;
    Options = archiveOptions();

    previousN = 12;
    stableN = 6;
    [AF,AI] = syntheticPreviousArchive(Problem,previousN);
    [Population1,Offspring1] = syntheticNewBoundaryPairs(stableN);

    [NewAF,NewAI,Diag] = UpdateBoundaryArchive_BDG(Problem,AF,AI, ...
        Population1,Offspring1,[],[],W,perRef,nearTau,Options);

    assert(Diag.archive_pair_direction_keep_count == stableN, ...
        'test_CCMO_GAN_BDG_archive_shrink_guard:StableShrinkSetupFailed', ...
        'The stable-shrink fixture must rebuild a non-tiny archive with exactly %d direction-kept pairs.', ...
        stableN);
    assertArchivePairCount(NewAF,NewAI,stableN, ...
        'test_CCMO_GAN_BDG_archive_shrink_guard:StableHalfShrinkFrozen', ...
        'A stable 12-to-6 archive shrink must accept the rebuilt archive instead of freezing the previous archive.');
    assert(all(NewAF.objs(:) < 3), ...
        'test_CCMO_GAN_BDG_archive_shrink_guard:StableHalfShrinkKeptOldObjs', ...
        'A stable shrink must keep the new low-objective archive pairs.');
end

function assertTinyShrinkRetainsPreviousArchive()
    Problem = struct('N',100,'D',2,'M',2);
    W = [1 0;0 1;0.5 0.5];
    perRef = 20;
    nearTau = -1;
    Options = archiveOptions();

    previousN = 12;
    tinyN = 5;
    [AF,AI] = syntheticPreviousArchive(Problem,previousN);
    [Population1,Offspring1] = syntheticNewBoundaryPairs(tinyN);

    [NewAF,NewAI,Diag] = UpdateBoundaryArchive_BDG(Problem,AF,AI, ...
        Population1,Offspring1,[],[],W,perRef,nearTau,Options);

    assert(Diag.archive_pair_direction_keep_count == tinyN, ...
        'test_CCMO_GAN_BDG_archive_shrink_guard:TinyShrinkSetupFailed', ...
        'The tiny-shrink fixture must rebuild a tiny archive with exactly %d direction-kept pairs.', ...
        tinyN);
    assertArchivePairCount(NewAF,NewAI,previousN, ...
        'test_CCMO_GAN_BDG_archive_shrink_guard:TinyShrinkNotRetained', ...
        'A shrink below the tiny support threshold must retain the previous archive.');
    assert(isequal(NewAF.objs,AF.objs) && isequal(NewAI.objs,AI.objs), ...
        'test_CCMO_GAN_BDG_archive_shrink_guard:TinyShrinkChangedArchive', ...
        'Tiny shrink protection must retain the previous AF/AI objective rows.');
end

function Options = archiveOptions()
    Options = struct( ...
        'paretoFilterMode',"global_af_nd", ...
        'pairDirectionMode',"ai_dominates_af");
end

function [AF,AI] = syntheticPreviousArchive(Problem,n)
    t = (1:n)';
    AF = struct( ...
        'decs',[0.70 + 0.001*t,0.30 + 0.001*t], ...
        'objs',[10 + t,10 + t], ...
        'ref',ones(n,1), ...
        'score',zeros(n,1));
    AI = struct( ...
        'decs',[0.75 + 0.001*t,0.35 + 0.001*t], ...
        'objs',AF.objs - 1, ...
        'ref',ones(n,1), ...
        'score',zeros(n,1));
    assert(size(AF.decs,2) == Problem.D && size(AF.objs,2) == Problem.M, ...
        'test_CCMO_GAN_BDG_archive_shrink_guard:BadPreviousArchiveFixture', ...
        'Previous archive fixture dimensions must match the synthetic problem.');
end

function [Population1,Offspring1] = syntheticNewBoundaryPairs(n)
    t = (1:n)';
    FDecs = [0.10 + 0.01*t,0.20 + 0.01*t];
    IDecs = FDecs + 0.001;
    FObjs = [1 + 0.01*t,2 - 0.01*t];
    IObjs = FObjs - 0.1;
    Population1 = SOLUTION(FDecs,FObjs,zeros(n,1));
    Offspring1 = SOLUTION(IDecs,IObjs,ones(n,1));
end

function assertArchivePairCount(AF,AI,expectedCount,identifier,message)
    actualCount = min(size(AF.decs,1),size(AI.decs,1));
    assert(actualCount == expectedCount,identifier, ...
        '%s Expected %d pairs, got %d.',message,expectedCount,actualCount);
end
