function test_CBS_initdecs
%TEST_CBS_INITDECS Guards for the optional initDecs injection parameter.
%   The parameter must default to empty (native random initialization,
%   proven drift-free by the mainline fingerprint test), reproduce
%   bitwise under a fixed seed, actually change the trajectory when the
%   injected matrix changes, spend the budget exactly, and reject
%   malformed input.

    testDefaultEmpty();
    testInjectionDeterminism();
    testBadInputs();
    fprintf('CBS initDecs guards passed.\n');
end

function testDefaultEmpty()
    A = CBS_RegionWGAN_GP('save',0,'outputFcn',@quietOutput);
    assert(isempty(A.effectiveInitDecs()),'initDecs must default to empty.');
end

function testInjectionDeterminism()
    zeroPar = zeroParameter();
    rng(7,'twister');
    Probe = LIRCMOP8_BC('N',20,'maxFE',1200);
    DecsA = unifrnd(repmat(Probe.lower,20,1),repmat(Probe.upper,20,1));
    DecsB = unifrnd(repmat(Probe.lower,20,1),repmat(Probe.upper,20,1));
    a = runOnce(DecsA,zeroPar);
    b = runOnce(DecsA,zeroPar);
    c = runOnce(DecsB,zeroPar);
    assert(isequal(a,b),'same initDecs and seed must reproduce bitwise.');
    assert(~isequal(a,c),'different initDecs must change the trajectory.');
end

function decs = runOnce(InitDecs,zeroPar)
    rng(4242,'twister');
    Problem = LIRCMOP8_BC('N',20,'maxFE',1200);
    Algorithm = CBS_RegionWGAN_GP('save',0,'outputFcn',@quietOutput, ...
        'guideMode','off','boundarySearch','off','scoutMode','off', ...
        'blsWindow','late','blsFeed','off','parameter',zeroPar, ...
        'initDecs',InitDecs);
    Algorithm.Solve(Problem);
    assert(Problem.FE == 1200,'the budget must be spent exactly.');
    decs = Algorithm.result{end,2}.decs;
end

function testBadInputs()
    typed = false;
    try
        CBS_RegionWGAN_GP('save',0,'outputFcn',@quietOutput, ...
            'initDecs','nope');
    catch Error
        typed = strcmp(Error.identifier,'CBSRegionGAN:BadInitDecs');
    end
    assert(typed,'non-numeric initDecs must error at construction.');
    sized = false;
    try
        rng(4242,'twister');
        Problem = LIRCMOP8_BC('N',20,'maxFE',1200);
        Algorithm = CBS_RegionWGAN_GP('save',0,'outputFcn',@quietOutput, ...
            'guideMode','off','boundarySearch','off','scoutMode','off', ...
            'blsWindow','late','blsFeed','off','parameter',zeroParameter(), ...
            'initDecs',ones(3,4));
        Algorithm.Solve(Problem);
    catch Error
        sized = strcmp(Error.identifier,'CBSRegionGAN:BadInitDecsSize');
    end
    assert(sized,'size-mismatched initDecs must error at Solve.');
end

function zeroPar = zeroParameter()
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    zeroPar = {0,Defaults.zDim,Defaults.ganIter,Defaults.ganMiniBatch, ...
        Defaults.nCritic,Defaults.minGANTrainCount,Defaults.sampleSigma};
end

function quietOutput(~,~)
end
