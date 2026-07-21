function run_CBS_mw_pilot()
%RUN_CBS_MW_PILOT Quick MW_BC pilot: A00 versus mainline, three seeds.
%   Runs MW1_BC..MW14_BC with N=100, maxFE=200000, runs 1:3, ten workers,
%   native save=2 files (IGD at ~100k and final). Results live under
%   <PlatEMO root>/mw_pilot_v1/Data/<class> so the formal Data/ campaigns
%   stay untouched. Resume is on: rerun after an interruption and finished
%   runs are reused. The cheap A00 arm runs first to fail fast.
    rootDir = fileparts(which('platemo'));
    assert(~isempty(rootDir),'platemo must be on the path.');
    problems   = "MW" + string((1:14)') + "_BC";
    algorithms = ["CBS_RegionWGAN_GP_A00","CBS_RegionWGAN_GP"];
    for a = 1:numel(algorithms)
        assert(~isempty(which(char(algorithms(a)))), ...
            '%s must be on the path.',algorithms(a));
    end
    for a = 1:numel(algorithms)
        outDir = fullfile(rootDir,'mw_pilot_v1','Data',char(algorithms(a)));
        run_CBS_RegionWGAN_GP_mainline(outDir,10,problems,100,[],200000, ...
            1:3,struct('resume',true,'algorithm',algorithms(a)));
    end
    disp('MW pilot complete.');
end
