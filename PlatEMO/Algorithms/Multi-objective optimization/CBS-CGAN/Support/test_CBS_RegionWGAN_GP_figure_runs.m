function test_CBS_RegionWGAN_GP_figure_runs()
%TEST_CBS_REGIONWGAN_GP_FIGURE_RUNS Smoke test multi-run capture and plots.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    outDir = fullfile(tempdir, ...
        ['cbs_region_wgan_figure_smoke_',char(java.util.UUID.randomUUID)]);
    cleanup = onCleanup(@()removeFolderIfPresent(outDir));
    Config = struct( ...
        'problemNames',"LIRCMOP5_BC", ...
        'N',20, ...
        'D',30, ...
        'maxFE',1200, ...
        'stageTargets',[400 800]);
    [Summary,Figures,RunFigures] = ...
        run_CBS_RegionWGAN_GP_figure_runs(outDir,2,1:2,Config);
    assert(height(Summary) == 2 && all(string(Summary.status) == "ok"));
    assert(all(double(Summary.finalFE) == Config.maxFE));
    assert(height(Figures) == 2 && all(isfile(Figures.figure_file)));
    assert(all(double(Figures.stage_count) == 2));
    assert(height(RunFigures) == 2 && ...
        all(isfile(RunFigures.figure_file)));
    assert(isfile(fullfile(outDir,'figure_manifest.csv')));
    assert(isfile(fullfile(outDir,'run_summary_figure_manifest.csv')));
    Config.renderOnly = true;
    [Summary2,Figures2,RunFigures2] = ...
        run_CBS_RegionWGAN_GP_figure_runs(outDir,2,1:2,Config);
    assert(height(Summary2) == 2 && height(Figures2) == 2 && ...
        height(RunFigures2) == 2);
    clear cleanup
    removeFolderIfPresent(outDir);
end

function removeFolderIfPresent(folder)
    if isfolder(folder)
        rmdir(folder,'s');
    end
end
