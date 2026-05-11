function [ConfigSummary,RawResults,ConfigTable] = run_PRBCCMO_t_igd_refinement(runs,workerCount,N,maxFE,outDir)
% Second-round PRBCCMO_t IGD refinement over all DASCMOP_BC/LIRCMOP_BC problems.

    if nargin < 1 || isempty(runs)
        runs = 5;
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 10;
    end
    if nargin < 3 || isempty(N)
        N = 100;
    end
    if nargin < 4 || isempty(maxFE)
        maxFE = 200000;
    end
    if nargin < 5 || isempty(outDir)
        rootDir = fileparts(which('platemo'));
        outDir = fullfile(rootDir,'Data','PRBCCMO_t', ...
            ['igd_refinement_',char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end

    ConfigTable = refinementConfigTable();
    problemNames = allBCProblems();
    [ConfigSummary,RawResults,ConfigTable] = run_PRBCCMO_t_igd_rangefinding( ...
        runs,workerCount,N,maxFE,outDir,problemNames,ConfigTable);
end

function problemNames = allBCProblems()
    das = arrayfun(@(i)sprintf('DASCMOP%d_BC',i),1:9,'UniformOutput',false);
    lir = arrayfun(@(i)sprintf('LIRCMOP%d_BC',i),1:14,'UniformOutput',false);
    problemNames = [das,lir];
end

function ConfigTable = refinementConfigTable()
    Names = {'config_id','name','hidden','epoch','lr','betaB','etaB','Tretrain','Gstart'};
    Rows = { ...
        1,  "top01_lhs24",          26,  56, 0.0786871753946444,  2.25, 0.38, 51, 141; ...
        2,  "top02_lhs02",         136, 193, 0.0605448397648544,  3.50, 0.92, 24, 150; ...
        3,  "top03_lhs10",         138, 122, 0.0962647971082955,  3.75, 0.61, 39, 133; ...
        4,  "top04_lhs14",          40,  25, 0.0034416925205873,  6.50, 0.18, 44, 161; ...
        5,  "top05_lhs20",          77, 146, 0.0071337112272430, 10.75, 0.99, 58, 109; ...
        6,  "top06_lhs06",          88, 185, 0.0048677732528965,  5.75, 0.68, 26,  37; ...
        7,  "top07_lhs09",          79, 100, 0.0027201568408858,  8.50, 0.10, 40, 185; ...
        8,  "local_best_eta",       32,  60, 0.0700000000000000,  2.50, 0.45, 50, 140; ...
        9,  "local_best_mid",       40,  80, 0.0500000000000000,  4.00, 0.10, 20, 150; ...
        10, "highnet_regularized", 120, 140, 0.0300000000000000,  4.00, 0.75, 40, 150; ...
        11, "midnet_late",          80, 100, 0.0150000000000000,  6.00, 0.45, 60, 170; ...
        12, "compact_boundary",     48,  50, 0.0100000000000000,  6.00, 0.20, 45, 160};

    ConfigTable = cell2table(Rows,'VariableNames',Names);
end
