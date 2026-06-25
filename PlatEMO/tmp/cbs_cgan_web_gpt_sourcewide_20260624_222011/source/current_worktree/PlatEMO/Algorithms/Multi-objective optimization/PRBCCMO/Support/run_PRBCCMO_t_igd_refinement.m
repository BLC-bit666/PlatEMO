function [ConfigSummary,RawResults,ConfigTable] = run_PRBCCMO_t_igd_refinement(runs,workerCount,N,maxFE,outDir)
% Run the current PRBCCMO_t baseline over all DASCMOP_BC/LIRCMOP_BC problems.

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
        1, "current_baseline", 64, 200, 0.001, 3.0, 0.10, 10, 0};

    ConfigTable = cell2table(Rows,'VariableNames',Names);
end
