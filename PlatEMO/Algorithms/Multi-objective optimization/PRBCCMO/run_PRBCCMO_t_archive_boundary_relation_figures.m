function Summary = run_PRBCCMO_t_archive_boundary_relation_figures(benchmarkCsv,outDir,problemNames,stages)
% Redraw archive-vs-true-boundary figures from existing PRBCCMO_t traces.

    if nargin < 1 || isempty(benchmarkCsv)
        rootDir = fileparts(which('platemo'));
        benchmarkCsv = fullfile(rootDir,'Data','PRBCCMO_t', ...
            'generation_region_current_20260501_latest','benchmark_200k.csv');
    end
    if nargin < 2 || isempty(outDir)
        rootDir = fileparts(which('platemo'));
        outDir = fullfile(rootDir,'Data','PRBCCMO_t','archive_boundary_relation_20260501');
    end
    if nargin < 3 || isempty(problemNames)
        problemNames = {'LIRCMOP7_BC','LIRCMOP8_BC','LIRCMOP9_BC', ...
            'LIRCMOP10_BC','LIRCMOP11_BC','LIRCMOP12_BC'};
    end
    if nargin < 4 || isempty(stages)
        stages = [10000 50000 100000 150000 180000 200000];
    end

    if ~isfolder(outDir)
        mkdir(outDir);
    end
    Benchmark = readtable(benchmarkCsv,'TextType','string');
    problemNames = string(problemNames(:));
    rows = cell(0,11);
    row = 0;
    for p = 1 : numel(problemNames)
        idx = find(Benchmark.problem == problemNames(p),1,'first');
        if isempty(idx)
            continue;
        end
        runFolder = char(Benchmark.analysis_folder(idx));
        T = readtable(fullfile(runFolder,'objective_snapshot.csv'),'TextType','string');
        Meta = readtable(fullfile(runFolder,'run_meta.csv'),'TextType','string');
        for targetFE = stages(:)'
            [generation,actualFE] = nearestGenerationAtFE(T,targetFE);
            Stage = T(T.generation == generation,:);
            bCount = sum(Stage.role == "archive_b");
            evidenceCount = sum(Stage.role == "boundary_evidence");
            figDir = fullfile(outDir,char(problemNames(p)),sprintf('run%d',double(Meta.run(1))));
            pngFile = plot_PRBCCMO_t_archive_boundary_relation(runFolder,figDir,generation);
            row = row + 1;
            rows(row,:) = {problemNames(p),double(Meta.run(1)),targetFE,actualFE, ...
                generation,bCount,evidenceCount,bCount > 0,evidenceCount > 0,string(runFolder),string(pngFile)};
        end
    end
    Summary = cell2table(rows,'VariableNames',{ ...
        'problem','run','target_fe','actual_fe','generation', ...
        'archive_b_count','boundary_evidence_count','has_archive_b','has_boundary_evidence','analysis_folder','png_file'});
    writetable(Summary,fullfile(outDir,'archive_boundary_relation_summary.csv'));
end

function [generation,actualFE] = nearestGenerationAtFE(T,targetFE)
    G = unique(double(T.generation),'stable');
    FE = zeros(numel(G),1);
    for i = 1 : numel(G)
        Stage = T(T.generation == G(i),:);
        FE(i) = median(double(Stage.fe),'omitnan');
    end
    [~,idx] = min(abs(FE - targetFE));
    generation = G(idx);
    actualFE = FE(idx);
end
