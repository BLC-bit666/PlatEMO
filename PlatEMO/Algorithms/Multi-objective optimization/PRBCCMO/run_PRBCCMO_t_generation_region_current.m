function [Benchmark,PlotSummary] = run_PRBCCMO_t_generation_region_current(problemNames,runId,N,maxFE,outDir,reuseBenchmark)
% Run current PRBCCMO_t and plot generation-region evidence for selected BC problems.

    if nargin < 1 || isempty(problemNames)
        problemNames = { ...
            'DASCMOP1_BC','DASCMOP2_BC','DASCMOP3_BC','DASCMOP4_BC','DASCMOP5_BC','DASCMOP6_BC', ...
            'LIRCMOP1_BC','LIRCMOP2_BC','LIRCMOP3_BC','LIRCMOP4_BC', ...
            'LIRCMOP7_BC','LIRCMOP8_BC','LIRCMOP9_BC','LIRCMOP10_BC','LIRCMOP11_BC','LIRCMOP12_BC'};
    end
    if nargin < 2 || isempty(runId)
        runId = 1;
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
            ['generation_region_current_',char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 6 || isempty(reuseBenchmark)
        reuseBenchmark = false;
    end

    problemNames = cellstr(string(problemNames(:)));
    stages = [10000 50000 100000 150000 180000 200000];
    stages = stages(stages <= maxFE);
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    benchCsv = fullfile(outDir,'benchmark_200k.csv');
    if reuseBenchmark && isfile(benchCsv)
        Benchmark = readtable(benchCsv,'TextType','string');
    else
        Benchmark = benchmark_PRBCCMO_t_suite(problemNames,runId,benchCsv,N,maxFE);
    end

    rows = cell(0,20);
    row = 0;
    for i = 1 : height(Benchmark)
        analysisFolder = char(Benchmark.analysis_folder(i));
        objectiveFile = fullfile(analysisFolder,'objective_snapshot.csv');
        metaFile = fullfile(analysisFolder,'run_meta.csv');
        if ~isfile(objectiveFile) || ~isfile(metaFile)
            continue;
        end
        T = readtable(objectiveFile,'TextType','string');
        Meta = readtable(metaFile,'TextType','string');
        for s = stages
            [generation,actualFE] = nearestGenerationAtFE(T,s);
            if isnan(generation)
                continue;
            end
            plotDir = fullfile(outDir,'figures',char(Benchmark.problem(i)),sprintf('run%d',Benchmark.run(i)));
            pngFile = plot_PRBCCMO_t_generation_region_evidence(analysisFolder,plotDir,generation);
            Check = checkGenerationPlotCondition(T,generation,char(Meta.problem(1)));
            row = row + 1;
            rows(row,:) = { ...
                string(Benchmark.problem(i)),Benchmark.run(i),s,actualFE,generation, ...
                Check.has_region_grid,Check.has_feasible_region,Check.has_infeasible_region, ...
                Check.pop_c_count,Check.pop_u_count,Check.archive_b_count,Check.boundary_evidence_count,Check.boundary_off_count, ...
                Check.has_pop_c,Check.has_pop_u,Check.has_archive_b,Check.has_boundary_evidence,Check.has_boundary_off, ...
                Check.qualified,string(pngFile)};
        end
    end

    PlotSummary = cell2table(rows,'VariableNames',{ ...
        'problem','run','target_fe','actual_fe','generation', ...
        'has_region_grid','has_feasible_region','has_infeasible_region', ...
        'pop_c_count','pop_u_count','archive_b_count','boundary_evidence_count','boundary_off_count', ...
        'has_pop_c','has_pop_u','has_archive_b','has_boundary_evidence','has_boundary_off', ...
        'qualified','png_file'});
    writetable(PlotSummary,fullfile(outDir,'generation_region_plot_condition_summary.csv'));

    if ~isempty(PlotSummary)
        ProblemSummary = groupsummary(PlotSummary,'problem','sum', ...
            {'qualified','has_archive_b','has_boundary_evidence','has_boundary_off'});
        writetable(ProblemSummary,fullfile(outDir,'generation_region_problem_condition_summary.csv'));
    end
end

function [generation,actualFE] = nearestGenerationAtFE(T,targetFE)
    if isempty(T) || ~any(string(T.Properties.VariableNames) == "generation") || ...
            ~any(string(T.Properties.VariableNames) == "fe")
        generation = NaN;
        actualFE = NaN;
        return;
    end
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

function Check = checkGenerationPlotCondition(T,generation,problemName)
    Stage = T(T.generation == generation,:);
    roles = string(Stage.role);
    Check = struct();
    Check.pop_c_count = sum(roles == "pop_c");
    Check.pop_u_count = sum(roles == "pop_u");
    Check.archive_b_count = sum(roles == "archive_b");
    Check.boundary_evidence_count = sum(roles == "boundary_evidence");
    Check.boundary_off_count = sum(roles == "boundary_off");
    Check.has_pop_c = Check.pop_c_count > 0;
    Check.has_pop_u = Check.pop_u_count > 0;
    Check.has_archive_b = Check.archive_b_count > 0;
    Check.has_boundary_evidence = Check.boundary_evidence_count > 0;
    Check.has_boundary_off = Check.boundary_off_count > 0;
    Check.has_region_grid = false;
    Check.has_feasible_region = false;
    Check.has_infeasible_region = false;
    try
        Problem = feval(problemName,'N',max(100,height(Stage)));
        PF = Problem.GetPF();
        Check.has_region_grid = iscell(PF) && numel(PF) >= 3 && ~isempty(PF{1}) && ~isempty(PF{2}) && ~isempty(PF{3});
        if ~Check.has_region_grid
            PF = buildLIRCMOP14RegionGridForCheck(problemName);
            Check.has_region_grid = iscell(PF) && numel(PF) >= 3 && ~isempty(PF{1}) && ~isempty(PF{2}) && ~isempty(PF{3});
        end
        if Check.has_region_grid
            Mask = isfinite(PF{3});
            Check.has_feasible_region = any(Mask(:));
            Check.has_infeasible_region = any(~Mask(:));
        end
    catch
        Check.has_region_grid = false;
    end
    Check.qualified = Check.has_region_grid && Check.has_feasible_region && Check.has_infeasible_region && ...
        Check.has_pop_c && Check.has_pop_u && Check.has_archive_b && Check.has_boundary_evidence && Check.has_boundary_off;
end

function PF = buildLIRCMOP14RegionGridForCheck(problemName)
    Tokens = regexp(char(problemName),'^LIRCMOP([1-4])_BC$','tokens','once');
    if isempty(Tokens)
        PF = {};
        return;
    end
    ProblemId = str2double(Tokens{1});
    [X,Y] = meshgrid(linspace(0.45,1.55,420),linspace(0.45,1.55,420));
    Mask = false(size(X));
    x1 = linspace(0,1,1200);
    if any(ProblemId == [1 3])
        base2 = 1 - x1.^2;
    else
        base2 = 1 - sqrt(x1);
    end
    if any(ProblemId == [3 4])
        x1 = x1(sin(20*pi*x1) >= 0.5);
        if any(ProblemId == [1 3])
            base2 = 1 - x1.^2;
        else
            base2 = 1 - sqrt(x1);
        end
    end
    dx = abs(X(1,2)-X(1,1));
    dy = abs(Y(2,1)-Y(1,1));
    TolX = max(0.002,0.55*dx);
    TolY = max(0.002,0.55*dy);
    for i = 1 : numel(x1)
        Mask = Mask | (X >= x1(i) + 0.5 - TolX & X <= x1(i) + 0.51 + TolX & ...
            Y >= base2(i) + 0.5 - TolY & Y <= base2(i) + 0.51 + TolY);
    end
    Z = nan(size(X));
    Z(Mask) = 0;
    PF = {X,Y,Z};
end
