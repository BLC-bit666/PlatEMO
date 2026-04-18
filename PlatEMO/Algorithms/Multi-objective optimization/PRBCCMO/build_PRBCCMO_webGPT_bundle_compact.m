function outFile = build_PRBCCMO_webGPT_bundle_compact(suiteDir,outFile,targetMB)
% Build a compact PRBCCMO webGPT bundle around a target size.

    if nargin < 1 || isempty(suiteDir)
        rootDir = fileparts(which('platemo'));
        suiteDir = resolveLatestSuiteDir(rootDir);
    end
    if nargin < 2 || isempty(outFile)
        outFile = fullfile(suiteDir, ...
            ['PRBCCMO_webGPT_bundle_compact_' datestr(now,'yyyymmdd') '.txt']);
    end
    if nargin < 3 || isempty(targetMB)
        targetMB = 20;
    end

    suiteDir = char(string(suiteDir));
    outFile  = char(string(outFile));
    assert(isfolder(suiteDir), ...
        'build_PRBCCMO_webGPT_bundle_compact:MissingSuiteDir', ...
        'Suite directory not found: %s', suiteDir);

    rootDir = fileparts(which('platemo'));
    algDir  = fullfile(rootDir,'Algorithms','Multi-objective optimization','PRBCCMO');
    benchmarkFile = fullfile(suiteDir,'benchmark_all.csv');
    assert(isfile(benchmarkFile), ...
        'build_PRBCCMO_webGPT_bundle_compact:MissingBenchmarkCsv', ...
        'benchmark_all.csv not found in %s', suiteDir);

    Benchmark = readtable(benchmarkFile,'TextType','string');
    mode = resolveBundleMode(targetMB);
    targetBytes = floor(targetMB * 1024 * 1024);
    reserveBytes = 96 * 1024;

    [sourceFiles,suiteFiles] = selectStaticFiles(algDir,suiteDir,mode);
    sourceFiles = existingFiles(sourceFiles);
    suiteFiles  = existingFiles(suiteFiles);

    fixedTraceFiles = selectFixedTraceFiles(unique(Benchmark.analysis_folder,'stable'),mode);
    fixedTraceFiles = existingFiles(fixedTraceFiles);
    mlpCandidates   = selectMlpCandidates(Benchmark,mode);

    fid = fopen(outFile,'wt');
    assert(fid >= 0, ...
        'build_PRBCCMO_webGPT_bundle_compact:OpenFailed', ...
        'Cannot open output file: %s', outFile);
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

    writePreamble(fid,rootDir,suiteDir,outFile,targetMB,Benchmark,sourceFiles,suiteFiles,mode);
    appendFileGroup(fid,sourceFiles);
    appendFileGroup(fid,suiteFiles);
    appendFileGroup(fid,fixedTraceFiles);

    [selectedMlp,selectedMlpRun,selectedMlpProblem] = appendSelectedFiles( ...
        fid,mlpCandidates,Benchmark,targetBytes,reserveBytes);

    selectedGen = strings(0,1);
    selectedGenRun = strings(0,1);
    selectedGenProblem = strings(0,1);
    if mode == "compact"
        generationFiles = orderedTraceFiles(Benchmark,'generation_summary.csv');
        [selectedGen,selectedGenRun,selectedGenProblem] = appendSelectedFiles( ...
            fid,generationFiles,Benchmark,targetBytes,reserveBytes);
    end

    writeTailSummary(fid,targetMB,targetBytes,ftell(fid),Benchmark, ...
        mode,selectedMlp,selectedMlpRun,selectedMlpProblem, ...
        selectedGen,selectedGenRun,selectedGenProblem);
end

function mode = resolveBundleMode(targetMB)
    if targetMB < 5.5
        mode = "mini";
    elseif targetMB <= 10.5
        mode = "ultra";
    else
        mode = "compact";
    end
end

function suiteDir = resolveLatestSuiteDir(rootDir)
    suites = dir(fullfile(rootDir,'Data','PRBCCMO_t_suite_*'));
    suites = suites([suites.isdir]);
    assert(~isempty(suites), ...
        'build_PRBCCMO_webGPT_bundle_compact:NoSuiteDir', ...
        'No PRBCCMO_t_suite_* directory found under %s/Data', rootDir);
    [~,ord] = sort({suites.name});
    suiteDir = fullfile(suites(ord(end)).folder,suites(ord(end)).name);
end

function files = existingFiles(files)
    files = cellfun(@char,files,'UniformOutput',false);
    mask = cellfun(@isfile,files);
    files = files(mask);
end

function [sourceFiles,suiteFiles] = selectStaticFiles(algDir,suiteDir,mode)
    sourceFiles = { ...
        fullfile(algDir,'PRBCCMO.m')
        fullfile(algDir,'PRBCCMO_t.m')
        fullfile(algDir,'test_PRBCCMO_semantics.m')
        fullfile(algDir,'test_PRBCCMO_t_metrics.m')
        fullfile(algDir,'summarize_PRBCCMO_t_run.m')
        fullfile(algDir,'summarize_PRBCCMO_t_data.m')
        fullfile(algDir,'benchmark_PRBCCMO_t_suite.m')
        };
    if mode == "compact"
        sourceFiles{end+1,1} = fullfile(algDir,'build_PRBCCMO_webGPT_bundle_compact.m');
    end

    suiteFiles = { ...
        fullfile(suiteDir,'benchmark_all.csv')
        fullfile(suiteDir,'run_summary.csv')
        fullfile(suiteDir,'problem_summary.csv')
        fullfile(suiteDir,'family_summary.csv')
        fullfile(suiteDir,'problem_performance.csv')
        fullfile(suiteDir,'family_performance.csv')
        };
    if mode == "compact"
        suiteFiles = [suiteFiles; { ...
            fullfile(suiteDir,'chunk1.csv')
            fullfile(suiteDir,'chunk2.csv')
            fullfile(suiteDir,'chunk3.csv')
            fullfile(suiteDir,'chunk4.csv')
            fullfile(suiteDir,'chunk5.csv')
            fullfile(suiteDir,'chunk6.csv')}];
    end
end

function files = selectFixedTraceFiles(runFolders,mode)
    files = {};
    if mode ~= "compact"
        return;
    end
    for i = 1 : numel(runFolders)
        folder = char(runFolders(i));
        files = [files; { ...
            fullfile(folder,'run_meta.csv')
            fullfile(folder,'boundary_event.csv')
            fullfile(folder,'archive_members.csv')}]; %#ok<AGROW>
    end
end

function files = selectMlpCandidates(Benchmark,mode)
    switch mode
        case "compact"
            runFolders = unique(Benchmark.analysis_folder,'stable');
            files = strings(numel(runFolders),1);
            for i = 1 : numel(runFolders)
                files(i) = fullfile(runFolders(i),'mlp_events.csv');
            end
        case "ultra"
            runFolders = unique(Benchmark.analysis_folder,'stable');
            files = strings(numel(runFolders),1);
            for i = 1 : numel(runFolders)
                files(i) = fullfile(runFolders(i),'mlp_events.csv');
            end
        otherwise
            files = orderedTraceFiles(Benchmark,'mlp_events.csv');
    end
end

function ordered = orderedTraceFiles(Benchmark,fileName)
    ordered = strings(0,1);
    problems = unique(Benchmark.problem,'stable');
    maxRun = max(Benchmark.run);
    for r = 1 : maxRun
        for i = 1 : numel(problems)
            mask = Benchmark.problem == problems(i) & Benchmark.run == r;
            if any(mask)
                folder = Benchmark.analysis_folder(find(mask,1,'first'));
                ordered(end+1,1) = fullfile(folder,fileName); %#ok<AGROW>
            end
        end
    end
    ordered = ordered(isfile(cellstr(ordered)));
end

function [selectedFiles,selectedRuns,selectedProblems] = appendSelectedFiles( ...
    fid,candidates,Benchmark,targetBytes,reserveBytes)

    selectedFiles = strings(0,1);
    selectedRuns = strings(0,1);
    selectedProblems = strings(0,1);

    for i = 1 : numel(candidates)
        nextFile = char(candidates(i));
        if ftell(fid) + estimateEmbeddedBytes(nextFile) + reserveBytes > targetBytes
            continue;
        end
        appendSingleFile(fid,nextFile);
        selectedFiles(end+1,1) = string(nextFile); %#ok<AGROW>
        [runFolder,~,~] = fileparts(nextFile);
        selectedRuns(end+1,1) = string(runFolder); %#ok<AGROW>
        match = Benchmark.analysis_folder == string(runFolder);
        if any(match)
            selectedProblems(end+1,1) = Benchmark.problem(find(match,1,'first')); %#ok<AGROW>
        else
            selectedProblems(end+1,1) = "<unknown>"; %#ok<AGROW>
        end
    end
end

function writePreamble(fid,rootDir,suiteDir,outFile,targetMB,Benchmark,sourceFiles,suiteFiles,mode)
    nl = newline;
    fullBundle = fullfile(suiteDir,'PRBCCMO_webGPT_bundle_20260417.txt');
    fprintf(fid,'# PRBCCMO / PRBCCMO_t 精简打包材料%s%s',nl,nl);
    fprintf(fid,'生成时间：%s%s',datestr(now,'yyyy-mm-dd HH:MM:SS'),nl);
    fprintf(fid,'项目根目录：%s%s',rootDir,nl);
    fprintf(fid,'当前文件：%s%s',outFile,nl);
    fprintf(fid,'目标体积：约 %.1f MB%s',targetMB,nl);
    fprintf(fid,'完整版 bundle：%s%s%s',fullBundle,nl,nl);

    fprintf(fid,'## 一、精简策略%s%s',nl,nl);
    fprintf(fid,'该 compact bundle 保留以下内容：%s',nl);
    fprintf(fid,'1. PRBCCMO / PRBCCMO_t 算法源码全文与验证脚本全文。%s',nl);
    switch mode
        case "mini"
            fprintf(fid,'2. suite 级关键 CSV：benchmark_all / run_summary / family_summary / problem_summary / family_performance / problem_performance。%s',nl);
            fprintf(fid,'3. mlp_events.csv 改为按问题优先、run 轮转抽样，直到整体体积逼近目标上限。%s',nl);
            fprintf(fid,'4. 为了压到 5 MB 以下，省略 chunk1-6、run_meta、boundary_event、archive_members、generation_summary，以及未被选中的 mlp_events。%s',nl);
            fprintf(fid,'5. 这些省略的原始 trace 仍保留在各 analysis_folder 目录，以及更大的 compact/full bundle 中。%s%s',nl,nl);
        case "ultra"
            fprintf(fid,'2. suite 级关键 CSV：benchmark_all / run_summary / family_summary / problem_summary / family_performance / problem_performance。%s',nl);
            fprintf(fid,'3. 138 个 run 的全量 mlp_events.csv。%s',nl);
            fprintf(fid,'4. 为了逼近 10 MB，省略 chunk1-6 与 run_meta / boundary_event / archive_members / generation_summary。%s',nl);
            fprintf(fid,'5. 这些省略的原始 trace 仍保留在各 analysis_folder 目录，以及更大的 compact/full bundle 中。%s%s',nl,nl);
        otherwise
            fprintf(fid,'2. suite 级完整 CSV：benchmark_all / run_summary / family_summary / problem_summary / family_performance / problem_performance / chunk1-6。%s',nl);
            fprintf(fid,'3. 138 个 run 的全量 mlp_events.csv、run_meta.csv、boundary_event.csv、archive_members.csv。%s',nl);
            fprintf(fid,'4. generation_summary.csv 只按“问题优先、run 轮转”的顺序加入，直到整体体积接近目标上限。%s',nl);
            fprintf(fid,'5. 完整原始 trace 仍保留在各 analysis_folder 对应目录，以及完整版 bundle 中。%s%s',nl,nl);
    end

    fprintf(fid,'这样压缩的目的不是改动实验结论，而是减少重复原始表体积，同时保留：%s',nl);
    fprintf(fid,'- 核心源码%s',nl);
    fprintf(fid,'- 验证脚本%s',nl);
    fprintf(fid,'- suite 级关键结论%s',nl);
    switch mode
        case "mini"
            fprintf(fid,'- 跨问题轮转采样的 MLP 启动证据%s',nl);
            fprintf(fid,'- 一个小于 5 MB 的网页版可投喂版本%s%s',nl,nl);
        case "ultra"
            fprintf(fid,'- 138 个 run 关于 MLP 是否启动的全量事件证据%s',nl);
            fprintf(fid,'- 一个接近 10 MB 的网页版可投喂版本%s%s',nl,nl);
        otherwise
            fprintf(fid,'- 138 个 run 关于 MLP 是否启动的全量事件证据%s',nl);
            fprintf(fid,'- 每个问题至少优先覆盖到的 generation 级动态样本%s%s',nl,nl);
    end

    fprintf(fid,'## 二、核心实验结论%s%s',nl,nl);
    fprintf(fid,'1. 当前 traced suite 不能证明“MLP 能稳定拟合边界”。%s',nl);
    fprintf(fid,'2. 这轮 138 个 run 的主要问题不是后续 BCE 或 calibration，而是更靠前的边界分支 bootstrap。%s',nl);
    fprintf(fid,'3. 在 suite 汇总里，max_seed_b_size / max_b_size / max_boundary_selected / max_train_size / max_train_can_train / max_model_trained 全部为 0。%s',nl);
    fprintf(fid,'4. boundary_budget 在绝大多数代存在，但 boundary_selected 始终为 0，说明边界预算被分配了，边界样本却没有成功进入链路。%s',nl);
    fprintf(fid,'5. 因此“MLP 是否启动”在这轮实验中的答案是：没有启动；“核心创新点是否得到验证”的答案是：没有。%s%s',nl,nl);

    fprintf(fid,'## 三、源码与 CSV 范围%s%s',nl,nl);
    fprintf(fid,'算法源码文件数：%d%s',numel(sourceFiles),nl);
    fprintf(fid,'suite 汇总 CSV 文件数：%d%s',numel(suiteFiles),nl);
    fprintf(fid,'总 run 数：%d%s%s',height(Benchmark),nl,nl);
end

function appendFileGroup(fid,files)
    for i = 1 : numel(files)
        appendSingleFile(fid,files{i});
    end
end

function appendSingleFile(fid,filePath)
    sep = repmat('=',1,60);
    fprintf(fid,'\n\n%s\nFILE: %s\n%s\n\n',sep,filePath,sep);
    txt = fileread(filePath);
    fwrite(fid,txt,'char');
    if isempty(txt) || txt(end) ~= newline
        fprintf(fid,'\n');
    end
end

function bytes = estimateEmbeddedBytes(filePath)
    info = dir(filePath);
    sep = repmat('=',1,60);
    header = sprintf('\n\n%s\nFILE: %s\n%s\n\n',sep,filePath,sep);
    bytes = info.bytes + numel(header) + 2;
end

function writeTailSummary(fid,targetMB,targetBytes,actualChars,Benchmark, ...
    mode,selectedMlp,selectedMlpRun,selectedMlpProblem, ...
    selectedGen,selectedGenRun,selectedGenProblem)

    %#ok<INUSD>
    nl = newline;
    fprintf(fid,'写入阶段字符计数：%d%s',actualChars,nl);
    fprintf(fid,'目标上限：%d bytes (%.2f MB)%s',targetBytes,targetBytes/1024/1024,nl);
    fprintf(fid,'实际文件字节大小请以生成后文件系统显示或 wc -c 为准。%s',nl);

    switch mode
        case "mini"
            fprintf(fid,'%s## 四、mlp_events 抽样结果%s%s',nl,nl);
            fprintf(fid,'选择规则：先覆盖每个问题的 run=1，再覆盖每个问题的 run=2，如此轮转，直到接近约 %.1f MB 的目标体积。%s',targetMB,nl);
            fprintf(fid,'纳入的 mlp_events.csv 数量：%d / %d%s%s',numel(selectedMlp),height(Benchmark),nl,nl);
            writePerProblemCounts(fid,'mlp_events',Benchmark.problem,selectedMlpProblem);
            writeSelectedPaths(fid,'mlp_events.csv',selectedMlp);
            skipped = setdiff(unique(Benchmark.analysis_folder,'stable'),unique(selectedMlpRun,'stable'),'stable');
            fprintf(fid,'%s未内嵌 mlp_events.csv 的 run 数：%d%s',nl,numel(skipped),nl);
            fprintf(fid,'这些 run 的完整 trace 仍在各 analysis_folder 目录、10MB 版 compact bundle 与完整版 bundle 中。%s',nl);
        case "ultra"
            fprintf(fid,'%s## 四、精简结果%s%s',nl,nl);
            fprintf(fid,'该档位为了逼近约 %.1f MB，不再内嵌 generation_summary.csv。%s',targetMB,nl);
            fprintf(fid,'保留重点转为：源码全文 + suite 关键汇总 CSV + 138 个 run 的 mlp_events.csv。%s',nl);
            fprintf(fid,'若网页版 GPT 需要逐代动态，请回到更大的 compact/full bundle。%s',nl);
        otherwise
            fprintf(fid,'%s## 四、generation_summary 选择结果%s%s',nl,nl);
            fprintf(fid,'选择规则：先覆盖每个问题的 run=1，再覆盖每个问题的 run=2，如此轮转，直到接近约 %.1f MB 的目标体积。%s',targetMB,nl);
            fprintf(fid,'纳入的 generation_summary.csv 数量：%d / %d%s%s',numel(selectedGen),height(Benchmark),nl,nl);
            writePerProblemCounts(fid,'generation_summary',Benchmark.problem,selectedGenProblem);
            writeSelectedPaths(fid,'generation_summary.csv',selectedGen);
            skipped = setdiff(unique(Benchmark.analysis_folder,'stable'),unique(selectedGenRun,'stable'),'stable');
            fprintf(fid,'%s未内嵌 generation_summary.csv 的 run 数：%d%s',nl,numel(skipped),nl);
            fprintf(fid,'这些 run 的完整 raw trace 仍在各 analysis_folder 目录，以及完整版 bundle 中。%s',nl);
    end

    fprintf(fid,'%s## 五、使用建议%s%s',nl,nl);
    fprintf(fid,'如果网页版 GPT 需要：%s',nl);
    fprintf(fid,'- 查看算法实现：直接阅读本文件中的源码全文。%s',nl);
    fprintf(fid,'- 验证 suite 级结论：优先看 benchmark_all.csv / run_summary.csv / family_summary.csv / problem_summary.csv。%s',nl);
    switch mode
        case "mini"
            fprintf(fid,'- 追踪 MLP 是否启动：先看本文件已内嵌的抽样 mlp_events.csv；若还不够，再回到 10MB/20MB/完整版。%s',nl);
            fprintf(fid,'- 查看逐代动态：回到 20MB compact bundle 或完整版 bundle。%s',nl);
        case "ultra"
            fprintf(fid,'- 追踪 MLP 是否启动：看 138 个 run 的 mlp_events.csv。%s',nl);
            fprintf(fid,'- 查看逐代动态：回到 20MB compact bundle 或完整版 bundle。%s',nl);
        otherwise
            fprintf(fid,'- 追踪 MLP 是否启动：看 138 个 run 的 mlp_events.csv。%s',nl);
            fprintf(fid,'- 查看逐代动态：先看本文件已内嵌的 generation_summary.csv；若仍不够，再回到完整版 bundle 或 analysis_folder 原目录。%s',nl);
    end
end

function writePerProblemCounts(fid,label,allProblems,selectedProblems)
    nl = newline;
    if isempty(selectedProblems)
        fprintf(fid,'按问题统计的 %s 纳入数：0%s',label,nl);
        return;
    end
    problems = unique(allProblems,'stable');
    fprintf(fid,'按问题统计的 %s 纳入数：%s',label,nl);
    for i = 1 : numel(problems)
        count = sum(selectedProblems == problems(i));
        fprintf(fid,'- %s: %d%s',problems(i),count,nl);
    end
end

function writeSelectedPaths(fid,label,selectedFiles)
    nl = newline;
    fprintf(fid,'%s纳入的 %s 绝对路径：%s',nl,label,nl);
    for i = 1 : numel(selectedFiles)
        fprintf(fid,'- %s%s',selectedFiles(i),nl);
    end
end
