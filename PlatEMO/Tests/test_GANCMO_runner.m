function test_GANCMO_runner
% 复制真实脚本，仅缩小顶部配置；验证更换算法/问题、保存和续跑。
    root = fileparts(fileparts(mfilename('fullpath')));
    source = fileread(fullfile(root,'test.m'));
    folder = tempname; mkdir(folder);
    oldPath = path; oldFolder = pwd;
    cleanup = onCleanup(@()restore(oldPath,oldFolder,folder));
    assert(isempty(gcp('nocreate')),'请在没有并行池的独立 MATLAB 会话中运行。');
    cases = {'GANCMO','LIRCMOP13_BC',200;'NSGAII','DTLZ2',4000};
    for k = 1:size(cases,1)
        dataDir = fullfile(folder,cases{k,1}); script = fullfile(folder,'run_case.m');
        code = regexprep(source,'(?m)^nWorker\s*=[^\n]*','nWorker = 2;');
        budget = cases{k,3};
        code = regexprep(code,'(?m)^maxFE\s*=[^\n]*',sprintf('maxFE = %d;',budget));
        code = regexprep(code,'(?m)^runs\s*=[^\n]*','runs = 1:2;');
        code = regexprep(code,'(?m)^algName\s*=[^\n]*',sprintf('algName = ''%s'';',cases{k,1}));
        code = regexprep(code,'proNames\s*=\s*\{[\s\S]*?\};',sprintf('proNames = {''%s''};',cases{k,2}));
        code = regexprep(code,'(?m)^rootPath = [^\n]*',sprintf('rootPath = ''%s'';',root));
        code = regexprep(code,'dataDir = [\s\S]*?;',sprintf('dataDir = ''%s'';',dataDir));
        writelines(code,script); runScript(script);
        files = dir(fullfile(dataDir,'*.mat')); assert(numel(files)==2);
        for j = 1:numel(files)
            saved = load(fullfile(files(j).folder,files(j).name));
            assert(strcmp(saved.meta.config.algorithm,cases{k,1}) && strcmp(saved.meta.problem,cases{k,2}));
            assert(saved.meta.FE==budget && saved.meta.threads==1 && saved.result{end,1}==budget);
            assert(isequal(size(saved.result),[1 2]) && ~isfield(saved.metric,'processIGD'));
            assert(size(saved.result{end,2}.decs,2)==30 && isscalar(saved.metric.IGD));
        end
        runScript(script);
        assert(isequal(files,dir(fullfile(dataDir,'*.mat'))),'续跑不应改写已完成结果。');
        assert(isempty(gcp('nocreate')),'入口创建的进程池应已关闭。');
        writelines(strrep(code,sprintf('maxFE = %d;',budget),sprintf('maxFE = %d;',budget+200)),script);
        caught = false;
        try
            runScript(script);
        catch exception
            assert(strcmp(exception.identifier,'Experiment:ExistingResult'),'%s',exception.message);
            caught = true;
        end
        assert(caught && isequal(files,dir(fullfile(dataDir,'*.mat'))));
    end
    fprintf('PASS simple experiment runner: editable script, GANCMO/NSGAII, LIRCMOP13_BC/DTLZ2, single-thread workers, final-only save/resume/config protection.\n');
end

function runScript(file)
% 隔离脚本开头 clear 对检查函数局部变量的影响。
    run(file);
end

function restore(oldPath,oldFolder,folder)
    path(oldPath); cd(oldFolder);
    if isfolder(folder); rmdir(folder,'s'); end
end
