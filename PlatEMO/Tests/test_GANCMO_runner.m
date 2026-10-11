function test_GANCMO_runner
% 复制真实脚本，仅缩小配置并隔离输出；验证末代保存、续跑和进程池清理。
    root = fileparts(fileparts(mfilename('fullpath')));
    source = fileread(fullfile(root,'test.m'));
    folder = tempname; mkdir(folder);
    oldPath = path; oldFolder = pwd;
    cleanup = onCleanup(@()restore(oldPath,oldFolder,folder));
    assert(isempty(gcp('nocreate')),'请在没有并行池的独立 MATLAB 会话中运行。');
    cases = {'GANCMO','LIRCMOP13_BC',200;'NSGAII','DTLZ2',4000};
    for k = 1:size(cases,1)
        dataDir = fullfile(folder,'Data',cases{k,1}); script = fullfile(folder,'run_case.m');
        code = regexprep(source,'(?m)^nWorker\s*=[^\n]*','nWorker = 2;');
        budget = cases{k,3};
        code = regexprep(code,'(?m)^maxFE\s*=[^\n]*',sprintf('maxFE = %d;',budget));
        code = regexprep(code,'(?m)^runs\s*=[^\n]*','runs = 1:2;');
        code = regexprep(code,'(?m)^algName\s*=[^\n]*',sprintf('algName = ''%s'';',cases{k,1}));
        code = regexprep(code,'proNames\s*=\s*\{[\s\S]*?\};',sprintf('proNames = {''%s''};',cases{k,2}));
        code = regexprep(code,'(?m)^rootPath = [^\n]*',sprintf('rootPath = ''%s'';',folder));
        code = strrep(code,'addRuntimePaths(rootPath);',sprintf('addRuntimePaths(''%s'');',root));
        writelines(code,script); runScript(script);
        files = dir(fullfile(dataDir,'*.mat')); assert(numel(files)==2);
        for j = 1:numel(files)
            assert(startsWith(files(j).name,sprintf('%s_%s_',cases{k,1},cases{k,2})));
            saved = load(fullfile(files(j).folder,files(j).name));
            assert(isfield(saved,'result') && isfield(saved,'metric'));
            assert(saved.result{end,1}==budget);
            assert(isequal(size(saved.result),[1 2]) && ~isfield(saved.metric,'processIGD'));
            assert(size(saved.result{end,2}.decs,2)==30 && isscalar(saved.metric.IGD));
        end
        runScript(script);
        assert(isequal(files,dir(fullfile(dataDir,'*.mat'))),'续跑不应改写已完成结果。');
        assert(isempty(gcp('nocreate')),'入口创建的进程池应已关闭。');
        writelines(strrep(code,sprintf('maxFE = %d;',budget),sprintf('maxFE = %d;',budget+200)),script);
        runScript(script);
        assert(isequal(files,dir(fullfile(dataDir,'*.mat'))),'仍按文件存在判定续跑，不增加配置校验。');
    end

    existingPool = parpool("Processes",2);
    poolCleanup = onCleanup(@()delete(existingPool));
    writelines(strrep(code,'runs = 1:2;','runs = 3;'),script);
    runWithExistingPool(script);
    assert(isvalid(existingPool) && isequal(gcp('nocreate'),existingPool),'不应关闭调用前已有的池。');
    assert(numel(dir(fullfile(dataDir,'*.mat')))==3);
    clear poolCleanup;

    badCode = strrep(code,'runs = 1:2;','runs = 4;');
    badCode = strrep(badCode,'nTask = size(tasks,1);', ...
        'error(''RunnerTest:ExpectedFailure'',''Expected failure after pool creation.''); nTask = size(tasks,1);');
    writelines(badCode,script);
    caught = false;
    try
        runScript(script);
    catch exception
        assert(strcmp(exception.identifier,'RunnerTest:ExpectedFailure'),'%s',exception.message);
        caught = true;
    end
    assert(caught && isempty(gcp('nocreate')),'运行报错也必须释放自建进程池。');
    fprintf('PASS experiment runner: algorithm directories/selection, final-only save, existence-based resume, owned pool cleanup, existing pool retained.\n');
end

function runScript(file)
% 隔离脚本开头 clear 对检查函数局部变量的影响。
    try
        run(file);
    catch exception
        assert(isempty(gcp('nocreate')),'报错后执行函数应已关闭自建池。');
        rethrow(exception);
    end
    assert(isempty(gcp('nocreate')),'脚本返回前执行函数应已关闭自建池。');
end

function runWithExistingPool(file)
    run(file);
end

function restore(oldPath,oldFolder,folder)
    path(oldPath); cd(oldFolder);
    if isfolder(folder); rmdir(folder,'s'); end
end
