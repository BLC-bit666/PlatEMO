function addCBSPaths(repoRoot)
%ADDCBSPATHS Add runtime roots without exposing frozen Data snapshots.

    repoRoot = char(repoRoot);
    dataRoot = string(fullfile(repoRoot,'Data'));
    currentPaths = string(strsplit(path,pathsep));
    frozen = currentPaths == dataRoot | ...
        startsWith(currentPaths,dataRoot+filesep);
    for item = reshape(currentPaths(frozen),1,[])
        rmpath(char(item));
    end

    addpath(repoRoot,'-begin');
    folders = {'Algorithms','Problems','Metrics'};
    for i = 1 : numel(folders)
        folder = fullfile(repoRoot,folders{i});
        if isfolder(folder)
            addpath(genpath(folder),'-begin');
        end
    end
end
