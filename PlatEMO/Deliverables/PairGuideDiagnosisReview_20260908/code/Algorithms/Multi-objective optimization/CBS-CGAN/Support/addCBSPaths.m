function addCBSPaths(repoRoot)
%ADDCBSPATHS Add runtime roots without exposing archived source snapshots.

    repoRoot = char(repoRoot);
    excludedRoots = string(fullfile(repoRoot,{'Data','Deliverables'}));
    currentPaths = string(strsplit(path,pathsep));
    frozen = false(size(currentPaths));
    for root = reshape(excludedRoots,1,[])
        frozen = frozen | currentPaths == root | ...
            startsWith(currentPaths,root+filesep);
    end
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
