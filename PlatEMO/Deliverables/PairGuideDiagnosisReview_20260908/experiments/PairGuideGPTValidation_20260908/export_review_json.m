function export_review_json
% Export existing cached arrays and model states; no training or oracle calls.
warning('off','all'); maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));
addCBSPaths(root);
out=fullfile(folder,'review_numeric'); if ~isfolder(out); mkdir(out); end
folders={fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run'), ...
    fullfile(folder,'stable_side'),fullfile(folder,'stable_no_side')};
arms={'original','stable_side','stable_no_side'};
count=0;
for a=1:3
    for n=5:8
        source=fullfile(folders{a},'analysis','figures',sprintf('LIRCMOP%d_BC',n),'run_01','plot_data.mat');
        R=load(source,'States');
        for j=1:numel(R.States)
            saveJSON(fullfile(out,sprintf('%s_LIRCMOP%d_BC_run01_plot%02d.json',arms{a},n,j)),R.States{j});
            count=count+1;
        end
    end
end
for name={'fixed_conditions','fresh_conditions'}
    files=dir(fullfile(folder,name{1},'*.mat'));
    for k=1:numel(files)
        if strcmp(files(k).name,'status.mat') || strcmp(files(k).name,'initial_attempt_status.mat'); continue; end
        R=load(fullfile(files(k).folder,files(k).name),'Snapshots','task');
        assert(numel(R.Snapshots)==5);
        [~,stem]=fileparts(files(k).name);
        for j=[1 5]
            value=struct('task',R.task,'snapshot',R.Snapshots{j});
            saveJSON(fullfile(out,sprintf('%s_%s_snapshot%d.json',name{1},stem,j)),value);
            count=count+1;
        end
    end
end
for n=7:8
    for seed=1:3
        stem=sprintf('LIRCMOP%d_BC_seed%02d',n,seed);
        R=load(fullfile(folder,'fixtures',[stem,'.mat']),'Data','W','Spans','Prefixes');
        saveJSON(fullfile(out,[stem,'_fixture.json']),R); count=count+1;
    end
end
fprintf('REVIEW_JSON_EXPORT_COMPLETE %d files; zero training and zero oracle calls\n',count);
end

function saveJSON(path,value)
encoded=jsonencode(plain(value));
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',encoded);
end

function result=plain(value)
if isa(value,'dlarray')
    array=gather(extractdata(value));
    result=struct('type','dlarray','dataClass',class(array),'size',size(array),'values',array);
elseif isa(value,'dlnetwork')
    result=struct('type','dlnetwork','layerNames',{string({value.Layers.Name})}, ...
        'layerClasses',{arrayfun(@(x)string(class(x)),value.Layers)}, ...
        'learnables',plain(value.Learnables),'state',plain(value.State));
elseif istable(value)
    result=plain(table2struct(value));
elseif isstruct(value)
    result=value; fields=fieldnames(value);
    for k=1:numel(value)
        for j=1:numel(fields); result(k).(fields{j})=plain(value(k).(fields{j})); end
    end
elseif iscell(value)
    result=cellfun(@plain,value,'UniformOutput',false);
elseif isnumeric(value) || islogical(value) || ischar(value) || isstring(value)
    result=value;
elseif isa(value,'function_handle')
    result=func2str(value);
else
    error('ReviewExport:UnsupportedType','Unsupported cached type: %s',class(value));
end
end
