% Reproducible launch: screened archive, four problems, three seeds.
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
rootPath='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
folder=fullfile(rootPath,'Data','PairGuideFilteredArchive_5to8_R3_20260907');
diary(fullfile(folder,'campaign.log'));
addpath(fullfile(rootPath,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));
addCBSPaths(rootPath);
manifest=jsondecode(fileread(fullfile(folder,'source_manifest.json')));
for k=1:numel(manifest)
    fid=fopen(fullfile(rootPath,manifest(k).path),'rb');
    assert(fid>=0,'Missing pinned source: %s',manifest(k).path);
    bytes=fread(fid,Inf,'*uint8'); fclose(fid);
    digest=java.security.MessageDigest.getInstance('SHA-256'); digest.update(bytes);
    actual=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2)',1,[]));
    assert(strcmp(actual,manifest(k).sha256),'Source changed: %s',manifest(k).path);
end
O=struct('problems',["LIRCMOP5_BC","LIRCMOP6_BC","LIRCMOP7_BC","LIRCMOP8_BC"], ...
    'seeds',1:3,'modes',"cgan",'N',100,'maxFE',100000, ...
    'checkpointFE',[10000 30000 50000 70000 100000], ...
    'distributionLayout',"training_pair",'outputDir',string(folder));
fprintf('CAMPAIGN ccmo/front1 CGAN tasks=12 workers=10 initialUpdates=200 retrainUpdates=20\n');
State=run_PairGuide_validation(rootPath,10,O);
assert(State.status=="complete" && State.completed==12 && State.failed==0);
fprintf('CAMPAIGN_ALL_12_COMPLETE\n');
