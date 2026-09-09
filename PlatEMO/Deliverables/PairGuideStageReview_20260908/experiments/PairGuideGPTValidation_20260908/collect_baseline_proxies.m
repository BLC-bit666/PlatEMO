function collect_baseline_proxies(nWorker)
if nargin<1; nWorker=2; end
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'; folder=fileparts(mfilename('fullpath'));
source=fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run');
files=dir(fullfile(source,'LIRCMOP*_seed*_*.mat'));
output=fullfile(folder,'boundary_proxies'); if ~isfolder(output); mkdir(output); end
pool=parpool('Processes',nWorker); cleanup=onCleanup(@()delete(pool));
parfor k=1:numel(files)
    maxNumCompThreads(1); addpath(folder);
    out=fullfile(output,replace(files(k).name,'.mat','.csv'));
    if isfile(out); continue; end
    R=load(fullfile(files(k).folder,files(k).name),'Record','Audit');
    mode=R.Record.mode; if mode=="cgan"; mode="original"; end
    Proxy=boundary_proxy_rows(R.Audit.evidence,R.Record.problem,R.Record.seed,mode);
    writetable(Proxy,out); fprintf('BASELINE_PROXY %s\n',files(k).name);
end
fprintf('BASELINE_PROXIES_COMPLETE\n');
end
