function first_use_baselines(seeds)
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root); addpath(fileparts(mfilename('fullpath')));
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907','first_use');
for seed=seeds
    for number=5:8
        name=sprintf('LIRCMOP%d_BC',number); available=dir(fullfile(folder,sprintf('%s_seed%02d_*.mat',name,seed)));
        assert(~isempty(available)); R=load(fullfile(folder,available(1).name),'M','Prefix');
        file=fullfile(fileparts(folder),sprintf('%s_seed%02d_first_fallback.mat',name,seed));
        if isfile(file); continue; end
        rng(seed,'twister'); ctor=str2func(name); P=ctor('N',100,'maxFE',R.M.firstUseFE,'maxRuntime',Inf);
        G=PairGuide('save',1,'run',seed,'outputFcn',@(varargin)[]); G.configureComparison('fallback_only');
        G.Solve(P); Audit=G.guideExperimentSnapshot(); E=Audit.evidence;
        assert(isequaln(E.population{end-1},R.Prefix));
        V=E.population{end}; Pop=SOLUTION(V.p1Decs,V.p1Objs,V.p1Cons);
        M=struct('problem',string(name),'seed',seed,'firstUseFE',P.FE, ...
            'IGD',P.CalMetric('IGD',Pop),'HV',P.CalMetric('HV',Pop));
        save(file,'M','Audit','-v7.3'); writetable(struct2table(M),[file,'.csv']);
        fprintf('FIRST_FALLBACK %s seed=%d FE=%d IGD=%.5g\n',name,seed,P.FE,M.IGD);
    end
end
end
