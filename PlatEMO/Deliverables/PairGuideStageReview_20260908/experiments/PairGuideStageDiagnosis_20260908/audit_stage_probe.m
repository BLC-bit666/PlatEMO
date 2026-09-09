function audit_stage_probe
warning('off','all'); maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));
addCBSPaths(root); addpath(folder);
rows={}; comparisons={};
for n=7:8
    F=load(fullfile(root,'Data','PairGuideResetAt99800_20260908','fixtures',sprintf('original_LIRCMOP%d_BC.mat',n)));
    for reset=-1:3
        M=F.Model; added=0;
        if reset>=0
            R=load(fullfile(root,'Data','PairGuideResetAt99800_20260908','runs',sprintf('original_LIRCMOP%d_BC_reset%d.mat',n,reset)),'Snapshots');
            M=R.Snapshots{end}.Model; added=10000;
        end
        saved=rng;
        [R,~]=PairBoundaryWGAN_StageProbe('gradientaudit',M,F.ActualData);
        assert(isequal(saved,rng));
        R.problemNumber=n; R.reset=reset; R.addedUpdates=added; rows{end+1}=R;
        fprintf('STATIC_GRADIENT problem=%d reset=%d cosine=%.6f derivative=%.6f finiteDiff=%.6f varianceFraction=%.6f\n', ...
            n,reset,R.gradientCosine,R.starAlongAdversarialDescent,R.finiteDifference,R.varianceFraction);
    end
    ctor=str2func(sprintf('LIRCMOP%d_BC',n)); P=ctor('N',100,'maxFE',100000);
    O=struct('retrainEpoch',20,'nCritic',5,'lrG',.001,'lrD',.001, ...
        'gpLambda',10,'miniBatch',32,'trainingSigma',0,'sampleSigma',0, ...
        'retrainChange',0,'retrainGenerations',1,'useSideCondition',true,'stableConditionSpan',false, ...
        'generation',F.Model.lastTrainGeneration+1,'probeLoss','adversarial','probeBase',F.Model.iterG);
    Gate=struct('eligible',true); rng(71234,'twister');
    [Native,S1]=PairBoundaryWGAN_RC('trainifneeded',F.Model,F.RawData,Gate,P,O); end1=rng;
    rng(71234,'twister');
    [Probe,S2]=PairBoundaryWGAN_StageProbe('trainifneeded',F.Model,F.RawData,Gate,P,O); end2=rng;
    equalG=isequaln(Native.netG.Learnables,Probe.netG.Learnables);
    equalD=isequaln(Native.netC.Learnables,Probe.netC.Learnables);
    equalAdam=isequaln(Native.avgG,Probe.avgG) && isequaln(Native.avgSqG,Probe.avgSqG) && isequaln(Native.avgC,Probe.avgC) && isequaln(Native.avgSqC,Probe.avgSqC);
    assert(equalG && equalD && equalAdam && isequal(end1,end2));
    assert(S1.updates==20 && S2.updates==20 && P.FE==0);
    comparisons{end+1}=struct('problemNumber',n,'equalG',equalG,'equalD',equalD,'equalAdam',equalAdam,'equalRNG',isequal(end1,end2));
    writetable(struct2table(Probe.probeRecords),fullfile(folder,sprintf('native_actual_steps_LIRCMOP%d.csv',n)));
end
T=struct2table(vertcat(rows{:}));
original=T(T.reset==-1,:); expected=[-.7421;-.1837];
assert(max(abs(original.gradientCosine-expected))<.002);
assert(max(abs(original.varianceFraction-[.0427;.0769]))<.002);
writetable(T,fullfile(folder,'source_gradient_reproduction.csv'));
writetable(struct2table(vertcat(comparisons{:})),fullfile(folder,'probe_neutrality.csv'));
fprintf('PROBE_AUDIT_COMPLETE 10 states; two 20-step identity checks; zero oracle FE\n');
end
