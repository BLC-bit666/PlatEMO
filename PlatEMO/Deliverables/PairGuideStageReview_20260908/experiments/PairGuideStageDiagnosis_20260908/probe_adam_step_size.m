function probe_adam_step_size
% Local counterfactual only: shorten the SAME actual Adam parameter displacement.
% Each case performs one native G/five D updates on copies. No oracle calls.
warning('off','all');maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));addCBSPaths(root);addpath(folder);
rows={};
for n=5:8
    for stage={'early','middle','late'}
        F=load(fullfile(folder,'fixtures',sprintf('LIRCMOP%d_BC_seed01_%s.mat',n,stage{1})));
        [~,E]=PairBoundaryWGAN_StageProbe('gradientaudit',F.Model,F.Data);
        C=single(E.conditions'); Real=single(2*E.X'-1); weights=E.weights';
        ctor=str2func(F.Meta.problem);P=ctor('N',100,'maxFE',100000);
        for stream=1:3
            O=struct('retrainEpoch',1,'nCritic',5,'lrG',.001,'lrD',.001,'gpLambda',10, ...
                'miniBatch',32,'trainingSigma',0,'sampleSigma',0,'retrainChange',0,'retrainGenerations',1, ...
                'useSideCondition',true,'stableConditionSpan',false,'probeLoss','adversarial','probeBase',F.Model.iterG, ...
                'generation',F.Model.lastTrainGeneration+1);
            rng(2100000+10000*stream+1,'twister');
            [M,S]=PairBoundaryWGAN_StageProbe('trainifneeded',F.Model,F.RawData,struct('eligible',true),P,O);
            assert(S.trained && S.updates==1 && P.FE==0);
            R=M.probeRecords(end);
            original=distance(F.Model.netG,C,Real,weights,F.Model.zDim);
            assert(abs(original-R.distanceBefore)<1e-8);
            for fraction=[.01 .1 1]
                G=F.Model.netG;
                for k=1:height(G.Learnables)
                    G.Learnables.Value{k}=G.Learnables.Value{k}+single(fraction)*(M.netG.Learnables.Value{k}-G.Learnables.Value{k});
                end
                value=distance(G,C,Real,weights,F.Model.zDim);
                if fraction==1;assert(abs(value-R.distanceAfter)<1e-7);end
                row=R;row.problem=string(F.Meta.problem);row.stage=string(stage{1});row.stream=stream;
                row.fraction=fraction;row.fractionalDistance=value;row.fractionalChange=value-original;rows{end+1}=row;
            end
        end
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,'adam_step_size.csv'));
fprintf('ADAM_STEP_SIZE_COMPLETE 36 one-step copies, 108 displacement probes; zero oracle calls\n');
end

function value=distance(G,C,Real,weights,zDim)
X=double(extractdata(forward(G,dlarray([zeros(zDim,size(C,2),'single');C],'CB'))));
value=sum(sqrt(sum((X-double(Real)).^2,1)+1e-12).*weights);
end
