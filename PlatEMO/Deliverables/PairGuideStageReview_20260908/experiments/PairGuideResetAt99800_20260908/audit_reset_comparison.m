function audit_reset_comparison
% Verify all matched reset interventions, RNG traces and offline costs.
warning('off','all'); folder=fileparts(mfilename('fullpath')); rows=cell(24,1); k=0;
for arm={'original','stable_side','stable_no_side'}
    for n=7:8
        F=load(fullfile(folder,'fixtures',sprintf('%s_LIRCMOP%d_BC.mat',arm{1},n)));
        Warm=load(fullfile(folder,'runs',sprintf('%s_LIRCMOP%d_BC_reset0.mat',arm{1},n)));
        for reset=0:3
            R=load(fullfile(folder,'runs',sprintf('%s_LIRCMOP%d_BC_reset%d.mat',arm{1},n,reset)));
            assert(R.Complete && height(R.Table)==5 && isequal(R.Table.addedUpdates,[0;1000;2000;5000;10000]));
            assert(isequal(R.RngEnds,Warm.RngEnds));
            Start=R.Snapshots{1}.Model; Last=R.Snapshots{5}.Model;
            assert(Last.iterG==R.base+10000 && Last.iterC==Start.iterC+50000);
            assert(isequal(Last.lastData,F.RawData) && isequal(Last.lastTrainingData.cF,F.ActualData.cF) && isequal(Last.lastTrainingData.cI,F.ActualData.cI));
            assert(isequal(Start.netG.Learnables.Layer,Last.netG.Learnables.Layer));
            if reset==0
                assert(isequaln(Start.netG.Learnables,F.Model.netG.Learnables) && isequaln(Start.avgG,F.Model.avgG));
                assert(isequaln(Start.netC.Learnables,F.Model.netC.Learnables) && isequaln(Start.avgC,F.Model.avgC));
                assert(isequaln(Start.avgSqG,F.Model.avgSqG) && isequaln(Start.avgSqC,F.Model.avgSqC));
            else
                assert(Start.iterG==0 && Start.iterC==0 && isempty(Start.avgG) && isempty(Start.avgSqG) && isempty(Start.avgC) && isempty(Start.avgSqC));
                assert(~isequaln(Start.netG.Learnables,F.Model.netG.Learnables) && ~isequaln(Start.netC.Learnables,F.Model.netC.Learnables));
            end
            assert(all(R.Table.offlineFullFE==size(F.QueryC,1)) && all(R.Table.offlineCalObjRows==2*R.Table.offlineFullFE) && all(R.Table.offlineCalConRows==R.Table.offlineFullFE));
            k=k+1; rows{k}=struct('arm',string(arm{1}),'problem',sprintf('LIRCMOP%d_BC',n),'reset',reset, ...
                'sameFrozenDataAndConditions',true,'sameNativeRNGTrace',true,'resetOrWarmStateVerified',true, ...
                'addedGUpdates',10000,'offlineFullFE',sum(R.Table.offlineFullFE),'verified',true);
        end
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,'verification.csv'));
fprintf('RESET_AUDIT_COMPLETE 24 matched cases\n');
end
