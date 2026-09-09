function audit_gradient_pairs(directory,numbers)
if nargin<1;directory='runs';end
if nargin<2;numbers=5:8;end
warning('off','all');folder=fileparts(mfilename('fullpath'));rows={};
for n=numbers
    for stage={'early','middle','late'}
        F=load(fullfile(folder,'fixtures',sprintf('LIRCMOP%d_BC_seed01_%s.mat',n,stage{1})));
        for stream=1:3
            stem=sprintf('LIRCMOP%d_BC_seed01_%s_',n,stage{1});
            A=load(fullfile(folder,directory,sprintf('%sadversarial_stream%d.mat',stem,stream)));
            B=load(fullfile(folder,directory,sprintf('%sdistance_stream%d.mat',stem,stream)));
            assert(A.Complete && B.Complete && isequal(A.RngEnds,B.RngEnds));
            assert(isequaln(A.Snapshots{1},B.Snapshots{1}));
            for result={A,B}
                R=result{1};M=R.Snapshots{end}.Model;
                if ~strcmp(directory,'runs')
                    Original=load(fullfile(folder,'runs',sprintf('%s%s_stream%d.mat',stem,R.task.loss,stream)));
                    assert(isequal(R.RngEnds,Original.RngEnds) && isequaln(R.Snapshots{1},Original.Snapshots{1}));
                    assert(R.task.generatorLR==.0001);
                    % Primary task metadata predates the configurable LR field;
                    % its .001 rate is fixed in the original experiment plan.
                    if isfield(Original.task,'generatorLR');assert(Original.task.generatorLR==.001);end
                end
                assert(M.iterG==F.Model.iterG+2000 && M.iterC==F.Model.iterC+10000);
                assert(isequal(M.lastData,F.RawData));
                assert(isequal(M.lastTrainingData.cF,F.Data.cF) && isequal(M.lastTrainingData.cI,F.Data.cI));
                assert(height(R.Table)==4 && all(R.Table.offlineFE==200));
                assert(all(R.Table.offlineObjRows==400) && all(R.Table.offlineConRows==200));
                assert(numel(M.probeRecords)==105 && M.probeRecords(end).addedUpdates==2000);
                P=struct2table(M.probeRecords);
                assert(max(abs(P.usedGradientNorm-P.adversarialGradientNorm)./max(P.adversarialGradientNorm,1e-12))<1e-5);
            end
            rows{end+1}=struct('problemNumber',n,'stage',string(stage{1}),'stream',stream, ...
                'complete',true,'sameInitialState',true,'sameRNGTrajectory',true,'frozenData',true, ...
                'matchedGradientNorm',true,'GUpdatesPerArm',2000,'DUpdatesPerArm',10000,'newOfflineFEPerPair',1200);
        end
    end
end
if strcmp(directory,'runs');file='gradient_pair_verification.csv';else;file=[directory,'_pair_verification.csv'];end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,file));
fprintf('GRADIENT_PAIRS_VERIFIED %s: %d pairs, %d cases\n',directory,numel(rows),2*numel(rows));
end
