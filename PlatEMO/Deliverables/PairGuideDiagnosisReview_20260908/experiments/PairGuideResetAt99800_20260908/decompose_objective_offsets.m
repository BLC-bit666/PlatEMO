function decompose_objective_offsets
% Decompose already evaluated objective differences; no new objective calls.
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder)); rows={};
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root); addpath(folder);
for arm={'original','stable_side','stable_no_side'}
    for n=7:8
        stem=sprintf('%s_LIRCMOP%d_BC',arm{1},n);
        H=load(fullfile(folder,'geometry',[stem,'.mat']),'codes','meanY','meanEnergy');
        F=load(fullfile(folder,'fixtures',[stem,'.mat']),'ActualData','W');
        trueC=[F.ActualData.cF;F.ActualData.cI]; trueY=[F.ActualData.yF;F.ActualData.yI];
        trueAngles=directionAngles(trueY,trueC,F);
        for reset=0:3
            file=fullfile(folder,'runs',sprintf('%s_reset%d.mat',stem,reset));
            if ~isfile(file); continue; end
            R=load(file,'Snapshots','Table');
            for step=[0 10000]
                if step==0 && reset~=0; continue; end
                at=find(R.Table.addedUpdates==step,1); if isempty(at); continue; end
                S=R.Snapshots{at}.Samples; [found,ix]=ismember(H.codes,S.conditions,'rows'); assert(all(found));
                X=S.X(ix,:); Y=S.Y(ix,:);
                if n==7; base=[X(:,1),1-sqrt(X(:,1))]; else; base=[X(:,1),1-X(:,1).^2]; end
                energy=Y-base-.7057; deltaEnergy=energy-H.meanEnergy;
                deltaBase=base-(H.meanY-H.meanEnergy-.7057); delta=Y-H.meanY;
                assert(max(abs(delta-deltaEnergy-deltaBase),[],'all')<1e-12);
                generatedAngles=directionAngles(Y,H.codes,F);
                rows{end+1}=struct('arm',string(arm{1}),'problem',sprintf('LIRCMOP%d_BC',n),'reset',reset,'updates',step, ...
                    'conditions',size(X,1),'meanDeltaF1',mean(delta(:,1)),'meanDeltaF2',mean(delta(:,2)), ...
                    'meanDeltaBaseF1',mean(deltaBase(:,1)),'meanDeltaBaseF2',mean(deltaBase(:,2)), ...
                    'meanDeltaEnergyF1',mean(deltaEnergy(:,1)),'meanDeltaEnergyF2',mean(deltaEnergy(:,2)), ...
                    'rmsBaseDifference',sqrt(mean(deltaBase.^2,'all')),'rmsEnergyDifference',sqrt(mean(deltaEnergy.^2,'all')), ...
                    'trueMeanDirectionError',mean(trueAngles),'generatedSeenMeanDirectionError',mean(generatedAngles));
            end
        end
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,'objective_offset_decomposition.csv'));
fprintf('OBJECTIVE_OFFSETS_DECOMPOSED %d rows; zero new oracle calls\n',numel(rows));
end

function angles=directionAngles(Y,C,F)
[~,~,Yn]=AssignReferenceVectors_CBS(Y,F.W,F.ActualData.referenceScale);
direction=Yn./max(vecnorm(Yn,2,2),eps); target=C(:,1:2); target=target./vecnorm(target,2,2);
angles=acosd(max(-1,min(1,sum(direction.*target,2))));
end
