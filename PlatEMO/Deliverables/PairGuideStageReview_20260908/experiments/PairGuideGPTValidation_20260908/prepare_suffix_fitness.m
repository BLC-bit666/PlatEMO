function prepare_suffix_fitness
% Recover selection fitness from the exact preceding union, not the survivors.
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuideGPTValidation_20260908');
for number=[7 8]
    for seed=1:3
        name=sprintf('LIRCMOP%d_BC',number);
        target=fullfile(folder,'fixtures',sprintf('%s_seed%02d_suffix.mat',name,seed));
        if isfile(target); continue; end
        source=fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run', ...
            sprintf('%s_seed%02d_cgan.mat',name,seed));
        R=load(source,'Audit'); E=R.Audit.evidence; Suffix=cell(2,1);
        for stage=1:2
            fe=[50000 70000]; fe=fe(stage); times=cellfun(@(p)p.observationFE,E.population);
            k=find(times==fe,1); generation=k-1; previous=E.population{k-1};
            O1=E.evaluations{2*generation}; O2=E.evaluations{2*generation+1};
            O=[SOLUTION(O1.decisions,O1.objectives,O1.constraints), ...
                SOLUTION(O2.decisions,O2.objectives,O2.constraints)];
            Prev1=SOLUTION(previous.p1Decs,previous.p1Objs,previous.p1Cons);
            Prev2=SOLUTION(previous.p2Decs,previous.p2Objs,previous.p2Cons);
            [P1,Fitness1]=EnvironmentalSelection_CBS([Prev1,O],100,true);
            [P2,Fitness2]=EnvironmentalSelection_CBS([Prev2,O],100,false);
            assert(isequal(P1.decs,E.population{k}.p1Decs) && isequal(P2.decs,E.population{k}.p2Decs));
            Suffix{stage}=struct('Fitness1',Fitness1,'Fitness2',Fitness2,'ExpectedNext',E.population{k+1});
        end
        save(target,'Suffix','source','-v7.3'); clear R E;
        fprintf('SUFFIX_FITNESS_SAVED %s seed=%d\n',name,seed);
    end
end
end
