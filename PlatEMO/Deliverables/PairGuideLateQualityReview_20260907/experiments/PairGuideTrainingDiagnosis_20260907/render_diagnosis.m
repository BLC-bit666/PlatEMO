function render_diagnosis
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
out=fullfile(root,'Data','PairGuideTrainingDiagnosis_20260907');
arms=["production20","warm1000","noAdv1000","uniformGap1000","lr1e3_1000", ...
 "noNoise1000","fresh32_1000","fresh128_1000","onehot32_1000"];
allRows={}; geometryRows={};
for number=[7 8]
 name=sprintf('LIRCMOP%d_BC',number); Values=cell(size(arms));
 for k=1:numel(arms)
  Values{k}=load(fullfile(out,sprintf('%s_FE099800_%s.mat',name,arms(k))));
  allRows{end+1}=Values{k}.row; %#ok<AGROW>
 end
 Base=Values{1}; D=Base.D; [~,order]=sort(D.w(:,1));
 Encoded=load(fullfile(out,sprintf('%s_FE099800_onehot32uniform5000.mat',name)));
 F=figure('Visible','off','Color','w','Position',[50 50 1450 900]); L=tiledlayout(2,2,'Padding','compact');
 nexttile; scatter(Base.yf(:,1),Base.yf(:,2),9,[0 .55 .15],'.'); hold on;
 scatter(Base.yi(:,1),Base.yi(:,2),9,[.8 .15 .1],'.');
 scatter(Base.ygf(1:16:end,1),Base.ygf(1:16:end,2),10,[.15 .4 .85],'o');
 scatter(Base.ygi(1:16:end,1),Base.ygi(1:16:end,2),10,[.65 .2 .7],'+');
 grid on; xlabel('f_1'); ylabel('f_2'); title('Original generator: endpoints before interpolation');
 legend('True F','True I','Generated F','Generated I','Location','best');
 coord=mean([Base.mf-D.xF;Base.mi-D.xI].^2,1); [~,worst]=max(coord);
 nexttile; plot(D.w(order,1),D.xF(order,worst),'o-','LineWidth',1); hold on;
 plot(D.w(order,1),Base.mf(order,worst),'-','LineWidth',2);
 plot(D.w(order,1),Encoded.mf(order,worst),'-','LineWidth',1.5);
 grid on; xlabel('Reference weight w_1'); ylabel(sprintf('x_{%d}',worst));
 title('Same directions: true targets and generated means'); legend('True F','Original','One-hot + uniform, 5000','Location','best');
 coldArms=["fresh32_5000","fresh128_5000","onehot32_5000", ...
     "fresh32uniform5000","fresh128uniform5000","onehot32uniform5000"];
 coldLabels=["32, original","128, original","One-hot, original", ...
     "32, uniform","128, uniform","One-hot, uniform"];
 coldError=zeros(size(coldArms));
 for j=1:numel(coldArms)
  C=load(fullfile(out,sprintf('%s_FE099800_%s.mat',name,coldArms(j))),'row'); coldError(j)=C.row.RMSE;
 end
 nexttile; bar(coldError); xticks(1:numel(coldLabels)); xticklabels(coldLabels); xtickangle(25); grid on;
 ylabel('Normalized endpoint RMSE'); title('Matched cold starts: 5000 updates, same fixed targets');
 nexttile; weights=1./max(sum(D.delta.^2,2),1e-6*size(D.xF,2));
 semilogy(D.w(:,1),weights/sum(weights),'o'); grid on; xlabel('Reference weight w_1'); ylabel('Fraction of endpoint loss weight');
 title(sprintf('Inverse gap weighting: effective %.2f / %d pairs',sum(weights)^2/sum(weights.^2),D.count));
 title(L,sprintf('%s | fixed training FE=99800 | seed 1 | controlled diagnostic experiments',name),'Interpreter','none');
 exportgraphics(F,fullfile(out,[name,'_diagnosis.png']),'Resolution',140); close(F);
 % Derive objective distortion of local decision averaging, without any oracle calls.
 R=load(fullfile(out,[name,'_nodes.mat']),'Nodes'); N=R.Nodes{end};
 target=[D.xF;D.xI]; smooth=[N.smoothF;N.smoothI];
 targetRad=radial(target); smoothRad=radial(smooth); generatedRad=radial([Base.mf;Base.mi]);
 row=struct('problem',string(name),'targetRadialEnergy',mean(targetRad), ...
  'neighborMeanRadialEnergy',mean(smoothRad),'generatedMeanRadialEnergy',mean(generatedRad), ...
  'targetX1Mean',mean(target(:,1)),'neighborX1Mean',mean(smooth(:,1)), ...
  'generatedX1Mean',mean([Base.mf(:,1);Base.mi(:,1)]));
 geometryRows{end+1}=row; %#ok<AGROW>
end
writetable(struct2table(vertcat(allRows{:})),fullfile(out,'fixed_data_comparisons.csv'));
writetable(struct2table(vertcat(geometryRows{:})),fullfile(out,'radial_energy.csv'));
F=figure('Visible','off','Color','w','Position',[50 50 1300 780]); L=tiledlayout(2,2,'Padding','compact');
for number=[7 8]
 for metric=["RMSE","midpointObjectiveRMSE"]
  nexttile; hold on;
  for prefix=["fresh32_","onehot32_"]
   values=zeros(1,3); budgets=[1000 5000 10000];
   for j=1:3
    C=load(fullfile(out,sprintf('LIRCMOP%d_BC_FE099800_%s%d.mat',number,prefix,budgets(j))),'row'); values(j)=C.row.(metric);
   end
   plot(budgets,values,'o-','LineWidth',1.8);
  end
  grid on; xlabel('Generator updates on the same fixed dataset'); ylabel(metric,'Interpreter','none');
  title(sprintf('LIRCMOP%d BC',number)); legend('Reference-vector input','One-hot direction input','Location','best');
 end
end
title(L,'Decision fit improves much faster than objective-space agreement; original loss, cold 32x32 networks');
exportgraphics(F,fullfile(out,'fit_vs_objective.png'),'Resolution',140); close(F);
fprintf('DIAGNOSIS_FIGURES_COMPLETE\n');
end
function v=radial(X)
dim=size(X,2); v=zeros(size(X,1),1);
for j=2:dim
 if mod(j,2), target=sin((.5*j/dim*pi)*X(:,1)); else, target=cos((.5*j/dim*pi)*X(:,1)); end
 v=v+(X(:,j)-target).^2;
end
end
