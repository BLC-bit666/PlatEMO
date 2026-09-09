function inspect_conditional_support
% Algebraic lower bound for endpoint MSE of a deterministic conditional output.
% It diagnoses the metric, not search value, and uses no objective calls.
warning('off','all');maxNumCompThreads(1);folder=fileparts(mfilename('fullpath'));rows=struct([]);
for n=7:8
 for seed=1:3
  F=load(fullfile(folder,'fixtures',sprintf('LIRCMOP%d_BC_seed%02d.mat',n,seed)),'Data');
  D=F.Data;X=[D.xF;D.xI];original=[D.cF;D.cI];
  for side=[true false]
   C=original;if ~side;C(:,end)=0;end
   [~,~,g]=unique(C,'rows'); centers=zeros(size(X));counts=zeros(max(g),1);
   distinct=counts;
   for k=1:max(g)
    ix=g==k;centers(ix,:)=repmat(mean(X(ix,:),1),nnz(ix),1);
    counts(k)=nnz(ix);distinct(k)=size(unique(X(ix,:),'rows'),1);
   end
   row=struct('problem',string(sprintf('LIRCMOP%d_BC',n)),'seed',seed,'useSide',side, ...
    'pairSlots',size(X,1),'conditionCodes',max(g),'codesWithMultipleDistinctX',nnz(distinct>1), ...
    'medianDistinctXPerCode',median(distinct),'maximumDistinctXPerCode',max(distinct), ...
    'minimumDeterministicEndpointRMSE',sqrt(mean((X-centers).^2,'all')));
   rows=[rows;row]; %#ok<AGROW>
  end
 end
end
writetable(struct2table(rows),fullfile(folder,'conditional_support.csv'));
fprintf('CONDITIONAL_SUPPORT_COMPLETE zero oracle calls\n');
end
