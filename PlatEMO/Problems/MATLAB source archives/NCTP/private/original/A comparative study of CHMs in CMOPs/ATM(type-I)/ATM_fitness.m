function final_new_fit = ATM_fitness(P,fit_obj_cv,percentage)
% Calculate the modified fitness value with method proposed in ATM

popsize = size(P,1);
column_size = size(fit_obj_cv,2);
fit_obj = fit_obj_cv(:,1:column_size - 1);
cv = fit_obj_cv(:,column_size);

f_sub = find(cv == 0);
inf_sub = find(cv > 0);
%feasible_set = P(f_sub,:);
%infeasible_set = P(inf_sub,:);
nor_new_fit = [];
for j = 1:column_size - 1
    fmin = min(fit_obj(:,j));
    fmax = max(fit_obj(:,j));
    
    for i = 1:popsize
        if ismember(i,f_sub)
            new_fit(i,j) = fit_obj(i,j);
        else
            new_fit(i,j) = max(percentage*fmin + (1-percentage)*fmax,fit_obj(i,j));
        end
    end
    nor_fit_j =(new_fit(:,j) - fmin)/(fmax - fmin+1E-30); 
    nor_new_fit = [nor_new_fit,nor_fit_j];
end
final_new_fit = nor_new_fit + cv*ones(1,column_size - 1);



























