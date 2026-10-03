function [new_P,new_fit,new_fit_obj] = ATM_sort_1(P,fit_obj_c)

selected_indiv = [];
selected_fit = [];
oroginal_size = size(fit_obj_c,1)/2;
while size(selected_indiv,1) < oroginal_size
% The number of individuals in population P
psize = size(fit_obj_c,1);

% The Number of objective funtions
m = size(fit_obj_c,2);

% The number of individuals that dominate individual i
N = [];

% Classify individuals into diffierent layer
    for i = 1:psize
        recorder_n = 0;
    
        for j = 1:size(fit_obj_c,1)
            less = 0;
            equal = 0;
            more = 0;
        
            for k = 1:m
                if fit_obj_c(i,k) < fit_obj_c(j,k)
                    less = less+1;
                elseif fit_obj_c(i,k) == fit_obj_c(j,k)
                    equal = equal+1;
                else 
                    more = more+1;
                end
            end
        
            % Individual i is dominated
            if less == 0 && equal ~= m 
                recorder_n = recorder_n+1;
            end
        end
        % Store the recorders
        N = [N,recorder_n];

    end

    Index1 = find(N == 0);
    num1 = length(Index1);
    
    % Sort the first layer basing on the cv    
    [~,Index2] = sort(fit_obj_c(Index1,m));
    sorted_index = Index1(Index2);
    selected_index = sorted_index(1:round(num1/2));
    
    % Selection
    selected_indiv = [selected_indiv;P(selected_index,:)];
    selected_fit = [selected_fit;fit_obj_c(selected_index,:)];
    
    % Set []
    P(selected_index,:) = [];
    fit_obj_c(selected_index,:) = [];
end

[x,fit,fit_obj] = TSR_sort(selected_indiv(1:oroginal_size,:),selected_fit(1:oroginal_size,1:2),selected_fit(1:oroginal_size,:),selected_fit(1:oroginal_size,3));

new_P = x;
new_fit = fit;
new_fit_obj = fit_obj;