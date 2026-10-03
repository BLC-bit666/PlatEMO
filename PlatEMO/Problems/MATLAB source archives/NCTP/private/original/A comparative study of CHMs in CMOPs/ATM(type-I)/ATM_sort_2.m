function [New_pop,New_fit,New_fit_obj] = ATM_sort_2(P,Fit,fit_obj)

% The number of individuals in population P
popsize = size(P,1);

% The Number of objective funtions
m = size(Fit,2)-1;

% The number of individuals that dominate individual i
N = [];

% The set of individuals which are dominated by individual i
S = cell(popsize,1);
%for i=1:20
%    S{i}=[];
%end

%
Temp_pop = [];

%
Temp_fit = [];

%
Temp_fit_obj = [];

%
New_pop = [];

%
New_fit = [];

%
New_fit_obj = [];

for i = 1:popsize
    recorder_n = 0;
    Recorder_s = [];
    
    for j = 1:popsize
        less = 0;
        equal = 0;
        more = 0;
        
        for k = 1:m
            if Fit(i,k) < Fit(j,k)
                less = less+1;
            elseif Fit(i,k) == Fit(j,k)
                equal = equal+1;
            else 
                more = more+1;
            end
        end
        
        % Individual i is dominated
        if less == 0 && equal ~= m 
            recorder_n = recorder_n+1;
            
        % Individual j is dominated by i
        elseif more == 0 && equal ~= m
            Recorder_s = [Recorder_s,j];
        end
    end
    % Store the recorders
    N = [N,recorder_n];
    S{i} = [Recorder_s];
end

% Classify individuals into diffierent layer
rank = 1;
while ~isempty(find(N ==0))
    Index1 = find(N == 0);
    N(Index1) = 100;
    size_temp = size(Index1,2);
    Temp_pop = P(Index1,:);
    Temp_fit = Fit(Index1,:);
    Temp_fit(:,m+1) = rank; 
    Temp_fit_obj = fit_obj(Index1,:); 
    rank = rank+1;
    
    % Reduce N
    for y = 1:length(Index1)
        cc = length(S{Index1(y)});
        for x = 1:cc
            be_reduced = S{Index1(y)}(x);
            N(be_reduced) = N(be_reduced)-1;  
        end
    end
    
    %Calculate the crowding distance
    for k = 1:m
        % Temporary population rerangement basing on the mth objective function value  
        [~,Index2] = sort(Temp_fit(:,k));
        Temp_pop = Temp_pop(Index2,:);
        Temp_fit = Temp_fit(Index2,:);
        Temp_fit_obj = Temp_fit_obj(Index2,:); 
        
        % Calculating
        Temp_fit(1,m+1+k) = inf; 
        Temp_fit(size_temp,m+1+k) = inf;
        
        for i = 2:size_temp-1
            last_fit = Temp_fit(i-1,k);
            next_fit = Temp_fit(i+1,k);
            Temp_fit(i,m+1+k) = next_fit - last_fit; 
        end
        
        % Normalizing
        min_k = Temp_fit(1,k);
        max_k = Temp_fit(size_temp,k);
        if (max_k - min_k == 0)
        	Temp_fit(:,m+1+k) = Inf;
        else
            Temp_fit(:,m+1+k) = Temp_fit(:,m+1+k)/(max_k-min_k);
        end
    end
    
    % Sumerizing
    Temp_fit(:,m+2) = sum(Temp_fit(:,m+2:2*m+1),2);
    Temp_fit = Temp_fit(:,1:m+2);
    [~,Index3] = sort(Temp_fit(:,m+2),'descend');
    New_pop = [New_pop;Temp_pop(Index3,:)];    
    New_fit = [New_fit;Temp_fit(Index3,:)];  
    New_fit_obj = [New_fit_obj;Temp_fit_obj(Index3,:)];
end        
New_pop = New_pop(1:popsize/2,:);
New_fit = New_fit(1:popsize/2,:);
New_fit_obj = New_fit_obj(1:popsize/2,:);
                