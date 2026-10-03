%SP-TypeII
clc;
clear all;
tic;

format long;
format compact;

% The number of test problems in this type
problemNum = 12;

% This program is run for 100 times
itrNum = 100;

% Record the number of IGD value in every interval(the interval number is set to be 13 empirically)
record = zeros(problemNum,13);

% Record the IGD value of every iteration in all test problems
IGDset = zeros(problemNum,itrNum);

for problem = 1:problemNum

switch problem

    %！！！！！！30D-S！！！！！！！！              
    case 1 % NCTP-7
        lu = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5];       
        n = 30;% The dimension number of the decision variable 
        m = 2; % The number of objectives in this test problem          
        PARETO = textread('PARETO2.txt');% Read the standard pareto front points
            
    case 2 % NCTP-8
        lu = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5];       
        n = 30;
        m = 2;      
        PARETO = textread('PARETO2.txt');
        
    case 3 % NCTP-9
        lu = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5];       
        n = 30;
        m = 2;      
        PARETO = textread('PARETO2.txt');
        
    %！！！！！！30D-L！！！！！！！！       
    case 4 % NCTP-10
        lu = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5];       
        n = 30;
        m = 2;      
        PARETO = textread('PARETO2.txt');
        
    case 5 % NCTP-11
        lu = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5];       
        n = 30;
        m = 2;      
        PARETO = textread('PARETO2.txt');
            
    case 6 % NCTP-12
        lu = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5];       
        n = 30;
        m = 2;      
        PARETO = textread('PARETO2.txt');
        
    %！！！！！！10D-S！！！！！！！！    
    case 7 % NCTP-7
        lu = [0 0 0 0 0 0 0 0 0 0;5 5 5 5 5 5 5 5 5 5];       
        n = 10;
        m = 2;      
        PARETO = textread('PARETO2.txt');
        
    case 8 % NCTP-8
        lu = [0 0 0 0 0 0 0 0 0 0;5 5 5 5 5 5 5 5 5 5];       
        n = 10;
        m = 2;      
        PARETO = textread('PARETO2.txt');
        
    case 9 % NCTP-9
        lu = [0 0 0 0 0 0 0 0 0 0;5 5 5 5 5 5 5 5 5 5];       
        n = 10;
        m = 2;      
        PARETO = textread('PARETO2.txt');
        
    %！！！！！！10D-L！！！！！！！！          
    case 10 % NCTP-10
        lu = [0 0 0 0 0 0 0 0 0 0 ;5 5 5 5 5 5 5 5 5 5];       
        n = 10;
        m = 2;      
        PARETO = textread('PARETO2.txt');
            
    case 11% NCTP-11
        lu = [0 0 0 0 0 0 0 0 0 0 ;5 5 5 5 5 5 5 5 5 5];       
        n = 10;
        m = 2;      
        PARETO = textread('PARETO2.txt');
        
    case 12 % NCTP-12
        lu = [0 0 0 0 0 0 0 0 0 0 ;5 5 5 5 5 5 5 5 5 5];       
        n = 10;
        m = 2;      
        PARETO = textread('PARETO2.txt');
end

itrCounter = 1;

while itrCounter<=itrNum
    
% maximum generation number
max_gen = 500;

% population size
popsize =100;

% crossover probability
pcrossover = 0.9;

% mutation probability
pmutation = 1/n;

% crossover distribution index
di = 20;

% mutation distribution index
dim = 20;
rand('seed',sum(100*clock));

% the first generation
gen = 1;

% initialize the population
x = ones(2 * popsize, 1) * lu(1, :) + rand(2 * popsize, n) .* (ones(2 * popsize, 1) * (lu(2, :) - lu(1, :)));

% evaluate the population
[fit,fit_obj] = fitness_(x,problem,m);

[x,fit,fit_obj] = fast_nondominated_sort(x,fit,fit_obj);
gen = gen + 1;

% evolutionary process
while gen<=max_gen

    pool_size = popsize;
    
    % tournament selection operator
    mate_pool_chromosome = tournament_selection(x, fit, pool_size, n, m+1, popsize);

    % genetic operator(simulated binary crossover && mutation)
    offspring_chromosome = genetic_operator(mate_pool_chromosome, pool_size, n, lu, pcrossover, pmutation, di, dim);

    x(popsize+1 : 2*popsize, :) =  offspring_chromosome;
    %fit_obj(popsize+1 : 2*popsize, :) = fit_offspring;

    % Calculate the fitness value of the mixed population
    [fit,fit_obj] = fitness_(x, problem,m);
    
    % Sort and select 
    [x,fit,fit_obj] = fast_nondominated_sort(x,fit,fit_obj);
    
    % Present the current generation number
    if mod(gen, 50) == 0
        fprintf('gen = %d\n',gen);
    end

    gen = gen + 1;

end

% store all feasible solutions in the final population
temp = find(fit_obj(:, m+1) == 0);
final_feasible_solutions = fit_obj( temp, : );

% store nondominated solutions in the final population
temp_1 = find (fit(temp, m+1) == 1);
final_nondominated_solutions = fit_obj(temp(temp_1), :);

%% plot
if m == 2
   figure(1);
        %h = ezplot('cos(-0.2*pi)*(y-1)-sin(-0.2*pi)*x-0.75*(abs(sin(10*pi*(sin(-0.2*pi)*(y-1)+cos(-0.2*pi)*x)))).^0.5',[0,5,-5,5]);
        %colormap([0 1 1]);
        %hold on
        %plot(final_nondominated_solutions(:,1),final_nondominated_solutions(:,2),'g.');
        %title('NCTP-11','Fontsize',25); 
        %xlabel('f_1','Fontsize',25);
        %ylabel('f_2','Fontsize',25);
end        

%% IGD calculation
R = PARETO;
W = final_nondominated_solutions(:,1:2);

Rsize = size(R,1);
Wsize = size(W,1);

dis_set = [];

for i = 1:Rsize
    dis = inf;
    
    for j = 1:Wsize
        current_dis = sqrt(sum((abs(R(i,:)-W(j,:))).^2));
        if current_dis<dis
            dis = current_dis;
        end
    end
    
    dis_set = [dis_set,dis];
end
    IGD = sum(dis_set)/Rsize;
    disp(sprintf('Problem: %u, iteration %u finished, time used: %u',problem, itrCounter, toc));
    
%% record && classification
IGDset(problem,itrCounter) = IGD;

interval_num = fix(IGD/0.25);

switch interval_num
    case 0
        record(problem,1) = record(problem,1)+1;
    case 1
        record(problem,2) = record(problem,2)+1;
    case 2
        record(problem,3) = record(problem,3)+1;
    case 3
        record(problem,4) = record(problem,4)+1;
    case 4
        record(problem,5) = record(problem,5)+1;     
    case 5
        record(problem,6) = record(problem,6)+1;
    case 6
        record(problem,7) = record(problem,7)+1;
    case 7
        record(problem,8) = record(problem,8)+1;
    case 8
        record(problem,9) = record(problem,9)+1;
    case 9
        record(problem,10) = record(problem,10)+1;
    case 10
        record(problem,11) = record(problem,11)+1;     
    case 11
        record(problem,12) = record(problem,12)+1;
end
  
itrCounter =itrCounter+1;
end

end

toc;