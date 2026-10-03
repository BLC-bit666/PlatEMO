function mate_pool_population = tournament_selection(x, fit_v, pool_size, n, M, popsize)

%x！！current population
%fit_v！individual fitness materix
%n！！number of decision variables
%M！！ M=m+1��m is the number of objevtive functions


% combine the population with its fitness
chromosome = [x fit_v];

% generate two index arrays
a1 = 1 : popsize;
a2 = 1 : popsize;

% genarate two random number matrix a1 and a2
for i = 1:popsize
    
    rand_index = floor(rand*popsize) + 1 ;
    temp = a1(rand_index);
    a1(rand_index) = a1(i);
    a1(i) = temp;

    rand_index = floor(rand*popsize) + 1 ;
    temp = a2(rand_index);
    a2(rand_index) = a2(i);
    a2(i) = temp;

end

% Initialize the mate pool
mate_pool_population = zeros(pool_size, n);

% Carry out the tournament selection operation
i=1;
% Pick pool_size parents and put them into the mate pool
while i <= pool_size
    
    temp_parent1 = chromosome(a1(i), :);
    temp_parent2 = chromosome(a1(i+1), :); 
    parent1 = tournament(temp_parent1, temp_parent2, n, M);
    
    temp_parent1 = chromosome(a1(i+2), :);
    temp_parent2 = chromosome(a1(i+3), :); 
    parent2 = tournament(temp_parent1, temp_parent2, n, M);
   
    mate_pool_population(i, 1:n) = parent1(1 : n);
    mate_pool_population(i+1, 1:n) = parent2(1 : n);
    
    temp_parent1 = chromosome(a2(i), :);
    temp_parent2 = chromosome(a2(i+1), :); 
    parent3 = tournament(temp_parent1, temp_parent2, n, M);
    
    temp_parent1 = chromosome(a2(i+2), :);
    temp_parent2 = chromosome(a2(i+3), :); 
    parent4 = tournament(temp_parent1, temp_parent2, n, M);
   
    mate_pool_population(i+2, 1:n) = parent3(1 : n);
    mate_pool_population(i+3, 1:n) = parent4(1 : n);
    
    i = i + 4;
    
end

mate_pool_population = mate_pool_population(1:popsize, :);
