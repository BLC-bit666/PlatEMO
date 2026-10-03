function offspring_chromosome = genetic_operator(mate_pool_chromosome, pool_size, n, lu, pcrossover, pmutation, di, dim)

%offspring！！offspring population
%popsize！！size of the offspring population
%n！！the number of decision variables
%lu！！ the range of the decision variables
%pmutation！！probability of mutation
%dim！！the distribution index of mutaion
%pcrossover！！probability of crossover
%di！！the distribution index of crossover

% Initialize the offspring population(pool_size * n)
offspring_chromosome = zeros( pool_size , n );

%crossover��pick two individual from the mate pool to be parent_1 and
%parent_2 and carry out the crossover operation
i = 1;
while i <= pool_size

    parent_1 = mate_pool_chromosome(i , :);
    parent_2 = mate_pool_chromosome(i+1 , :);

    [ child_1 , child_2 ] = crossover( parent_1 , parent_2 , n , lu , pcrossover , di );

    offspring_chromosome( i , :) = child_1;
    offspring_chromosome( i+1 , :) = child_2;

    i = i + 2;

end

% mutation
offspring_chromosome = mutation( offspring_chromosome, pool_size, n, lu, pmutation, dim );
