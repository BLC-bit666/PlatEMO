function offspring_chromosome = mutation(offspring, popsize, n, lu, pmutation, dim)

%offspring！！offspring population
%popsize！！size of the offspring population
%n！！the number of decision variables
%lu！！ the range of the decision variables
%pmutation！！probability of mutation
%dim！！the distribution index of mutaion

offspring_chromosome = offspring;

% run the mutation operator to every individual
for i = 1 : popsize

    for j = 1 : n

        if rand <= pmutation

            y = offspring(i,j);
            y_low = lu(1,j);
            y_upper = lu(2,j);

            delta1 = (y - y_low)/(y_upper - y_low);
            delta2 = (y_upper - y)/(y_upper - y_low);

            rnd = rand;

            indi = 1.0/(dim + 1.0);

            if rnd <= 0.5

                xy = 1.0 - delta1;
                val = 2.0*rnd + (1.0 - 2.0*rnd) * power(xy,(dim+1));
                deltaq =  power(val,indi) - 1.0;

            else

                xy = 1.0 - delta2;
                val = 2.0*(1.0-rnd) + 2.0*(rnd-0.5) * power(xy,(dim+1));
                deltaq = 1.0 - power(val,indi);

            end

            y = y + deltaq * (y_upper - y_low);

            if y < y_low
                y = y_low;
            end

            if y > y_upper
                y = y_upper;
            end

            offspring_chromosome(i , j) = y;

        end

    end

end
