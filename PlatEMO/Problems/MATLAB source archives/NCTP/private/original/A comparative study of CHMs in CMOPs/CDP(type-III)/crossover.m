function [child1, child2] = crossover(parent1, parent2, n, lu, pcrossover, di)

parent_1 = parent1(1 : n);
parent_2 = parent2(1 : n);

%pcrossover=0.9 is the probability of crossover
if rand <= pcrossover 

    child1 = zeros(1,n);
    child2 = zeros(1,n);

    % crossover for every dimension
    for i = 1:n

        %abs(parent_1(i)-parent_2(i))>=1.0e-14,while the parents are not
        %the same
        if rand<=0.5
            if abs(parent_1(i)-parent_2(i))>1.0e-14
                if parent_2(i)>parent_1(i)
                    y2 = parent_2(i);
                    y1 = parent_1(i);
                else
                    y2 = parent_1(i);
                    y1 = parent_2(i);
                end

                y_low=lu(1,i);
                y_upper=lu(2,i);

                beta = 1.0 + (2.0*(y1-y_low)/(y2-y1));%%%%%%%%%%%%%%%%%%%%

                expp = -(di + 1.0);
                alpha = 2.0 - power(beta,expp);

                rnd=rand;
                if rnd <= 1.0/alpha

                    alpha = alpha*rnd;
                    expp = 1.0/(di+1.0);
                    betaq = power(alpha,expp);

                else

                    alpha = alpha*rnd;
                    alpha = 1.0/(2.0-alpha);
                    expp = 1.0/(di+1.0);
                    betaq = power(alpha,expp);

                end

                temp_child1 = 0.5*((y1+y2) - betaq*(y2-y1));

                beta = 1.0 + (2.0*(y_upper-y2)/(y2-y1));

                expp = -(di + 1.0);
                alpha = 2.0 - power(beta,expp);

                rnd=rand;
                if rnd <= 1.0/alpha

                    alpha = alpha*rnd;
                    expp = 1.0/(di+1.0);
                    betaq = power(alpha,expp);

                else

                    alpha = alpha*rnd;
                    alpha = 1.0/(2.0-alpha);
                    expp = 1.0/(di+1.0);
                    betaq = power(alpha,expp);

                end

                temp_child2 = 0.5*((y1+y2) + betaq*(y2-y1));

                if temp_child1 < y_low
                    temp_child1 = y_low;
                end
                if temp_child1 > y_upper
                    temp_child1 = y_upper;
                end
                if temp_child2 < y_low
                    temp_child2 = y_low;
                end
                if temp_child2 > y_upper
                    temp_child2 = y_upper;
                end

                if rand <= 0.5
                    child1(i) = temp_child2;
                    child2(i) = temp_child1;
                else
                    child1(i) = temp_child1;
                    child2(i) = temp_child2;
                end

            else

                child1(i) = parent_1(i);
                child2(i) = parent_2(i);
            end

        else

            child1(i) = parent_1(i);
            child2(i) = parent_2(i);

        end

    end

else

    child1 = parent_1;
    child2 = parent_2;

end
