if fit(:, 2)>0

                indexSel = replacement(fit, lambda, mu);
                
                % Choose the individuals for the next population
                x = x(indexSel, :);
                eta = eta(indexSel, :);
                fit = fit(indexSel, :);

            % The population contains some feasible solutions
            elseif length(find(fit(:, 2) == 0)) < size(fit, 1)

                % Record the subscripts of the feasible solutions
                Z1 = find(fit(:, 2) == 0);
                
                % Record the subscripts of the infeasible solutions
                Z2 = find(fit(:, 2) > 0);

                % objFun denotes the objective function of the population
                objFun = fit(:, 1);
                
                % conVio denotes the degree of constraint violation of the
                % population
                conVio = fit(:, 2);

                % Compute the minimum and maximum objective function values
                % of the feasible solutions
                minObjFun = min(objFun(Z1));
                maxObjFun = max(objFun(Z1));

                % Compute the minimum and maximum degree of constraint
                % violation of the infeasible solutions
                minConVio = min(conVio(Z2));
                maxConVio = max(conVio(Z2));

                % Modify the objective function
                objFun(Z2) = max(objFun(Z2), ones(size(Z2, 1), 1) * (percent * minObjFun + (1 - percent) * maxObjFun));
                
                % Normalize the objective function
                norObjFun = (objFun- min(objFun)) ./ ((max(objFun) - min(objFun)) + 1E-30);

                % Normalize the degree of constraint violations
                norConVio = conVio;
                norConVio(Z2) = (conVio(Z2) - minConVio) ./ (maxConVio - minConVio + 1E-30);

                % Obtain the final objective function
                finalFit = norObjFun + norConVio;

                [valueSel, indexSel] = sort(finalFit);
                
                % Choose the individuals for the next population
                x = x(indexSel(1 : mu),  :);
                eta = eta(indexSel(1 : mu), :);
                fit = fit(indexSel(1 : mu), :);

            % The population contains only feasible solutions
            else

                [valueSel, indexSel] = sort(fit(:, 1));
                
                % Choose the individuals for the next population
                x = x(indexSel(1 : mu), :);
                eta = eta(indexSel(1 : mu), :);
                fit = fit(indexSel(1 : mu), :);

            end