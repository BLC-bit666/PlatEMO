classdef NCTP6_BC < NCTP6
% <2016> <multi> <real> <constrained>
% NCTP6 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP6(obj,X);
            C = double(any(C>0,2));
        end
    end
end
