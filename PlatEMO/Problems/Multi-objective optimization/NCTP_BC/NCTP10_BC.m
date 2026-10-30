classdef NCTP10_BC < NCTP10
% <2016> <multi> <real> <constrained>
% NCTP10 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP10(obj,X);
            C = double(any(C>0,2));
        end
    end
end
