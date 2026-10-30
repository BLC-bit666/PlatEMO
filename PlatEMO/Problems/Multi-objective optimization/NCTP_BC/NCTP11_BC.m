classdef NCTP11_BC < NCTP11
% <2016> <multi> <real> <constrained>
% NCTP11 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP11(obj,X);
            C = double(any(C>0,2));
        end
    end
end
