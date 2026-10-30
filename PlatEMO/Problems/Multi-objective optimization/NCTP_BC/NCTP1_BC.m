classdef NCTP1_BC < NCTP1
% <2016> <multi> <real> <constrained>
% NCTP1 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP1(obj,X);
            C = double(any(C>0,2));
        end
    end
end
