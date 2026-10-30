classdef NCTP15_BC < NCTP15
% <2016> <multi> <real> <constrained>
% NCTP15 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP15(obj,X);
            C = double(any(C>0,2));
        end
    end
end
