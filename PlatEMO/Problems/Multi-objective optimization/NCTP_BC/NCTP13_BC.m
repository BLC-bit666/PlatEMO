classdef NCTP13_BC < NCTP13
% <2016> <multi> <real> <constrained>
% NCTP13 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP13(obj,X);
            C = double(any(C>0,2));
        end
    end
end
