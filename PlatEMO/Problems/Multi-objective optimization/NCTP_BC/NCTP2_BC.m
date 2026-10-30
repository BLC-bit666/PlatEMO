classdef NCTP2_BC < NCTP2
% <2016> <multi> <real> <constrained>
% NCTP2 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP2(obj,X);
            C = double(any(C>0,2));
        end
    end
end
