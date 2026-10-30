classdef NCTP9_BC < NCTP9
% <2016> <multi> <real> <constrained>
% NCTP9 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP9(obj,X);
            C = double(any(C>0,2));
        end
    end
end
