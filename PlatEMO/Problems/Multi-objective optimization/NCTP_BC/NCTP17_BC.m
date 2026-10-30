classdef NCTP17_BC < NCTP17
% <2016> <multi> <real> <constrained>
% NCTP17 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP17(obj,X);
            C = double(any(C>0,2));
        end
    end
end
