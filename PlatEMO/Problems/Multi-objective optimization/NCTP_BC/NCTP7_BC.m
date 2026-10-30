classdef NCTP7_BC < NCTP7
% <2016> <multi> <real> <constrained>
% NCTP7 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP7(obj,X);
            C = double(any(C>0,2));
        end
    end
end
