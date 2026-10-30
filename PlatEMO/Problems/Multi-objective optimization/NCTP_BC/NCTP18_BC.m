classdef NCTP18_BC < NCTP18
% <2016> <multi> <real> <constrained>
% NCTP18 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP18(obj,X);
            C = double(any(C>0,2));
        end
    end
end
