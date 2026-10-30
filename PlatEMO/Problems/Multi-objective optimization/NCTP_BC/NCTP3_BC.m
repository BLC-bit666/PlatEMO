classdef NCTP3_BC < NCTP3
% <2016> <multi> <real> <constrained>
% NCTP3 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP3(obj,X);
            C = double(any(C>0,2));
        end
    end
end
