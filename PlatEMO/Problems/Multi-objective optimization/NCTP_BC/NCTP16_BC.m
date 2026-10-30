classdef NCTP16_BC < NCTP16
% <2016> <multi> <real> <constrained>
% NCTP16 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP16(obj,X);
            C = double(any(C>0,2));
        end
    end
end
