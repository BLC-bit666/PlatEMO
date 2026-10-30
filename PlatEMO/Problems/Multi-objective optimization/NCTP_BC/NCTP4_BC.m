classdef NCTP4_BC < NCTP4
% <2016> <multi> <real> <constrained>
% NCTP4 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP4(obj,X);
            C = double(any(C>0,2));
        end
    end
end
