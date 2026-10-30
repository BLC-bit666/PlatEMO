classdef NCTP8_BC < NCTP8
% <2016> <multi> <real> <constrained>
% NCTP8 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP8(obj,X);
            C = double(any(C>0,2));
        end
    end
end
