classdef NCTP12_BC < NCTP12
% <2016> <multi> <real> <constrained>
% NCTP12 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP12(obj,X);
            C = double(any(C>0,2));
        end
    end
end
