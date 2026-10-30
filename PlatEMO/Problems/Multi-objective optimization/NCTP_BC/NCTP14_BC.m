classdef NCTP14_BC < NCTP14
% <2016> <multi> <real> <constrained>
% NCTP14 with scalar binary feasibility: 0 feasible, 1 infeasible.
    methods
        function C = CalCon(obj,X)
            C = CalCon@NCTP14(obj,X);
            C = double(any(C>0,2));
        end
    end
end
