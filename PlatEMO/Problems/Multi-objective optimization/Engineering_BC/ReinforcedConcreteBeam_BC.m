classdef ReinforcedConcreteBeam_BC < PROBLEM
% <2026> <multi> <real> <constrained>
% Reinforced concrete beam: PDF cost and steel area, binary feasibility
%
% Source: 经典工程优化例题.pdf, section 34.2.12, pp. 19-21.
% x=[As,h,b], all continuous. The PDF numerical variable order, bounds,
% original cost and both inequalities are retained. Names follow the
% equations: x2 is depth and x3 is width. Steel area is added as objective 2.
% Every in-box design, including infeasible ones, has two finite objectives.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. PlatEMO research-use and citation terms apply.
%--------------------------------------------------------------------------

    methods
        function Setting(obj)
            obj.M        = 2;
            obj.D        = 3;
            obj.lower    = [1,28,5];
            obj.upper    = [28,40,10];
            obj.encoding = ones(1,obj.D);
        end
        function PopObj = CalObj(~,X)
            As = X(:,1); h = X(:,2); b = X(:,3);
            PopObj = [2.9.*As+0.6.*h.*b,As];
        end
        function PopCon = CalCon(~,X)
            As = X(:,1); h = X(:,2); b = X(:,3);
            RawCon = [h./b-4,180+7.375.*As.^2./b-As.*h];
            PopCon = double(any(RawCon./[4,180]>1e-12,2));
        end
        function R = GetOptimum(obj,~)
            % Fixed box upper bounds for native HV; not a Pareto front.
            R = obj.CalObj(obj.upper);
        end
        function score = CalMetric(obj,metName,Population)
            if strcmp(metName,'HV')
                score = CalMetric@PROBLEM(obj,metName,Population);
            else
                score = feval(metName,Population,[]);
            end
        end
    end
end
