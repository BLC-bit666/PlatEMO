classdef ThreeBarTruss_BC < PROBLEM
% <2026> <multi> <real> <constrained>
% Three-bar truss: material volume and normalized displacement, binary label
%
% Source: 经典工程优化例题.pdf, section 34.2.5, pp. 8-9.
% x=[A1,A2], with symmetric diagonal areas A1=A3, H=100, P=sigma=2.
% All three PDF stress inequalities and the volume objective are retained.
% Objective 2 is E/(P*H) times displacement magnitude under the 45-deg load.
% The PDF does not give E; this normalized response requires no invented E.
% A1 lower bound changes from 0 to 0.1 to exclude singular zero sections.
% Every original feasible design has A1>=0.6589186225978911 for A2<=1,
% so this removes no original feasible design. A2=0 remains supported.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. PlatEMO research-use and citation terms apply.
%--------------------------------------------------------------------------

    methods
        function Setting(obj)
            obj.M        = 2;
            obj.D        = 2;
            obj.lower    = [0.1,0];
            obj.upper    = [1,1];
            obj.encoding = ones(1,obj.D);
        end
        function PopObj = CalObj(~,X)
            a = X(:,1); b = X(:,2);
            volume = 100.*(2*sqrt(2).*a+b);
            displacement = hypot(1./a,1./(a+sqrt(2).*b));
            PopObj = [volume,displacement];
        end
        function PopCon = CalCon(~,X)
            a = X(:,1); b = X(:,2);
            denominator = sqrt(2).*a.^2+2.*a.*b;
            RawCon = [2.*(sqrt(2).*a+b)./denominator-2, ...
                      2.*b./denominator-2,2./(a+sqrt(2).*b)-2];
            PopCon = double(any(RawCon./2>1e-12,2));
        end
        function R = GetOptimum(obj,~)
            F = obj.CalObj(obj.upper);
            % stress1=1/A1+1/(A1+sqrt(2)*A2)<=2 bounds the displacement norm.
            R = [F(1),2];
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
