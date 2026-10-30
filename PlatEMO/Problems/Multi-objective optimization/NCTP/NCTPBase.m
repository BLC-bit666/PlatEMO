classdef NCTPBase < PROBLEM
% Shared definition for the eighteen NCTP problems. Use NCTP1--NCTP18.
%
% J.-P. Li, Y. Wang, S. Yang, and Z. Cai. A comparative study of
% constraint-handling techniques in evolutionary constrained multiobjective
% optimization. IEEE CEC, 2016, 4175-4182. DOI:10.1109/CEC.2016.7744320.
% Adapted from the authors' CDP(type-I/II/III)/fit_cv.m and main.m.
% Original archive and attribution: Problems/MATLAB source archives/NCTP.

    properties(SetAccess = protected)
        problemIndex;
    end
    methods
        function Setting(obj,index)
            obj.problemIndex = index;
            obj.M = 2;
            if isempty(obj.D); obj.D = 30; end
            validateattributes(obj.D,{'numeric'},{'scalar','integer','>=',3});
            obj.lower = zeros(1,obj.D);
            obj.upper = 5*ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
        end
        function F = CalObj(obj,X)
            g = sum(100*(X(:,3:end)-X(:,2:end-1).^2).^2 ...
                +(1-X(:,2:end-1)).^2,2);
            [~,~,z] = parameters(obj.problemIndex);
            % Algebraically identical for g>0; continuous and finite at g=0.
            F = [X(:,1),g-sqrt(X(:,1))+z];
        end
        function C = CalCon(obj,X)
            C = constraints(obj.CalObj(X),obj.problemIndex);
        end
        function R = GetOptimum(obj,N)
            % Deterministic numerical CPF reference, not the authors' coarse
            % PARETO files. Keep only attainable, feasible nondominated points.
            persistent fronts resolutions
            if isempty(fronts); fronts = cell(1,18); resolutions = zeros(1,18); end
            index = obj.problemIndex;
            resolution = max(50000,20*N);
            if resolutions(index)<resolution
                fronts{index} = referenceFront(index,resolution);
                resolutions(index) = resolution;
            end
            R = fronts{index};
            if size(R,1)>N
                R = R(unique(round(linspace(1,size(R,1),N))),:);
            end
        end
        function R = GetPF(obj)
            R = obj.GetOptimum(1000);
        end
    end
end

function [a,d,z] = parameters(index)
    j = mod(index-1,3)+1;
    amplitudes = [0.2,0.75,2]; powers = [0.5,0.5,6];
    shifts = [-0.5*ones(1,6),ones(1,6),2,2.5,4,2,2.5,4];
    a = amplitudes(j); d = powers(j); z = shifts(index);
end

function C = constraints(F,index)
    [a,d] = parameters(index);
    theta = -0.2*pi;
    u = sin(theta)*(F(:,2)-1)+cos(theta)*F(:,1);
    v = cos(theta)*(F(:,2)-1)-sin(theta)*F(:,1);
    C = a*abs(sin(10*pi*u)).^d-v;
    if mod(index-1,6)<3
        bounds = [4,4,6];
        C(:,2) = F(:,2)+0.73*F(:,1)-bounds(mod(index-1,3)+1);
    end
end

function R = referenceFront(index,resolution)
    [a,d,z] = parameters(index);
    x = linspace(0,5,resolution)';
    R = [x,z-sqrt(x)];
    if index<=12
        theta = -0.2*pi;
        % Parametrize the first-constraint boundary in rotated coordinates.
        % A small interior offset avoids infeasible roundoff at sine cusps.
        u = unique([linspace(-abs(tan(theta))*a,5/cos(theta),resolution)'; ...
            ((ceil(-10*abs(tan(theta))*a):floor(50/cos(theta)))/10)']);
        v = a*abs(sin(10*pi*u)).^d+1e-7;
        B = [cos(theta)*u-sin(theta)*v,1+sin(theta)*u+cos(theta)*v];
        B = B(B(:,1)>=0 & B(:,1)<=5,:);
        R = [R;B;edgePoint(0,index,z);edgePoint(5,index,z)];
    end
    g = R(:,2)+sqrt(R(:,1))-z;
    % g<=1600 is constructively attainable for every supported D: set
    % x2...x(D-1)=1 and xD=1+sqrt(g/100), within the original [0,5] bounds.
    R = R(g>=0 & g<=1600 & all(constraints(R,index)<=0,2),:);
    R = unique(R,'rows');
    R = R(R(:,2)<[inf;cummin(R(1:end-1,2))],:);
    assert(~isempty(R),'NCTP:EmptyReference','No feasible reference points.');
end

function F = edgePoint(x,index,z)
    y = linspace(z-sqrt(x),6,20001)';
    C = constraints([repmat(x,size(y)),y],index);
    j = find(all(C<=0,2),1);
    if isempty(j); F = zeros(0,2); return; end
    if j>1
        lo = y(j-1); hi = y(j);
        for k = 1:52
            mid = (lo+hi)/2;
            if all(constraints([x,mid],index)<=0); hi=mid; else; lo=mid; end
        end
        y(j) = hi;
    end
    F = [x,y(j)];
end
