function Fitness = CalFitness_CBS(PopObj,PopCon)
%CALFITNESS_CBS SPEA2-style fitness with optional constraint dominance.
%   This is an algorithm-specific fitness assignment rather than a variation
%   operator. Pairwise strengths are required by the CCMO/SPEA2 selection;
%   they cannot be reconstructed from the platform NDSort front numbers.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate constraint violation
    N = size(PopObj,1);
    if nargin < 2 || isempty(PopCon)
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end

    %% Detect pairwise dominance
    left  = reshape(PopObj,N,1,[]);
    right = reshape(PopObj,1,N,[]);
    objectiveDominates = all(left <= right,3) & any(left < right,3);
    Dominate = CV < CV' | (CV == CV' & objectiveDominates);
    Dominate(1:N+1:end) = false;

    %% Calculate strength and raw fitness
    Strength = sum(Dominate,2);
    Raw = Strength'*double(Dominate);

    %% Calculate density and final fitness
    Distance2 = max(sum(PopObj.^2,2) + sum(PopObj.^2,2)' - ...
        2*(PopObj*PopObj'),0);
    Distance2(1:N+1:end) = inf;
    kth = max(1,min(N,floor(sqrt(N))));
    Nearest2 = mink(Distance2,kth,2);
    Density = 1./(sqrt(Nearest2(:,end))+2);
    Fitness = Raw + Density';
end
