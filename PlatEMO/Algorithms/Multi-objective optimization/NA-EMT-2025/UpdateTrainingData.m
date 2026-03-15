function [DataX,DataY] = UpdateTrainingData(DataX,DataY,ArchiveSet,N1,SampleQuota)
% Update the training data following the paper's FIFO strategy

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    FeasibleMask  = ~any(ArchiveSet.cons>0,2);
    FeasibleSet   = ArchiveSet(FeasibleMask);
    InfeasibleSet = ArchiveSet(~FeasibleMask);
    if numel(FeasibleSet) < SampleQuota || numel(InfeasibleSet) < SampleQuota
        error('NAEMT2025:InsufficientArchiveSamples', ...
            ['Paper-faithful retraining requires at least %d feasible and %d ' ...
             'infeasible samples in the 10-generation archive.'], ...
            SampleQuota,SampleQuota);
    end

    FeasibleIndex   = randperm(numel(FeasibleSet),SampleQuota);
    InfeasibleIndex = randperm(numel(InfeasibleSet),SampleQuota);
    NewX            = [FeasibleSet(FeasibleIndex).decs; InfeasibleSet(InfeasibleIndex).decs];
    NewY            = [ones(SampleQuota,1); zeros(SampleQuota,1)];

    Keep = N1 - size(NewX,1);
    if Keep > 0
        DataX = DataX(end-Keep+1:end,:);
        DataY = DataY(end-Keep+1:end,:);
    else
        DataX = zeros(0,size(DataX,2));
        DataY = zeros(0,1);
    end
    DataX = [DataX; NewX];
    DataY = [DataY; NewY];
end
