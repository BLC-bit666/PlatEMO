function [DataX,DataY] = NAEMTUpdateData(DataX,DataY,Archive,N1,quota)
% Update the fixed-size training data by balanced sampling and FIFO.

    if isempty(Archive)
        error('NAEMT:EmptyArchive', ...
            'The paper-prescribed MLP update cannot use an empty archive.');
    end
    if 2*quota > N1
        error('NAEMT:InvalidSampleQuota', ...
            'The feasible and infeasible quotas together must not exceed N1.');
    end
    feasible        = all(Archive.cons<=0,2);
    feasibleIndex  = find(feasible);
    infeasibleIndex = find(~feasible);
    if isempty(feasibleIndex) || isempty(infeasibleIndex)
        error('NAEMT:MissingArchiveClass', ...
            ['The paper requires both feasible and infeasible archive samples ' ...
             'for retraining, but one class is absent.']);
    end
    feasibleIndex   = SampleClass(feasibleIndex,quota);
    infeasibleIndex = SampleClass(infeasibleIndex,quota);

    NewX = [Archive(feasibleIndex).decs;Archive(infeasibleIndex).decs];
    NewY = [ones(quota,1);zeros(quota,1)];
    keep = N1 - size(NewX,1);
    if keep > 0
        DataX = DataX(end-keep+1:end,:);
        DataY = DataY(end-keep+1:end,:);
    else
        DataX = zeros(0,size(DataX,2));
        DataY = zeros(0,1);
    end
    DataX = [DataX;NewX];
    DataY = [DataY;NewY];
end

function index = SampleClass(candidates,quota)
% Use ordinary sampling, or bootstrap only when the paper's quota is scarce.

    if numel(candidates) >= quota
        index = candidates(randperm(numel(candidates),quota));
    else
        index = candidates(randi(numel(candidates),1,quota));
    end
end
