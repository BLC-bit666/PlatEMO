function [TrainDec,TrainLabel] = UpdateTrainingArchive(TrainDec,TrainLabel,NewSolutions,BoundaryArchive,MaxTrain)
% Update a balanced FIFO training archive for the boundary MLP.

    if nargin < 1 || isempty(TrainDec)
        TrainDec = [];
        TrainLabel = [];
    end
    if nargin < 4 || isempty(BoundaryArchive)
        BoundaryArchive = [];
    end
    if nargin < 5 || MaxTrain <= 0
        TrainDec   = [];
        TrainLabel = [];
        return;
    end

    NewDec = [];
    NewLab = [];
    if ~isempty(NewSolutions)
        NewDec = NewSolutions.decs;
        NewLab = double(all(NewSolutions.cons<=0,2));
    end

    ArcDec = [];
    ArcLab = [];
    if ~isempty(BoundaryArchive)
        ArcDec = BoundaryArchive.decs;
        ArcLab = double(all(BoundaryArchive.cons<=0,2));
    end

    AllDec = [TrainDec;NewDec;ArcDec];
    AllLab = [TrainLabel(:);NewLab(:);ArcLab(:)];
    if isempty(AllDec)
        return;
    end
    Keep = KeepLatestDecisionRows(AllDec);
    AllDec = AllDec(Keep,:);
    AllLab = AllLab(Keep);

    [TrainDec,TrainLabel] = TrimBalancedFIFO(AllDec,AllLab,MaxTrain);
end

function [Dec,Lab] = TrimBalancedFIFO(Dec,Lab,MaxTrain)
    Total = length(Lab);
    if Total <= MaxTrain
        Lab = double(Lab(:));
        return;
    end

    Feasible   = find(Lab==1);
    Infeasible = find(Lab==0);
    if isempty(Feasible) || isempty(Infeasible)
        Keep = max(1,Total-MaxTrain+1) : Total;
        Dec  = Dec(Keep,:);
        Lab  = double(Lab(Keep));
        return;
    end

    Quota = floor(MaxTrain/2);
    KeepF = Feasible(max(1,end-Quota+1):end);
    KeepI = Infeasible(max(1,end-Quota+1):end);
    Keep  = sort([KeepF(:);KeepI(:)])';

    if length(Keep) < MaxTrain
        ExtraNeed = MaxTrain - length(Keep);
        Remain    = setdiff(1:Total,Keep,'stable');
        Extra     = Remain(max(1,end-ExtraNeed+1):end);
        Keep      = sort([Keep,Extra]);
    elseif length(Keep) > MaxTrain
        Keep = Keep(end-MaxTrain+1:end);
    end

    Dec = Dec(Keep,:);
    Lab = double(Lab(Keep));
end
