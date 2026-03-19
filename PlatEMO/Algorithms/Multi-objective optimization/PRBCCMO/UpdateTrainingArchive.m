function [TrainDec,TrainLabel] = UpdateTrainingArchive(TrainDec,TrainLabel,ProtectedDec,ProtectedLabel,NewSolutions,HoldoutDec,MaxTrain)
% Update a balanced FIFO training archive for the boundary MLP.

    if nargin < 1 || isempty(TrainDec)
        TrainDec = [];
        TrainLabel = [];
    end
    if nargin < 3 || isempty(ProtectedDec)
        ProtectedDec = [];
        ProtectedLabel = [];
    end
    if nargin < 6 || isempty(HoldoutDec)
        HoldoutDec = [];
    end
    if nargin < 7 || MaxTrain <= 0
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

    AllDec = [TrainDec;NewDec;ProtectedDec];
    AllLab = [TrainLabel(:);NewLab(:);ProtectedLabel(:)];
    ProtectFlag = [false(size(TrainDec,1)+size(NewDec,1),1);true(size(ProtectedDec,1),1)];
    if isempty(AllDec)
        return;
    end
    Keep = KeepLatestDecisionRows(AllDec);
    AllDec = AllDec(Keep,:);
    AllLab = AllLab(Keep);
    ProtectFlag = ProtectFlag(Keep);
    if ~isempty(HoldoutDec)
        Keep = ~ismember(AllDec,HoldoutDec,'rows');
        AllDec = AllDec(Keep,:);
        AllLab = AllLab(Keep);
        ProtectFlag = ProtectFlag(Keep);
    end

    [TrainDec,TrainLabel] = TrimBalancedFIFO(AllDec,AllLab,ProtectFlag,MaxTrain);
end

function [Dec,Lab] = TrimBalancedFIFO(Dec,Lab,ProtectFlag,MaxTrain)
    Total = length(Lab);
    Lab   = double(Lab(:));
    if Total <= MaxTrain
        return;
    end

    HardQuota = min(sum(ProtectFlag),min(max(1,round(0.25*MaxTrain)),MaxTrain));
    Protected = find(ProtectFlag);
    if HardQuota > 0
        Protected = Protected(max(1,end-HardQuota+1):end);
    else
        Protected = zeros(0,1);
    end
    RemainMask = true(Total,1);
    RemainMask(Protected) = false;

    Feasible   = find(Lab==1 & RemainMask);
    Infeasible = find(Lab==0 & RemainMask);
    Keep       = Protected;
    Space      = MaxTrain - length(Keep);
    if Space <= 0
        Keep = Keep(end-MaxTrain+1:end);
        Dec  = Dec(Keep,:);
        Lab  = Lab(Keep);
        return;
    end

    if isempty(Feasible) || isempty(Infeasible)
        Remain = find(RemainMask);
        Extra  = Remain(max(1,end-Space+1):end);
        Keep   = sort([Keep;Extra(:)]);
        Dec  = Dec(Keep,:);
        Lab  = Lab(Keep);
        return;
    end

    Quota = floor(Space/2);
    KeepF = Feasible(max(1,end-Quota+1):end);
    KeepI = Infeasible(max(1,end-Quota+1):end);
    Keep  = sort([KeepF(:);KeepI(:);Keep]);

    if length(Keep) < MaxTrain
        ExtraNeed = MaxTrain - length(Keep);
        Remain    = setdiff(1:Total,Keep,'stable');
        Extra     = Remain(max(1,end-ExtraNeed+1):end);
        Keep      = sort([Keep;Extra(:)]);
    elseif length(Keep) > MaxTrain
        Keep = Keep(end-MaxTrain+1:end);
    end

    Dec = Dec(Keep,:);
    Lab = Lab(Keep);
end
