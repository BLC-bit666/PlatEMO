function [CalDec,CalLabel,CalNear] = UpdateCalibrationBuffer(CalDec,CalLabel,CalNear,NewSolutions,NewInfo,MaxCal)
% Update a held-out labeled buffer used for calibration or evaluation.

    if nargin < 1 || isempty(CalDec)
        CalDec = [];
        CalLabel = [];
        CalNear = [];
    end
    if nargin < 6 || MaxCal <= 0
        CalDec = [];
        CalLabel = [];
        CalNear = [];
        return;
    end
    if isempty(NewSolutions)
        [CalDec,CalLabel,CalNear] = TrimCalibrationBuffer(CalDec,CalLabel,CalNear,MaxCal);
        return;
    end

    NewDec  = NewSolutions.decs;
    NewLab  = double(all(NewSolutions.cons<=0,2));
    NewNear = true(size(NewLab));
    if nargin >= 5 && isstruct(NewInfo) && isfield(NewInfo,'prob') && numel(NewInfo.prob) == numel(NewSolutions)
        NewNear = abs(NewInfo.prob(:)-0.5) <= 0.1;
    end

    AllDec  = [CalDec;NewDec];
    AllLab  = [CalLabel(:);NewLab(:)];
    AllNear = [CalNear(:);NewNear(:)];
    Keep = KeepLatestDecisionRows(AllDec);
    AllDec  = AllDec(Keep,:);
    AllLab  = AllLab(Keep);
    AllNear = AllNear(Keep);

    [CalDec,CalLabel,CalNear] = TrimCalibrationBuffer(AllDec,AllLab,AllNear,MaxCal);
end

function [Dec,Lab,Near] = TrimCalibrationBuffer(Dec,Lab,Near,MaxCal)
    Total = numel(Lab);
    Lab  = double(Lab(:));
    Near = logical(Near(:));
    if Total <= MaxCal
        return;
    end

    NearQuota = min(sum(Near),floor(0.5*MaxCal));
    KeepNear  = find(Near);
    KeepNear  = KeepNear(max(1,end-NearQuota+1):end);

    RemainMask = true(Total,1);
    RemainMask(KeepNear) = false;
    Space = MaxCal - numel(KeepNear);
    Feasible   = find(Lab==1 & RemainMask);
    Infeasible = find(Lab==0 & RemainMask);
    Quota = floor(Space/2);
    KeepF = Feasible(max(1,end-Quota+1):end);
    KeepI = Infeasible(max(1,end-Quota+1):end);
    Keep  = [KeepNear(:);KeepF(:);KeepI(:)];

    if numel(Keep) < MaxCal
        ExtraNeed = MaxCal - numel(Keep);
        Remain = setdiff(find(RemainMask),[KeepF(:);KeepI(:)],'stable');
        Extra  = Remain(max(1,end-ExtraNeed+1):end);
        Keep   = [Keep;Extra(:)];
    elseif numel(Keep) > MaxCal
        Keep = Keep(end-MaxCal+1:end);
    end

    Keep = sort(Keep);
    Dec  = Dec(Keep,:);
    Lab  = Lab(Keep);
    Near = Near(Keep);
end
