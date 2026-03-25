function [CalDec,CalLabel,CalNear,Status] = UpdateCalibrationBuffer( ...
    CalDec,CalLabel,CalNear,NewSolutions,NewInfo,MaxCal,FallbackDec,FallbackLabel)
% Update a held-out labeled buffer used for calibration or evaluation.

    if nargin < 1 || isempty(CalDec)
        CalDec = [];
        CalLabel = [];
        CalNear = [];
    end
    if nargin < 7 || isempty(FallbackDec)
        FallbackDec = zeros(0,size(CalDec,2));
        FallbackLabel = zeros(0,1);
    end
    if nargin < 6 || MaxCal <= 0
        CalDec = [];
        CalLabel = [];
        CalNear = [];
        Status = InitCalibrationBufferStatus(CalLabel,CalNear,MaxCal);
        return;
    end
    if isempty(NewSolutions)
        [CalDec,CalLabel,CalNear] = TrimCalibrationBuffer(CalDec,CalLabel,CalNear,MaxCal);
        Status = InitCalibrationBufferStatus(CalLabel,CalNear,MaxCal);
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
    [CalDec,CalLabel,CalNear] = RepairCalibrationBuffer( ...
        CalDec,CalLabel,CalNear,FallbackDec,FallbackLabel,MaxCal);
    Status = InitCalibrationBufferStatus(CalLabel,CalNear,MaxCal);
end

function [Dec,Lab,Near] = TrimCalibrationBuffer(Dec,Lab,Near,MaxCal)
    Total = numel(Lab);
    Lab  = double(Lab(:));
    Near = logical(Near(:));
    if Total > MaxCal
        Keep = SelectBalancedCalibrationRows(Lab,Near,MaxCal);
        Keep = sort(Keep);
        Dec  = Dec(Keep,:);
        Lab  = Lab(Keep);
        Near = Near(Keep);
    end
end

function Keep = SelectBalancedCalibrationRows(Lab,Near,MaxCal)
    Total = numel(Lab);
    MaxCal = min(MaxCal,Total);
    Keep = zeros(0,1);
    ClassValues = [1,0];

    % Reserve the newest sample from each class when both classes exist.
    for i = 1 : numel(ClassValues)
        ClassIdx = find(Lab == ClassValues(i));
        if ~isempty(ClassIdx)
            Keep(end+1,1) = ClassIdx(end); %#ok<AGROW>
        end
    end
    Keep = unique(Keep,'stable');
    if numel(Keep) >= MaxCal
        Keep = Keep(end-MaxCal+1:end);
        return;
    end

    Remaining = setdiff((1:Total)',Keep,'stable');
    NearQuota = min(sum(Near(Remaining)),max(0,floor(0.5*MaxCal)));
    NearIdx = Remaining(Near(Remaining));
    if NearQuota > 0
        TakeNear = min(NearQuota,numel(NearIdx));
        Keep = [Keep;NearIdx(max(1,end-TakeNear+1):end)];
        Keep = unique(Keep,'stable');
    end

    Remaining = setdiff((1:Total)',Keep,'stable');
    SlotsLeft = MaxCal - numel(Keep);
    if SlotsLeft <= 0
        Keep = Keep(end-MaxCal+1:end);
        return;
    end

    ClassFillCell = cell(numel(ClassValues),1);
    TargetPerClass = floor(SlotsLeft/2);
    for i = 1 : numel(ClassValues)
        ClassIdx = Remaining(Lab(Remaining) == ClassValues(i));
        Take = min(numel(ClassIdx),TargetPerClass);
        if Take > 0
            ClassFillCell{i} = ClassIdx(max(1,end-Take+1):end);
        end
    end
    ClassFill = vertcat(ClassFillCell{:});
    Keep = unique([Keep;ClassFill],'stable');

    Remaining = setdiff((1:Total)',Keep,'stable');
    SlotsLeft = MaxCal - numel(Keep);
    if SlotsLeft > 0
        Keep = [Keep;Remaining(max(1,end-SlotsLeft+1):end)];
    end
    Keep = unique(Keep,'stable');
    if numel(Keep) > MaxCal
        Keep = Keep(end-MaxCal+1:end);
    end
end

function [Dec,Lab,Near] = RepairCalibrationBuffer(Dec,Lab,Near,FallbackDec,FallbackLabel,MaxCal)
    if isempty(FallbackDec) || isempty(FallbackLabel)
        return;
    end
    Status = InitCalibrationBufferStatus(Lab,Near,MaxCal);
    if Status.valid
        return;
    end

    FallbackLabel = double(FallbackLabel(:) > 0);
    PresentClass = unique(double(Lab(:) > 0));
    NeededClass = setdiff([1;0],PresentClass,'stable');
    if isempty(Lab)
        NeededClass = [1;0];
    end

    AddIdx = zeros(0,1);
    for i = 1 : numel(NeededClass)
        ClassIdx = find(FallbackLabel == NeededClass(i));
        if ~isempty(ClassIdx)
            AddIdx(end+1,1) = ClassIdx(end); %#ok<AGROW>
        end
    end
    if isempty(AddIdx)
        return;
    end

    Dec = [Dec;FallbackDec(AddIdx,:)];
    Lab = [double(Lab(:) > 0);FallbackLabel(AddIdx)];
    Near = [logical(Near(:));false(numel(AddIdx),1)];
    Keep = KeepLatestDecisionRows(Dec);
    Dec = Dec(Keep,:);
    Lab = Lab(Keep);
    Near = Near(Keep);
    [Dec,Lab,Near] = TrimCalibrationBuffer(Dec,Lab,Near,MaxCal);
end

function Status = InitCalibrationBufferStatus(Label,Near,MaxCal)
    Label = double(Label(:));
    Near = logical(Near(:));
    ClassValues = unique(Label);
    ClassValues = ClassValues(isfinite(ClassValues));
    FeasibleCount = sum(Label > 0);
    InfeasibleCount = sum(Label <= 0);

    Status = struct();
    Status.count = numel(Label);
    Status.nearCount = sum(Near);
    Status.maxCount = MaxCal;
    Status.feasibleCount = FeasibleCount;
    Status.infeasibleCount = InfeasibleCount;
    Status.classCount = numel(ClassValues);
    Status.valid = false;
    Status.singleClass = false;
    Status.status = 'invalid_empty';
    if Status.count == 0
        return;
    end

    if Status.classCount < 2
        Status.singleClass = true;
        Status.status = 'invalid_single_class';
        return;
    end

    Status.valid = true;
    Status.status = 'valid_dual_class';
end
