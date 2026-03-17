function HardNegativeArchive = UpdateHardNegativeArchive(HardNegativeArchive,NewBatch,MaxSize)
% Update the archive of locally confirmed hard-negative neighborhoods.

    if nargin < 1 || isempty(HardNegativeArchive)
        HardNegativeArchive.Dec    = zeros(0,0);
        HardNegativeArchive.Radius = zeros(0,1);
    end
    if nargin < 3 || MaxSize <= 0
        HardNegativeArchive.Dec    = zeros(0,size(HardNegativeArchive.Dec,2));
        HardNegativeArchive.Radius = zeros(0,1);
        return;
    end
    if isempty(NewBatch) || ~isstruct(NewBatch) || ~isfield(NewBatch,'Dec') || isempty(NewBatch.Dec)
        if size(HardNegativeArchive.Dec,1) > MaxSize
            Keep = size(HardNegativeArchive.Dec,1)-MaxSize+1 : size(HardNegativeArchive.Dec,1);
            HardNegativeArchive.Dec = HardNegativeArchive.Dec(Keep,:);
            HardNegativeArchive.Radius = HardNegativeArchive.Radius(Keep);
        end
        return;
    end

    AllDec = [HardNegativeArchive.Dec;NewBatch.Dec];
    AllRad = [HardNegativeArchive.Radius(:);NewBatch.Radius(:)];
    Keep = KeepLatestDecisionRows(AllDec);
    AllDec = AllDec(Keep,:);
    AllRad = AllRad(Keep);

    if size(AllDec,1) > MaxSize
        Keep = size(AllDec,1)-MaxSize+1 : size(AllDec,1);
        AllDec = AllDec(Keep,:);
        AllRad = AllRad(Keep);
    end

    HardNegativeArchive.Dec = AllDec;
    HardNegativeArchive.Radius = AllRad;
end
