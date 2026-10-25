function Dec = GANCMO_V1Candidates(Model,W,Problem,quota,allowUse)
% One candidate per visited direction; send finite proposals directly to evaluation.
    if nargin<5; allowUse=true; end
    Dec=zeros(0,Problem.D);
    if quota<=0 || isempty(Model); return; end
    directionCount=size(W,1);
    % Preserve the tested single-candidate noise mapping and query stream.
    refs=randperm(directionCount)'; refs=refs(randperm(directionCount));
    noise=Model.settings.noiseSigma*randn(directionCount,Problem.D);
    directionOrder=randperm(directionCount);
    if ~allowUse; return; end
    Dec=zeros(quota,Problem.D);count=0;next=1;
    while count<quota && next<=directionCount
        last=min(directionCount,next+max(8,quota-count)-1);
        directions=directionOrder(next:last);next=last+1;
        slots=find(ismember(refs,directions));
        % Repeated padding inputs retain the numerical kernel; their outputs
        % are discarded and do not create additional candidates or evaluations.
        sampleSlots=[slots;repmat(slots(end),max(0,min(16,directionCount)-numel(slots)),1)];
        X=GANCMO_V1Model.sample(Model,double(W(refs(sampleSlots),:)),Problem,noise(sampleSlots,:));
        X=X(1:numel(slots),:);
        valid=all(isfinite(X),2);X=min(Problem.upper,max(Problem.lower,X));
        for ref=directions
            row=find(refs(slots)==ref & valid,1);
            if isempty(row);continue;end
            count=count+1;Dec(count,:)=double(X(row,:));
            if count>=quota;break;end
        end
    end
    Dec=Dec(1:count,:);
end
