function probe_critic_conditions(directory,updates)
% Forward-only direction mismatch probe; no training or objective evaluation.
if nargin<1; directory='runs'; end
if nargin<2; updates=10000; end
warning('off','all'); folder=fileparts(mfilename('fullpath')); rows={};
arms={'original','stable_side','stable_no_side'};
output='critic_condition_probe.csv';
if ~strcmp(directory,'runs'); arms={'original'}; output=sprintf('critic_condition_probe_mismatch_%d.csv',updates); end
for arm=arms
    for n=7:8
        stem=sprintf('%s_LIRCMOP%d_BC',arm{1},n); F=load(fullfile(folder,'fixtures',[stem,'.mat']));
        rows{end+1}=measure(F.Model,F,arm{1},n,0,0);
        for reset=0:3
            file=fullfile(folder,directory,sprintf('%s_reset%d.mat',stem,reset));
            csvFile=strrep(file,'.mat','.csv');
            if ~isfile(file) || ~isfile(csvFile); continue; end
            T=readtable(csvFile); at=find(T.addedUpdates==updates,1); if isempty(at); continue; end
            info=dir(file); if now-info.datenum<10/86400; continue; end
            R=load(file,'Snapshots','Complete'); if updates==10000 && ~R.Complete; continue; end
            rows{end+1}=measure(R.Snapshots{at}.Model,F,arm{1},n,reset,updates);
        end
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,output));
fprintf('CRITIC_CONDITION_PROBE_COMPLETE %d models; zero oracle and zero training\n',numel(rows));
end

function R=measure(M,F,arm,n,reset,updates)
D=F.ActualData; X=[D.xF;D.xI]; C=[D.cF;D.cI];
[~,keep]=unique([X,C(:,end)],'rows','stable'); X=X(keep,:); C=C(keep,:); side=C(:,end);
if ~F.useSide; C(:,end)=0; end
wrong=C; near=C; moderate=C;
for s=0:1
    ix=find(side==s); [~,order]=sort(C(ix,1)); ix=ix(order);
    wrong(ix,1:2)=C(circshift(ix,floor(numel(ix)/2)),1:2);
    paired=reshape(ix(1:2*floor(numel(ix)/2)),2,[]);
    near(paired(:),1:2)=C(reshape(flipud(paired),[],1),1:2);
    for start=1:10:numel(ix)
        block=ix(start:min(start+9,numel(ix))); moderate(block,1:2)=C(flipud(block),1:2);
    end
end
assert(isequal(sortrows(C),sortrows(wrong)) && isequal(sortrows(C),sortrows(near)) && isequal(sortrows(C),sortrows(moderate)));
real=single(2*X'-1); cond=single(C'); bad=single(wrong');
fake=extractdata(forward(M.netG,dlarray([zeros(M.zDim,size(X,1),'single');cond],'CB')));
scoreReal=double(extractdata(forward(M.netC,dlarray([real;cond],'CB'))));
scoreFake=double(extractdata(forward(M.netC,dlarray([fake;cond],'CB'))));
scoreWrong=double(extractdata(forward(M.netC,dlarray([real;bad],'CB'))));
scoreNear=double(extractdata(forward(M.netC,dlarray([real;single(near')],'CB'))));
scoreModerate=double(extractdata(forward(M.netC,dlarray([real;single(moderate')],'CB'))));
if F.useSide
    flipped=cond; flipped(end,:)=1-flipped(end,:);
    scoreSide=double(extractdata(forward(M.netC,dlarray([real;flipped],'CB'))));
    sideGap=mean(scoreReal-scoreSide); sidePreference=mean(scoreReal>scoreSide);
else; sideGap=NaN; sidePreference=NaN; end
R=struct('arm',string(arm),'problem',sprintf('LIRCMOP%d_BC',n),'reset',reset,'updates',updates, ...
    'deduplicatedRows',size(X,1),'fractionDirectionsChanged',mean(any(C(:,1:2)~=wrong(:,1:2),2)), ...
    'realMinusFakeScore',mean(scoreReal-scoreFake),'realMinusWrongDirectionScore',mean(scoreReal-scoreWrong), ...
    'correctDirectionPreference',mean(scoreReal>scoreWrong),'realMinusFlippedSideScore',sideGap, ...
    'correctSidePreference',sidePreference, ...
    'realMinusNearbyDirectionScore',mean(scoreReal-scoreNear), ...
    'realMinusModerateDirectionScore',mean(scoreReal-scoreModerate), ...
    'nearbyDirectionChangeDegrees',mean(directionChange(C,near)), ...
    'moderateDirectionChangeDegrees',mean(directionChange(C,moderate)), ...
    'farDirectionChangeDegrees',mean(directionChange(C,wrong)));
end

function angle=directionChange(A,B)
A=A(:,1:2)./vecnorm(A(:,1:2),2,2); B=B(:,1:2)./vecnorm(B(:,1:2),2,2);
angle=acosd(max(-1,min(1,sum(A.*B,2))));
end
