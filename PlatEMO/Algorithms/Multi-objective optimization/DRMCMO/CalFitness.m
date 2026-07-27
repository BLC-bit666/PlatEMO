function Fitness = CalFitness(Problem,PopObj,PopCon,archiveObj,flag,r,maxR,alpha,PopDecs,arcDecs,maxCV)
    N = size(PopObj,1);
    if flag == 0
        CV = zeros(N,1);
    elseif flag == 1
        Cons                 = max(0,PopCon);
        NormCons             = Cons./repmat(max(1,max(Cons,[],1)),N,1);
        CV                   = sum(NormCons,2);
    elseif flag == 2
        Cons                 = max(0,PopCon);
        NormCons             = Cons./repmat(max(1,max(Cons,[],1)),N,1);
        CV                   = sum(NormCons,2);
        if alpha < 0.98    
            Dis = pdist2(PopObj,archiveObj+alpha*sqrt(r*r/2));
            B = any(Dis < r, 2);
            CV(B) = 0;
        end
    end
    

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
