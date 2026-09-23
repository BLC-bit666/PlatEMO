function [z,quota,cutoff] = GANCMOControl(P1,P2,W,Scale,z,maxFE,batch)
% One coverage-weighted state controls quota and the last usable FE batch.
% No objective/constraint evaluation or random draw is performed here.
    Current = [P1,P2];
    X = double(Current.decs); Y = double(Current.objs); G = double(Current.cons);
    valid = ~isempty(X) && ~isempty(W) && all(isfinite(X),'all') && ...
        all(isfinite(Y),'all') && all(isfinite(G),'all') && ...
        all(isfinite(Scale.minimum)) && all(isfinite(Scale.span)) && all(Scale.span>0);
    if valid
        [~,keep] = unique(X,'rows','stable'); Y = Y(keep,:); G = G(keep,:);
        feasible = all(G<=0,2);
        if any(feasible)
            fu = NDSort(Y,inf); iu = fu(:)==1;
            fc = NDSort(Y(feasible,:),inf); ic = false(size(iu));
            indices = find(feasible); ic(indices(fc(:)==1)) = true;
            S = 1-nnz(iu&ic)/nnz(iu|ic);
            ru = GANCMOReference(Y(iu,:),W,Scale);
            rc = GANCMOReference(Y(ic,:),W,Scale);
            C = min(numel(unique(ru)),numel(unique(rc)))/size(W,1);
            if C > 0
                gain = .04*C; target = 10+20*S;
                z = (1-gain)*z+gain*target;
            end
        end
    end
    quota = min(30,max(10,round(z)));
    fraction = min(.7,max(.2,.2+.5*(z-10)/20));
    cutoff = floor(fraction*maxFE/batch+1e-10)*batch;
end
