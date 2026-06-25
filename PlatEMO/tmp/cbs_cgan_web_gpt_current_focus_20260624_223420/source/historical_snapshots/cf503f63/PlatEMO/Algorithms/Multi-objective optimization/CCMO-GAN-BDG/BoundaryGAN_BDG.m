function varargout = BoundaryGAN_BDG(action,varargin)
% BoundaryGAN_BDG - Public dispatcher for boundary GAN training and sampling.
    switch lower(action)
        case 'train'
            varargout{1} = TrainBoundaryGAN_BDG(varargin{:});
        case 'sample'
            [varargout{1:nargout}] = SampleBoundaryGAN_BDG(varargin{:});
        case 'probe'
            varargout{1} = ProbeBoundaryGAN_BDG(varargin{:});
        otherwise
            error('BoundaryGAN_BDG:UnknownAction','Unknown action "%s".',action);
    end
end

function GAN = TrainBoundaryGAN_BDG(X,lower,upper,zDim,maxIter,GAN)
% Standard GAN trained only on AF.decs

    X = ScaleToTanh_BDG(X,lower,upper);
    X = single(X);
    D = size(X,2);

    if nargin < 6 || isempty(GAN)
        netG = CreateGenerator_BDG(zDim,D);
        netD = CreateDiscriminator_BDG(D);
    else
        netG = GAN.netG;
        netD = GAN.netD;
    end

    miniBatch = min(16,size(X,1));
    numData   = size(X,1);

    trailingAvgG   = [];
    trailingAvgSqG = [];
    trailingAvgD   = [];
    trailingAvgSqD = [];

    lrG = 2e-4;
    lrD = 1e-4;
    beta1 = 0.5;
    beta2 = 0.999;

    for iter = 1 : maxIter
        idx = randperm(numData,miniBatch);
        XBatch = dlarray(X(idx,:)',"CB");
        Z      = dlarray(randn(zDim,miniBatch,'single'),"CB");

        [lossD,gradD] = dlfeval(@DiscriminatorGradients_BDG,netG,netD,XBatch,Z);
        [netD,trailingAvgD,trailingAvgSqD] = adamupdate(...
            netD,gradD,trailingAvgD,trailingAvgSqD,iter,lrD,beta1,beta2);

        Z = dlarray(randn(zDim,miniBatch,'single'),"CB");
        [lossG,gradG] = dlfeval(@GeneratorGradients_BDG,netG,netD,Z);
        [netG,trailingAvgG,trailingAvgSqG] = adamupdate(...
            netG,gradG,trailingAvgG,trailingAvgSqG,iter,lrG,beta1,beta2);

        extractdata(lossD);
        extractdata(lossG);
    end

    GAN.netG  = netG;
    GAN.netD  = netD;
    GAN.lower = lower;
    GAN.upper = upper;
    GAN.zDim  = zDim;
end

function [Offspring,Diag] = SampleBoundaryGAN_BDG(Problem,GAN,AF,~,W,nGen,repairK)
% Sample from G, evaluate, and repair infeasible samples by short bisection

    Diag = EmptySampleDiag_BDG();
    if isempty(GAN) || isempty(AF.decs)
        Offspring = [];
        return;
    end

    X = GenerateRawDecisions_BDG(GAN,nGen);

    S    = Problem.Evaluation(X);
    good = IsFeasibleSet_BDG(S);
    Diag.raw_count    = numel(S);
    Diag.raw_decs     = S.decs;
    Diag.raw_objs     = S.objs;
    Diag.raw_feasible = good;

    Good = S(good);
    Bad  = S(~good);
    Repaired = S([]);
    Diag.repair_count = numel(Bad);

    for i = 1 : numel(Bad)
        refAll = AssignRefFromObj_BDG([Bad(i).objs;AF.objs],W);
        refBad = refAll(1);
        refAF  = refAll(2:end);

        cand = find(refAF == refBad);
        if isempty(cand)
            cand = 1 : size(AF.decs,1);
        end

        Y = NormalizeObjMat_BDG([AF.objs(cand,:);Bad(i).objs]);
        d = pdist2(Y(end,:),Y(1:end-1,:));
        [~,p] = min(d);
        anchor = cand(p);

        sF = Problem.Evaluation(AF.decs(anchor,:));
        [sF,~] = RefinePair_BDG(Problem,sF,Bad(i),repairK);
        Repaired(end+1) = sF; %#ok<AGROW>
    end

    Offspring = [Good,Repaired];
    if numel(Offspring) > nGen
        Offspring = Offspring(1:nGen);
    end
    Diag.repaired_count  = numel(Repaired);
    Diag.offspring_count = numel(Offspring);
end

function Diag = ProbeBoundaryGAN_BDG(Problem,GAN,nProbe)
% Generate and evaluate raw GAN samples without repair or population injection.
    Diag = struct( ...
        'probe_raw_count',0, ...
        'probe_raw_decs',zeros(0,Problem.D), ...
        'probe_raw_objs',zeros(0,Problem.M), ...
        'probe_raw_feasible',false(0,1), ...
        'probe_raw_FE',0);
    if isempty(GAN) || nProbe <= 0
        return;
    end
    nProbe = max(0,round(double(nProbe)));
    if nProbe == 0
        return;
    end
    X = GenerateRawDecisions_BDG(GAN,nProbe);
    S = Problem.Evaluation(X);
    Diag.probe_raw_count = numel(S);
    Diag.probe_raw_decs = S.decs;
    Diag.probe_raw_objs = S.objs;
    Diag.probe_raw_feasible = IsFeasibleSet_BDG(S);
    Diag.probe_raw_FE = numel(S);
end

function Diag = EmptySampleDiag_BDG()
    Diag = struct( ...
        'raw_count',0, ...
        'raw_decs',zeros(0,0), ...
        'raw_objs',zeros(0,0), ...
        'raw_feasible',false(0,1), ...
        'repair_count',0, ...
        'repaired_count',0, ...
        'offspring_count',0);
end

function X = GenerateRawDecisions_BDG(GAN,n)
    Z = dlarray(randn(GAN.zDim,n,'single'),"CB");
    X = predict(GAN.netG,Z);
    X = double(extractdata(X)');
    X = UnscaleFromTanh_BDG(X,GAN.lower,GAN.upper);
end

function netG = CreateGenerator_BDG(zDim,D)
    layersG = [
        featureInputLayer(zDim,Normalization="none",Name="in")
        fullyConnectedLayer(64,Name="fc1")
        reluLayer(Name="relu1")
        fullyConnectedLayer(64,Name="fc2")
        reluLayer(Name="relu2")
        fullyConnectedLayer(D,Name="fc3")
        tanhLayer(Name="tanh")];
    netG = dlnetwork(layerGraph(layersG));
end

function netD = CreateDiscriminator_BDG(D)
    layersD = [
        featureInputLayer(D,Normalization="none",Name="in")
        fullyConnectedLayer(64,Name="fc1")
        leakyReluLayer(0.2,Name="lrelu1")
        fullyConnectedLayer(32,Name="fc2")
        leakyReluLayer(0.2,Name="lrelu2")
        fullyConnectedLayer(1,Name="fc3")];
    netD = dlnetwork(layerGraph(layersD));
end

function [lossD,gradD] = DiscriminatorGradients_BDG(netG,netD,XReal,Z)
    XFake = forward(netG,Z);
    YReal = forward(netD,XReal);
    YFake = forward(netD,XFake);

    PReal = 1./(1+exp(-YReal));
    PFake = 1./(1+exp(-YFake));

    epsVal = single(1e-8);
    lossD = -mean(log(PReal+epsVal) + log(1-PFake+epsVal),'all');
    gradD = dlgradient(lossD,netD.Learnables);
end

function [lossG,gradG] = GeneratorGradients_BDG(netG,netD,Z)
    XFake = forward(netG,Z);
    YFake = forward(netD,XFake);
    PFake = 1./(1+exp(-YFake));

    epsVal = single(1e-8);
    lossG = -mean(log(PFake+epsVal),'all');
    gradG = dlgradient(lossG,netG.Learnables);
end

function X = ScaleToTanh_BDG(X,lower,upper)
    X = 2*((X - lower)./(upper - lower + 1e-12)) - 1;
    X = min(max(X,-1),1);
end

function X = UnscaleFromTanh_BDG(X,lower,upper)
    X     = double(X);
    lower = double(lower);
    upper = double(upper);
    X = (X + 1)/2 .* (upper - lower) + lower;
    X = min(max(X,lower),upper);
end

function flag = IsFeasibleSet_BDG(P)
    if isempty(P)
        flag = false(0,1);
        return;
    end
    if isempty(P.cons)
        flag = true(numel(P),1);
    else
        flag = all(P.cons <= 0,2);
    end
end

function [sF,sI] = RefinePair_BDG(Problem,sF,sI,K)
    for t = 1 : K
        sM = Problem.Evaluation((sF.decs + sI.decs)/2);
        if IsFeasibleSet_BDG(sM)
            sF = sM;
        else
            sI = sM;
        end
    end
end

function ref = AssignRefFromObj_BDG(PopObj,W,zmin,zmax)
    if nargin < 3
        Y = NormalizeObjMat_BDG(PopObj);
    else
        Y = NormalizeObjMat_BDG(PopObj,zmin,zmax);
    end
    Wn = W ./ sqrt(sum(W.^2,2) + 1e-12);
    Yn = Y ./ sqrt(sum(Y.^2,2) + 1e-12);
    [~,ref] = max(Yn*Wn',[],2);
end

function Y = NormalizeObjMat_BDG(Obj,zmin,zmax)
    if isempty(Obj)
        Y = Obj;
        return;
    end
    if nargin < 2
        zmin = min(Obj,[],1);
        zmax = max(Obj,[],1);
    end
    Y = (Obj - zmin) ./ (zmax - zmin + 1e-12);
    Y = min(max(Y,0),1);
end
