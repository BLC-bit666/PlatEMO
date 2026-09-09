function M=fresh_model(Template,initSeed)
% Same exact network factories, random weights and zero Adam moments.
saved=rng;cleanup=onCleanup(@()rng(saved));rng(initSeed,'twister');
M=Template;
M.netG=createGenerator(M.C+M.zDim,M.D,M.generatorHidden);
M.netC=createCritic(M.D+M.C,M.criticHidden);
M.avgG=[];M.avgSqG=[];M.avgC=[];M.avgSqC=[];M.iterG=0;M.iterC=0;M.version=0;
% Compatible architecture bypasses production initialization; training RNG is separate.
M.ready=true;
end

function netG = createGenerator(inputDim,D,hidden)
%CREATEGENERATOR Two hidden layers and bounded absolute output.

    layers = featureInputLayer(inputDim, ...
        'Normalization','none','Name','pair_guide_g_in');
    for i = 1 : numel(hidden)
        layers = [layers;fullyConnectedLayer(hidden(i), ...
            'Name',sprintf('pair_guide_g_fc%d',i)); ...
            leakyReluLayer(0.2, ...
            'Name',sprintf('pair_guide_g_lrelu%d',i))]; %#ok<AGROW>
    end
    layers = [layers;fullyConnectedLayer(D,'Name','pair_guide_g_out'); ...
        tanhLayer('Name','pair_guide_g_tanh')];
    netG = dlnetwork(layerGraph(layers));
end

function netC = createCritic(inputDim,hidden)
%CREATECRITIC Conditional endpoint critic with linear output.

    layers = featureInputLayer(inputDim, ...
        'Normalization','none','Name','pair_guide_c_in');
    for i = 1 : numel(hidden)
        layers = [layers;fullyConnectedLayer(hidden(i), ...
            'Name',sprintf('pair_guide_c_fc%d',i)); ...
            leakyReluLayer(0.2, ...
            'Name',sprintf('pair_guide_c_lrelu%d',i))]; %#ok<AGROW>
    end
    layers = [layers;fullyConnectedLayer(1,'Name','pair_guide_c_out')];
    netC = dlnetwork(layerGraph(layers));
end
