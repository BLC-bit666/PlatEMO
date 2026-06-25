function test_CBS_CGAN_network_options()
%TEST_CBS_CGAN_NETWORK_OPTIONS Verify configurable CGAN hidden layers.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));

    rng(113,'twister');
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    TrainX = repmat(Problem.lower,5,1) + rand(5,Problem.D).* ...
        repmat(Problem.upper - Problem.lower,5,1);
    TrainC = rand(5,3);
    Options = struct( ...
        'zDim',2, ...
        'iter',1, ...
        'miniBatch',5, ...
        'advWeight',0, ...
        'reconstructionWeight',1, ...
        'generatorHidden',[16 8], ...
        'discriminatorHidden',[12]);

    GAN = BoundaryCGAN_CBS('train',[],TrainX,TrainC,Problem,Options);

    assert(layerOutputSize(GAN.netG,'g_fc1') == 16 && ...
            layerOutputSize(GAN.netG,'g_fc2') == 8, ...
        'Generator hidden layers must follow Options.generatorHidden.');
    assert(layerOutputSize(GAN.netD,'d_fc1') == 12, ...
        'Discriminator hidden layers must follow Options.discriminatorHidden.');
    assert(~hasLayer(GAN.netG,'g_fc3') && ~hasLayer(GAN.netD,'d_fc2'), ...
        'Custom hidden-layer vectors must not keep default extra layers.');
    fprintf('CBS-CGAN network option regressions passed.\n');
end

function tf = hasLayer(net,name)
    tf = any(string({net.Layers.Name}) == string(name));
end

function value = layerOutputSize(net,name)
    idx = find(string({net.Layers.Name}) == string(name),1,'first');
    assert(~isempty(idx),'Missing layer: %s.',name);
    value = double(net.Layers(idx).OutputSize);
end
