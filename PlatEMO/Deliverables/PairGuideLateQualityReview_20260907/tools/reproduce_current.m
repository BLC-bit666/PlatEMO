function reproduce_current(mode,nWorker)
% Portable runner; writes NEW results, never starts automatically on review.
% mode: 'first_use' (one LIRCMOP7 seed 1 case) or 'full' (36 searches).
if nargin<1; mode='first_use'; end
if nargin<2; nWorker=1; end
packageRoot=fileparts(fileparts(mfilename('fullpath')));
root=fullfile(packageRoot,'code');
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));
addCBSPaths(root);
saved=jsondecode(fileread(fullfile(packageRoot,'experiments', ...
    'PairGuideSinglePoint_20260907','selected_configuration.json')));
O=saved.trainingOptions;
O.generatorHidden=reshape(O.generatorHidden,1,[]);
O.criticHidden=reshape(O.criticHidden,1,[]);
folder=tempname(fullfile(packageRoot,'reproduced'));
if ~isfolder(fileparts(folder)); mkdir(fileparts(folder)); end
mkdir(folder);
switch mode
    case 'first_use'
        run_PairGuide_single_first_use(root,folder,7,1,'review1000',O);
    case 'full'
        Options=struct('problems',["LIRCMOP5_BC","LIRCMOP6_BC","LIRCMOP7_BC","LIRCMOP8_BC"], ...
            'seeds',1:3,'N',100,'maxFE',100000, ...
            'modes',["cgan","fallback_only","pair_only"], ...
            'checkpointFE',[10000 30000 50000 70000 100000], ...
            'distributionLayout',"training_pair",'outputDir',string(folder), ...
            'trainingOptions',O);
        run_PairGuide_validation(root,nWorker,Options);
    otherwise
        error('Review:UnknownMode','Use first_use or full.');
end
fprintf('Reproduction output: %s\n',folder);
end
