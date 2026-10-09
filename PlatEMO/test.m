% Current GANCMO local regressions. Complete paired-search evidence is kept
% in Experiments/GANCMO_Activation_20261008 and GANCMO_Speed_20261009.
rootPath=fileparts(mfilename('fullpath'));
addpath(fullfile(rootPath,'Tests'));
test_GANCMO_mainline;
test_GANCMO_coverage;
