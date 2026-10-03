% Direct-CGAN research campaign: four problems, five paired seeds, ten workers.
% Run in a dedicated MATLAB session. Frozen cases never reuse old result data.
rootPath = fileparts(mfilename('fullpath'));
addpath(fullfile(rootPath,'Experiments','GANCMO_Direct_20260927'));
caseName = 'gpt_v1'; % Baseline only; final selected configuration is not known yet.
run_campaign(caseName);
analyze_case(caseName);
