function reproduce_offset(modelFile)
% Deterministic symptom check through the production generator forward path.
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
if nargin==0, modelFile=fullfile(root,'Data','PairGuideTrainingDiagnosis_20260907','LIRCMOP7_BC_FE099800_model.mat'); end
R=load(modelFile);
if isfield(R,'B'), M=R.B.Model; D=R.B.Data; O=R.B.Options; else, M=R.M; D=R.D; O=R.O; end
P=LIRCMOP7_BC(); rng(20260907,'twister'); repeat=64;
[~,Q]=PairBoundaryWGAN_RC('sample',M,repelem(D.cF,repeat,1),P,O);
error=sqrt(mean([Q.generatedF-repelem(D.xF,repeat,1);Q.generatedI-repelem(D.xI,repeat,1)].^2,'all'));
fprintf('OFFSET_REPRO pairs=%d normalized_endpoint_RMSE=%.9f\n',D.count,error);
assert(error<0.02,'OffsetRepro:EndpointFit', ...
    'Actual trained model misses its own endpoint targets by %.4f RMS of the decision range (diagnostic threshold: 0.02).',error);
end
