classdef UC1_C3_DTLZ3 < UC_DTLZ
% <2024> <multi/many> <real> <constrained>
% UC1-C3-DTLZ3 with the five noisy binary copies in supplement Eq. (40)
% referenceMode --- 1 --- 1: unfiltered NBI paper comparison; 0: feasible PF
%
%------------------------------- Reference --------------------------------
% S. Li, K. Li, W. Li, and M. Yang. Evolutionary alternating direction method
% of multipliers for constrained multi-objective optimization with unknown
% constraints. IEEE Transactions on Evolutionary Computation.
% DOI: 10.1109/TEVC.2024.3425629. Supplement Appendix A and Table I.
%--------------------------------------------------------------------------

    methods
        function Setting(obj)
            obj.Setting@UC_DTLZ(1,3,3);
        end
    end
end

