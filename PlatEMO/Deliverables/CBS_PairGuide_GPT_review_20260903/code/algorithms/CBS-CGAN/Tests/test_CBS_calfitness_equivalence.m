function test_CBS_calfitness_equivalence()
%TEST_CBS_CALFITNESS_EQUIVALENCE Compare vectorized and legacy fitness.

    rng(19,'twister');
    sizes = [1 2 7 31 100];
    for n = sizes
        for trial = 1 : 10
            objectives = rand(n,3);
            constraints = randn(n,2);
            actualConstrained = CalFitness_CBS(objectives,constraints);
            expectedConstrained = legacyFitness(objectives,constraints);
            assert(isequaln(actualConstrained,expectedConstrained));

            actualUnconstrained = CalFitness_CBS(objectives);
            expectedUnconstrained = legacyFitness(objectives,[]);
            assert(isequaln(actualUnconstrained,expectedUnconstrained));
        end
    end
    fprintf('CBS CalFitness vectorization equivalence test passed.\n');
end

function Fitness = legacyFitness(PopObj,PopCon)
    N = size(PopObj,1);
    if nargin < 2 || isempty(PopCon)
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                iDominates = all(PopObj(i,:) <= PopObj(j,:)) && ...
                    any(PopObj(i,:) < PopObj(j,:));
                jDominates = all(PopObj(j,:) <= PopObj(i,:)) && ...
                    any(PopObj(j,:) < PopObj(i,:));
                Dominate(i,j) = iDominates;
                Dominate(j,i) = jDominates;
            end
        end
    end
    Strength = sum(Dominate,2);
    Raw = zeros(1,N);
    for i = 1 : N
        Raw(i) = sum(Strength(Dominate(:,i)));
    end
    Distance = sqrt(max(sum(PopObj.^2,2) + ...
        sum(PopObj.^2,2)' - 2*(PopObj*PopObj'),0));
    Distance(logical(eye(N))) = inf;
    Distance = sort(Distance,2);
    kth = max(1,min(size(Distance,2),floor(sqrt(N))));
    Density = 1./(Distance(:,kth)+2);
    Fitness = Raw + Density';
end
