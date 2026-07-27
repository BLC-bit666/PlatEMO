classdef DRMCMO < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained>
    methods
        function main(Algorithm,Problem)
           %% Parameter setting
            type = Algorithm.ParameterSet(1);
            %% Generate random population
            Population = Problem.Initialization();
            Zmin       = min(Population.objs,[],1);
            archive = Population;
            coord = Population.objs;
            distances = sqrt(sum(coord.^2, 2));
            maxR = min(distances);
            Cons                 = max(0,Population.cons);
            NormCons             = Cons./repmat(max(1,max(Cons,[],1)),Problem.N,1);
            CV                   = sum(NormCons,2);
            maxCV =  max(CV);
            size(Population.decs)
            Fitness    = CalFitness(Problem,Population.objs,Population.cons,archive.objs,2,maxR,maxR,0,Population.decs,archive.decs,maxCV);
            findFeasible = 0;
            sFE = 0;
            %% Optimization
            while Algorithm.NotTerminated(archive)
                if findFeasible == 0
                    coord = Population.objs;
                    distances = sqrt(sum(coord.^2, 2));
                    maxR = min(distances);
                end
                CVa = sum(max(0,archive.cons),2);
                if any(CVa == 0) && sFE == 0 
                    findFeasible = 1; 
                    coord = archive.objs;
                    distances = sqrt(sum(coord.^2, 2));
                    maxR = min(distances);
                    Cons                 = max(0,Population.cons);
                    NormCons             = Cons./repmat(max(1,max(Cons,[],1)),Problem.N,1);
                    CV                   = sum(NormCons,2);
                    maxCV =  max(CV);
                    sFE = Problem.FE;
                end
               
                delta = sigmoid(10 * ((Problem.FE-sFE) / (Problem.maxFE-sFE) - 0.6))
                r = maxR *(1- delta);
                alpha = sigmoid(10 * ((Problem.FE-sFE) / (Problem.maxFE-sFE) - 0.6))
                Ra = 1 - Problem.FE / Problem.maxFE;
                if type ==1
                    MatingPool = TournamentSelection(2,Problem.N,Fitness);
                    [Mate1,Mate2,Mate3,Mate4] = Neighbor_Pairing_Strategy(Population(MatingPool),Zmin,alpha,Problem.N);
                    Offspring1  = OperatorGA(Problem,[Mate1,Mate2]);
                    Offspring = [];
                else
                    Nt = floor(Ra*Problem.N);
                    MatingPool = [Population(randsample(Problem.N,Nt)),archive(randsample(Problem.N,Problem.N-Nt))];
                    [Mate1,Mate2,Mate3,Mate4] = Neighbor_Pairing_Strategy(MatingPool,Zmin,alpha,Problem.N);
                    if rand > 0.5
                        Offspring = OperatorDE(Problem,Mate1,Mate2,Mate3);
                    else
                        Offspring = OperatorDE(Problem,Mate1,Mate2,Mate3,{0.5,0.5,0.5,0.75});
                    end
                    Zmin = min([Zmin;Offspring.objs],[],1);
                    Offspring1 = [];
                end
                [archive,Fitness2] = EnvironmentalSelection(Problem,[Population,Offspring,archive],archive,Problem.N,1,r,maxR,alpha,maxCV);
                if findFeasible == 1
                    [Population,Fitness] = EnvironmentalSelection(Problem,[Population,Offspring,Offspring1,archive],archive,Problem.N,2,r,maxR,alpha,maxCV);
                else
                    [Population,Fitness] = EnvironmentalSelection(Problem,[Population,Offspring,Offspring1,archive],archive,Problem.N,1,r,maxR,alpha,maxCV);
                end
            end
        end
    end
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
