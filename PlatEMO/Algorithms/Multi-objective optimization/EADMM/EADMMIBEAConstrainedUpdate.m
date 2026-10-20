function Population = EADMMIBEAConstrainedUpdate(Parents,Incoming,N,kappa)
% Apply IBEA to feasible candidates; keep infeasible parents only as fillers.

    feasible = all(Incoming.cons<=0,2);
    if any(feasible)
        parentFeasible = all(Parents.cons<=0,2);
        Pool = [Parents(parentFeasible),Incoming(feasible)];
        if numel(Pool) >= N
            Population = EADMMIBEASelection(Pool,N,kappa);
        else
            % The paper does not specify how to fill an undersized feasible
            % pool. Preserve all feasible solutions and select only the
            % remaining slots from existing infeasible parents.
            Fill = EADMMIBEASelection(Parents(~parentFeasible),N-numel(Pool),kappa);
            Population = [Pool,Fill];
        end
    else
        Population = Parents;
    end
end
