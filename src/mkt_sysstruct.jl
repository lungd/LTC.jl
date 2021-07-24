function ModelingToolkit.structural_simplify(sys::ODESystem)
    sys = initialize_system_structure(alias_elimination(sys))
    # check_consistency(sys)
    if sys isa ODESystem
        sys = dae_index_lowering(sys)
    end
    sys = tearing(sys)
    fullstates = [map(eq->eq.lhs, observed(sys)); states(sys)]
    ModelingToolkit.@set! sys.observed = ModelingToolkit.topsort_equations(observed(sys), fullstates)
    return sys
end


function ModelingToolkit.initialize_system_structure(sys::ODESystem)
    sys = flatten(sys)

    iv = independent_variable(sys)
    eqs = copy(equations(sys))
    neqs = length(eqs)
    algeqs = trues(neqs)
    dervaridxs = OrderedSet{Int}()
    var2idx = Dict{Any,Int}()
    symbolic_incidence = []
    fullvars = []
    var_counter = Ref(0)
    addvar! = let fullvars=fullvars, var_counter=var_counter
        var -> begin
            get!(var2idx, var) do
                push!(fullvars, var)
                var_counter[] += 1
            end
        end
    end

    vars = OrderedSet()
    for (i, eq′) in enumerate(eqs)
        if ModelingToolkit._iszero(eq′.lhs)
            eq = eq′
        else
            eq = 0 ~ eq′.rhs - eq′.lhs
        end
        ModelingToolkit.vars!(vars, eq.rhs)
        isalgeq = true
        statevars = []
        for var in vars
            hasmetadata(var, VariableOutput) && continue
            isequal(var, iv) && continue
            if ModelingToolkit.isparameter(var) || (ModelingToolkit.istree(var) && ModelingToolkit.isparameter(ModelingToolkit.operation(var)))
                continue
            end
            varidx = addvar!(var)
            push!(statevars, var)

            dvar = var
            idx = varidx
            while ModelingToolkit.isdifferential(dvar)
                if !(idx in dervaridxs)
                    push!(dervaridxs, idx)
                end
                isalgeq = false
                dvar = ModelingToolkit.arguments(dvar)[1]
                idx = addvar!(dvar)
            end
        end
        push!(symbolic_incidence, copy(statevars))
        empty!(statevars)
        empty!(vars)
        algeqs[i] = isalgeq
        if isalgeq
            eqs[i] = eq
        end
    end

    # sort `fullvars` such that the mass matrix is as diagonal as possible.
    dervaridxs = collect(dervaridxs)
    sorted_fullvars = OrderedSet(fullvars[dervaridxs])
    for dervaridx in dervaridxs
        dervar = fullvars[dervaridx]
        diffvar = ModelingToolkit.arguments(dervar)[1]
        if !(diffvar in sorted_fullvars)
            push!(sorted_fullvars, diffvar)
        end
    end
    for v in fullvars
        if !(v in sorted_fullvars)
            push!(sorted_fullvars, v)
        end
    end
    fullvars = collect(sorted_fullvars)
    var2idx = Dict(fullvars .=> eachindex(fullvars))
    dervaridxs = 1:length(dervaridxs)

    nvars = length(fullvars)
    diffvars = []
    vartype = fill(ModelingToolkit.SystemStructures.DIFFERENTIAL_VARIABLE, nvars)
    varassoc = zeros(Int, nvars)
    inv_varassoc = zeros(Int, nvars)
    for dervaridx in dervaridxs
        vartype[dervaridx] = ModelingToolkit.SystemStructures.DERIVATIVE_VARIABLE
        dervar = fullvars[dervaridx]
        diffvar = ModelingToolkit.arguments(dervar)[1]
        diffvaridx = var2idx[diffvar]
        push!(diffvars, diffvar)
        varassoc[diffvaridx] = dervaridx
        inv_varassoc[dervaridx] = diffvaridx
    end

    algvars = setdiff(states(sys), diffvars)
    for algvar in algvars
        # hasmetadata(algvar, VariableOutput) && continue
        # it could be that a variable appeared in the states, but never appeared
        # in the equations.
        algvaridx = get(var2idx, algvar, 0)

        algvaridx == 0 && throw(ModelingToolkit.SystemStructures.InvalidSystemException("The system is missing "
            * "an equation for $algvar."
        ))
        vartype[algvaridx] = ModelingToolkit.SystemStructures.ALGEBRAIC_VARIABLE
    end

    graph = BipartiteGraph(neqs, nvars, Val(false))
    for (ie, vars) in enumerate(symbolic_incidence), v in vars
        jv = var2idx[v]
        LightGraphs.add_edge!(graph, ie, jv)
    end

    ModelingToolkit.@set! sys.eqs = eqs
    ModelingToolkit.@set! sys.structure = SystemStructure(
        fullvars = fullvars,
        vartype = vartype,
        varassoc = varassoc,
        inv_varassoc = inv_varassoc,
        varmask = iszero.(varassoc),
        algeqs = algeqs,
        graph = graph,
        solvable_graph = BipartiteGraph(ModelingToolkit.nsrcs(graph), ModelingToolkit.ndsts(graph), Val(false)),
        assign = Int[],
        inv_assign = Int[],
        scc = Vector{Int}[],
        partitions = ModelingToolkit.SystemStructures.SystemPartition[],
    )
    return sys
end








@variables v(t)=0.0 I(t)=0.0 out1(t)=0.0 out2(t)=0.0 dout1(t) dout2(t)
eqs = [
    D(v) ~ 0.1 + I
    I ~ 0.35*v*v
    D(out1) ~ dout1
    dout1 ~ 0.1 + I
    D(out2) ~ dout2
    out2 ~ I
]
test = ODESystem(eqs,t;name=:test)
sys = structural_simplify(test)
defs = ModelingToolkit.get_defaults(sys)
prob = ODAEProblem(sys,defs,(0.0,1.0))
sol = solve(prob,Tsit5())
plot(sol)
plot(sol.t, sol[1,:])
plot(sol, vars=[out1])
plot(sol, vars=[I])
plot(sol, vars=[out2])
plot(sol, vars=[dout1])

eqs = [D(v) ~ 0.1]
observed = [o1.x ~ v
            I ~ 0.1v
            o2.x ~ I]

=>

eqs = [D(v) ~ dv
       D(o1.x) ~ dv
       D(o2.x) ~ 0.1*dv]
observed = [I ~ 0.1*v
            dv ~ 0.1]

# outpins = [o1.x, o2.x]
# diffeqs = filter(e -> ModelingToolkit.isdiffeq(e), eqs)
# diffeqdict = Dict([e.lhs => e.rhs for e in diffeqs])
function move_observed_state_output(oeq)
    # create variable dv(iv...)
    # push!(observed, dv ~ diffeqdict[D(oeq.rhs)])
    # push!(eqs, D(oeq.lhs) ~ dv
    # remove oeq from observed
end
function move_observed_intermediate_output(oeq)
    # create variable 'du1x'
    # for eq in observed
    #   eq.lhs != oeq.rhs && continue
    #   rhs = eq.rhs
    #   diff_terms = []
    #   new_rhs = substutude
    #   new_oeq = du1x ~ new_rhs
    # end
    # push!(observed, ds1x ~ diffeqdict[D(oeq.rhs)])
    # push!(eqs, D(oeq.lhs) ~ du1x
    # remove oeq from observed
end
function create_states_for_outpins()
    for eq in observed
        # eq.lhs ∉ outpins && continue
        # if eq ∈ diffeqs
        #   move_observed_state_output()
        # else
        #   move_observed_intermediate_output()
        # end
    end
end
