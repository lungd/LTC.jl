function optimize(model, loss, cb, opt, AD, train_dl; normalize=false)
  pp = initial_params(model)
  lb, ub = get_bounds(model)

  @show length(train_dl)
  @show size(first(train_dl)[1])
  @show size(first(train_dl)[1][1])
  @show length(pp)
  @show length(lb)
  @show length(ub)

  optfun = GalacticOptim.OptimizationFunction((θ,p,x,y) -> loss(θ,model,x,y), AD)
  optfunc = GalacticOptim.instantiate_function(optfun, pp, AD, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, pp, lb=lb, ub=ub,
                                #grad = true, hess = true, sparse = true,
                                #parallel=ModelingToolkit.MultithreadedForm()
                                )

  if normalize
    solve_normalized(optprob, opt, train_dl, cb = cb)
  else
    GalacticOptim.solve(optprob, opt, train_dl, cb = cb)
  end
end


function optimize(model::Flux.Chain, loss, cb, opt, AD, train_dl)
  pp, re = LTC.destructure(model)
  lb, ub = get_bounds(model)

  @show length(train_dl)
  @show size(first(train_dl)[1])
  @show size(first(train_dl)[1][1])
  @show length(pp)
  @show length(lb)
  @show length(ub)
  @show sum(length.(pp))

  optfun = GalacticOptim.OptimizationFunction((θ,p,x,y) -> loss(θ,re,x,y), AD)
  optfunc = GalacticOptim.instantiate_function(optfun, pp, AD, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, pp, lb=lb, ub=ub,
                                #grad = true, hess = true, sparse = true,
                                #parallel=ModelingToolkit.MultithreadedForm()
                                )
  GalacticOptim.solve(optprob, opt, train_dl, cb = cb)
end


# https://github.com/SciML/GalacticOptim.jl/issues/146
function solve_normalized(prob,args... ;cb=(args...)->(false),kwargs...)
    θ_start = copy(prob.u0)
    @assert !any(θ_start .== 0) "TODO: error text"

    _normalize = (θ) ->  θ./θ_start #TODO: inplace?
    _inv_normalize = (α) -> α.*θ_start

    normalized_f(α,args...) = prob.f.f(_inv_normalize(α),args...)
    normalized_cb(α,args...) = cb(_inv_normalize(α),args...)

    lb = isnothing(prob.lb) ? nothing : _normalize(prob.lb)
    ub = isnothing(prob.ub) ? nothing : _normalize(prob.ub)
    _prob = remake(prob,u0=_normalize(prob.u0),lb=lb,ub=ub,f=OptimizationFunction(normalized_f,prob.f.adtype))
    # TODO: passing other fields of prob.f

    optsol = GalacticOptim.solve(_prob,args...;cb=normalized_cb,kwargs...)
    optsol.u .= _inv_normalize(optsol.u)
    optsol
end
