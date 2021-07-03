function optimize(model, loss, cb, opt, AD, train_dl)
  pp = initial_params(model)
  lb, ub = get_bounds(model)

  @show length(train_dl)
  @show size(first(train_dl)[1])
  @show size(first(train_dl)[1][1])
  @show length(pp)
  @show length(lb)
  @show length(ub)

  optfun = GalacticOptim.OptimizationFunction((θ, p, x, y) -> loss(θ,x,y, model), AD)
  optfunc = GalacticOptim.instantiate_function(optfun, pp, AD, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, pp, lb=lb, ub=ub,
                                #grad = true, hess = true, sparse = true,
                                #parallel=ModelingToolkit.MultithreadedForm()
                                )
  GalacticOptim.solve(optprob, opt, train_dl, cb = cb)
end


function optimize(model::Flux.Chain, loss, cb, opt, AD, train_dl)
  lb, ub = get_bounds(model)

  @show length(train_dl)
  @show size(first(train_dl)[1])
  @show size(first(train_dl)[1][1])
  @show length(pp)
  @show length(lb)
  @show length(ub)

  optfun = GalacticOptim.OptimizationFunction((θ, p, x, y) -> loss(p,re, x,y), AD)
  optfunc = GalacticOptim.instantiate_function(optfun, pp, AD, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, pp, lb=lb, ub=ub,
                                #grad = true, hess = true, sparse = true,
                                #parallel=ModelingToolkit.MultithreadedForm()
                                )
  GalacticOptim.solve(optprob, opt, train_dl, cb = cb)
end
