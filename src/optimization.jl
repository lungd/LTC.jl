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
    solve_normalized(optprob, opt, train_dl, scale=true, cb = cb)
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
# Skip the DiffEqBase handling
struct InverseScale{T}
    scale::T
end

(inv_scale::InverseScale)(x, compute_inverse::Bool = false) =
    compute_inverse ? x .* inv_scale.scale : x ./ inv_scale.scale

function solve_normalized(prob::OptimizationProblem, opt, args...;
               scale::Bool = false, scaling_function = nothing,
               cb = (args...) -> (false), kwargs...)
    !scale && return GalacticOptim.solve(prob, opt, args...; kwargs...)

    θ_start = copy(prob.u0)
    for (i,p) in enumerate(θ_start)
      if iszero(p)
        @show i
        @show p
        θ_start[i] += 0.00001f0
      end
    end

    if isnothing(scaling_function) && any(iszero.(θ_start))
        error("Default Inverse Scaling is not compatible with `0` as initial guess")
    end

    scaling_function = if isnothing(scaling_function)
            InverseScale(θ_start)
        else
            # Check if arguments are compatible
            # First arg is the parameter
            # 2nd one denotes inverse computation or not
            scaling_function(θ_start, false)
            scaling_function
        end

    normalized_f(α, args...) = prob.f.f(scaling_function(α, true), args...)
    normalized_cb(α, args...) = cb(scaling_function(α, true), args...)

    lb = isnothing(prob.lb) ? nothing : scaling_function(prob.lb, false)
    ub = isnothing(prob.ub) ? nothing : scaling_function(prob.ub, false)

    u0s = scaling_function(prob.u0, false)
    optfun = OptimizationFunction(normalized_f, prob.f.adtype,
                             grad = prob.f.grad, hess = prob.f.hess,
                             hv = prob.f.hv, cons = prob.f.cons,
                             cons_j = prob.f.cons_j, cons_h = prob.f.cons_h)
    optfunc = GalacticOptim.instantiate_function(optfun, u0s, prob.f.adtype, nothing)

    _prob = remake(prob, u0 = u0s, lb = lb,
                   ub = ub,
                   f = optfunc)

    optsol = GalacticOptim.solve(_prob, opt, args...; cb = normalized_cb, kwargs...)
    optsol.u .= scaling_function(optsol.u, true)
    optsol
end
