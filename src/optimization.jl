mutable struct ClampBoundOptim{V} <: Flux.Optimise.AbstractOptimiser
  os::Vector{Any}
  lb::V
  ub::V
end
ClampBoundOptim(lb,ub,o...) = ClampBoundOptim{typeof(lb)}(Any[o...],lb,ub)

Flux.@forward ClampBoundOptim.os Base.getindex, Base.first, Base.last, Base.lastindex, Base.push!, Base.setindex!
Flux.@forward ClampBoundOptim.os Base.iterate

Base.getindex(c::ClampBoundOptim, i::AbstractArray) = ClampBoundOptim(c.lb,c.ub,c.os[i]...)

function Flux.Optimise.apply!(o::ClampBoundOptim{V}, x, Δ) where V
  for opt in o.os
    Δ = Flux.Optimise.apply!(opt, x, Δ)
  end
  return Δ
end

function Flux.Optimise.update!(opt::ClampBoundOptim{V}, x, x̄) where V
  x̄r = Flux.Optimise.ArrayInterface.restructure(x, x̄) # address some cases where Zygote's
                                          # output are not mutable, see #1510
  x .-= Flux.Optimise.apply!(opt, x, x̄r)
  x .= map(i -> clamp(x[i], opt.lb[i], opt.ub[i]), 1:length(x))
end


load_model(chain::Flux.Chain, T::DataType=Float32) = (LTC.destructure(chain)..., get_bounds(chain,T)...)
load_model(chain::FastChain, T::DataType=Float32) = (initial_params(chain), chain, get_bounds(chain,T)...)


function optimize(chain, loss, cb, opt, train_dl, epochs=1, T::DataType=Float32, AD=GalacticOptim.AutoZygote())
  pp, model, lb, ub = load_model(chain, T)

  println("--------------- optimize ---------------")
  println("# training samples:         $(length(train_dl))")
  println("# parameters:               $(length(pp))")
  println("typeof(p):                  $(typeof(pp))")
  println("# epochs:                   $(epochs)")
  println("# lb:                       $(length(lb))")
  println("# ub:                       $(length(ub))")


  # mycb = LTC.MyCallback(T, cb, epochs, length(train_dl))
  train_dl = epochs > 1 ? ncycle(train_dl, epochs) : train_dl

  f(θ,p,x,y) = loss(θ,model,x,y)
  optfun = GalacticOptim.OptimizationFunction(f, AD)
  optfunc = GalacticOptim.instantiate_function(optfun, pp, AD, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, pp, lb=lb, ub=ub,
                                grad = true, hess = true, #sparse = true,
                                #parallel=ModelingToolkit.MultithreadedForm()
                                )

  GalacticOptim.solve(optprob, opt, train_dl, cb = cb)
end


# function optimize(chain, loss, cb, ecb, opt, AD, train_dl, epochs; normalize=false, T=Float32)
#   pp, model, lb, ub = load_model(chain, T)
#
#   # @show length(Flux.destructure(chain)[1])
#
#   n_states = 0
#   for l in chain.layers
#     if l isa Flux.Chain || l isa FastChain
#     end
#     !isdefined(l, :cell) && continue
#     !isdefined(l.cell, :sys) && continue
#     n_states = length(ModelingToolkit.states(l.cell.sys))
#   end
#
#
#   println("------- optimize -------")
#   println("# training samples: $(length(train_dl))")
#   println("# parameters:       $(length(pp))")
#   n_states != 0 && println("# mtksys states:    $(n_states)")
#   println("typeof(p):          $(typeof(pp))")
#   println("# epochs:           $(epochs)")
#
#   # @show length(train_dl)
#   # @show size(first(train_dl)[1])
#   # @show size(first(train_dl)[1][1])
#   # @show length(pp)
#   # @show length(lb)
#   # @show length(ub)
#
#   losses = T[]
#   function _cb(p,l,args...; kwargs...)
#     if length(train_dl) > 1
#       push!(losses,l)
#       ProgressMeter.next!(porgress; showvalues = [(:loss,"$(losses[end])")])
#     end
#     cb(p,l,args...; kwargs...)
#   end
#
#   # f = tspan == nothing ? (θ,p,x,y) -> loss(θ,model,x,y) : (θ,p,x,y) -> loss(θ,model,x,y,tspan)
#   f = (θ,p,x,y) -> loss(θ,model,x,y)
#   optfun = GalacticOptim.OptimizationFunction(f, AD)
#   optfunc = GalacticOptim.instantiate_function(optfun, pp, AD, nothing)
#   optprob = GalacticOptim.OptimizationProblem(optfunc, pp, lb=lb, ub=ub,
#                                 grad = true, hess = true, #sparse = true,
#                                 #parallel=ModelingToolkit.MultithreadedForm()
#                                 )
#   # sys = ModelingToolkit.modelingtoolkitize(optprob)
#   # optprob = OptimizationProblem(sys,pp,lb=lb, ub=ub,grad=true,hess=true)
#   if length(train_dl) > 1
#     porgress = ProgressMeter.Progress(length(train_dl); showspeed=true)
#     res = GalacticOptim.solve(optprob, opt, train_dl, cb = _cb)
#     ecb(losses,1,res)
#     losses = T[]
#     for epoch in 2:epochs
#       prob = remake(optprob,u0=res.u)
#       porgress = ProgressMeter.Progress(length(train_dl); showspeed=true)
#       #res = normalize ? solve_normalized(prob, opt, train_dl, scale=true, cb = cb) : GalacticOptim.solve(prob, opt, train_dl, cb = _cb)
#       res = GalacticOptim.solve(prob, opt, train_dl, cb = _cb)
#       ecb(losses,epoch,res)
#       losses = T[]
#     end
#     return res
#   else
#     res = GalacticOptim.solve(optprob, opt, ncycle(train_dl,epochs), cb = _cb)
#     return res
#   end
# end


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

    scaling_function === nothing && any(iszero.(θ_start)) && error("Default Inverse Scaling is not compatible with `0` as initial guess")


    scaling_function = if scaling_function === nothing
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

    lb = prob.lb === nothing ? nothing : scaling_function(prob.lb, false)
    ub = prob.ub === nothing ? nothing : scaling_function(prob.ub, false)



    u0s = scaling_function(prob.u0, false)
    optfun = OptimizationFunction(normalized_f, prob.f.adtype,
                             grad = prob.f.grad, hess = prob.f.hess,
                             hv = prob.f.hv, cons = prob.f.cons,
                             cons_j = prob.f.cons_j, cons_h = prob.f.cons_h)
    optfunc = GalacticOptim.instantiate_function(optfun, u0s, prob.f.adtype, nothing)

    # _prob = remake(prob, u0 = u0s, lb = lb,
                   # ub = ub,
                   # f = optfunc)
    _prob = GalacticOptim.OptimizationProblem(optfunc, u0s, lb=prob.lb, ub=prob.ub,
                                  grad = true, hess = true, #sparse = true,
                                  #parallel=ModelingToolkit.MultithreadedForm()
                                  )

    optsol = GalacticOptim.solve(_prob, opt, args...; cb = normalized_cb, kwargs...)
    optsol.u .= scaling_function(optsol.u, true)
    optsol
end



# function Symbolics.toexpr(p::Symbolics.SpawnFetch{ModelingToolkit.MultithreadedForm}, st)
#     args = isnothing(p.args) ?
#               Iterators.repeated((), length(p.exprs)) : p.args
#     spawns = map(p.exprs, args) do thunk, a
#         ex = :($Funcall($(Symbolics.@RuntimeGeneratedFunction(toexpr(thunk, st))),
#                        ($(toexpr.(a, (st,))...),)))
#         quote
#             let
#                 task = Base.Task($ex)
#                 task.sticky = false
#                 Base.schedule(task)
#                 task
#             end
#         end
#     end
#     quote
#         $(toexpr(p.combine, st))(map(fetch, ($(spawns...),))...)
#     end
# end
