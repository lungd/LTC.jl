mutable struct MTKNODE{T,V,S}
  cell::T
  p::V
  paramlength::Int
  state::S
  MTKNODE(cell, p, paramlength, state) = new{typeof(cell),typeof(p),typeof(state)}(cell,p,paramlength,state)
end
function MTKNODE(cell;)
  p = initial_params(cell)
  MTKNODE(cell, p, length(p), cell.state0)
end
function (m::MTKNODE)(x, p=m.p)
  m.state, y = m.cell(m.state, x, p)
  Inf ∈ m.state && return fill(Inf32,size(y)...)
  NaN ∈ m.state && return fill(Inf32,size(y)...)
  # m.state = h
  return y
end
Base.show(io::IO, m::MTKNODE) = print(io, "MTKNODE(", m.cell, ")")
initial_params(m::MTKNODE) = m.p
paramlength(m::MTKNODE) = m.paramlength
# Flux.@functor MTKNODE (cell,)
# Flux.functor(m::MTKNODE) = (m.cell,), re -> MTKNODE(re..., m.p, m.paramlength, m.state)
Flux.functor(m::MTKNODE) = (m.p,), re -> MTKNODE(m.cell, re..., m.paramlength, m.state)
# Flux.functor(m::MTKNODE) = (m.cell,m.p,m.state), re -> MTKNODE(re[1:2]..., m.paramlength, re[3])
Flux.trainable(m::MTKNODE) = (m.p,)
get_bounds(m::MTKNODE{C,<:AbstractArray{T},S}, ::DataType=nothing) where {C,T,S} = get_bounds(m.cell, T)
reset!(m::MTKNODE, p=m.p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))
reset_state!(m::MTKNODE, p=m.p) = (m.state = reshape(p[end-size(m.cell.state0,1)+1:end], :, 1))
# TODO: reset_state! for cell with train_u0=false

struct MTKNODECell{B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,S}
  in::Int
  out::Int
  wiring::W
  net::NET
  sys::SYS
  prob::PROB
  prob_f::PROBF
  solver::SOLVER
  kwargs::KW
  p::V
  paramlength::Int
  outpins::OP
  train_u0::B
  state0::S

  function MTKNODECell(in, out, wiring, net, sys, prob, prob_f, solver, kwargs, p, paramlength, outpins, train_u0, state0)
    new{typeof(train_u0), typeof(wiring), typeof(net),typeof(sys),typeof(prob),typeof(prob_f),typeof(solver),typeof(kwargs),typeof(p),typeof(outpins),typeof(state0)}(
                       in, out, wiring, net, sys, prob, prob_f, solver, kwargs, p, paramlength, outpins, train_u0, state0)
  end
end
function MTKNODECell(wiring::Wiring{<:AbstractMatrix{T},S2}, solver; train_u0=true, kwargs...) where {T,S2}
  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)::ModelingToolkit.ODESystem
  MTKNODECell(wiring, net, sys, solver; train_u0, kwargs...)
end
function MTKNODECell(wiring::Wiring{<:AbstractMatrix{T},S2}, net::S, sys::S, solver; train_u0=true, kwargs...) where {T, S2, S <: ModelingToolkit.AbstractSystem}

  in::Int = wiring.n_in
  out::Int = wiring.n_out

  tspan = (T(0), T(1))
  defs = ModelingToolkit.get_defaults(sys) # inpins and u0 is always included
  prob = ODEProblem(sys, defs, tspan, tgrad=true, jac=false) # TODO: jac, sparse ???
  # jac = eval(ModelingToolkit.generate_jacobian(sys)[2])
  # tgrad =eval(ModelingToolkit.generate_tgrad(sys)[2])
  prob_f = ODEFunction(sys, states(sys), parameters(sys), tgrad=true)

  @show prob.f.syms
  @show parameters(sys)

  _states = collect(states(sys))
  input_idxs = Int8[findfirst(x->contains(string(x), string(Symbol("x$(i)_InPin"))), _states) for i in 1:in]
  param_names = ["placeholder"]
  outpins = 1f0

  p_ode = prob.p
  u0 = prob.u0[in+1:end]
  state0 = reshape(u0, :, 1)

  p = train_u0 == true ? vcat(p_ode, u0) : p_ode
  infs = fill(T(Inf), size(state0,1))

  cell = MTKNODECell(in, out, wiring, net, sys, prob, prob_f, solver, kwargs, p, length(p), outpins, train_u0, state0)
  LTC.print_cell_info(cell, train_u0)
  cell
end

function (m::MTKNODECell{B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}})(h, inputs::AbstractArray{T,3}, p) where {B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}
  # size(h) == (N,1) at the first MTKNODECell invocation. Need to duplicate batchsize times
  num_reps = size(inputs,3)-size(h,2)+1
  hr = repeat(h, 1, num_reps)
  p_ode = m.train_u0 == true ? p[1:end-size(hr,1)] : p
  solve_ensemble_full_seq(m,hr,inputs,p_ode)
end
#
# function (m::MTKNODECell{true,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}})(h, inputs::AbstractArray{T,3}, p) where {W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}
#   # size(h) == (N,1) at the first MTKNODECell invocation. Need to duplicate batchsize times
#   num_reps = size(inputs,3)-size(h,2)+1
#   hr = repeat(h, 1, num_reps)
#   p_ode = p[1:end-size(hr,1)]
#   solve_ensemble_full_seq(m,hr,inputs,p_ode)
# end

function get_dIs(xs)
  T = eltype(xs)
  batchsize = size(xs,3)
  ϵ = T(1e-5)
  ts = collect(T(0.0):size(xs,2))
  Is = [LinearInterpolation(hcat((@view xs[:,:,i]), (@view xs[:,end,i]).+ϵ),ts) for i in 1:batchsize]
  _dI(I,t) = ForwardDiff.derivative(t->I(t), t)
  _dIxs(i,ts) = reduce(hcat, [_dI(Is[i],t) for t in ts])
  dIs = [LinearInterpolation(_dIxs(i,ts),ts) for i in 1:batchsize]
  Is, dIs
end
Zygote.@nograd get_dIs # TODO



function solve_ensemble_full_seq(m::MTKNODECell{B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}}, u0s, xs::AbstractArray{T,3}, p_ode) where {B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}

  nfs = size(xs,1)
  nobs = size(xs,2)
  batchsize = size(u0s,2)
  nout = size(u0s,1)
  tspan = (T(0), T(nobs))

  dosetimes = collect(1f0:T(nobs))

  ts1 = collect(T(0):T(nobs))

  prob_f = m.prob_f
  mprob = m.prob.f

  function _f(du,_u,p,t,I,dI)
    # u = vcat(I(t), (@view _u[m.in+1:end]))
    mprob.f(du,_u,p,t)
    du[1:m.in] .= dI(t) #ForwardDiff.derivative(t->I(t), t) # D(inpin.x) ~ 0
    # return nothing
  end
  # function _f_jac(J,_u,p,t,I,dI)
  #   # u = vcat(I(t), (@view _u[m.in+1:end]))
  #   mprob.jac(J,_u,p,t)
  #   du[1:m.in] .= dI(t) #ForwardDiff.derivative(t->I(t), t) # D(inpin.x) ~ 0
  #   return nothing
  # end

  Is, dIs = get_dIs(xs)
  u0b = [vcat(Is[1](0),(@view u0s[:,i])) for i in 1:batchsize]
  # fs = [ODEFunction{true,false}((du,u,p,t)->_f(du,u,p,t,Is[i],dIs[i]),jac=(du,u,p,t)->_f_jac(du,u,p,t,Is[i],dIs[i])) for i in 1:batchsize]
  fs = [ODEFunction{true,false}((du,u,p,t)->_f(du,u,p,t,Is[i],dIs[i])) for i in 1:batchsize]


  function prob_func(prob,i,repeat)
    u0 = u0b[i]
    f = fs[i]
    remake(prob; f, u0)
  end

  function output_func(sol,i)
    # @show sol.retcode
    sol.retcode != :Success && return fill(T(Inf), nout,nobs+1), false # dt <= dtmin not causing retcode != Success (VCABM)
    sol, false
  end

  _prob = ODEProblem{true}(fs[1], u0b[1], tspan, p_ode,
    saveat=1f0, save_everystep=false, save_end=true, save_start=true; m.kwargs...)

  ensemble_prob = EnsembleProblem(_prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sol = solve(ensemble_prob, m.solver, EnsembleThreads(); trajectories=batchsize,
              saveat=1f0, save_everystep=false, save_start=true, save_end=true,
              m.kwargs...,
	)
  get_quantities_of_interest(Array(sol), m, dosetimes)
end

function get_quantities_of_interest(sol::AbstractArray, m::MTKNODECell, dosetimes)
  NaN ∈ sol && return fill(Inf32, size(sol[m.in+1:end,end,:])...), fill(Inf32, size(sol[end-m.out+1:end, end-length(dosetimes)+1:end, :])...)
  Inf ∈ sol && return fill(Inf32, size(sol[m.in+1:end,end,:])...), fill(Inf32, size(sol[end-m.out+1:end, end-length(dosetimes)+1:end, :])...)
  h = sol[m.in+1:end,end,:]
  # h = sol[:,end,:]
  out = @view sol[end-m.out+1:end, end-length(dosetimes)+1:end, :]
  return h, out
end

Base.show(io::IO, m::MTKNODECell) = print(io, "MTKNODECell(", m.in, ",", m.out, ")")
initial_params(m::MTKNODECell) = m.p
paramlength(m::MTKNODECell) = m.paramlength
# Flux.@functor MTKNODECell (p,)
Flux.trainable(m::MTKNODECell) = (m.p,)
Flux.functor(m::MTKNODECell{B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}}) where {B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}  = (m.p,), re -> MTKNODECell(m.in,m.out,m.wiring,m.net,m.sys,m.prob,m.prob_f,m.solver,m.kwargs, re..., m.paramlength,m.outpins,m.train_u0,m.state0)

### Already defined in mtk_recur.jl
# function _get_bounds(T, default_lb, default_ub, vars)
#   cell_lb = T[]
#   cell_ub = T[]
#   for v in vars
#     contains(string(v), "InPin") && continue
#     contains(string(v), "OutPin") && continue
#     hasmetadata(v, ModelingToolkit.VariableOutput) && continue
#     lower = hasmetadata(v, VariableLowerBound) ? getmetadata(v, VariableLowerBound) : default_lb
#     upper = hasmetadata(v, VariableUpperBound) ? getmetadata(v, VariableUpperBound) : default_ub
#     push!(cell_lb, lower)
#     push!(cell_ub, upper)
#   end
#   return cell_lb, cell_ub
# end


# function get_bounds(m::MTKNODECell{B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}}, ::DataType=nothing; default_lb = -Inf, default_ub = Inf) where {B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}
#   params = collect(parameters(m.sys))
#   _get_bounds(T, default_lb, default_ub, params)
# end


function get_bounds(m::MTKNODECell{B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}}, ::DataType=nothing; default_lb = -Inf, default_ub = Inf) where {B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}
  params = collect(parameters(m.sys))
  states = collect(ModelingToolkit.states(m.sys))[m.in+1:end]
  _get_bounds(T, default_lb, default_ub, vcat(params,states))
end

function MTKNODEMapped(chainf::C, wiring, solver; kwargs...) where C
  chainf(LTC.MapperIn(wiring),
          MTKNODE(MTKNODECell(wiring, solver; kwargs...)),
          LTC.MapperOut(wiring))
end
function MTKNODEMapped(chainf::C, wiring, net, sys, solver; kwargs...) where C
  chainf(LTC.MapperIn(wiring),
          MTKNODE(MTKNODECell(wiring, net, sys, solver; kwargs...)),
          LTC.MapperOut(wiring))
end
