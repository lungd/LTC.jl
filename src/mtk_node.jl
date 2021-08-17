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
Flux.@functor MTKNODE (p,)
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
  tspan::Tuple
  p::V
  paramlength::Int
  param_names::Vector{String}
  outpins::OP
  infs::Vector
  state0::S

  function MTKNODECell(in, out, wiring, net, sys, prob, prob_f, solver, kwargs, tspan, p, paramlength, param_names, outpins, infs, train_u0, state0)
    cell = new{train_u0, typeof(wiring), typeof(net),typeof(sys),typeof(prob),typeof(prob_f),typeof(solver),typeof(kwargs),typeof(p),typeof(outpins),typeof(state0)}(
                       in, out, wiring, net, sys, prob, prob_f, solver, kwargs, tspan, p, paramlength, param_names, outpins, infs, state0)
    LTC.print_cell_info(cell, train_u0)
    cell
  end
end
function MTKNODECell(wiring::Wiring{T}, solver; train_u0=true, kwargs...) where T
  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)::ModelingToolkit.ODESystem
  MTKNODECell(wiring, net, sys, solver; train_u0, kwargs...)
end
function MTKNODECell(wiring::Wiring{T}, net::S, sys::S, solver; train_u0=true, kwargs...) where {T, S <: ModelingToolkit.AbstractSystem}

  in::Int = wiring.n_in
  out::Int = wiring.n_out

  tspan = (T(0), T(1))
  defs = ModelingToolkit.get_defaults(sys) # inpins and u0 is always included
  prob = ODEProblem(sys, defs, tspan) # TODO: jac, sparse ???
  prob_f = ODEFunction(sys, states(sys), parameters(sys), tgrad=true)

  _states = collect(states(sys))
  input_idxs = Int8[findfirst(x->contains(string(x), string(Symbol("x$(i)_InPin"))), _states) for i in 1:in]
  param_names = ["placeholder"]
  outpins = 1f0

  p_ode = prob.p
  u0 = prob.u0[in+1:end] # use input when calling as u0[1:in]
  state0 = reshape(u0, :, 1)

  p = train_u0 == true ? vcat(p_ode, u0) : p_ode
  infs = fill(T(Inf), size(state0,1))

  MTKNODECell(in, out, wiring, net, sys, prob, prob_f, solver, kwargs, tspan, p, length(p), param_names, outpins, infs, train_u0, state0)
end

function (m::MTKNODECell{false,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}})(h, inputs::AbstractArray{T,3}, p) where {W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}
  # size(h) == (N,1) at the first MTKNODECell invocation. Need to duplicate batchsize times
  num_reps = size(inputs,3)-size(h,2)+1
  hr = repeat(h, 1, num_reps)
  p_ode = p
  solve_ensemble_full_seq(m,hr,inputs,p_ode)
end

function (m::MTKNODECell{true,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}})(h, inputs::AbstractArray{T,3}, p) where {W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}
  # size(h) == (N,1) at the first MTKNODECell invocation. Need to duplicate batchsize times
  num_reps = size(inputs,3)-size(h,2)+1
  hr = repeat(h, 1, num_reps)
  p_ode = @view p[1:end-size(hr,1)]
  solve_ensemble_full_seq(m,hr,inputs,p_ode)
end

function solve_ensemble_full_seq(m::MTKNODECell{B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}}, u0s, xs::AbstractArray{T,3}, p_ode) where {B,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}

  nfs = size(xs,1)
  nobs = size(xs,2)
  batchsize = size(u0s,2)
  nout = size(u0s,1)
  tspan = (T(0), T(nobs))

  infs = fill(T(Inf), nout,nobs+1)
  infu0 = fill(T(Inf), size(u0s,1),nobs)

  dosetimes = collect(1f0:T(nobs))

  ts1 = collect(T(0):T(nobs))

  prob_f = m.prob_f

  # basic_tgrad(u,p,t) = zero(u)

  function _f(u,p,t,I)
    du = similar(u)
    _f(du,u,p,t,I)
    return du
  end

  function _f(du,u,p,t,I)
    u_mtk = u
    prob_f.f(du,u_mtk,p,t)
    du[1:m.in] .= ForwardDiff.derivative(t->I(t), t)
    return nothing
  end
  f = ODEFunction(_f)
  # f = ODEFunction(_f,tgrad=basic_tgrad)

  function prob_func(prob,i,repeat)
    ϵ = T(1e-5)
    x = xs[:,:,i]
    x1 = hcat(x, x[:,end].+ϵ)
    I = LinearInterpolation(x1,ts1)

    u0 = vcat((@view x[:,1]),(@view u0s[:,i]))
    p = p_ode
    ODEProblem((du,u,p,t)->f(du,u,p,t,I), u0, tspan, p,
      saveat=1f0, save_everystep=false, save_end=true, save_start=true)
  end

  function output_func(sol,i)
    # @show sol.retcode
    sol.retcode != :Success && return infs, false # dt <= dtmin not causing retcode != Success (VCABM)
    sol, false
  end

  ensemble_prob = EnsembleProblem(m.prob; prob_func, output_func, safetycopy=true) # TODO: safetycopy ???
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
Flux.@functor MTKNODECell (p,)
Flux.trainable(m::MTKNODECell) = (m.p,)


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


function get_bounds(m::MTKNODECell{false,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}}, ::DataType=nothing; default_lb = -Inf, default_ub = Inf) where {W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}
  params = collect(parameters(m.sys))
  _get_bounds(T, default_lb, default_ub, params)
end


function get_bounds(m::MTKNODECell{true,W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,<:AbstractMatrix{T}}, ::DataType=nothing; default_lb = -Inf, default_ub = Inf) where {W,NET,SYS,PROB,PROBF,SOLVER,KW,V,OP,T}
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
