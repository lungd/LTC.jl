mutable struct RecurMTK{T,V,S}
  cell::T
  p::V
  paramlength::Int
  state::S
end
function RecurMTK(cell)
  p = initial_params(cell)
  RecurMTK(cell, p, length(p), cell.state0)
end
function (m::RecurMTK)(x, p=m.p)
  h, y = m.cell(m.state, x, p)
  Inf ∈ h && return h
  m.state = h
  return y
end
Base.show(io::IO, m::RecurMTK) = print(io, "RecurMTK(", m.cell, ")")
initial_params(m::RecurMTK) = m.p
paramlength(m::RecurMTK) = m.paramlength
Flux.@functor RecurMTK (p,)
Flux.trainable(m::RecurMTK) = (m.p,)
get_bounds(m::RecurMTK{C,<:AbstractArray{T},S}, ::DataType=nothing) where {C,T,S} = get_bounds(m.cell, T)
reset!(m::RecurMTK, p=m.p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))
reset_state!(m::RecurMTK, p=m.p) = (m.state = reshape(p[end-size(m.cell.state0,1)+1:end], :, 1))
# TODO: reset_state! for cell with train_u0=false


struct MTKCell{B,W,NET,SYS,PROB,SOLVER,KW,V,OP,S}
  in::Int
  out::Int
  wiring::W
  net::NET
  sys::SYS
  prob::PROB
  solver::SOLVER
  kwargs::KW
  tspan::Tuple
  p::V
  paramlength::Int
  param_names::Vector{String}
  outpins::OP
  infs::Vector
  state0::S

  function MTKCell(in, out, wiring, net, sys, prob, solver, kwargs, tspan, p, paramlength, param_names, outpins, infs, train_u0, state0)
    cell = new{train_u0, typeof(wiring), typeof(net),typeof(sys),typeof(prob),typeof(solver),typeof(kwargs),typeof(p),typeof(outpins),typeof(state0)}(
                       in, out, wiring, net, sys, prob, solver, kwargs, tspan, p, paramlength, param_names, outpins, infs, state0)
    LTC.print_cell_info(cell, train_u0)
    cell
  end
end

function MTKCell(wiring::Wiring{T}, solver; train_u0=true, kwargs...) where T
  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)::ModelingToolkit.ODESystem
  MTKCell(wiring, net, sys, solver; train_u0, kwargs...)
end
function MTKCell(wiring::Wiring{T}, net::S, sys::S, solver; train_u0=true, kwargs...) where {T, S <: ModelingToolkit.AbstractSystem}

  in::Int = wiring.n_in
  out::Int = wiring.n_out

  tspan = (T(0), T(1))
  defs = ModelingToolkit.get_defaults(sys)
  prob = ODEProblem(sys, defs, tspan) # TODO: jac, sparse ???

  _states = collect(states(sys))
  input_idxs = Int8[findfirst(x->contains(string(x), string(Symbol("x$(i)_InPin"))), _states) for i in 1:in]
  param_names = ["placeholder"]
  outpins = 1f0

  p_ode = prob.p
  u0 = prob.u0[in+1:end]
  state0 = reshape(u0, :, 1)

  p = train_u0 == true ? vcat(p_ode, u0) : p_ode
  infs = fill(T(Inf), size(state0,1))

  MTKCell(in, out, wiring, net, sys, prob, solver, kwargs, tspan, p, length(p), param_names, outpins, infs, train_u0, state0)
end

function (m::MTKCell{false,W,NET,SYS,PROB,SOLVER,KW,V,OP,<:AbstractMatrix{T}})(h, inputs::AbstractArray, p) where {W,NET,SYS,PROB,SOLVER,KW,V,OP,T}
  # size(h) == (N,1) at the first MTKNODECell invocation. Need to duplicate batchsize times
  num_reps = size(inputs,3)-size(h,2)+1
  hr = repeat(h, 1, num_reps)
  p_ode = p
  solve_ensemble_full_seq(m,hr,inputs,p_ode)
end

function (m::MTKCell{true,W,NET,SYS,PROB,SOLVER,KW,V,OP,<:AbstractMatrix{T}})(h, inputs::AbstractArray, p) where {W,NET,SYS,PROB,SOLVER,KW,V,OP,T}
  # size(h) == (N,1) at the first MTKNODECell invocation. Need to duplicate batchsize times
  num_reps = size(inputs,3)-size(h,2)+1
  hr = repeat(h, 1, num_reps)
  p_ode = @view p[1:end-size(hr,1)]
  solve_ensemble(m,hr,inputs,p_ode)
end


function solve_ensemble(m::MTKCell{B,W,NET,SYS,PROB,SOLVER,KW,V,OP,<:AbstractMatrix{T}}, u0s, xs, p_ode) where {B,W,NET,SYS,PROB,SOLVER,KW,V,OP,T}
  elapsed = 1.0 # TODO
  tspan = (T(0), T(elapsed))

  batchsize = size(xs,2)
	infs = m.infs

  function prob_func(prob,i,repeat)
    u0 = vcat(xs[:,i], u0s[:,i])
    p = p_ode
    remake(prob; tspan, p, u0)
  end

  function output_func(sol,i)
    sol.retcode != :Success && return infs, false
		# sol[:,end], false
    sol[m.in+1:end, end], false
  end

  ensemble_prob = EnsembleProblem(m.prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sol = DiffEqBase.solve(ensemble_prob, m.solver, EnsembleThreads(), trajectories=batchsize,
                         saveat=1f0; m.kwargs...
	)

	get_quantities_of_interest(Array(sol), m)
end


function get_quantities_of_interest(sol::AbstractArray, m::MTKCell)
  return sol, sol[end-m.out+1:end, :]
end


Base.show(io::IO, m::MTKCell) = print(io, "MTKCell(", m.in, ",", m.out, ")")
initial_params(m::MTKCell) = m.p
paramlength(m::MTKCell) = m.paramlength
Flux.@functor MTKCell (p,)
Flux.trainable(m::MTKCell) = (m.p,)


function _get_bounds(T, default_lb, default_ub, vars)
  cell_lb = T[]
  cell_ub = T[]
  for v in vars
    contains(string(v), "InPin") && continue
    contains(string(v), "OutPin") && continue
    hasmetadata(v, ModelingToolkit.VariableOutput) && continue
    lower = hasmetadata(v, VariableLowerBound) ? getmetadata(v, VariableLowerBound) : default_lb
    upper = hasmetadata(v, VariableUpperBound) ? getmetadata(v, VariableUpperBound) : default_ub
    push!(cell_lb, lower)
    push!(cell_ub, upper)
  end
  return cell_lb, cell_ub
end


function get_bounds(m::MTKCell{false,W,NET,SYS,PROB,SOLVER,KW,V,OP,<:AbstractMatrix{T}}, ::DataType=nothing; default_lb = -Inf, default_ub = Inf) where {W,NET,SYS,PROB,SOLVER,KW,V,OP,T}
  params = collect(parameters(m.sys))
  _get_bounds(T, default_lb, default_ub, params)
end


function get_bounds(m::MTKCell{true,W,NET,SYS,PROB,SOLVER,KW,V,OP,<:AbstractMatrix{T}}, ::DataType=nothing; default_lb = -Inf, default_ub = Inf) where {W,NET,SYS,PROB,SOLVER,KW,V,OP,T}
  params = collect(parameters(m.sys))
  states = collect(ModelingToolkit.states(m.sys))[m.in+1:end]
  _get_bounds(T, default_lb, default_ub, vcat(params,states))
end

function MTKRecurMapped(chainf::C, wiring, solver; kwargs...) where C
  chainf(LTC.MapperIn(wiring),
          RecurMTK(MTKCell(wiring, solver; kwargs...)),
          LTC.MapperOut(wiring))
end

function MTKRecurMapped(chainf::C, wiring, net, sys, solver; kwargs...) where C
  chainf(LTC.MapperIn(wiring),
          RecurMTK(MTKCell(wiring, net, sys, solver; kwargs...)),
          LTC.MapperOut(wiring))
end
