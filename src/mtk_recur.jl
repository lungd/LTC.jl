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
  m.state, y = m.cell(m.state, x, p)
  return y
end
Base.show(io::IO, m::RecurMTK) = print(io, "RecurMTK(", m.cell, ")")
initial_params(m::RecurMTK) = m.p
paramlength(m::RecurMTK) = m.paramlength
Flux.@functor RecurMTK (p,)
Flux.trainable(m::RecurMTK) = (m.p,)
reset!(m::RecurMTK, p=m.p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))
reset_state!(m::RecurMTK, p=m.p) = (m.state = reshape(p[end-size(m.cell.state0,1)+1:end], :, 1))
function get_bounds(m::RecurMTK)
  T = eltype(m.p)
  T = Float32
  cell_lb = T[]
  cell_ub = T[]

  params = collect(parameters(m.cell.sys))
  states = collect(ModelingToolkit.states(m.cell.sys))[m.cell.in+1:end]
  for v in vcat(params,states)
    contains(string(v), "InPin") && continue
    contains(string(v), "OutPin") && continue
    hasmetadata(v, VariableOutput) && continue
    lower = hasmetadata(v, VariableLowerBound) ? getmetadata(v, VariableLowerBound) : -Inf
    upper = hasmetadata(v, VariableUpperBound) ? getmetadata(v, VariableUpperBound) : Inf
    push!(cell_lb, lower)
    push!(cell_ub, upper)
  end
  return cell_lb, cell_ub
end

struct MTKCell{NET,SYS,PROB,SOLVER,SENSEALG,V,OP,S}
  in::Int
  out::Int
  net::NET
  sys::SYS
  prob::PROB
  solver::SOLVER
  sensealg::SENSEALG
  tspan::Tuple
  p::V
  paramlength::Int
  param_names::Vector{String}
  outpins::OP
	infs::Vector
  return_sequence::Bool
  state0::S
end
function MTKCell(wiring::Wiring, net, sys::ModelingToolkit.AbstractSystem, solver, sensealg, T=Float32; return_sequence=true)

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

  p = vcat(p_ode, u0)
  infs = fill(T(Inf), size(state0,1))

  @show param_names
  @show prob.u0
  @show size(state0)
  @show prob.f.syms
  @show length(prob.p)
  @show length(prob.p)
  @show input_idxs
  @show outpins

  @show typeof(p_ode)
  @show typeof(prob.u0)
  @show eltype(p_ode)
  @show eltype(prob.u0)

  MTKCell(in, out, net, sys, prob, solver, sensealg, tspan, p, length(p), param_names, outpins, infs, return_sequence, state0)
end

function (m::MTKCell)(h, xs::AbstractArray, p) where {PROB,SOLVER,SENSEALG,V}
  # size(h) == (N,1) at the first MTKCell invocation. Need to duplicate batchsize times
  num_reps = size(xs,2)-size(h,2)+1
  hr = repeat(h, 1, num_reps)
  p_ode_l = size(p)[1] - size(hr)[1]
  p_ode = @view p[1:p_ode_l]
  solve_ensemble(m,hr,xs,p_ode)
end


function solve_ensemble(m, u0s, xs, p_ode)
  T = Float32
  elapsed = 1.0 # TODO
  tspan = T.((0.0, elapsed))

  batchsize = size(xs,2)
	infs = m.infs

  function prob_func(prob,i,repeat)
    u0 = vcat((@view xs[:,i]), (@view u0s[:,i]))
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
                         saveat=0.5f0, reltol=1e-3, abstol=1e-3,
                         sensealg=m.sensealg,
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



function MTKRecurMapped(chainf, wiring, net, sys, solver, sensealg)
  chainf(LTC.MapperIn(wiring),
          RecurMTK(MTKCell(wiring, net, sys, solver, sensealg)),
          LTC.MapperOut(wiring))
end
