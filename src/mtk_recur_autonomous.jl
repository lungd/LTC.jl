mutable struct RecurMTKAut{T,V,S}
  cell::T
  p::V
  paramlength::Int
  state::S
end
function RecurMTKAut(cell;)
  p = initial_params(cell)
  RecurMTKAut(cell, p, length(p), cell.state0)
end
function (m::RecurMTKAut)(x, p=m.p)
  m.state, y = m.cell(m.state, p)
  return y
end
Base.show(io::IO, m::RecurMTKAut) = print(io, "RecurMTKAut(", m.cell, ")")
initial_params(m::RecurMTKAut) = m.p
paramlength(m::RecurMTKAut) = m.paramlength

Flux.@functor RecurMTKAut (p,)
Flux.trainable(m::RecurMTKAut) = (m.p,)


struct MTKCellAut{NET,SYS,PROB,SOLVER,SENSEALG,V,OP,S}
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
  state0::S
end
function MTKCellAut(out::Int, net, sys, solver, sensealg, tspan; seq_len=1)

  defs = ModelingToolkit.get_defaults(sys)
	prob = ODEProblem(sys, defs, tspan) # TODO: jac, sparse ???

  param_names = collect(parameters(sys))


  outpins = [getproperty(net, Symbol("x", i, "_OutPin"), namespace=false) for i in 1:out]

  state0 = reshape(prob.u0, :, 1)
	infs = fill(Inf, size(state0,1))

  p_ode = prob.p
  @show prob.u0
  p = Float64.(vcat(p_ode, prob.u0))
  # p = p_ode

  @show param_names
  @show prob.u0
  @show size(state0)
  @show prob.f.syms
  @show length(prob.p)
  @show outpins
	# @show sys.states

  MTKCellAut(out, net, sys, prob, solver, sensealg, tspan, p, length(p), string.(param_names), infs, outpins, state0)
end

function (m::MTKCellAut)(h, p) where {PROB,SOLVER,SENSEALG,V}
  # size(h) == (N,1) at the first MTKCellAut invocation. Need to duplicate batchsize times
  # num_reps = size(saveat,1)-size(h,2)+1
  # hr = repeat(h, 1, num_reps)
  hr = h
  p_ode_l = size(p,1) - size(hr,1)
  p_ode = p[1:p_ode_l]
  _solve(m,hr,p_ode)
end


function _solve(m, u0s, p_ode)

  batchsize = 1
  # batchsize = size(u0s,2)
  infs = m.infs
  # infs = reshape([Inf for _ in 1:m.out*size(saveats[1],1)], m.out,:)
  tspan = (0.0, 1.0)

  function prob_func(prob,i,repeat)
    remake(prob; tspan, u0=u0s[:,i], p=p_ode)
  end

  function output_func(sol,i)
    # sol.retcode != :Success && println("##############")
    sol.retcode != :Success && return infs, false
    # @show size(sol[:,:])
		sol[:,end], false
  end

  ensemble_prob = EnsembleProblem(m.prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sol = solve(ensemble_prob, m.solver, EnsembleThreads(), trajectories=batchsize, sensealg=m.sensealg,
              #saveat=1.0,
              # dense=true,
              save_everystep=false, save_start=false,
              abstol=1e-3, reltol=1e-3,
              )

	get_quantities_of_interest(Array(sol), m)
  # get_quantities_of_interest(saved_values, saveats, m)
end


function get_quantities_of_interest(sol::AbstractArray, m::MTKCellAut)
  # @show size(sol)
  return sol[:, end, :], sol[end-m.out+1:end, end, :]
end

Base.show(io::IO, m::MTKCellAut) = print(io, "MTKCellAut(", m.out, ")")
initial_params(m::MTKCellAut) = m.p
paramlength(m::MTKCellAut) = m.paramlength

Flux.@functor MTKCellAut (p,)
Flux.trainable(m::MTKCellAut) = (m.p,)


reset!(m::RecurMTKAut, p=m.p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))

reset_state!(m::RecurMTKAut, p=m.p) = (m.state = reshape(p[end-size(m.cell.state0,1)+1:end], :, 1))

function get_bounds(m::RecurMTKAut)
  cell_lb = Float64[]
  cell_ub = Float64[]

  params = collect(parameters(m.cell.sys))
  states = collect(ModelingToolkit.states(m.cell.sys))
  for v in vcat(params,states)
		contains(string(v), "OutPin") && continue
		hasmetadata(v, VariableOutput) && continue
    lower = hasmetadata(v, VariableLowerBound) ? getmetadata(v, VariableLowerBound) : -Inf
    upper = hasmetadata(v, VariableUpperBound) ? getmetadata(v, VariableUpperBound) : Inf
    push!(cell_lb, lower)
    push!(cell_ub, upper)
  end
  return cell_lb, cell_ub
end
