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
function MTKCell(in::Int, out::Int, net, sys::ModelingToolkit.AbstractSystem, solver, sensealg, T=Float32; return_sequence=true)

  tspan = (T(0), T(1))
  defs = ModelingToolkit.get_defaults(sys)
	prob = ODEProblem(sys, defs, tspan) # TODO: jac, sparse ???

  # _pn = collect(parameters(sys))
  _states = collect(states(sys))
  input_idxs = Int8[findfirst(x->contains(string(x), string(Symbol("x$(i)_InPin"))), _states) for i in 1:in]
	# pn = _pn[in+1:end]
  # param_names = string.(pn)

  param_names = ["placeholder"]

  #outpins = [getproperty(net, Symbol("x", i, "_OutPin"), namespace=false) for i in 1:out]
  outpins = 1f0

  p_ode = prob.p
  u0 = prob.u0[in+1:end]

  state0 = reshape(u0, :, 1)

  # p_ode = prob.p[in+1:end]::Vector{T}

  p = vcat(p_ode, u0)
  # p = p_ode
  infs = fill(T(Inf), size(state0,1))



  @show param_names
  @show prob.u0
  @show size(state0)
  @show prob.f.syms
  @show length(prob.p)
  @show length(prob.p)
  @show input_idxs
  @show outpins
	# @show sys.states

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
  # @show batchsize
	infs = m.infs

  function prob_func(prob,i,repeat)
    x = @view xs[:,i]
    # u0 = @view u0s[:,i]
    # p = vcat(x, p_ode)
    u0i = @view u0s[:,i]
    u0 = vcat(x, u0i)
    p = p_ode
    remake(prob; tspan, p, u0, jac=true)
  end

  function output_func(sol,i)
    sol.retcode != :Success && return infs, false
		# sol[:,end], false
    sol[m.in+1:end, end], false
  end

  ensemble_prob = EnsembleProblem(m.prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sol = DiffEqBase.solve(ensemble_prob, m.solver, EnsembleThreads(), trajectories=batchsize,
                         saveat=0.2, reltol=1e-3, abstol=1e-3,
                         sensealg=m.sensealg,

              # dense = false,
	  # save_on = false,
	  # save_end = true,
	  #alias_u0 = true,
	  # calck = false
		) # TODO: saveat ?

	get_quantities_of_interest(Array(sol), m)
end


function get_quantities_of_interest(sol::AbstractArray, m::MTKCell)
  # @show size(sol)
  return sol, sol[end-m.out+1:end, :]

  # outpins = m.outpins
	# outb = Flux.Zygote.Buffer(u0s, m.out, batchsize)
	# for j in 1:batchsize
	# 	solj = sol[j]
	# 	for i in 1:m.out
	# 		# @show outpins[j].x               # x1_OutPin₊x(t)
	# 		# @show sol[j][outpins[j].x, end]  # e.g., -0.06400018f0
	# 		# obs = observed_var_sol(m.sys, sol[j], outpins[j].x)
	# 		obs = solj[outpins[i].x,1]
	# 		# @show obs
	# 		# obs = sol[j][outpins[j].x]
	# 		outb[i,j] = obs
	# 		# @show sol[j][outpins[j].x]
	# 		# outb[i,j] = sol[j][outpins[j].x][end]
	# 	end
	# end
	# out = copy(outb)
	# return Array(sol)[:,end,:], out
end

Base.show(io::IO, m::MTKCell) = print(io, "MTKCell(", m.in, ",", m.out, ")")
initial_params(m::MTKCell) = m.p
paramlength(m::MTKCell) = m.paramlength

Flux.@functor MTKCell (p,)
Flux.trainable(m::MTKCell) = (m.p,)


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
