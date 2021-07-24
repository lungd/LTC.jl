mutable struct RecurMTK{T,V,S}
  cell::T
  p::V
  paramlength::Int
  state::S
end
function RecurMTK(cell; seq_len=1)
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
  p::V
  paramlength::Int
  param_names::Vector{String}
  outpins::OP
	infs::Vector
  state0::S
end
function MTKCell(in::Int, out::Int, net, sys, solver, sensealg; seq_len=1)

  defs = ModelingToolkit.get_defaults(sys)
	prob = ODEProblem(sys, defs, Float32.((0,1))) # TODO: jac, sparse ???

  param_names = collect(parameters(sys))
  input_idxs = Int8[findfirst(x->contains(string(x), string(Symbol("x$(i)_InPin"))), param_names) for i in 1:in]
	param_names = param_names[in+1:end]


  outpins = [getproperty(net, Symbol("x", i, "_OutPin"), namespace=false) for i in 1:out]

  state0 = reshape(prob.u0, :, 1)
	infs = fill(Inf32, size(state0,1))

  p_ode = prob.p[in+1:end]
  @show prob.u0
  p = Float32.(vcat(p_ode, prob.u0))
  # p = p_ode

  @show param_names
  @show prob.u0
  @show size(state0)
  @show prob.f.syms
  @show length(prob.p)
  @show length(prob.p[in+1:end])
  @show input_idxs
  @show outpins
	# @show sys.states

  MTKCell(in, out, net, sys, prob, solver, sensealg, p, length(p), string.(param_names), infs, outpins, state0)
end
function (m::MTKCell)(h, xs::AbstractVecOrMat{T}, p) where {PROB,SOLVER,SENSEALG,V,T}
  # size(h) == (N,1) at the first MTKCell invocation. Need to duplicate batchsize times
  num_reps = size(xs,2)-size(h,2)+1
  hr = repeat(h, 1, num_reps)
  p_ode_l = size(p)[1] - size(hr)[1]
  p_ode = p[1:p_ode_l]
  solve_ensemble(m,hr,xs,p_ode)
end

function solve_ensemble(m, u0s, xs, p_ode, tspan=(0f0,1f0))

  batchsize = size(xs,2)
  # @show batchsize
	infs = m.infs

  function prob_func(prob,i,repeat)
    x = xs[:,i]
    u0 = u0s[:,i]
    p = vcat(x, p_ode)
    remake(prob; tspan, p, u0)
  end

  function output_func(sol,i)
    sol.retcode != :Success && return infs, false
		sol[:,end], false
  end

  ensemble_prob = EnsembleProblem(m.prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sol = solve(ensemble_prob, m.solver, EnsembleThreads(), trajectories=batchsize,
              sensealg=m.sensealg, save_everystep=false, save_start=false,
              dense = false,
	  # save_on = false,
	  # save_end = true,
	  #alias_u0 = true,
	  # calck = false
		) # TODO: saveat ?

	get_quantities_of_interest(sol, m)
end

function get_quantities_of_interest(sol::EnsembleSolution, m::MTKCell)
	s = Array(sol)
  return s, s[end-m.out+1:end, :]

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
  cell_lb = Float32[]
  cell_ub = Float32[]

  params = collect(parameters(m.cell.sys))[m.cell.in+1:end]
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
