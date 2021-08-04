mutable struct MTKNODE{T,V,S}
  cell::T
  p::V
  paramlength::Int
  state::S
end
function MTKNODE(cell; seq_len=1)
  p = initial_params(cell)
  MTKNODE(cell, p, length(p), cell.state0)
end
function (m::MTKNODE)(x, p=m.p)
  m.state, y = m.cell(m.state, x, p)
  return y
end
Base.show(io::IO, m::MTKNODE) = print(io, "MTKNODE(", m.cell, ")")
initial_params(m::MTKNODE) = m.p
paramlength(m::MTKNODE) = m.paramlength

Flux.@functor MTKNODE (p,)
Flux.trainable(m::MTKNODE) = (m.p,)


struct MTKNODECell{NET,SYS,PROB,SOLVER,SENSEALG,V,OP,S}
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
  state0::S
end
function MTKNODECell(in::Int, out::Int, net, sys, solver, sensealg, tspan=(0f0,1f0), T=Float32; seq_len=1)

  defs = ModelingToolkit.get_defaults(sys)
  prob = ODEProblem(sys, defs, tspan) # TODO: jac, sparse ???

  # param_names = collect(parameters(sys))
  # input_idxs = Int8[findfirst(x->contains(string(x), string(Symbol("x$(i)_InPin"))), param_names) for i in 1:in]
	# param_names = param_names[in+1:end]
  # outpins = [getproperty(net, Symbol("x", i, "_OutPin"), namespace=false) for i in 1:out]

  _states = collect(states(sys))
  input_idxs = Int8[findfirst(x->contains(string(x), string(Symbol("x$(i)_InPin"))), _states) for i in 1:in]
  param_names = ["placeholder"]
  outpins = 1f0

  # state0 = reshape(prob.u0, :, 1)
	# infs = fill(Inf, size(state0,1))
  #
  # p_ode = prob.p[in+1:end]
  # @show prob.u0
  # p = Float64.(vcat(p_ode, prob.u0))
  # p = p_ode

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
  @show input_idxs
  @show outpins
	# @show sys.states

  MTKNODECell(in, out, net, sys, prob, solver, sensealg, tspan, p, length(p), param_names, outpins, infs, state0)
end

function (m::MTKNODECell)(h, inputs::AbstractArray{T,3}, p) where {PROB,SOLVER,SENSEALG,V, T<:AbstractFloat}
  # size(h) == (N,1) at the first MTKNODECell invocation. Need to duplicate batchsize times
  num_reps = size(inputs,3)-size(h,2)+1
  hr = repeat(h, 1, num_reps)
  p_ode_l = size(p,1) - size(hr,1)
  p_ode = p[size(hr,1)+1:end]
  solve_ensemble_full_seq(m,hr,inputs,p_ode)
end


function solve_ensemble_full_seq(m, u0s, xs, p_ode) #where T <: AbstractFloat

  @show size(p_ode)
  @show p_ode[1:42]
  T = Float32
  # xs = Flux.stack(xs, 2)
	# tspan = (T(0), T(size(xs,2)))
  tspan = (0f0, 20f0)
  # tspan = (0.0, 1.0)
  batchsize = size(xs,3)
	infs = m.infs

  dosetimes = collect(1f0:tspan[2]-1)

  function prob_func(prob,i,repeat)
    x = xs[:,1,i]
    u0i = u0s[:,i]
    u0 = vcat(x,u0i)
    # p = vcat(x, p_ode)
    p = p_ode

		# condition(u,t,integrator) = t < tspan[2]
		function affect!(integrator)
			idx = trunc(Int,integrator.t)+1
      # println("i=$(i), t_trunc=$(t)")
			# xt = t > size(xs,2) ? xs[:,t,i] : xs[:,end,i]
      for f in 1:size(xs,1)
        integrator.u[f] = idx > size(xs,2) ? xs[f, end, i] : xs[f, idx, i]
      end
      # xt = xs[:,Int(integrator.t),i]
			# integrator.p[1:size(xs,1)] = xs[:,i,t]

		end
		# callback = PeriodicCallback(affect!,1f0,initial_affect=false,save_positions=(true,false))
    callback = PresetTimeCallback(dosetimes,affect!,save_positions=(false,false))

    remake(prob; u0, tspan, p, callback, saveat=1.0, save_everystep=false, save_start=false)
  end

  function output_func(sol,i)
    sol.retcode != :Success && return infs, false
    # @show sol.t
    # @show size(sol)
		sol, false
  end

  ensemble_prob = EnsembleProblem(m.prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sol = solve(ensemble_prob, m.solver, EnsembleThreads(), trajectories=batchsize,
              sensealg=m.sensealg, saveat=collect(1f0:Float32(size(xs,2))),
	  # save_on = false,
	  # save_end = true,
	  #alias_u0 = true,
	  # calck = false
		) # TODO: saveat ?
  @show sol[1].t
  s = Array(sol)
  @show size(sol)
  @show size(sol[1])
  @show size(s)
	get_quantities_of_interest(s, m, Int.(dosetimes))
end



function get_quantities_of_interest(sol::AbstractArray, m::MTKNODECell, dosetimes)
  @show size(sol)
  @show size(sol[1])
  @show size(sol[1:2])
  @show size(sol[:, 1, 1:2])
  @show size(sol[:, :, 1:2])
  # out = Array{Float32}(undef, m.out, 20, size(sol,3))
  # for t in dosetimes
  #   out[:,t,:] .= sol[end-m.out+1:end,t,:]
  # end
  # out[:,end,:] = sol[end-m.out+1:end, end, :]
  # out = size(sol,2) == 21 ? sol[end-m.out+1:end, 2:end, :] : sol[end-m.out+1:end, :, :]
  out = sol[end-m.out+1:end, end-length(dosetimes):end, :]
  return sol[m.in+1:end,end,:], out
end

Base.show(io::IO, m::MTKNODECell) = print(io, "MTKNODECell(", m.in, ",", m.out, ")")
initial_params(m::MTKNODECell) = m.p
paramlength(m::MTKNODECell) = m.paramlength

Flux.@functor MTKNODECell (p,)
Flux.trainable(m::MTKNODECell) = (m.p,)


reset!(m::MTKNODE, p=m.p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))

reset_state!(m::MTKNODE, p=m.p) = (m.state = reshape(p[end-size(m.cell.state0,1)+1:end], :, 1))

function get_bounds(m::MTKNODE, T=Float32, default_lb = -Inf, default_ub = Int)
  cell_lb = T[]
  cell_ub = T[]

  params = collect(parameters(m.cell.sys))
  states = collect(ModelingToolkit.states(m.cell.sys))[m.cell.in+1:end]
  for v in vcat(params,states)
    contains(string(v), "InPin") && continue
		contains(string(v), "OutPin") && continue
		hasmetadata(v, VariableOutput) && continue
    lower = hasmetadata(v, VariableLowerBound) ? getmetadata(v, VariableLowerBound) : default_lb
    upper = hasmetadata(v, VariableUpperBound) ? getmetadata(v, VariableUpperBound) : default_ub
    push!(cell_lb, lower)
    push!(cell_ub, upper)
  end
  return cell_lb, cell_ub
end
