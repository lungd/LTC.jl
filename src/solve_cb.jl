using ModelingToolkit
using ModelingToolkit: get_defaults
using OrdinaryDiffEq
using DiffEqSensitivity
using Zygote
using DiffEqCallbacks: PresetTimeCallback, ContinuousCallback, PeriodicCallback


@variables t
D = Differential(t)

function InSPin(; name)
  @variables x(t)=13.0
  ODESystem(Equation[D(x)~0],t,[x],Num[]; name)
end

function Neuron(; name)
  @variables v(t)=0.0 I(t)
	@parameters s=0.4
  eqs = D(v) ~ I/s
  ODESystem([eqs],t,[v,I],[s];name)
end

function Network(;name)
	@named in1 = InSPin()
  @named n1 = Neuron()
  eqs = n1.I ~ in1.x
  ODESystem([eqs],t,[],[];name,systems=[in1,n1])
end

@named net = Network()
sys = structural_simplify(net)
prob = ODEProblem(sys, ModelingToolkit.get_defaults(sys), (0.0, 20.0), sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),)
p = prob.p
u0 = prob.u0

function predict(xs,u0s,p_ode; solve_saveat=1.0, save_positions=(true,false), kwargs...)
	tspan = (0.0, Float64(size(xs,2)))
  @show size(xs)

  @show tspan
	batchsize = size(xs,3)
  @show batchsize
	dosetimes = [i for i in 1.0:size(xs,2)-1]
  @show dosetimes

	function prob_func(prob,i,repeat)
  	x = xs[:,1,i]
  	u0i = u0s[:,i]
  	u0 = vcat(x,u0i)
  	p = p_ode

    condition(u,t,integrator) = t in dosetimes
  	function affect!(integrator)
  		idx = trunc(Int,integrator.t)+1
  	  for f in 1:size(xs,1)
  			integrator.u[f] = idx > size(xs,2) ? xs[f, end, i] : xs[f, idx, i]
  	  end
  	end
    # callback = PeriodicCallback(affect!,1f0,save_positions=save_positions)
  	callback = PresetTimeCallback(dosetimes,affect!,save_positions=save_positions)

  	remake(prob; u0, tspan, p, kwargs...)
	end

	function output_func(sol,i)
    sol, false
	end

  solver = VCABM()
  sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

	ensemble_prob = EnsembleProblem(prob; prob_func, output_func, safetycopy=true) # TODO: safetycopy ???
	sol = solve(ensemble_prob, solver, EnsembleThreads(), trajectories=batchsize,
    sensealg=sensealg, saveat=1.0, save_everystep=false, save_start=false)

  s = Array(sol)[2:end,:,:]
  @show size(s)
  @show sol[1].t
  return s
end

function loss(x,θ; solve_saveat=1.0, save_positions=(true,false), kwargs...)
  u0 = repeat(θ[1:1],1,size(x,3))
  p = θ[2:end]
  ŷ = predict(x,u0,p; solve_saveat, save_positions, kwargs...)[:,end-9:end,:]
  @show size(ŷ)
	sum(ŷ)
end

xs = rand(1,10,2)
θ = [u0[2:end]; p]
loss(xs, θ, saveat=1.0)

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=1.0, save_positions=(false,false), saveat=1.0)
end

@time gs = Zygote.gradient(θ) do p
  loss(xs,p; save_positions=(false,false))
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=1.0, save_positions=(false,false), saveat=collect(1.0:20.0))
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=1.0, save_positions=(false,false), saveat=1.0, save_start=false)
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=1.0, save_positions=(false,false), save_everystep=false)
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=1.0, save_positions=(false,false), save_everystep=false, saveat=1.0)
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=1.0, save_positions=(false,false), save_everystep=false, saveat=collect(1.0:20.0))
end


@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=collect(1.0:20.0), save_positions=(false,false))
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=collect(1.0:20.0), save_positions=(false,false), saveat=1.0)
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=collect(1.0:20.0), save_positions=(false,false), saveat=collect(1.0:20.0))
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=collect(1.0:20.0), save_positions=(false,false), saveat=1.0, save_start=false)
end





@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=1.0, save_positions=(true,false))
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=1.0, save_positions=(true,false), saveat=1.0)
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=1.0, save_positions=(true,false), saveat=collect(1.0:20.0))
end

@time gs = Zygote.gradient(θ) do p
	loss(xs,p; solve_saveat=1.0, save_positions=(true,false), saveat=1.0, save_start=false)
end




struct MMMyDiscreteCallback{F1,F2,F3,F4} <: DiffEqBase.AbstractDiscreteCallback
  condition::F1
  affect!::F2
  initialize::F3
  finalize::F4
  save_positions::Vector
  MMMyDiscreteCallback(condition::F1,affect!::F2,
                   initialize::F3,finalize::F4,save_positions) where {F1,F2,F3,F4} = new{F1,F2,F3,F4}(condition,
                                                                                   affect!,initialize,finalize, save_positions)
end
MMMyDiscreteCallback(condition,affect!;
                 initialize = (cb,u,t,integrator)->nothing, finalize = DiffEqBase.FINALIZE_DEFAULT,
                 save_positions=(true,true)) = MMMyDiscreteCallback(condition,affect!,initialize,finalize,[save_positions[1],save_positions[2]])



Zygote.@nograd PeriodicCallback
Zygote.@nograd DiffEqBase.add_tstop!

function predict_single(xs,u0s,p_ode; solve_saveat=1.0, save_positions=(true,false), kwargs...)
	tspan = (0.0, Float64(size(xs,2)))
  @show size(xs)
  @show tspan

	dosetimes = [i for i in 1.0:size(xs,2)-1]
  # @show dosetimes

  x = xs[:,1]
  u0i = u0s[:]
  u0 = vcat(x,u0i)
  p = p_ode

  condition(u,t,integrator) = t in dosetimes
  function affect!(integrator)
    idx = trunc(Int,integrator.t)+1
    for f in 1:size(xs,1)
      integrator.u[f] = idx > size(xs,2) ? xs[f, end] : xs[f, idx]
    end
  end
  callback = MMMyDiscreteCallback(affect!,1f0)
  # callback = PresetTimeCallback(dosetimes,affect!,save_positions=save_positions)

  solver = Tsit5()
  sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

	sol = solve(prob, solver,
    sensealg=sensealg, saveat=0.1)
  @show size(sol)
  s = Array(sol)[2:end,:]
  @show size(s)
  @show sol.t
  return s
end

function loss_single(x,θ; solve_saveat=1.0, save_positions=(true,false), kwargs...)
  u0 = θ[1:1]
  p = θ[2:end]
  ŷ = predict_single(x,u0,p; solve_saveat, save_positions, kwargs...)#[:,end-9:end]
  @show size(ŷ)
	sum(ŷ)
end

p = prob.p
u0 = prob.u0
xs = rand(1,10)
θ = [1.0; p]
loss_single(xs, θ, saveat=1.0)

@time gs = Zygote.gradient(θ) do p
	loss_single(xs,p; solve_saveat=1.0, save_positions=(false,false), saveat=1.0)
end
