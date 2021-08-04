using ModelingToolkit
using ModelingToolkit: get_defaults
using OrdinaryDiffEq
using DiffEqSensitivity
using Zygote
using DiffEqCallbacks: PresetTimeCallback, ContinuousCallback



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
prob = ODEProblem(sys, ModelingToolkit.get_defaults(sys), (0.0, 1.0), sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),)
p = prob.p
u0 = prob.u0

function predict(xs,u0s,p_ode)
	tspan = (0.0, Float64(size(xs,2)))
	batchsize = size(xs,3)

	dosetimes = collect(1.0:tspan[2]-1)

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
  	# callback = PeriodicCallback(affect!,1f0,initial_affect=false,save_positions=(true,false))
  	callback = PresetTimeCallback(dosetimes,affect!,save_positions=(false,false))
    # callback = ContinuousCallback(condition,affect!,save_positions=(false,false))

  	remake(prob; u0, tspan, p, callback, saveat=1.0, save_everystep=false, save_start=false)
	end

	function output_func(sol,i)
    sol, false
	end

  solver = VCABM()
  sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

	ensemble_prob = EnsembleProblem(prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
	sol = solve(ensemble_prob, solver, EnsembleThreads(), trajectories=batchsize,
    sensealg=sensealg, saveat=collect(1.0:Float64(size(xs,2))))

  s = Array(sol)[2:end,:,:]
  @show size(s)
  @show sol[1].t
  return s
end

function loss(x,θ)
  u0 = θ[1:1]
  p = θ[2:end]
	sum(predict(x,u0,p))
end

xs = rand(1,10,1)
θ = [u0[2:end]; p]
loss(xs, θ)

@time gs = Zygote.gradient(θ) do p
	loss(xs,p)
end
