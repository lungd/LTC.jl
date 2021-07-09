using LTC
using ModelingToolkit
using LTC: rand_uniform

@parameters t
D = Differential(t)

function ExternalInput(;name)
  vars = @variables x(t)
  ps = @parameters val
  eqs = Equation[
    x ~ val
  ]
  defaults = Dict(
    val => 13.37f0
  )
  ODESystem(eqs,t,vars,ps; defaults, name)
end

@register Flux.sigmoid(t)
# ModelingToolkit.derivative(::typeof(GalacticOptim.Flux.sigmoid), (x,)) = sigs(x) * (1 - sigs(x))
# ModelingToolkit.derivative(::typeof(GalacticOptim.Flux.sigmoid), (x,), ::Val{1}) = sigs(x) * (1 - sigs(x))#GalacticOptim.Flux.sigmoid(x) * (1 - GalacticOptim.Flux.sigmoid(x))

function SigmoidSynapse(;name)
  vars = @variables I(t), v_pre(t), v_post(t)
  ps = @parameters μ, σ, G, E
  eqs = [
    I ~ G * GalacticOptim.Flux.sigmoid((v_pre - μ) * σ) * (v_post - E)
  ]
  defaults = Dict(
    μ => rand_uniform(Float32, 0.3, 0.8, 1)[1],
    σ => rand_uniform(Float32, 3, 8, 1)[1],
    G => rand_uniform(Float32, 0.001, 1, 1)[1],
    E => rand_uniform(Float32, -0.3, 0.3, 1)[1],
  )
  systems = ODESystem[]
  ODESystem(eqs, t, vars, ps; systems, defaults, name)
end

function LeakChannel(;name)
  vars = @variables I(t), v(t)
  ps = @parameters G, E
  eqs = [
    I ~ G * (v - E)
  ]
  defaults = Dict(
    G => rand_uniform(Float32, 0.001, 1, 1)[1],
    E => rand_uniform(Float32, -0.3, 0.3, 1)[1],
  )# : Dict()
  # setmetadata(G, OptimRange, Float32.([0,1]))
  # setmetadata(E, OptimRange, Float32.([-1,1]))
  systems = ODESystem[]
  ODESystem(eqs, t, vars, ps; systems, defaults, name)
end

function Neuron(; name)
  vars = @variables v(t), I_comps(t)
  ps = @parameters Cm
  @named leak = LeakChannel()
  eqs = [
    D(v) ~ -Cm * (leak.I + I_comps)
    leak.v ~ v
  ]
  defaults = Dict(
    v => rand_uniform(Float32, 0.001, 0.2, 1)[1],
    Cm => rand_uniform(Float32, 1, 3, 1)[1],
  )
  systems = ODESystem[leak]
  ODESystem(eqs, t, vars, ps; systems, defaults, name)
end

function Net(wiring; name)
  vars = Num[]
  ps = Num[]
  eqs = Equation[]
  systems = ODESystem[]

  N = wiring.n_total
  inputs = [ExternalInput(;name=Symbol("x_$(i)_ExternalInput")) for i in 1:wiring.n_in]
  systems = ODESystem[inputs...]

  n = 1
  neurons = ODESystem[]
  for i in 1:wiring.n_sensory
    push!(neurons, Neuron(;name=Symbol("n$(n)_SensoryNeuron")))
    n += 1
  end
  for i in 1:wiring.n_inter
    push!(neurons, Neuron(;name=Symbol("n$(n)_InterNeuron")))
    n += 1
  end
  for i in 1:wiring.n_command
    push!(neurons, Neuron(;name=Symbol("n$(n)_CommandNeuron")))
    n += 1
  end
  for i in 1:wiring.n_motor
    push!(neurons, Neuron(;name=Symbol("n$(n)_MotorNeuron")))
    n += 1
  end
  # neurons = ODESystem[Neuron(;name=Symbol("n$(n)_Neuron")) for n in 1:N]
  push!(systems, neurons...)

  n = 1
  for dst in 1:length(neurons)
    I_comps_dst = 0
    for src in 1:length(wiring.sens_mask[:,dst])
      wiring.sens_mask[src,dst] == 0 && continue
      syn = SigmoidSynapse(;name=Symbol("$(n)_x_$(src)-->n$(dst)_SigmoidSynapse"))
      n += 1
      push!(eqs, syn.v_pre ~ inputs[src].x)
      # push!(eqs, syn.v_pre ~ f_fun(src,t))
      push!(eqs, syn.v_post ~ neurons[dst].v)
      push!(systems, syn)
      I_comps_dst += syn.I
    end
    for src in 1:length(wiring.syn_mask[:,dst])
      wiring.syn_mask[src,dst] == 0 && continue
      syn = SigmoidSynapse(;name=Symbol("$(n)_n$(src)-->n$(dst)_SigmoidSynapse"))
      n += 1
      push!(eqs, syn.v_pre ~ neurons[src].v)
      push!(eqs, syn.v_post ~ neurons[dst].v)
      push!(systems, syn)
      I_comps_dst += syn.I
    end
    push!(eqs, neurons[dst].I_comps ~ I_comps_dst)
  end
  defaults = Dict()
  ODESystem(eqs, t, vars, ps; systems, defaults, name=name)
end
