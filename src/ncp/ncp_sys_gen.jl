using LTC
using ModelingToolkit
using LTC: rand_uniform

@parameters t
D = Differential(t)

function InPin(;name)
  vars = @variables x(t)
  ps = @parameters val=13.37f0
  eqs = [x ~ val]
  ODESystem(eqs,t,vars,ps; name)
end

function OutPin(;name)
  vars = @variables x(t)=1111.0f0
  ODESystem(Equation[],t,vars,[]; name)
end

@register Flux.sigmoid(t)

function SigmoidSynapse(;name)
  vars = @variables I(t), v_pre(t), v_post(t)
  ps = @parameters begin
    μ = rand_uniform(Float32, 0.3, 0.8), [lower=0.1f0, upper=1.0f0]
    σ = rand_uniform(Float32, 3, 8), [lower=1f0, upper=10f0]
    G = rand_uniform(Float32, 0.001, 1), [lower=0.0001f0, upper=1.1f0]
    E = rand_uniform(Float32, -0.3, 0.3), [lower=-1f0, upper=1f0]
  end
  eqs = [
    I ~ G * GalacticOptim.Flux.sigmoid((v_pre - μ) * σ) * (v_post - E)
  ]
  ODESystem(eqs, t, vars, ps; name)
end

function LeakChannel(;name)
  vars = @variables I(t), v(t)
  ps = @parameters begin
    G = rand_uniform(Float32, 0.001, 1), [lower = 0.0001f0, upper = 1.1f0]
    E = rand_uniform(Float32, -0.3, 0.3), [lower = -1f0, upper = 1f0]
  end
  eqs = [
    I ~ G * (v - E)
  ]
  ODESystem(eqs, t, vars, ps; name)
end

function Neuron(; name)
  @variables begin
    (v(t) = rand_uniform(Float32, 0.001, 0.2)), [lower=0.0001f0, upper=0.3f0]
    (I_comps(t))
  end
  ps = @parameters begin
    Cm = rand_uniform(Float32, 1, 3), [lower = 0.8f0, upper = 5f0]
  end
  @named leak = LeakChannel()
  eqs = [
    D(v) ~ -Cm * (leak.I + I_comps)
    leak.v ~ v
  ]
  systems = ODESystem[leak]
  ODESystem(eqs, t, [v, I_comps], ps; systems, name)
end

function Net(wiring; name)
  vars = Num[]
  ps = Num[]
  eqs = Equation[]
  systems = ODESystem[]

  N = wiring.n_total
  inputs = [InPin(;name=Symbol("x$(i)_InPin")) for i in 1:wiring.n_in]
  outputs = [OutPin(;name=Symbol("x$(i)_OutPin")) for i in 1:wiring.n_out]
  push!(systems, inputs...)

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
  push!(systems, neurons...)

  n = 1
  for dst in 1:length(neurons)
    I_comps_dst = 0
    for src in 1:length(wiring.sens_mask[:,dst])
      wiring.sens_mask[src,dst] == 0 && continue
      syn = SigmoidSynapse(;name=Symbol("s$(n)_x$(src)-->n$(dst)_SigmoidSynapse"))
      n += 1
      push!(eqs, syn.v_pre ~ inputs[src].x)
      push!(eqs, syn.v_post ~ neurons[dst].v)
      push!(systems, syn)
      I_comps_dst += syn.I
    end
    for src in 1:length(wiring.syn_mask[:,dst])
      wiring.syn_mask[src,dst] == 0 && continue
      syn = SigmoidSynapse(;name=Symbol("s$(n)_n$(src)-->n$(dst)_SigmoidSynapse"))
      n += 1
      push!(eqs, syn.v_pre ~ neurons[src].v)
      push!(eqs, syn.v_post ~ neurons[dst].v)
      push!(systems, syn)
      I_comps_dst += syn.I
    end
    push!(eqs, neurons[dst].I_comps ~ I_comps_dst)
  end

  # connect output pins with n_out last neurons
  for i in 1:wiring.n_out
    push!(eqs, outputs[i].x ~ neurons[end-wiring.n_out-i].v)
  end
  push!(systems, outputs...)

  ODESystem(eqs, t, vars, ps; systems, name=name)
end
