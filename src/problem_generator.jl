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

# @register Int64(t)
# @register floor(x)
# get_xt(val,t) = val[Int(floor(t))+1]
# @register get_xt(val,x)

function ExternalInput2(; name, seq_len)
  vars = @variables x(t)
  if seq_len > 1
    ps = @parameters val[1:seq_len]
    ps = vcat(ps...)
  else
    ps = @parameters val
  end
  eqs = Equation[
    x ~ get_xt(val,t)
  ]
  val_defs = seq_len > 1 ? [val[i] => 13.37f0 for i in 1:length(val)] : [val => 13.37f0]
  defaults = Dict(
    #x => 0f0,
    val_defs...,
  )
  ODESystem(eqs,t,vars,ps; defaults, name)
end

# value_vector = zeros(17,1)
# f_fun(src,t) = value_vector[src,Int(floor(t))+1]
# @register f_fun(t)


sigs(x) = 1 / (1 + exp(-x))

@register GalacticOptim.Flux.sigmoid(t)
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


function GapJunction(;name)
  vars = @variables I(t), v_pre(t), v_post(t)
  ps = @parameters G
  eqs = [
    I ~ G * (v_pre - v_post)
  ]
  defaults = Dict(
    G => rand_uniform(Float32, 0.001, 1, 1)[1],
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


function vCaNeuron(; name)
  vars = @variables v(t), Ca(t), I_comps(t)
  ps = @parameters Cm
  @named leak = LeakChannel()
  eqs = [
    D(v) ~ -Cm * (leak.I + I_comps)
    D(Ca) ~ α*exp(-τ) - β*Ca
    leak.v ~ v
  ]
  defaults = Dict(
    v => rand_uniform(Float32, -80, -30, 1)[1],
    Ca => rand_uniform(Float32, 0.001, 0.2, 1)[1],
    Cm => rand_uniform(Float32, 1, 3, 1)[1],
  )
  systems = ODESystem[leak]
  ODESystem(eqs, t, vars, ps; systems, defaults, name)
end



function Net(wiring; name, seq_len=1)
  # @parameters f(t)

  vars = Num[]
  ps = Num[]
  eqs = Equation[]
  systems = ODESystem[]

  N = wiring.n_total
  inputs = seq_len > 1 ? [ExternalInput2(;name=Symbol("x_$(i)_ExternalInput"), seq_len) for i in 1:wiring.n_in] :
    [ExternalInput(;name=Symbol("x_$(i)_ExternalInput")) for i in 1:wiring.n_in]
  # inputs = [ExternalInput(name=Symbol("x_$(i)_ExternalInput")) for i in 1:wiring.n_in]
  # inputs = [ExternalInput2(name=Symbol("x_$(i)_ExternalInput"), seq_len) for i in 1:wiring.n_in]

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







####################################
# OBSOLETE CODE
####################################


# function Mapper(;name)
#   vars = @variables v(t), x(t)
#   ps = @parameters W, b, val
#   eqs = [
#     v ~ W * (x + b)
#     x ~ val
#   ]
#   defaults = Dict(
#     val => 0f0,
#     W => rand_uniform(Float32, -1, 1, 1)[1],
#     b => 0f0,
#   )# : Dict()
#   systems = ODESystem[]
#   ModelingToolkit.ODESystem(eqs, t, vars, ps; systems, defaults, name)
# end

# function Mapout(;name)
#   vars = @variables v(t), x(t)
#   ps = @parameters W, b
#   eqs = [
#     v ~ W * (x + b)
#   ]
#   defaults = Dict(
#     W => rand_uniform(Float32, -1, 1, 1)[1],
#     b => 0f0,
#   )# : Dict()
#   systems = ODESystem[]
#   ModelingToolkit.ODESystem(eqs, t, vars, ps; systems, defaults, name)
# end




function ExternalInput(batchsize; name)
  vars = @variables x[1:batchsize](t)
  vars = vcat(vars...)
  ps = @parameters val[1:batchsize]
  ps = vcat(ps...)
  eqs = Equation[
    #[x[i] ~ val[i] for i in 1:n_in]...
    [x[i] ~ val[i] for i in 1:batchsize]...,
  ]
  defaults = Dict(
    [val[i] => 13.37f0 for i in 1:batchsize]...,
    #x => 0f0
  )
  ODESystem(eqs,t,vars,ps; defaults, name)
end

function SigmoidSynapse(batchsize; name)
  vars = @variables I[1:batchsize](t), v_pre[1:batchsize](t), v_post[1:batchsize](t)
  vars = vcat(vars...)
  ps = @parameters μ, σ, G, E
  eqs = [
    # I ~ G * (1f0 / (1f0 + exp(-((v_pre - μ) * σ)))) * (v_post - E)
    [I[i] ~ G * GalacticOptim.Flux.sigmoid((v_pre[i] - μ) * σ) * (v_post[i] - E) for i in 1:batchsize]...,
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

function LeakChannel(batchsize; name)
  vars = @variables I[1:batchsize](t), v[1:batchsize](t)
  vars = vcat(vars...)
  ps = @parameters G, E
  eqs = [
    [I[i] ~ G * (v[i] - E) for i in 1:batchsize]...,
  ]
  defaults = Dict(
    G => rand_uniform(Float32, 0.001, 1, 1)[1],
    E => rand_uniform(Float32, -0.3, 0.3, 1)[1],
  )# : Dict()
  systems = ODESystem[]
  ModelingToolkit.ODESystem(eqs, t, vars, ps; systems, defaults, name)
end

function Neuron(batchsize; name)
  vars = @variables v[1:batchsize](t), I_comps[1:batchsize](t)
  vars = vcat(vars...)
  ps = @parameters Cm
  @named leak = LeakChannel(batchsize;)
  #@named mapout = Mapout()
  @variables I[1:batchsize]
  eqs = [
    [D(v[i]) ~ -Cm * (getproperty(leak, Symbol(I[i])) + I_comps[i]) for i in 1:batchsize]...,
    [getproperty(leak, Symbol(replace(string(v[i]),"(t)"=>""))) ~ v[i] for i in 1:batchsize]...,
  ]
  defaults = Dict(
    [v[i] => rand_uniform(Float32, 0.001, 0.2, 1)[1] for i in 1:batchsize]...,
    Cm => rand_uniform(Float32, 1, 3, 1)[1],
  )
  systems = ODESystem[leak]
  #systems = ODESystem[leak, mapout]
  ODESystem(eqs, t, vars, ps; systems, defaults, name=name)
end

function Net(wiring, batchsize; name)
  vars = Num[]
  ps = Num[]
  eqs = Equation[]
  systems = ODESystem[]

  N = wiring.n_total

  inputs = [ExternalInput(batchsize; name=Symbol("x_$(i)")) for i in 1:wiring.n_in]

  systems = ODESystem[inputs...]


  neurons = ODESystem[Neuron(batchsize; name=Symbol("n$(n)")) for n in 1:N]
  push!(systems, neurons...)

  @variables v_pre[1:batchsize]
  @variables v_post[1:batchsize]
  @variables x[1:batchsize]
  @variables v[1:batchsize]
  @variables I[1:batchsize]
  @variables I_comps[1:batchsize]


  for dst in 1:length(neurons)
    t = Term
    I_comps_dst = Num[0 for b in 1:batchsize]
    for src in 1:length(wiring.sens_mask[:,dst])
      wiring.sens_mask[src,dst] == 0 && continue
      syn = SigmoidSynapse(batchsize; name=Symbol("x_$(src)-->n$(dst)"))
      push!(systems, syn)
      for b in 1:batchsize
        push!(eqs, getproperty(syn, Symbol(v_pre[b])) ~ getproperty(inputs[src], Symbol(x[b])))
        push!(eqs, getproperty(syn, Symbol(v_post[b])) ~ getproperty(neurons[dst], Symbol(v[b])))
        I_comps_dst[b] += getproperty(syn, Symbol(I[b]))
      end
    end
    for src in 1:length(wiring.syn_mask[:,dst])
      wiring.syn_mask[src,dst] == 0 && continue
      syn = SigmoidSynapse(batchsize; name=Symbol("n$(src)-->n$(dst)"))
      push!(systems, syn)
      for b in 1:batchsize
        push!(eqs, getproperty(syn, Symbol(v_pre[b])) ~ getproperty(neurons[src], Symbol(v[b])))
        push!(eqs, getproperty(syn, Symbol(v_post[b])) ~ getproperty(neurons[dst], Symbol(v[b])))
        I_comps_dst[b] += getproperty(syn, Symbol(I[b]))
      end
    end
    for b in 1:batchsize
      push!(eqs, getproperty(neurons[dst], Symbol(I_comps[b])) ~ I_comps_dst[b])
    end
  end
  defaults = Dict(
    #[input[i] => 0f0 for i in 1:size(input,1)]...,
  )
  ODESystem(eqs, t, vars, ps; systems, defaults, name=name)
end


function finish_prob(net,x,ks,p)
  @variables x_input[1:size(x,1)](t)
  push!(net.eqs, [x_input[i] ~ x[i] for i in 1:size(x,1)]...)
  sys = structural_simplify(net)
  defs = vcat([ks[i] => ps[i] for i in 1:length(ps)])
  prob = ODEProblem(sys, defs, Float32.((0,1)))
  sys, prob
end


function generate_sys(wiring, external_input; create_defaults=true)
  @named net = Net(wiring)
  #@variables x_input[1:size(external_input,1)](t)
  #push!(net.eqs, [x_input[i] ~ external_input[i] for i in 1:size(external_input,1)]...)
  #sys = structural_simplify(net)
  net
end

function generate_prob(wiring, external_input, _ks=nothing, _ps=nothing; create_defaults=true)
  sys = generate_sys(wiring, external_input; create_defaults)
  ks, ps = _ks, _ps
  if create_defaults && ps == nothing
    ks, ps = get_params(sys)
  end
  defs = vcat([ks[i] => ps[i] for i in 1:length(ps)])
  prob = ODEProblem(sys, defs, Float32.((0,1)))
end


function get_params(sys)
  defs = ModelingToolkit.get_defaults(sys)
  collect(keys(defs)), Float32.(collect(values(defs)))
end

# wiring = NCPWiring(17,2,
#          n_sensory=4, n_inter=4, n_command=7, n_motor=2,
#          sensory_in=-1, rec_sensory=0, sensory_inter=2, sensory_command=0, sensory_motor=0,
#          inter_in=2, rec_inter=2, inter_command=3, inter_motor=1,                       # inter_in = sensory_out
#          command_in=0, rec_command=4, command_motor=2,                   # command_in = inter_out
#          motor_in=0, rec_motor=3)
# ncp = LTC.LTCNet(wiring, nothing, nothing)

# x = rand(17)
# @time prob = generate_prob(wiring, x)
#
# sol = solve(prob, Tsit5())
# sol.t
# plot(sol)










#
# function ModelingToolkit.ODESystem(
#                    deqs::AbstractVector{<:Equation}, iv, dvs, ps;
#                    observed = Num[],
#                    systems = ODESystem[],
#                    name=gensym(:ODESystem),
#                    default_u0=Dict(),
#                    default_p=Dict(),
#                    defaults=ModelingToolkit._merge(Dict(default_u0), Dict(default_p)),
#                    connection_type=nothing,
#                   )
#     iv′ = ModelingToolkit.value(iv)
#     dvs′ = ModelingToolkit.value.(dvs)
#     ps′ = ModelingToolkit.value.(ps)
#
#     if !(isempty(default_u0) && isempty(default_p))
#         Base.depwarn("`default_u0` and `default_p` are deprecated. Use `defaults` instead.", :ODESystem, force=true)
#     end
#     defaults = ModelingToolkit.todict(defaults)
#     defaults = Dict(ModelingToolkit.value(k) => ModelingToolkit.value(v) for (k, v) in pairs(defaults))
#
#     tgrad = ModelingToolkit.RefValue(Vector{Num}(undef, 0))
#     jac = ModelingToolkit.RefValue{Any}(Matrix{Num}(undef, 0, 0))
#     Wfact   = ModelingToolkit.RefValue(Matrix{Num}(undef, 0, 0))
#     Wfact_t = ModelingToolkit.RefValue(Matrix{Num}(undef, 0, 0))
#     if !isempty(systems)
#       sysnames = [ModelingToolkit.nameof(s) for s in systems]
#       @show "abc"
#       @show sysnames
#       @show unique(sysnames)
#       @show length(unique(sysnames))
#       @show length(sysnames)
#       if length(unique(sysnames)) != length(sysnames)
#         throw(ArgumentError("System names must be unique."))
#       end
#     end
#     ODESystem(deqs, iv′, dvs′, ps′, observed, tgrad, jac, Wfact, Wfact_t, name, systems, defaults, nothing, connection_type)
# end
