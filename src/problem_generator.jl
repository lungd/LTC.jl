using LTC
using ModelingToolkit
using LTC: rand_uniform

@parameters t
D = Differential(t)


function ExternalInput(;name)
  vars = @variables x(t)#, val(t)
  ps = @parameters val#[1:n_in]
  #ps = []
  eqs = [
    #[x[i] ~ val[i] for i in 1:n_in]...
    x ~ val
  ]
  defaults = Dict(
    val => 0f0
  )
  ODESystem(eqs,t,vars,ps; defaults, name)
end

# @register sigmoid(rand(Float32))
@register sigmoid(t)

function SigmoidSynapse(;name)
  vars = @variables I(t), v_pre(t), v_post(t)
  ps = @parameters μ, σ, G, E
  eqs = [
    I ~ G * (1 / (1 + exp(-((v_pre - μ) * σ)))) * (v_post - E)
    #I ~ G * GalacticOptim.Flux.sigmoid((v_pre - μ) * σ) * (v_post - E)
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

function LeakChannel(;name,)
  vars = @variables I(t), v(t)
  ps = @parameters G, E
  eqs = [
    I ~ G * (v - E)
  ]
  defaults = Dict(
    G => rand_uniform(Float32, 0.001, 1, 1)[1],
    E => rand_uniform(Float32, -0.3, 0.3, 1)[1],
  )# : Dict()
  systems = ODESystem[]
  ModelingToolkit.ODESystem(eqs, t, vars, ps; systems, defaults, name)
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
    Cm => rand_uniform(Float32, 1, 3, 1)[1]
  )
  systems = ODESystem[leak]
  ODESystem(eqs, t, vars, ps; systems, defaults, name=name)
end

function Net(wiring; name)
  N = wiring.n_total
  #vars = @variables x_input[1:size(wiring.sens_mask,1)](t)
  vars = []

  inputs = [ExternalInput(name=Symbol("x_$(i)")) for i in 1:wiring.n_in]

  ps = Num[]
  systems = ODESystem[inputs...]

  #@variables x_input[1:size(wiring.sens_mask,1)](t)

  eqs = Equation[]
  neurons = ODESystem[Neuron(;name=Symbol("n$(n)")) for n in 1:N]
  push!(systems, neurons...)
  #push!(eqs, [D(x_input[i]) ~ 0 for i in 1:size(x_input,1)]...)

  for dst in 1:length(neurons)
    I_comps_dst = 0
    for src in 1:length(wiring.sens_mask[:,dst])
      wiring.sens_mask[src,dst] == 0 && continue
      syn = SigmoidSynapse(;name=Symbol("x_$(src)-->n$(dst)"))
      push!(eqs, syn.v_pre ~ inputs[src].x)                         # TODO: param + callback for input ?
      #push!(eqs, D(syn.v_pre) ~ 0)
      push!(eqs, syn.v_post ~ neurons[dst].v)
      push!(systems, syn)
      I_comps_dst += syn.I
    end
    for src in 1:length(wiring.syn_mask[:,dst])
      wiring.syn_mask[src,dst] == 0 && continue
      syn = SigmoidSynapse(;name=gensym("n$(src)-->n$(dst)"))
      push!(eqs, syn.v_pre ~ neurons[src].v)
      push!(eqs, syn.v_post ~ neurons[dst].v)
      push!(systems, syn)
      I_comps_dst += syn.I
    end
    push!(eqs, neurons[dst].I_comps ~ I_comps_dst)
  end
  # defaults = Dict()
  ODESystem(eqs, t; systems, name=name)
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
