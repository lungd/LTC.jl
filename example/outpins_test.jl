using ModelingToolkit
@variables t
D = Differential(t)

@variables xxx(t)
[xxx]

function InPin(;name)
  @parameters x
  defaults = Dict(x => 13.37)
  ODESystem(Equation[],t,Num[],[x]; name, defaults)
end

function OutPin(;name)
  vars = @variables x(t) xx(t)
  defaults = Dict(x => 0.0)
  ODESystem(Equation[D(x)~xx],t,[x,xx],Num[]; name, defaults)
end

function create_pins(in::Integer, out::Integer)
  inpins = [InPin(;name=Symbol("x$(i)_InPin")) for i in 1:in]
  outpins = [OutPin(;name=Symbol("x$(i)_OutPin")) for i in 1:out]
  inpins, outpins
end

function Network(;name)
    @variables x(t)=0.0 a(t)
    @parameters p=0.4
    eqs = [D(x) ~ a, a ~ 0.1p]
    ODESystem(eqs;name)
end

function Model(;name)
    op = OutPin(;name=:op)
    net = Network(;name=:net)
    eqs = [op.xx ~ net.x]
    # eqs = Equation[]
    systems = [op,net]
    ODESystem(eqs,t;name,systems)
end

# using Symbolics

model = Model(;name=:model)
sys = structural_simplify(model)
@nonamespace n = model.net
@nonamespace o = model.op
u0 = Dict(
    n.x => 0.0,
    o.x => 0.0,
)
p = Dict(
    n.p => 0.4,
)
defs = ModelingToolkit.get_defaults(sys)
prob = ODEProblem(sys, u0, (0.0,1.0), p)
sol = solve(prob, Tsit5())
plot(sol[o.x])
plot(sol[n.x])
sol
plot(sol(sol.t,Val{1})[1,:])
plot(sol(sol.t,Val{1})[2,:])
plot(sol[1,:])
plot(sol[2,:])

p = [0.1]
u0b = rand(1,10)
function prob_func(prob,i,repeat)
    u0 = @view u0b[:,i]
    remake(prob; u0)
end

function output_func(sol,i)
    sol[:, end], false
end

function loss(p,model)
    ensemble_prob = EnsembleProblem(prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
    sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=size(u0b,2),
              sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), save_everystep=false, save_start=false)
    sum(Array(sol))
end



cbg = function (p,l;doplot=false)
    display(l)
    return false
end

optfun = GalacticOptim.OptimizationFunction((θ,p) -> loss(θ,model), GalacticOptim.AutoZygote())
optfunc = GalacticOptim.instantiate_function(optfun, p, GalacticOptim.AutoZygote(), nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, p)
GalacticOptim.solve(optprob, ADAM(), cb = cbg, maxiters=100)









using DiffEqCallbacks, OrdinaryDiffEq, LinearAlgebra
prob = ODEProblem((du,u,p,t) -> du .= u, rand(4), (0.0,1.0))
saved_values = SavedValues(Float64, Vector{Float64})
cb = SavingCallback((u,t,integrator)->integrator(t,Val{1})[:,1], saved_values, saveat = 0.0:0.1:1.0)
sol = solve(prob, Tsit5(), saveat=0.1, callback=cb)
as = Array(sol)
plot(s)
sol[:]
saved_values.saveval
plot(saved_values.saveval)



using ModelingToolkit
using ModelingToolkit: get_defaults
using OrdinaryDiffEq

@variables t
D = Differential(t)

function SubSys(;name)
  @variables x(t)=0.0
  @parameters p=0.3
  ODESystem([x~p],t,[x],[p];name)
end
function Network(;name)
  @variables x(t)=0.0
  @named subsys = SubSys()
  ODESystem([D(x)~subsys.x],t,[x],[];name,systems=[subsys])
end

@named net = Network()
sys = structural_simplify(net)
prob = ODEProblem(sys,get_defaults(sys),(0.0,0.1))
sol = solve(prob,Tsit5())

@nonamespace x = net.x
@nonamespace subsys = net.subsys
sol[x]
sol[subsys.x]


sol[net.x]
sol[net.subsys.x]
