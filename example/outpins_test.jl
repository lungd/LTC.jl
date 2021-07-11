@variables t
D = Differential(t)

function OutPin(;name)
  @variables x(t)=0f0
  ODESystem(Equation[],t,[x],[]; name)
end

function Network(;name)
    @variables x(t)=0.0 a(t)
    @parameters p=0.1
    eqs = [D(x) ~ a, a ~ 0.1p]
    ODESystem(eqs;name)
end

function Model(;name)
    @named op = OutPin()
    @named net = Network()
    eqs = [op.x ~ net.x]
    systems = [op,net]
    ODESystem(eqs;name,systems)
end

@named model = Model()
sys = structural_simplify(model)

defs = ModelingToolkit.get_defaults(sys)
prob = ODEProblem(sys, defs, (0.0,1.0))

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
