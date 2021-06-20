using LTC
using Plots
gr()
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using BlackBoxOptim

include("half_cheetah_data_loader.jl")


function train_cheetah(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true));)

  function lg(p,x,y, m)
    LTC.reset_state!(m,p)
    # reset!(m)

    # ŷ = m.(x, [p])

    ŷb = GalacticOptim.Flux.Zygote.Buffer(rand(eltype(x[1]),1,1,1), size(y,1), size(y[1],1), size(y[1],2))
    # ŷb = rand(eltype(x[1]), size(x,1), size(x[1],1), size(x[1],2))
    for (i, xi) in enumerate(x)#size(ŷb,1)
      # xi = x[i]
      ŷi = m(xi, p)
      # @show size(ŷi)
      # @show size(ŷb[i,:,:])
      ŷb[i,:,:] = ŷi
      Inf32 ∈ ŷi && return Inf32, ŷi, y
    end
    # ŷ = copy(ŷb)
    # Inf32 ∈ ŷ && return Inf32, [ŷ[1,:,:]], y
    ŷ = Flux.unstack(copy(ŷb),1)
    mean(GalacticOptim.Flux.Losses.mse.(ŷ,y)), ŷ, y
  end

  cbg = function (p,l,pred,y;doplot=true)
    display(l)
    if doplot
      fig = plot([ŷ[1,1] for ŷ in pred], label="ŷ1")
      plot!(fig, [ŷ[2,1] for ŷ in pred], label="ŷ2")
      plot!(fig, [yi[1,1] for yi in y], label="y1")
      plot!(fig, [yi[2,1] for yi in y], label="y2")
      display(fig)
    end
    return false
  end

  batchsize=16

  train_dl, test_dl, valid_dl = get_dl(batchsize=batchsize, seq_len=32)

  wiring = NCPWiring(17,2,
    n_sensory=5, n_inter=3, n_command=4, n_motor=17,
    sensory_in=-1, rec_sensory=2, sensory_inter=2, sensory_command=0, sensory_motor=0,
    inter_in=0, rec_inter=2, inter_command=3, inter_motor=1,                       # inter_in = sensory_out
    command_in=0, rec_command=4, command_motor=4,                   # command_in = inter_out
    motor_in=0, rec_motor=3)

  model = DiffEqFlux.FastChain(#LTC.Mapper(17),
                               #DiffEqFlux.FastDense(17,17,σ),
                               LTC.LTCNet(wiring, solver, sensealg),
                               (x,p)->x[end-wiring.n_motor+1:end, :],
                               #LTC.Mapper(wiring.n_motor),
                               DiffEqFlux.FastDense(wiring.n_motor,17)
                               )

  opt = Flux.Optimiser(ClipValue(0.50), ADAM(0.006))
  # opt = Optim.LBFGS()
  # opt = BBO()
  # opt = ParticleSwarm(;lower=lb, upper=ub)
  # opt = Fminbox(GradientDescent())
  AD = GalacticOptim.AutoZygote()
  # AD = GalacticOptim.AutoModelingToolkit()
  LTC.optimize(model, lg, cbg, opt, AD, ncycle(train_dl,n)), model
end

@time res1,model = train_cheetah(100)
# @time res1,model = train_cheetah(500, AutoTsit5(Rosenbrock23()))
# @time res1,model = train_cheetah(5, VCABM(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(5, Tsit5(), InterpolatingAdjoint(checkpointing=true))
# @time res1,model = train_cheetah(5, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(5, Tsit5(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(1, VCABM(), BacksolveAdjoint())























using DifferentialEquations, Plots

function lotka_volterra!(du,u,p,t)
    rab, wol = u
    α,β,γ,δ=p
    du[1] = drab = α*rab - β*rab*wol
    du[2] = dwol = γ*rab*wol - δ*wol
    du[3] = u[3] * u[3]
    nothing
end

u0 = [1.0,1.0, 0.5]
# u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra!,u0,tspan,p)
sol = solve(prob,saveat=0.1)
plot(sol)

dataset = Array(sol)
scatter!(sol.t,dataset')

tmp_prob = remake(prob, p=[1.2,0.8,2.5,0.8])
tmp_sol = solve(tmp_prob)
plot(tmp_sol)
scatter!(sol.t,dataset')

function loss(p)
  tmp_prob = remake(prob, p=p)
  tmp_sol = solve(tmp_prob,Tsit5(),saveat=0.1)
  if tmp_sol.retcode == :Success
    return sum(abs2,Array(tmp_sol) - dataset)
  else
    return Inf
  end
end



function prob_func(prob,i,repeat)
  remake(prob,u0=rand()*prob.u0)
end
function reduction(u,data,I)
  u = append!(u,data)
  finished = false
  u, finished
end
ensemble_prob = EnsembleProblem(prob;prob_func)
sim = solve(ensemble_prob,Tsit5(),EnsembleThreads(),trajectories=100,save_start=false, save_everystep=false)



using DiffEqFlux, Optim

pinit = [1.2,0.8,2.5,0.8]
res = DiffEqFlux.sciml_train(loss,pinit,ADAM(), maxiters = 1000)

#try Newton method of optimization
res = DiffEqFlux.sciml_train(loss,pinit,Newton(), GalacticOptim.AutoForwardDiff())
