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
    reset_state!(m,p)
    # reset!(m)
    ŷ = m.(x, [p])
    # mean([GalacticOptim.Flux.Losses.mse(ŷ[i],y[i]) for i in 1:length(y)]), ŷ, y
    mean(GalacticOptim.Flux.Losses.mse.(ŷ,y)), ŷ, y
  end

  cbg = function (p,l,pred,y;doplot=true)
    display(l)
    if doplot
      fig = plot([ŷ[1,1] for ŷ in pred], label="ŷ1")
      plot!(fig, [ŷ[2,1] for ŷ in pred], label="ŷ2")
      plot!(fig, [yi[1,1] for yi in y], label="y2")
      plot!(fig, [yi[2,1] for yi in y], label="y2")
      display(fig)
    end
    return false
  end

  batchsize=32

  train_dl, test_dl, valid_dl = get_dl(batchsize=batchsize, seq_len=32)

  wiring = NCPWiring(17,2,
    n_sensory=17, n_inter=8, n_command=6, n_motor=2,
    sensory_in=4, rec_sensory=2, sensory_inter=2, sensory_command=0, sensory_motor=0,
    inter_in=0, rec_inter=2, inter_command=3, inter_motor=1,                       # inter_in = sensory_out
    command_in=0, rec_command=4, command_motor=2,                   # command_in = inter_out
    motor_in=0, rec_motor=3)

  model = DiffEqFlux.FastChain(#LTC.Mapper(17),
                               LTC.LTCNet(wiring, solver, sensealg),
                               (x,p)->x[end-wiring.n_motor+1:end, :],
                               #LTC.Mapper(wiring.n_motor),
                               DiffEqFlux.FastDense(wiring.n_motor,17)
                               )

  opt = Flux.Optimiser(ClipValue(0.70), ADAM(0.008))
  # opt = Optim.LBFGS()
  # opt = BBO()
  # opt = ParticleSwarm(;lower=lb, upper=ub)
  # opt = Fminbox(GradientDescent())
  AD = GalacticOptim.AutoZygote()
  # AD = GalacticOptim.AutoModelingToolkit()
  LTC.optimize(model, lg, cbg, opt, AD, ncycle(train_dl,n)), model
end

@time res1,model = train_cheetah(10)
# @time res1,model = train_cheetah(500, AutoTsit5(Rosenbrock23()))
# @time res1,model = train_cheetah(5, VCABM(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(5, Tsit5(), InterpolatingAdjoint(checkpointing=true))
# @time res1,model = train_cheetah(5, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(5, Tsit5(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(1, VCABM(), BacksolveAdjoint())
