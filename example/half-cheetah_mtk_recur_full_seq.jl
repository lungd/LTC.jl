using LTC
using Plots
gr()
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using BlackBoxOptim
using ModelingToolkit

include("half_cheetah_data_loader.jl")


function train_cheetah(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true));)

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

  batchsize=10
  seq_len=20
  train_dl, test_dl, valid_dl = get_dl(batchsize=batchsize, seq_len=seq_len)

  wiring = LTC.NCPWiring(17,17;
    n_sensory=17, n_inter=5, n_command=4, n_motor=17,
    sensory_in=-1, rec_sensory=4, sensory_inter=2, sensory_command=1, sensory_motor=0,
    inter_in=2, rec_inter=2, inter_command=3, inter_motor=1,                       # inter_in = sensory_out
    command_in=0, rec_command=4, command_motor=4,                   # command_in = inter_out
    motor_in=0, rec_motor=2)

  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = DiffEqFlux.FastChain((x,p) -> Flux.stack(x, 3),
                               LTC.Mapper(wiring.n_in),
                               LTC.RecurMTK(LTC.MTKCell(wiring.n_in, wiring.n_out, net, sys, solver, sensealg)),
                               LTC.Mapper(wiring.n_out),
                               (x,p) -> Flux.unstack(x, 2),
                               )

  opt = Flux.Optimiser(ClipValue(1.00), ExpDecay(1, 0.1, 200, 1e-4), ADAM())
  # opt = Optim.LBFGS()
  # opt = BBO()
  # opt = ParticleSwarm(;lower=lb, upper=ub)
  # opt = Fminbox(GradientDescent())
  AD = GalacticOptim.AutoZygote()
  # AD = GalacticOptim.AutoModelingToolkit()
  LTC.optimize(model, LTC.loss_full_seq, cbg, opt, AD, ncycle(train_dl,n)), model
end

@time res1,model = train_cheetah(100)
# @time res1,model = train_cheetah(500, AutoTsit5(Rosenbrock23()))
# @time res1,model = train_cheetah(5, VCABM(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(5, Tsit5(), InterpolatingAdjoint(checkpointing=true))
# @time res1,model = train_cheetah(5, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(5, Tsit5(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(1, VCABM(), BacksolveAdjoint())
