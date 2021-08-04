using LTC
using Plots
gr()
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
# using BlackBoxOptim
using ModelingToolkit
using IterTools: ncycle

include("half_cheetah_data_loader.jl")

# function train_cheetah(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP());)
function train_cheetah_node(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true));)

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

  T = Float32
  batchsize=8
  seq_len=20
  train_dl, _, _, _ = get_dl(T, batchsize=batchsize, seq_len=seq_len)

  fx, fy = first(train_dl)
  @show size(fx)
  @show size(fx[1])
  @show size(fx[1][1])

  wiring = LTC.DiagSensNCPWiring(17, 17;
    n_sensory=17, n_inter=4, n_command=4, n_motor=17, # total = 17
    rec_sensory=0, sensory_inter=2, sensory_command=0, sensory_motor=0,
    rec_inter=0, inter_command=2, inter_motor=0,                       # inter_in = sensory_out
    rec_command=4, command_motor=4,                   # command_in = inter_out
    rec_motor=0, orig=true)

  LTC.plot(wiring)

  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = DiffEqFlux.FastChain((x,p) -> Flux.stack(x,2),
                               LTC.Mapper(wiring.n_in),
                               LTC.MTKNODE(LTC.MTKNODECell(wiring.n_in, wiring.n_out, net, sys, solver, sensealg)),
                               LTC.Mapper(wiring.n_out),
                               (x,p) -> Flux.unstack(x,2),
                               )

  opt = Flux.Optimiser(ClipValue(1.00), ADAM(0.02))
  # opt = Optim.LBFGS()
  # opt = BBO()
  # opt = ParticleSwarm(;lower=lb, upper=ub)
  # opt = Fminbox(GradientDescent())
  AD = GalacticOptim.AutoZygote()
  # AD = GalacticOptim.AutoModelingToolkit()
  LTC.optimize(model, LTC.loss_seq_node, cbg, opt, AD, ncycle(train_dl,n), normalize=false), model
end

# @time res1,model = train_cheetah(100)

train_cheetah_node(500, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(autodiff=true,
  autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(5, VCABM(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(5, Tsit5(), InterpolatingAdjoint(checkpointing=true))
# @time res1,model = train_cheetah(5, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(5, Tsit5(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = train_cheetah(1, VCABM(), BacksolveAdjoint())


using JETTest
using Cthulhu
@descend_code_warntype DiffEqFlux.FastChain((x,p) -> Flux.stack(x,2), LTC.Mapper(3))
