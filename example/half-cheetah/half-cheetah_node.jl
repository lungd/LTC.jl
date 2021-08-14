using LTC
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using ModelingToolkit

# Not in Project.toml
import IterTools: ncycle
using Plots
gr()

include("half_cheetah_data_loader.jl")

function train_cheetah_node(epochs, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); T=Float32)

  cb = function (p,l,pred,y;doplot=true)
    display(l)
    if doplot
      fig = plot([ŷ[1,1] for ŷ in Flux.unstack(pred,2)], label="ŷ1")
      plot!(fig, [ŷ[2,1] for ŷ in Flux.unstack(pred,2)], label="ŷ2")
      plot!(fig, [yi[1,1] for yi in Flux.unstack(y,2)], label="y1")
      plot!(fig, [yi[2,1] for yi in Flux.unstack(y,2)], label="y2")
      display(fig)
    end
    return false
  end

  ecb = function (losses,epoch,res;doplot=true)
    # display(l)
    println("Epoch $(string(epoch, pad = length(digits(epochs))))/$(epochs), train_loss:$(losses[end])")
    if doplot
      x,y = first(train_dl)
      LTC.reset_state!(model, res.u)
      pred = model(x,res.u)
      fig = plot([ŷ[1,1] for ŷ in pred], label="ŷ1")
      plot!(fig, [ŷ[2,1] for ŷ in pred], label="ŷ2")
      plot!(fig, [yi[1,1] for yi in y], label="y1")
      plot!(fig, [yi[2,1] for yi in y], label="y2")
      display(fig)
    end
    return false
  end

  batchsize=15
  seq_len=32
  train_dl, _, _, _ = get_dl(T, batchsize=batchsize, seq_len=seq_len)

  train_dl = ncycle(train_dl, epochs)

  fx, fy = first(train_dl)
  @show size(fx)
  @show size(fx[1])
  @show size(fx[1][1])

  # wiring = LTC.DiagSensNCPWiring(17, 17, T;
  #   n_sensory=17, n_inter=6, n_command=6, n_motor=17, # total = 17
  #   rec_sensory=0, sensory_inter=4, sensory_command=0, sensory_motor=0,
  #   rec_inter=0, inter_command=3, inter_motor=0,                       # inter_in = sensory_out
  #   rec_command=4, command_motor=4,                   # command_in = inter_out
  #   rec_motor=0, orig=true)

  wiring = LTC.FWiring(17,5)

  LTC.plot_wiring(wiring)

  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = FastChain((x,p) -> Flux.stack(x,2),
                    LTC.MTKNODEMapped(FastChain, wiring, net, sys, solver, sensealg),
                    LTC.FluxLayerWrapper(Flux.Dense(randn(T, 17, wiring.n_out), true)),
                    (x,p) -> Flux.unstack(x,2),
  )

  opt = Flux.Optimiser(ClipValue(0.80), ADAM(0.02))
  AD = GalacticOptim.AutoZygote()
  LTC.optimize(model, LTC.loss_seq_node2, cb, opt, AD, train_dl; T), model
end

train_cheetah_node(1)
train_cheetah_node(100)
