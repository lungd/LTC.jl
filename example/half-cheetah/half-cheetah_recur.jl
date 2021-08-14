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

function plot_wiring(wiring::Wiring)
  display(heatmap(wiring.sens_mask))
  display(heatmap(wiring.sens_pol))
  display(heatmap(wiring.syn_mask))
  display(heatmap(wiring.syn_pol))
end

function train_cheetah(epochs, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); T = Float32)

  cb = function (p,l,ŷ,y;doplot=true)
    display(l)
    if doplot
      y = Flux.stack(y,2)
      ŷ = Flux.stack(ŷ,2)
      fig = plot(y[1,:,1], label="y1")
      plot!(fig, y[2,:,1], label="y2")
      plot!(fig, ŷ[1,:,1], label="ŷ1")
      plot!(fig, ŷ[2,:,1], label="ŷ2")
      display(fig)
    end
    return false
  end

  batchsize=15
  seq_len=32
  train_dl, _, _, _ = get_dl(T, batchsize=batchsize, seq_len=seq_len)
  train_dl = ncycle(train_dl, epochs)

  wiring = LTC.FWiring(17,5)

  plot_wiring(wiring)

  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = FastChain(LTC.MTKRecurMapped(FastChain, wiring, net, sys, solver, sensealg),
                    FastDense(wiring.n_out, 17),
  )

  opt = Flux.Optimiser(ClipValue(1.00), ADAM(0.02))
  AD = GalacticOptim.AutoZygote()
  LTC.optimize(model, LTC.loss_seq, cb, opt, AD, train_dl), model
end

@time train_cheetah(1)
@time train_cheetah(100)
