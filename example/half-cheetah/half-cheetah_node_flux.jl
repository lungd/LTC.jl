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

function train_cheetah_node(epochs, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); T=Float32)

  cb = function (p,l,ŷ,y;doplot=true)
    display(l)
    if doplot
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

  train_dl = ncycle(train_dl,epochs)

  fx, fy = first(train_dl)
  @show size(fx)
  @show size(fx[1])
  @show size(fx[1][1])

  wiring = LTC.FWiring(17,5)

  plot_wiring(wiring)

  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = Flux.Chain(x -> Flux.stack(x,2),
                     LTC.MTKNODEMapped(Chain, wiring, net, sys, solver, sensealg),
                     Flux.Dense(randn(T, 17, wiring.n_out), true),
                     x -> Flux.unstack(x,2),
  )

  opt = Flux.Optimiser(ClipValue(0.80), ADAM(0.02))
  AD = GalacticOptim.AutoZygote()
  LTC.optimize(model, LTC.loss_seq_node2, cb, opt, AD, train_dl; T), model
end


train_cheetah_node(1)
train_cheetah_node(100)
