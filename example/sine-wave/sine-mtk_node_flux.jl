using LTC
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using ModelingToolkit
import Flux: Data.DataLoader

# Not in Project.toml
import IterTools: ncycle
using Plots
gr()

include("sine_wave_dataloader.jl")

function plot_wiring(wiring::Wiring)
  display(heatmap(wiring.sens_mask))
  display(heatmap(wiring.sens_pol))
  display(heatmap(wiring.syn_mask))
  display(heatmap(wiring.syn_pol))
end

function train_sine_node_s(epochs, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); T=Float32)

  cb = function (p,l,ŷ,y;doplot=true)
    display(l)
    if doplot
      fig = plot(y[1,:,1], label="y")
      plot!(fig, ŷ[1,:,1], label="ŷ")
      display(fig)
    end
    return false
  end

  batchsize = 1
  train_dl = generate_3d_data(T)

  wiring = LTC.FWiring(2,8,T)
  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = Flux.Chain(LTC.MTKNODEMapped(Flux.Chain, wiring, net, sys, solver, sensealg),
                     Flux.Dense(ones(T, 1, wiring.n_out), true, identity),
                    )

  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.8), ADAM(0.02))
  AD = GalacticOptim.AutoZygote()
  LTC.optimize(model, LTC.loss_seq_node, cb, opt, AD, ncycle(train_dl,epochs); T), model
end

@time train_sine_node_s(1)
@time train_sine_node_s(100)
