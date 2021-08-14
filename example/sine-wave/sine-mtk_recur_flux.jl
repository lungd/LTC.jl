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

include("sine_wave_dataloader.jl")

function plot_wiring(wiring::Wiring)
  display(heatmap(wiring.sens_mask))
  display(heatmap(wiring.sens_pol))
  display(heatmap(wiring.syn_mask))
  display(heatmap(wiring.syn_pol))
end

function train_sine(epochs, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true));T=Float32)

  cb = function (p,l,ŷ,y;doplot=true)
    display(l)
    if doplot
      fig = plot(Flux.stack(y,2)[1,:,1], label="y")
      plot!(fig, Flux.stack(ŷ,2)[1,:,1], label="ŷ")
      display(fig)
    end
    return false
  end



  batchsize = 1
  wiring = LTC.FWiring(2,8)

  plot_wiring(wiring)
  train_dl = generate_2d_arr_data(T)

  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = Flux.Chain(LTC.MTKRecurMapped(Flux.Chain, wiring, net, sys, solver, sensealg),
                     Flux.Dense(wiring.n_out,1),
  )


  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.01))
  AD = GalacticOptim.AutoZygote()
  LTC.optimize(model, LTC.loss_seq, cb, opt, AD, ncycle(train_dl, epochs)), model
end

@time train_sine(1)
@time train_sine(100)
# @time traintest(1000, QNDF())
# @time traintest(1000, TRBDF2())
# @time traintest(1000, AutoTsit5(Rosenbrock23()))
