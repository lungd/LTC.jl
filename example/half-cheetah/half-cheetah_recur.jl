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

function train_cheetah(epochs, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); T = Float32)

  cb = function (p,l,pred,y;doplot=true)
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

  batchsize=15
  seq_len=32
  train_dl, _, _, _ = get_dl(T, batchsize=batchsize, seq_len=seq_len)
  train_dl = ncycle(train_dl, epochs)

  wiring = LTC.FWiring(17,5)

  LTC.plot_wiring(wiring)

  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = FastChain(LTC.MTKRecurMapped(FastChain, wiring, net, sys, solver, sensealg),
                    FastDense(wiring.n_out, 17),
  )

  opt = Flux.Optimiser(ClipValue(1.00), ADAM(0.02))
  AD = GalacticOptim.AutoZygote()
  LTC.optimize(model, LTC.loss_seq, cb, opt, AD, train_dl), model
end

@time res1,model = train_cheetah(1)
@time res1,model = train_cheetah(100)
