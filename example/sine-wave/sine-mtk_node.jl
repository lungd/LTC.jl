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

function train_sine_node_s(epochs, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); T=Float32)

  cb = function (p,l,pred,y;doplot=true)
    display(l)
    if doplot
      fig = plot([ŷ[end,1] for ŷ in Flux.unstack(pred,2)], label="ŷ")
      plot!(fig, [yi[end,1] for yi in Flux.unstack(y,2)], label="y")
      display(fig)
    end
    return false
  end

  batchsize = 1
  train_dl = generate_3d_data(T)

  wiring = LTC.FWiring(2,8,T)
  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = FastChain(LTC.MTKNODEMapped(FastChain, wiring, net, sys, solver, sensealg),
                    LTC.FluxLayerWrapper(Flux.Dense(ones(T, 1, wiring.n_out), true, identity)),
                    # FastDense(ones(T, 1, wiring.n_out), true, identity), # does not work for 3d input

  )

  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.8), ADAM(0.02))
  AD = GalacticOptim.AutoZygote()
  sol = LTC.optimize(model, LTC.loss_seq_node, cb, opt, AD, ncycle(train_dl,epochs); T)
end

# 36.107867 seconds (111.82 M allocations: 6.876 GiB, 3.87% gc time)

@time model = train_sine_node_s(1)
@time model = train_sine_node_s(100)

# @time model = train_sine_node_s(10000, VCABM(), InterpolatingAdjoint(autojacvec=ZygoteVJP()))
# @time model = train_sine_node_s(10000, VCABM(), InterpolatingAdjoint(autojacvec=EnzymeVJP()); T=Float64)
# @time model = train_sine_node_s(10000, Rosenbrock23(autodiff=false))
