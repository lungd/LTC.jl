using LTC
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using ModelingToolkit

# Not in Project.toml
using Plots
gr()

include("half_cheetah_data_loader.jl")
include("../example_utils.jl")

function train_cheetah_node(epochs, solver=VCABM(); T::DataType=Float32, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), kwargs...)
  batchsize=15
  seq_len=32
  train_dl, _, _, _ = get_dl(T, batchsize=batchsize, seq_len=seq_len)

  wiring = LTC.FWiring(17,8, T)
  plot_wiring(wiring)
  model = Flux.Chain(x -> Flux.stack(x,2),
                     LTC.MTKNODEMapped(Flux.Chain, wiring, solver; sensealg, kwargs...),
                     Flux.Dense(rand(T, 17, wiring.n_out), zeros(T,17), identity),
                     x -> Flux.unstack(x,2),
  )
  cb = LTC.MyCallback(T; cb=mycb, ecb=LTC.DEFAULT_ECB, nepochs=epochs, nsamples=length(train_dl))
  opt = Flux.Optimiser(ClipValue(0.80), ADAM(0.02))
  LTC.optimize(model, LTC.loss_seq_node, cb, opt, train_dl, epochs, T), model
end


train_cheetah_node(1)
train_cheetah_node(100)
train_cheetah_node(100, Tsit5(); abstol=1e-3, reltol=1e-3)
train_cheetah_node(100, Tsit5(); abstol=1e-3, reltol=1e-3, dtmin=1e-8)
