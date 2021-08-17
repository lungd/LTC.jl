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

function train_cheetah(epochs, solver=VCABM(); T::DataType=Float32, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), kwargs...)
  batchsize=15
  seq_len=32
  train_dl, _, _, _ = get_dl(T, batchsize=batchsize, seq_len=seq_len)

  wiring = LTC.FWiring(17,5)
  plot_wiring(wiring)
  model = Flux.Chain(LTC.MTKRecurMapped(Chain, wiring, solver; sensealg, kwargs...),
                    Flux.Dense(randn(T, 17, wiring.n_out), zeros(T,17), identity),
  )
  cb = LTC.MyCallback(T; cb=mycb, ecb=LTC.DEFAULT_ECB, nepochs=epochs, nsamples=length(train_dl))
  opt = Flux.Optimiser(ClipValue(1.00), ADAM(0.01))
  LTC.optimize(model, LTC.loss_seq, cb, opt, train_dl, epochs, T), model
end


@time train_cheetah(1)
@time train_cheetah(100)
