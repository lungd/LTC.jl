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

function train_cheetah(epochs; T=Float32, model_size=5, batchsize=15, seq_len=32)
  train_dl, _, _, _ = get_2d_dl(T; batchsize, seq_len)
  model = Flux.Chain(LTC.Mapper(T, 17),
                     Flux.LSTM(17, model_size),
                     LTC.Mapper(T, model_size),
                     LTC.Dense(model_size, 17),
  )
  cb = LTC.MyCallback(T; cb=mycb, ecb=LTC.DEFAULT_ECB, nepochs=epochs, nsamples=length(train_dl))
  opt = Flux.Optimiser(ClipValue(0.80), ADAM(0.02))
  LTC.optimize(model, LTC.loss_seq, cb, opt, train_dl, epochs, T), model
end


@time train_cheetah(1)
@time train_cheetah(100; model_size=30)
