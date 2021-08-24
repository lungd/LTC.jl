using LTC
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using ModelingToolkit
import Flux: Data.DataLoader

# Not in Project.toml
using Plots
gr()

include("sine_wave_dataloader.jl")
include("../example_utils.jl")

function train_sine_lstm(epochs;
  T=Float32, model_size=8, kwargs...)

  train_dl = generate_2d_data(T)
  model = Flux.Chain(LTC.Mapper(T, 2),
                     Flux.LSTM(2, model_size),
                     LTC.Mapper(T, model_size),
                     Flux.Dense(model_size, 1),
  )
  cb = LTC.MyCallback(T; cb=mycb, ecb=(_)->nothing, nepochs=epochs, nsamples=length(train_dl))
  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.8), ADAM(0.02))
  LTC.optimize(model, LTC.loss_seq, cb, opt, train_dl, epochs, T), model
end

@time train_sine_lstm(1; model_size=5)
@time train_sine_lstm(100; model_size=5)
