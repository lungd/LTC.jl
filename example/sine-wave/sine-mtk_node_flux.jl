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

function train_sine_node(epochs, solver=VCABM(); T::DataType=Float32, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), kwargs...)
  train_dl = generate_3d_data(T)
  wiring = LTC.FWiring(2,8,T)
  model = Flux.Chain(LTC.MTKNODEMapped(Flux.Chain, wiring, solver; sensealg, kwargs...),
                     Flux.Dense(ones(T, 1, wiring.n_out), zeros(T,1), identity),
  )
  cb = LTC.MyCallback(T; cb=mycb, ecb=(_)->nothing, nepochs=epochs, nsamples=length(train_dl))
  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.8), ADAM(0.02))
  LTC.optimize(model, LTC.loss_seq_node, cb, opt, train_dl, epochs, T), model
end

@time train_sine_node(1)
@time train_sine_node(100)
