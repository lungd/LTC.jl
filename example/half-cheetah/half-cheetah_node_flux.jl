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

function train_cheetah_node(epochs, solver=nothing; sensealg=nothing,
  T=Float32, model_size=5, batchsize=1, seq_len=32, normalise=true,
  kwargs...)

  train_dl, _, _, _ = get_3d_dl(T; batchsize, seq_len, normalise)
  wiring = LTC.FWiring(17,model_size, T)
  plot_wiring(wiring)
  model = Flux.Chain(LTC.MTKNODEMapped(Flux.Chain, wiring, solver; sensealg, kwargs...),
                     Flux.Dense(rand(T, 17, wiring.n_out), false, identity),
  )
  cb = LTC.MyCallback(T; cb=mycb, ecb=LTC.DEFAULT_ECB, nepochs=epochs, nsamples=length(train_dl))
  # opt = Flux.Optimiser(ClipValue(0.80), ADAM(0.005))
  opt = LTC.ClampBoundOptim(LTC.get_bounds(model,T)..., ClipValue(T(0.8)), ADAM(T(0.02)))
  LTC.optimize(model, LTC.loss_seq_node, cb, opt, train_dl, epochs, T), model
end


@time train_cheetah_node(3 ; batchsize=1, model_size=10, abstol=1e-3, reltol=1e-3
)
train_cheetah_node(3, AutoTsit5(Rosenbrock23()); batchsize=2, model_size=8, abstol=1e-4, reltol=1e-4
)
