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

function train_sine_node_s(epochs, solver=VCABM();
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  T=Float32, model_size=5,
  kwargs...)

  train_dl = generate_3d_data(T)
  wiring = LTC.FWiring(2,model_size,T)
  model = FastChain(LTC.MTKNODEMapped(FastChain, wiring, solver; sensealg, kwargs...),
                    LTC.FluxLayerWrapper(Flux.Dense(ones(T, 1, wiring.n_out), false, identity), T),
                    # FastDense(ones(T, 1, wiring.n_out), true, identity), # FastDense does not work for 3d input
  )
  cb = LTC.MyCallback(T; cb=mycb, ecb=(_)->nothing, nepochs=epochs, nsamples=length(train_dl))
  opt = LTC.ClampBoundOptim(LTC.get_bounds(model,T)..., ClipValue(T(0.8)), ADAM(T(0.03)))
  LTC.optimize(model, LTC.loss_seq_node, cb, opt, train_dl, epochs, T), model
end

@time train_sine_node_s(1)
@time train_sine_node_s(100)
