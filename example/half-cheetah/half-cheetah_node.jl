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


function train_cheetah_node(epochs, solver=VCABM();
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  T=Float32, model_size=5, batchsize=1, seq_len=32, normalise=true,
  kwargs...)

  train_dl, _, _, _ = get_3d_dl(T; batchsize, seq_len, normalise)

  wiring = LTC.FWiring(17,model_size,T)
  plot_wiring(wiring)
  model = FastChain(LTC.MTKNODEMapped(FastChain, wiring, solver; sensealg, kwargs...),
                    LTC.FluxLayerWrapper(Flux.Dense(randn(T, 17, wiring.n_out), false, identity)),
  )
  cb = LTC.MyCallback(T; cb=mycb, ecb=LTC.DEFAULT_ECB, nepochs=epochs, nsamples=length(train_dl))
  # opt = Flux.Optimiser(ClipValue(0.80), ADAM(0.02))
  opt = LTC.ClampBoundOptim(LTC.get_bounds(model,T)..., ClipValue(T(1.0)), ADAM(T(0.02)))
  LTC.optimize(model, LTC.loss_seq_node, cb, opt, train_dl, epochs, T), model
end

train_cheetah_node(1)
train_cheetah_node(100)
train_cheetah_node(100, AutoTsit5(Rosenbrock23(autodiff=false)); model_size=8, abstol=1e-5, reltol=1e-5)
train_cheetah_node(100, AutoVern7(Rodas5(autodiff=false)); model_size=8, abstol=1e-5, reltol=1e-5)
