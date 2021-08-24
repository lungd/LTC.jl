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

include("sine_wave_dataloader.jl")
include("../example_utils.jl")

function train_sine(epochs, solver=VCABM();
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  T=Float32, model_size=5,
  kwargs...)

  train_dl = generate_2d_data(T)
  wiring = LTC.FWiring(2,model_size, T)
  model = Flux.Chain(LTC.MTKRecurMapped(Flux.Chain, wiring, solver; sensealg, kwargs...),
                     Flux.Dense(ones(T, 1, wiring.n_out), false, identity),
  )
  cb = LTC.MyCallback(T; cb=mycb, ecb=(_)->nothing, nepochs=epochs, nsamples=length(train_dl))
  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.02))
  plot_wiring(wiring)
  LTC.optimize(model, LTC.loss_seq, cb, opt, train_dl, epochs, T), model
end

@time train_sine(1)
@time train_sine(100)
@time train_sine(100; abstol=1e-3, reltol=1e-3)
@time train_sine(100, AutoTsit5(Rosenbrock23()); model_size=5, abstol=1e-4, reltol=1e-4)
