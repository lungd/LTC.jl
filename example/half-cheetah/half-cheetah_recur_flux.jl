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

function train_cheetah(epochs, solver=VCABM();
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  T=Float32, model_size=5, batchsize=1, seq_len=32, normalise=true,
  kwargs...)

  train_dl, _, _, _ = get_2d_dl(T; batchsize, seq_len, normalise)
  wiring = LTC.FWiring(17,model_size)
  plot_wiring(wiring)
  model = Flux.Chain(LTC.MTKRecurMapped(Flux.Chain, wiring, solver; sensealg, kwargs...),
                    Flux.Dense(randn(T, 17, wiring.n_out), false, identity),
  )
  cb = LTC.MyCallback(T; cb=mycb, ecb=LTC.DEFAULT_ECB, nepochs=epochs, nsamples=length(train_dl))
  opt = Flux.Optimiser(ClipValue(1.00), ADAM(0.01))
  LTC.optimize(model, LTC.loss_seq, cb, opt, train_dl, epochs, T), model
end


@time train_cheetah(1)
@time train_cheetah(2, AutoTsit5(Rosenbrock23(autodiff=false)); model_size=8, batchsize=1, abstol=1e-4, reltol=1e-4
)
