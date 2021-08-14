using LTC
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using ModelingToolkit

# Not in Project.toml
import IterTools: ncycle
using Plots
gr()

include("sine_wave_dataloader.jl")

# function train_sine(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP());)
function train_sine(epochs, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); T=Float32)

  cb = function (p,l,pred,y;doplot=true)
    display(l)
    if doplot
      fig = plot([ŷ[end,1] for ŷ in pred], label="ŷ")
      plot!(fig, [yi[end,1] for yi in y], label="y")
      display(fig)
    end
    return false
  end

  batchsize = 1

  wiring = LTC.FWiring(2,8)
  LTC.plot_wiring(wiring)

  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = FastChain(LTC.MTKRecurMapped(FastChain, wiring, net, sys, solver, sensealg),
                    FastDense(wiring.n_out,1),
  )

  train_dl = generate_2d_arr_data(T)
  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.8), ADAM(0.02))
  # opt = Optim.LBFGS()
  # opt = BBO()
  # opt = ParticleSwarm(;lower=lb, upper=ub)
  # opt = Fminbox(GradientDescent())
  AD = GalacticOptim.AutoZygote()
  # AD = GalacticOptim.AutoModelingToolkit()

  # return model

  sol = LTC.optimize(model, LTC.loss_seq, cb, opt, AD, ncycle(train_dl,epochs); T)

end

# 36.107867 seconds (111.82 M allocations: 6.876 GiB, 3.87% gc time)

@time model = train_sine(1)
@time model = train_sine(100)

# @time model = train_sine(100, VCABM(), InterpolatingAdjoint(checkpointing=true, autodiff=false, autojacvec=false))
# @time model = train_sine(100, VCABM(), InterpolatingAdjoint(checkpointing=true, autodiff=false, autojacvec=ReverseDiffVJP(true))) #  38.617758 seconds (113.64 M allocations: 7.143 GiB, 3.63% gc time)
# @time model = train_sine(100, VCABM(), InterpolatingAdjoint(checkpointing=true, autodiff=true, autojacvec=ReverseDiffVJP(true))) #  35.214501 seconds (113.69 M allocations: 7.149 GiB, 3.92% gc time)
# @time model = train_sine(100, VCABM(), InterpolatingAdjoint(checkpointing=false, autodiff=true, autojacvec=ReverseDiffVJP(true)))
# @time model = train_sine(100, Tsit5(), InterpolatingAdjoint(autojacvec=DiffEqSensitivity.EnzymeVJP()))
# @time traintest(1000, QNDF())
# @time traintest(1000, TRBDF2())
# @time traintest(1000, AutoTsit5(Rosenbrock23()))
