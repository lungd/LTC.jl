using LTC
using Plots
gr()
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using ModelingToolkit
import Flux: Data.DataLoader
import IterTools: ncycle

function generate_data()
    in_features = 2
    out_features = 1
    N = 48
    data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
    data_x = [reshape([Float32(data_x[1][i]),Float32(data_x[2][i])],2,1) for i in 1:N]# |> f32
    data_y = [reshape([Float32(y)],1,1) for y in sin.(range(0,stop=6π,length=N))]# |> f32
    dl = DataLoader((data_x, data_y), batchsize=N)
    @show length(dl)
    fx, fy = first(dl)
    @show size(fx)
    @show size(fx[1])
    @show size(fy)
    @show size(fy[1])
    fig = plot([x[1,1] for x in fx], label="x1")
    plot!(fig, [x[2,1] for x in fx], label="x2")
    plot!(fig, [y[1,1] for y in fy], label="y1")
    display(fig)
    dl
end

function data(iter; data_x=nothing, data_y=nothing, short=false, noisy=false)
    ncycle(generate_data(), iter)
end
# function train_sine(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP());)
function train_sine(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

  cbg = function (p,l,pred,y;doplot=true)
    display(l)
    if doplot
      fig = plot([ŷ[end,1] for ŷ in pred], label="ŷ")
      plot!(fig, [yi[end,1] for yi in y], label="y")
      #frame(anim)
      display(fig)
    end
    return false
  end

  #train_dl = generate_data()
  batchsize = 1

  wiring = LTC.FWiring(2,1)
  net = LTC.Net(wiring, name=:net)
  sys = ModelingToolkit.structural_simplify(net)

  model = DiffEqFlux.FastChain(LTC.Mapper(wiring.n_in),
                               LTC.RecurMTK(LTC.MTKCell(wiring.n_in, wiring.n_out, net, sys, solver, sensealg)),
                               LTC.Mapper(wiring.n_out),
                               )

  train_dl = data(n)
  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.04))
  # opt = Optim.LBFGS()
  # opt = BBO()
  # opt = ParticleSwarm(;lower=lb, upper=ub)
  # opt = Fminbox(GradientDescent())
  AD = GalacticOptim.AutoZygote()
  # AD = GalacticOptim.AutoModelingToolkit()

  # return model

  LTC.optimize(model, LTC.loss_seq, cbg, opt, AD, train_dl, normalize=false)

end

# 36.107867 seconds (111.82 M allocations: 6.876 GiB, 3.87% gc time)

@time model = train_sine(100)
@time model = train_sine(100, VCABM(), InterpolatingAdjoint(checkpointing=true, autodiff=false, autojacvec=false))
@time model = train_sine(100, VCABM(), InterpolatingAdjoint(checkpointing=true, autodiff=false, autojacvec=ReverseDiffVJP(true))) #  38.617758 seconds (113.64 M allocations: 7.143 GiB, 3.63% gc time)
@time model = train_sine(100, VCABM(), InterpolatingAdjoint(checkpointing=true, autodiff=true, autojacvec=ReverseDiffVJP(true))) #  35.214501 seconds (113.69 M allocations: 7.149 GiB, 3.92% gc time)
@time model = train_sine(100, VCABM(), InterpolatingAdjoint(checkpointing=false, autodiff=true, autojacvec=ReverseDiffVJP(true)))
# @time model = train_sine(100, Tsit5(), InterpolatingAdjoint(autojacvec=DiffEqSensitivity.EnzymeVJP()))
# @time traintest(1000, QNDF())
# @time traintest(1000, TRBDF2())
# @time traintest(1000, AutoTsit5(Rosenbrock23()))
