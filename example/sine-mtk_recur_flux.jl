using LTC
using Plots
gr()
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using Juno
using Cthulhu
using Profile
using BlackBoxOptim
#using PProf
using ProfileView

function generate_data()
    in_features = 2
    out_features = 1
    N = 48
    data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
    data_x = [reshape([Float32(data_x[1][i]),Float32(data_x[2][i])],2,1) for i in 1:N]# |> f32
    data_y = [reshape([Float32(y)],1,1) for y in sin.(range(0,stop=6π,length=N))]# |> f32
    DataLoader((data_x, data_y), batchsize=N)
end

function data(iter; data_x=nothing, data_y=nothing, short=false, noisy=false)
    ncycle(generate_data(), iter)
end

function train_sine(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

  function loss_flux_chain(p, re, x, y)
    m = re(p)
    LTC.reset_state!(m, p) # use initial conditions from current params

    ŷb = Flux.Zygote.Buffer([y[1]], size(y,1))
    for i in 1:size(x,1)
      xi = x[i]
      ŷi = m(xi)
      Inf32 ∈ ŷi && return Inf32, copy(ŷb), y
      ŷb[i] = ŷi
    end
    ŷ = copy(ŷb)

    # mean(sum.(abs2, (ŷ .- y))), ŷ, y
    return mean(Flux.Losses.mse.(ŷ,y, agg=mean)), ŷ, y
  end

  cbg = function (p,l,pred,y;doplot=false)
    display(l)
    if doplot
      fig = plot([ŷ[end,1] for ŷ in pred], label="ŷ")
      plot!(fig, [yi[end,1] for yi in y], label="y")
      #frame(anim)
      display(fig)
    end
    return false
  end

  batchsize = 1

  wiring = LTC.FWiring(2,1)
  net = LTC.Net(wiring; name=:net)

  model = DiffEqFlux.Chain(Flux.Dense(wiring.n_in,wiring.n_in,tanh), LTC.Mapper(wiring.n_in),
                               LTC.RecurMTK(LTC.MTKCell(wiring.n_in, wiring.n_out, net, solver, sensealg)),
                               LTC.Mapper(wiring.n_out),
                               )

  train_dl = data(n)
  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.01))
  # opt = Optim.LBFGS()
  # opt = BBO()
  # opt = ParticleSwarm(;lower=lb, upper=ub)
  # opt = Fminbox(GradientDescent())
  AD = GalacticOptim.AutoZygote()
  # AD = GalacticOptim.AutoModelingToolkit()

  LTC.optimize(model, loss_flux_chain, cbg, opt, AD, train_dl)

end

@time train_sine(100)
# @time traintest(1000, QNDF())
# @time traintest(1000, TRBDF2())
# @time traintest(1000, AutoTsit5(Rosenbrock23()))
