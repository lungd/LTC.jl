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

  #anim = Animation()

  function loss(p,x,y, m)
    # ŷ = m.(x, [p])

    ŷb = GalacticOptim.Flux.Zygote.Buffer(rand(eltype(y[1]),1,1,1), size(y,1), size(y[1])...)
    for (i, xi) in enumerate(x)
      ŷi = m(xi, p)
      ŷb[i,:,:] = ŷi
      Inf32 ∈ ŷi && return Inf32, Flux.unstack(copy(ŷb),1), y # TODO: what if a layer after MTKRecur can handle Infs?
    end
    ŷ = Flux.unstack(copy(ŷb),1)

    LTC.reset_state!(m, p)
    mean(GalacticOptim.Flux.Losses.mse.(ŷ, y)), ŷ, y
  end

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
  net = LTC.Net(wiring; name=:net)

  model = DiffEqFlux.FastChain(LTC.Mapper(wiring.n_in),
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

  LTC.optimize(model, loss, cbg, opt, AD, train_dl)

end

@time train_sine(100)
# @time traintest(1000, QNDF())
# @time traintest(1000, TRBDF2())
# @time traintest(1000, AutoTsit5(Rosenbrock23()))
