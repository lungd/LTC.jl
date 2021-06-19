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
    data_y = [reshape([Float32(y)],1) for y in sin.(range(0,stop=6π,length=N))]# |> f32
    DataLoader((data_x, data_y), batchsize=N)
end

function data(iter; data_x=nothing, data_y=nothing, short=false, noisy=false)
    ncycle(generate_data(), iter)
end

function traintest(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

  #anim = Animation()

  function lg(p,x,y,model)
    reset_state!(model,p)
    # reset!(model)
    ŷ = model.(x,[p])
    # sum(sum([(ŷ[i][end,:] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y), ŷ, y
    mean(GalacticOptim.Flux.Losses.mse.(ŷ,y)), ŷ, y
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
  model = LTC.LTCNet(Wiring(2,1), solver, sensealg)

  train_dl = data(n)
  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.008))
  # opt = Optim.LBFGS()
  # opt = BBO()
  # opt = ParticleSwarm(;lower=lb, upper=ub)
  # opt = Fminbox(GradientDescent())
  AD = GalacticOptim.AutoZygote()
  # AD = GalacticOptim.AutoModelingToolkit()

  LTC.optimize(model, lg, cbg, opt, AD, train_dl)

end

@time traintest(2000)
# @time traintest(1000, QNDF())
# @time traintest(1000, TRBDF2())
# @time traintest(1000, AutoTsit5(Rosenbrock23()))
