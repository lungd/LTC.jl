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

  function loss(p,x,y, m)

    # ŷ = m.(x, [p])

    LTC.reset_state!(m, p)

    ŷb = Flux.Zygote.Buffer(y[1], size(y)[1], size(y[1])[1], size(y[1])[2])
    for (i, xi) in enumerate(x)
      ŷi = m(xi, p)#::Matrix{T}
      ŷb[i,:,:] = ŷi
      Inf32 ∈ ŷi && return Inf32, Flux.unstack(copy(ŷb),1), y # TODO: what if a layer after MTKRecur can handle Infs?
    end

    # ŷ = copy(ŷb)
    # ŷ = [ŷ[i,:,:] for i in 1:size(ŷ)[1]]
    ŷ = Flux.unstack(copy(ŷb),1)


    # GalacticOptim.Flux.Losses.mse(ŷ, y, agg=mean)[1], ŷ, y
    mean(sum.(abs2, (ŷ .- y))), ŷ, y
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

@time train_sine(10)
# @time traintest(1000, QNDF())
# @time traintest(1000, TRBDF2())
# @time traintest(1000, AutoTsit5(Rosenbrock23()))






















using ModelingToolkit, OrdinaryDiffEq
@variables t y₁(t) y₂(t) y₃(t)
@parameters k₁ k₂ k₃
D = Differential(t)
eqs = [D(y₁) ~ -k₁*y₁+k₃*y₂*y₃
       D(y₂) ~  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
       D(y₃) ~  k₂*y₂^2]
sys = ODESystem(eqs, t)
prob = ODEProblem(sys,[y₁=>1f0,y₂=>0f0,y₃=>0f0],(0f0,500f0),
                      [k₁=>4f-2,k₂=>3f7,k₃=>1f4],jac=true)
N = 1000
y₁s = rand(Float32,N)
y₂s = 1f-4 .* rand(Float32,N)
y₃s = rand(Float32,N)
function prob_func(prob,i,repeat)
    remake(prob,p=[y₁s[i],y₂s[i],y₃s[i]])
end
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
solve(monteprob,Rodas5(),EnsembleThreads(),trajectories=1000)
@time solve(monteprob,Rodas5(),EnsembleThreads(),trajectories=1000)
# 0.286984 seconds (579.92 k allocations: 39.468 MiB)
# 0.298308 seconds (580.01 k allocations: 39.474 MiB)
# 0.322972 seconds (586.44 k allocations: 40.250 MiB)
solve(monteprob,RadauIIA5(),EnsembleThreads(),trajectories=1000)
@profview  solve(monteprob,RadauIIA5(),EnsembleThreads(),trajectories=1000)
@btime solve(monteprob,RadauIIA5(),EnsembleThreads(),trajectories=1000)
# 0.245401 seconds (456.89 k allocations: 35.456 MiB)
# 0.226091 seconds (456.91 k allocations: 35.456 MiB)
# 0.226375 seconds (456.94 k allocations: 35.457 MiB)
