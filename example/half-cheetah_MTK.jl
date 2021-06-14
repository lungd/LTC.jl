using LTC
using Plots
gr()
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using BlackBoxOptim

include("half_cheetah_data_loader.jl")


function traintest(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)); _pp = nothing, _model=nothing)

  function lg(p,x,y, m)
  	# reset_state!(m,p)
  	reset!(m)

  	# ŷt = [m.([xi[:,i] for i in 1:size(xi,2)],[p]) for xi in x]
  	# ŷ = [hcat(ŷts...) for ŷts in ŷt]

  	# ŷ = [m(xi[:,1], p) for xi in x]
  	ŷ = m.(x, [p])
    # mean([GalacticOptim.Flux.Losses.mse(ŷ[i],y[i]) for i in 1:length(y)]), ŷ, y

  	mean(GalacticOptim.Flux.Losses.mse.(ŷ,y)), ŷ, y

  	# @show size(ŷ)
  	# @show size(ŷ[1])
  	# @show size(ŷ[1][1])
  	# @show size(y)
  	# @show size(y[1])

  	# ŷ = map(xi -> m(xi,p), x)
  	# mean([GalacticOptim.Flux.Losses.mse(ŷ[i],y[i]) for i in 1:length(y)]), ŷ, y
    #sum(sum([(ŷ[i] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y), ŷ, y
  end
  cbg = function (p,l,pred,y;doplot=true)
   display(l)
    if doplot
  	  fig = plot([ŷ[1,1] for ŷ in pred])
          plot!(fig, [ŷ[2,1] for ŷ in pred])
  	  plot!(fig, [yi[1,1] for yi in y])
          plot!(fig, [yi[2,1] for yi in y])
  	  display(fig)
  	end
  	return false
  end

  batchsize=64

  train_dl, test_dl, valid_dl = get_dl(batchsize=batchsize, seq_len=32)

  @show length(train_dl)
  @show size(first(train_dl)[1])
  @show size(first(train_dl)[1][1])

  ncp = LTC.LTCNet(NCPWiring(17,2,
          n_sensory=8, n_inter=8, n_command=8, n_motor=17,
          sensory_in=-1, rec_sensory=0, sensory_inter=2, sensory_command=0, sensory_motor=0,
          inter_in=2, rec_inter=2, inter_command=3, inter_motor=1,                       # inter_in = sensory_out
          command_in=0, rec_command=4, command_motor=2,                   # command_in = inter_out
          motor_in=0, rec_motor=3), solver, sensealg)

  dense = DiffEqFlux.FastDense(ncp.cell.wiring.n_motor,17)

  # model = (x,p) -> ncp(x,p)[end-16:end, :]
  model = DiffEqFlux.FastChain(ncp, (x,p)->x[end-16:end, :], dense)
  # model = DiffEqFlux.FastChain(ncp, dense)


  pp = initial_params(model)
  @show length(pp)
  lower,upper = [],[]
  b = get_bounds(model)
  lb, ub = b
  @show length(lower)
  @show length(lb)
  @show length(ub)

  optfun = OptimizationFunction((θ, p, x, y) -> lg(θ,x,y, model), GalacticOptim.AutoZygote())
  # optfun = OptimizationFunction((θ, p, x, y) -> lg(θ,x,y, model), GalacticOptim.AutoModelingToolkit())

  optprob = OptimizationProblem(optfun, pp, lb=lb, ub=ub,
                                #grad = true, hess = true, sparse = true,
                                #parallel=ModelingToolkit.MultithreadedForm()
                                )
  # res1 = GalacticOptim.solve(optprob, Flux.Optimiser(ClipValue(0.50), ADAM(0.005)), ncycle(train_dl,n), cb = cbg)
  # res1 = GalacticOptim.solve(optprob, Optim.LBFGS(), ncycle(train_dl,n), cb = cbg)
  res1 = GalacticOptim.solve(optprob, BBO(), ncycle(train_dl,n), cb = cbg)
  # res1 = GalacticOptim.solve(optprob, ParticleSwarm(;lower=lb, upper=ub), ncycle(train_dl,n), cb = cbg)
  # res1 = GalacticOptim.solve(optprob, Fminbox(GradientDescent()), ncycle(train_dl,n), cb = cbg)

  # res1 = GalacticOptim.solve(optprob, CMAES(μ =40 , λ = 100), ncycle(train_dl,n), cb = cbg)
  # res1 = GalacticOptim.solve(optprob, Opt(:LD_LBFGS, length(pp)), ncycle(train_dl,n), cb = cbg)

  res1, model
end

@time res1,model = traintest(1)
# @time res1,model = traintest(5, VCABM(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = traintest(5, Tsit5(), InterpolatingAdjoint(checkpointing=true))
# @time res1,model = traintest(5, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = traintest(5, Tsit5(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# @time res1,model = traintest(1, VCABM(), BacksolveAdjoint())
