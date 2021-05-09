using LTC
using Plots
gr()
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim

include("half_cheetah_data_loader.jl")


function traintest(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
	function loss(x,y,m)
	  Flux.reset!(m)
	  #m = re(θ)
	  #ŷ = m.(x)
	  ŷ = map(xi -> m(xi), x)
	  #ŷ = [m(xi)[end-m.cell.wiring.n_motor+1:end, :] for xi in x]
	  #ŷ = m.(x)
	  #length(ŷ) < length(y) && return Inf
	  #sum(Flux.Losses.mse.(ŷ, y; agg=mean))
	  sum(sum([(ŷ[i] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y)#, ŷ
	  #sum([sum((ŷ[i] .- y[i]) .^ 2) for i in 1:length(y)])/length(y)
	end

  function cb(x,y,l,m)
    println(l)
    # pred = m.(x)
    # # isnan(l) && return false
    # fig = plot([ŷ[1,1] for ŷ in pred])
	#   plot!(fig, [ŷ[end,1] for ŷ in pred])
    # plot!(fig, [yi[1,1] for yi in y])
	#   plot!(fig, [yi[end,1] for yi in y])
    # display(fig)
    return false
  end

  function lg(p,x,y)
	m = model
	#Flux.reset!(m)
	reset_state!(model,p)
	ŷ = map(xi -> m(xi,p), x)
    sum(sum([(ŷ[i] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y), ŷ, y
  end
  cbg = function (p,l,pred,y;doplot=true) #callback function to observe training
   display(l)
    # plot current prediction against data
    if doplot
  	  # pl = scatter(first(train_dl)[1],label="data")
	  # scatter!(pl,pred[1,:],label="prediction")
	  # display(plot(pl))

	  fig = plot([ŷ[end,1] for ŷ in pred])
	  plot!(fig, [yi[end,1] for yi in y])
	  display(fig)
	end
	return false
  end

  train_dl, test_dl, valid_dl = get_dl(batchsize=32, seq_len=32)
  ncp = LTC.LTCNet(NCPWiring(17,2,
          n_sensory=4, n_inter=4, n_command=7, n_motor=2,
          sensory_in=-1, rec_sensory=0, sensory_inter=2, sensory_command=0, sensory_motor=0,
          inter_in=2, rec_inter=2, inter_command=3, inter_motor=1,                       # inter_in = sensory_out
          command_in=0, rec_command=4, command_motor=2,                   # command_in = inter_out
          motor_in=0, rec_motor=3), solver, sensealg)
  model1 = ncp
  model = ncp

  model2 = DiffEqFlux.FastDense(2,17)
  #model = (x,p) -> model1(x,p)[end-16:end, :]
  model = DiffEqFlux.FastChain(model1, (x,p) -> x[end-1:end,:], model2)
  # model2 = Dense(2,17)
  # model = Chain(model1, x -> x[end-1:end, :], model2)
  # model = model1
  # model = DiffEqFlux.FastChain(ncp,(x,p) -> x[end-1:end, :],DiffEqFlux.FastDense(2,17))
  #θ = Flux.params(model1,model2)
  # θ = Flux.params(model)
  # @show sum(length.(θ))
  pp = initial_params(model)
  @show length(pp)
  # pp = DiffEqFlux.initial_params(model)
  lower,upper = get_bounds(model.layers[1])
  lower,upper = get_bounds(model)
  # lower,upper = [],[]
  @show length(lower)

  # @show length(train_dl)
  # @show size(first(train_dl)[1])
  # @show size(first(train_dl)[1][1])
  #
  # display(Plots.heatmap(ncp.cell.wiring.sens_mask))
  # display(Plots.heatmap(ncp.cell.wiring.syn_mask))
  #
  # display(Plots.heatmap(ncp.cell.wiring.sens_pol))
  # display(Plots.heatmap(ncp.cell.wiring.syn_pol))

  #@show sum([length(p) for p in θ])
  #@show length(lower)
  # @show length(pp)

  # model.(first(train_dl)[1])
  # model.(first(train_dl)[1])
  # model.(first(train_dl)[1])

  opt = Flux.Optimiser(ClipValue(0.5), ADAM(0.01))

  optfun = OptimizationFunction((θ, p, x, y) -> lg(θ,x,y), GalacticOptim.AutoZygote())
  optprob = OptimizationProblem(optfun, pp, lb=lower, ub=upper)
  #using IterTools: ncycle
  res1 = GalacticOptim.solve(optprob, opt, ncycle(train_dl,n), cb = cbg, maxiters = n)
  # return res1

  #my_custom_train!(model, (x,y) -> loss(x,y,model), θ, train_dl, opt; cb, lower, upper)
  # Flux.@epochs n my_custom_train!(model, (x,y) -> loss(x,y,model), θ, train_dl, opt; cb, lower, upper)
end


@time traintest(10)

# @time traintest(10, VCABM(), ForwardDiffSensitivity())
# @time traintest(10, VCABM(), ReverseDiffAdjoint())
# @time traintest(10, VCABM(), QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
# # 192.343950 seconds (206.01 M allocations: 64.016 GiB, 4.65% gc time, 22.36% compilation time)
# # 138.803175 seconds (159.75 M allocations: 51.048 GiB, 5.04% gc time)
# # 89.533896 seconds (346.13 M allocations: 138.572 GiB, 23.55% gc time)
# @time traintest(10, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
#
#
# @time traintest(1, VCABM(), InterpolatingAdjoint(checkpointing=true, autojacvec=ZygoteVJP()))
# @time traintest(1, VCABM(), TrackerAdjoint())
# @time traintest(1, VCABM(), InterpolatingAdjoint(autojacvec=ZygoteVJP()))
# @time traintest(1, VCABM(), QuadratureAdjoint(compile=true, abstol=1e-3,reltol=1e-3, autojacvec=ZygoteVJP()))
# @time traintest(1, VCABM(), QuadratureAdjoint(compile=true, abstol=1e-3,reltol=1e-3, autojacvec=ReverseDiffVJP(true)))
# @time traintest(1, Tsit5(), ZygoteAdjoint())
# # 142.623700 seconds (1.43 G allocations: 380.410 GiB, 45.09% gc time, 0.02% compilation time)
#
# @time traintest(1, VCABM(), ReverseDiffAdjoint())
# @time traintest(1, CVODE_BDF(linear_solver=:GMRES), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
# @time traintest(1, VCABM(), InterpolatingAdjoint(checkpointing=true, autojacvec=ReverseDiffVJP(true)))
# # 161.160516 seconds (985.58 M allocations: 201.055 GiB, 22.29% gc time, 9.22% compilation time)
#
# #@time traintest(3, VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
#



#@time traintest(3, Rosenbrock23(), InterpolatingAdjoint(checkpointing=false))

#@time traintest(3, ImplicitEuler(), InterpolatingAdjoint(autojacvec=ZygoteVJP()))
#@time traintest(3, VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
