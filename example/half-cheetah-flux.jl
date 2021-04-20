using LTC
using Plots
gr()
using BenchmarkTools

include("half_cheetah_data_loader.jl")


function traintest(n, solver, sensealg)
  function loss(x,y,m)
    Flux.reset!(m)

    ŷ = [m(xi) for xi in x]

    sum(sum([(ŷ[i] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y)
  end

  function cb(x,y,l,m)
    println(l)
    #pred = m.(x)
    # isnan(l) && return false
    # fig = plot([ŷ[1,1] for ŷ in pred])
	# plot!(fig, [ŷ[end,1] for ŷ in pred])
    # plot!(fig, [yi[1,1] for yi in y])
	# plot!(fig, [yi[end,1] for yi in y])
    # display(fig)
    return false
  end

  train_dl, test_dl, valid_dl = get_dl(batchsize=32, seq_len=32)
  ncp = NCP(NCPWiring(17,2,
          n_sensory=2, n_inter=3, n_command=2, n_motor=2,
          sensory_in=-1, rec_sensory=0, sensory_inter=2, sensory_command=0, sensory_motor=0,
          inter_in=0, rec_inter=2, inter_command=2, inter_motor=0,                       # inter_in = sensory_out
          command_in=0, rec_command=1, command_motor=10,                   # command_in = inter_out
          motor_in=0, rec_motor=1), solver, sensealg)
  model1 = ncp
  model2 = Dense(2,17,σ)
  model = Chain(model1, model2)
  #θ = Flux.params(model1,model2)
  θ = Flux.params(model)
  lower,upper = get_bounds(model)

  # @show length(train_dl)
  # @show size(first(train_dl)[1])
  # @show size(first(train_dl)[1][1])
  #
  # display(display(Plots.heatmap(ncp.cell.wiring.sens_mask)))
  # display(display(Plots.heatmap(ncp.cell.wiring.syn_mask)))
  #
  # display(display(Plots.heatmap(ncp.cell.wiring.sens_pol)))
  # display(display(Plots.heatmap(ncp.cell.wiring.syn_pol)))

  @show sum([length(p) for p in θ])
  @show length(lower)


  opt = Flux.Optimiser(ClipValue(0.1), ADAM(0.001))
  my_custom_train!(model, (x,y) -> loss(x,y,model), θ, train_dl, opt; cb, lower, upper)
  my_custom_train!(model, (x,y) -> loss(x,y,model), θ, train_dl, opt; cb, lower, upper)
end


@time traintest(3, VCABM(), InterpolatingAdjoint(autojacvec=ZygoteVJP()))





#@time traintest(3, Rosenbrock23(), InterpolatingAdjoint(checkpointing=false))

#@time traintest(3, ImplicitEuler(), InterpolatingAdjoint(autojacvec=ZygoteVJP()))
#@time traintest(3, VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
