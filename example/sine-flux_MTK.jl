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
#using PProf

function generate_data()
    in_features = 2
    out_features = 1
    N = 48
    data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
    data_x = [reshape([Float32(data_x[1][i]),Float32(data_x[2][i])],2,1) for i in 1:N]# |> f32
    data_y = [reshape([Float32(y)],1) for y in sin.(range(0,stop=6π,length=N))]# |> f32

    #data_x = [repeat(x,1,20) for x in data_x]
    #data_y = [repeat(x,1,20) for x in data_y]

    # data_x, data_y
    DataLoader((data_x, data_y), batchsize=N)
end

function data(iter; data_x=nothing, data_y=nothing, short=false, noisy=false)
    #noisy_data = Vector{Tuple{Vector{Matrix{Float64}}, Vector{Vector{Float64}}}}([])
    # if data_y === nothing
    #   data_x, data_y = generate_data()
    # end
    # noisy_data = Vector{Tuple{Vector{Matrix{eltype(data_x[1])}}, Vector{Vector{eltype(data_y[1])}}}}([])
    # for i in 1:iter
    #     x = data_x
    #     y = data_y
    #     if short isa Array
    #       x = x[short[1]:short[2]]
    #       y = y[short[1]:short[2]]
    #     end
    #     push!(noisy_data, (x , noisy ? add_gauss.(y,0.02) : y))
    # end
    # noisy_data


    ncycle(generate_data(), iter)
end

function traintest(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

  #anim = Animation()

  function lg(p,x,y,model)
    # reset_state!(model,p)
    reset!(model)
    #ŷ = [m(xi,p)[end-m.cell.wiring.n_motor+1:end, :] for xi in x]
    ŷ = model.(x,[p])
    #ŷ = map(xi -> m(xi,p), x)
    #losses = [Flux.Losses.mse(ŷ[i][end,:], y[i]) for i in 1:length(y)]
    #sum(losses)/length(losses), ŷ
    sum(sum([(ŷ[i][end,:] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y), ŷ, y
  end
  cbg = function (p,l,pred,y;doplot=true)
    display(l)
    if doplot
      fig = plot([ŷ[end,1] for ŷ in pred])
      plot!(fig, [yi[end,1] for yi in y])
      #frame(anim)
      display(fig)
    end
    return false
  end

  #train_dl = generate_data()
  model = LTC.LTCNet(Wiring(2,1), solver, sensealg)


  # lower,upper = get_bounds(model)
  lower,upper = [],[]

  pp = DiffEqFlux.initial_params(model)
  @show length(pp)
  @show length(pp)

  @show length(lower)

  train_dl = data(n)

  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.008))


  optfun = OptimizationFunction((θ, p, x, y) -> lg(θ,x,y,model), GalacticOptim.AutoZygote())
  # optfun = OptimizationFunction((θ, p, x, y) -> lg(θ,x,y,model), GalacticOptim.AutoModelingToolkit())
  # optprob = OptimizationProblem(optfun, pp, lb=lower, ub=upper)
  optprob = OptimizationProblem(optfun, pp)
  #using IterTools: ncycle
  #Juno.@profiler GalacticOptim.solve(optprob, opt, train_data, cb = cbg, maxiters = n) C = true
  @show size(first(train_dl)[1])
  @show size(first(train_dl)[1][1])
  GalacticOptim.solve(optprob, opt, train_dl, cb = cbg)



  # sciml_train(p->lg(p,model), pp, opt, cb = cbg, maxiters=100)



  #Juno.@profiler my_custom_train!(model, (x,y) -> loss(x,y,model), θ, train_data, opt; cb, lower, upper) C = true
end


@time traintest(1000)
#@time traintest(1000, QNDF())
# @time traintest(5000, AutoTsit5(Rosenbrock23()))
