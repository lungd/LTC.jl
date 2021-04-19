using LTC
using Plots
gr()
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq

function generate_data()
    in_features = 2
    out_features = 1
    N = 48
    data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
    data_x = [reshape([Float32(data_x[1][i]),Float32(data_x[2][i])],2,1) for i in 1:N]# |> f32
    data_y = [reshape([Float32(y)],1) for y in sin.(range(0,stop=6π,length=N))]# |> f32

    #data_x = [repeat(x,1,20) for x in data_x]
    #data_y = [repeat(x,1,20) for x in data_y]

    data_x, data_y
end

function data(iter; data_x=nothing, data_y=nothing, short=false, noisy=false)
    #noisy_data = Vector{Tuple{Vector{Matrix{Float32}}, Vector{Vector{Float32}}}}([])
    if data_y === nothing
      data_x, data_y = generate_data()
    end
    noisy_data = []
    for i in 1:iter
        x = data_x
        y = data_y
        if short isa Array
          x = x[short[1]:short[2]]
          y = y[short[1]:short[2]]
        end
        push!(noisy_data, (x , noisy ? add_gauss.(y,0.02) : y))
    end
    noisy_data
end



function traintest(n, solver, sensealg)
  function loss(x,y,m)
    ŷ = m.(x)
    sum(sum([(ŷ[i][end,:] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y)#, ŷ
  end

  function cb(x,y,l,m)
    println(l)
    pred = m.(x)
    # isnan(l) && return false
    fig = scatter([ŷ[end,1] for ŷ in pred])
    scatter!(fig, [yi[end,1] for yi in y])
    display(fig)
    return false
  end

  x,y = generate_data()
  model = NCP(NCPWiring(2,1), solver, sensealg)
  θ = Flux.params(model)
  lower,upper = get_bounds(model)

  display(display(Plots.heatmap(model.cell.wiring.sens_mask)))
  display(display(Plots.heatmap(model.cell.wiring.syn_mask)))

  opt = Flux.Optimiser(ClipValue(1), ADAM(0.01f0))
  my_custom_train!(model, (x,y) -> loss(x,y,model), θ, data(3), opt; cb, lower, upper)
  my_custom_train!(model, (x,y) -> loss(x,y,model), θ, data(n), opt; cb, lower, upper)
end


@btime traintest(3, VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
@btime traintest(3, VCABM(), InterpolatingAdjoint(autojacvec=ZygoteVJP()))
@time traintest(300, VCABM(), InterpolatingAdjoint(autojacvec=ZygoteVJP()))
@time traintest(300, VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

@time traintest(3, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
@time traintest(300, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))


#  2.118 s (6965793 allocations: 461.57 MiB)

#ltc = Flux.Chain(Dense(2,5),Flux.LSTM(5,5),Flux.Dense(5,1))

# Flux.reset!(ltc)
# opt = GalacticOptim.Flux.Optimiser(ClipValue(0.1), ADAM(0.001))
#
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(3), opt; cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(1000), opt; cb=()->cbf(data_x,data_y,ltc),lower,upper)

# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100), opt; data_range=[1,8], cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100), opt; data_range=[15,20], cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100), opt; data_range=[10,30], cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100), opt; data_range=[1,35], cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100, short=[1,20]), ADAM(0.001); cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100, short=[1,30]), ADAM(0.001); cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(3000), ADAM(0.001); cb=()->cbf(data_x,data_y,ltc),lower,upper)
#
# Flux.train!((x,y)->lossf(x,y)[1],Flux.params(ltc),data(200),ADAM(0.02f0); cb = ()->cbf(data_x,data_y,ltc))
