using LTC
using GalacticOptim
using Plots
using Functors

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


data_x,data_y = generate_data()


wiring = Wiring(2,1)

display(Plots.heatmap(wiring.sens_mask))
display(Plots.heatmap(wiring.sens_pol))
display(Plots.heatmap(wiring.syn_mask))
display(Plots.heatmap(wiring.syn_pol))

ltc = NCP(wiring)
θ = Flux.params(ltc)
@time ltc.([rand(Float32,2,1) for i in 1:50])

Flux.reset!(ltc)
@time loss_galactic(ltcp,[],data_x,data_y,ltcre)

lower, upper = get_bounds(ltc)

function lossf(x,y)
  ŷ = ltc.(x)
  sum(sum([(ŷ[i] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y), ŷ
end

function cbf(x,y,c)
    l,pred = lossf(x, y)
    println(l)
    isnan(l) && return false
    fig = scatter([(isinf(ŷ[end,1]) || isnan(ŷ[end,1])) ? missing : ŷ[1,1] for ŷ in pred])
    scatter!(fig, [yi[1,1] for yi in y])
    display(fig)
    return false
end


function my_custom_train!(m, loss, ps, data, opt; data_range=nothing, lower=nothing, upper=nothing, cb=()->nothing)
  ps = Params(ps)
  for d in data

    Flux.reset!(m)

    x, y = d

    if data_range !== nothing
      seq_start = data_range[1]
      seq_end = length(data_range) == 2 ? data_range[2] : length(x)

      @views x = x[1:seq_end]
      @views y = y[1:seq_end]

      if seq_start != 1
        skipx = x[1:seq_start-1]
        m.(skipx)
        @views x = x[seq_start:end]
        @views y = y[seq_start:end]
      end
    end

    # back is a method that computes the product of the gradient so far with its argument.
    train_loss, back = Zygote.pullback(() -> loss(x,y), ps)
    cb()
    # Insert whatever code you want here that needs training_loss, e.g. logging.
    # logging_callback(training_loss)
    # Apply back() to the correct type of 1.0 to get the gradient of loss.
    gs = back(one(train_loss))
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.

    GalacticOptim.Flux.Optimise.update!(opt, ps, gs)

    # Here you might like to check validation set accuracy, and break out to do early stopping.

    lower == nothing && continue
    upper == nothing && continue


    for (i,p) in enumerate(ps[1])
        ps[1][i] = max(lower[i],ps[1][i])
        ps[1][i] = min(upper[i],ps[1][i])
    end
  end
end


#ltc = Flux.Chain(Dense(2,5),Flux.LSTM(5,5),Flux.Dense(5,1))

Flux.reset!(ltc)
opt = GalacticOptim.Flux.Optimiser(ClipValue(0.01), ADAM(0.001))

my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(3), opt; cb=()->cbf(data_x,data_y,ltc),lower,upper)
my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(1000), opt; cb=()->cbf(data_x,data_y,ltc),lower,upper)

# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100), opt; data_range=[1,8], cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100), opt; data_range=[15,20], cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100), opt; data_range=[10,30], cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100), opt; data_range=[1,35], cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100, short=[1,20]), ADAM(0.001); cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(100, short=[1,30]), ADAM(0.001); cb=()->cbf(data_x,data_y,ltc),lower,upper)
# my_custom_train!(ltc, (x,y) -> lossf(x,y)[1], Flux.params(ltc), data(3000), ADAM(0.001); cb=()->cbf(data_x,data_y,ltc),lower,upper)
#
# Flux.train!((x,y)->lossf(x,y)[1],Flux.params(ltc),data(200),ADAM(0.02f0); cb = ()->cbf(data_x,data_y,ltc))
