using LTC

# typeof(LTC.NCPNet(2, 2,7,0,1,2,2,3,4).cell)
# @code_warntype LTC.NCPNet(2, 2,7,0,1,2,2,3,4).cell(rand(Float32,10,1),rand(Float32,2,1))

sum(length.(Flux.params(NCPNet(2, 2,7,0,1,2,2,3,4))))
length(Flux.destructure(NCPNet(2, 2,7,0,1,2,2,3,4))[1])
#sum(length.(Flux.trainable(NCPNet(2, 2,7,0,1,2,2,3,4).cell)))


using IterTools: ncycle
#using Flux: reset!, params, Chain, destructure, trainable, Losses, Optimiser, ClipValue, train!, ADAM
#import Flux: train!
using GalacticOptim
using Plots
using Noise


function generate_data()
    in_features = 2
    out_features = 1
    N = 48
    data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
    data_x = [reshape([Float32(data_x[1][i]),Float32(data_x[2][i])],2,1) for i in 1:N]# |> f32
    data_y = [reshape([Float32(y)],1) for y in sin.(range(0,stop=6π,length=N))]# |> f32


    data_x, data_y
end


function lossf(x,y,m)
  GalacticOptim.Flux.reset!(m)
  ŷ = m.(x)
  # sum([(ŷ[i][end,1] - y[i][1,1]) ^ 2 for i in 1:length(y)]), ŷ
  sum([(ŷ[i][end,1] .- y[i][end,1]) .^ 2 for i in 1:length(y)]), ŷ
end

function callback(x,y,c)
    l,pred = lossf(x, y, c)
    println(l)
    isnan(l) && return false
    fig = scatter([(isinf(ŷ[end,1]) || isnan(ŷ[end,1])) ? missing : ŷ[1,1] for ŷ in pred])
    scatter!(fig, [yi[1,1] for yi in y])
    display(fig)
    return false
end

function loss_sciml(θ)
  m = re(θ)
  x, y = data_x, data_y
  ŷ = m.(x)
  sum(Losses.mse.(ŷ,y)), ŷ
end

function loss_galactic(θ,re,x,y)
  m = re(θ)
  #x, y = data_x, data_y
  ŷ = m.(x)


  #sum([(ŷ[i][end,1] - y[i][1,1]) ^ 2 for i in 1:length(y)]), ŷ
  sum([(ŷ[i][1] .- y[i][1]) .^ 2 for i in 1:length(y)]), ŷ
end


function cbs(θ,l,pred;doplot=true) #callback function to observe training
  display(l)
  if doplot
    plot([x[1] for x in pred])
    #fig = scatter([(isinf(x[end,1]) || isnan(x[end,1])) ? missing : x[end,1] for x in pred])
    #scatter!(fig, [x[1,1] for x in y])
    #display(fig)
  end
  return false
end


function cbs2(θ,l;doplot=false) #callback function to observe training
  display(l)
  if doplot
    isnan(l) && return false
    fig = plot([x[end,1] for x in pred])
    #scatter!([x[1] for x in y])
    display(fig)
  end
  return false
end




tmp = NCPNet(2, 2,7,0,1,2,2,3,4)
tmpp, tmpre = GalacticOptim.Flux.destructure(tmp)
# tmp(ones(2))
#
# tmp2 = tmpre(tmpp)
# tmp2(ones(2))
#
tmpps = GalacticOptim.Flux.params(tmpp)
#
# gs = GalacticOptim.Flux.Zygote.gradient(ps) do
#     x = loss_galactic(tmpp, tmpre, [rand(Float32,2),rand(Float32,2)],[rand(Float32,1),rand(Float32,1)])
#     first(x)
# end
# for g in gs
#     println(g)
# end
#
#
#
# gs = GalacticOptim.Flux.Zygote.gradient(ps) do
#     x = lossf([rand(2)],[rand(1)],tmp)
#     first(x)
# end
# for g in gs
#     @show g
# end


function my_custom_train!(loss, ps, data, opt; lower=nothing, upper=nothing, cb=()->nothing)
  ps = Params(ps)
  for d in data
    # back is a method that computes the product of the gradient so far with its argument.
    train_loss, back = Zygote.pullback(() -> loss(d...), ps)
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


function train()
    #ncp = NCPNet(2, 2,7,0,1, connections="full")
    #ncp = NCPNet(2, 2,3,4,1,2,2,3,4)
    ncp = NCPNet(2, 2,7,0,1,2,2,3,4)
    chain = Chain(Mapper(2),ncp,Mapper(1))
    #c = Chain(Mapper(2),LTCNS(2,2),Mapper(2),LTC(2,7),Mapper(7),LTCNS(7,1),Mapper(1))
    #c = ncp
    c = ncp
    θ = params(c)
    @show length(θ)
    @show sum(length.(θ))

    ps,re = GalacticOptim.Flux.destructure(c)
    @show length(ps)
    #@show sum(length.(trainable(ncp.cell)))


    #sens_mask = ncp.cell.sens_mask
    syn_mask = ncp.cell.syn_mask()

    #@show sum(sens_mask)
    @show sum(syn_mask)

    #display(Plots.heatmap(sens_mask))
    display(Plots.heatmap(syn_mask))


    data_x, data_y = generate_data()

    @time loss_galactic(ps,re,data_x,data_y)
    @time loss_galactic(ps,re,data_x,data_y)
    @time loss_galactic(ps,re,data_x,data_y)
    @time loss_galactic(ps,re,data_x,data_y)


    @time lossf(data_x,data_y,c)
    @time lossf(data_x,data_y,c)
    @time lossf(data_x,data_y,c)


    fig = plot([x[1] for x in data_x])
    plot!(fig, [x[2] for x in data_x])
    plot!(fig, [x[1] for x in data_y])
    display(fig)

    # g = Zygote.gradient(Params(ps)) do
    #  loss_galactic(ps,re,data_x,data_y)[1]
    # end

    lower, upper = get_bounds(c)
    #lower, upper = [get_bounds(Mapper(2))...,get_bounds(ncp)...,get_bounds(Mapper(1))...]
    #@show size(lower)

    #data = (iter) -> ncycle([(data_x , data_y)], iter)
    #data = (iter) -> [(data_x , add_gauss.(data_y,0.02)) for _ in 1:iter]


    function data(iter; short=false, noisy=false)
        noisy_data = Vector{Tuple{Vector{Matrix{Float32}}, Vector{Vector{Float32}}}}([])
        #noisy_data = []
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
    @show typeof(data(2))
    @show typeof(data(2)[1])
    callback(data_x,data_y,c)

    #opt = GalacticOptim.Flux.Optimiser(ClipValue(0.0001),GalacticOptim.Flux.ExpDecay(0.01, 0.1, 200, 1e-4), ADAM(0.01))
    opt = GalacticOptim.Flux.Optimiser(GalacticOptim.Flux.ExpDecay(0.001, 0.1, 500, 1e-4), ADAM(0.001))
    #train!((x,y) -> lossf(x,y,c)[1], θ, ncycle([(data_x, data_y)], 100), ADAM(0.001f0),cb=()->callback(data_x,data_y,c))
    # GalacticOptim.Flux.train!((x,y) -> lossf(x,y,c)[1], θ, data(2), opt,cb=()->callback(data_x,data_y,c))
    # #GalacticOptim.Flux.train!((x,y) -> lossf(x,y,c)[1], θ, data(1000), GalacticOptim.Flux.Optimiser(ClipValue(0.0001f0), ADAM(0.001f0)),cb=()->callback(data_x,data_y,c))
    #my_custom_train!((x,y) -> lossf(x,y,c)[1], θ, data(2), opt; cb=()->callback(data_x,data_y,c),lower,upper)
    my_custom_train!((x,y) -> lossf(x,y,c)[1], θ, data(3), opt; cb=()->callback(data_x,data_y,c),lower,upper)
    my_custom_train!((x,y) -> lossf(x,y,c)[1], θ, data(3,short=[1,10]), opt; cb=()->callback(data_x,data_y,c),lower,upper)
    my_custom_train!((x,y) -> lossf(x,y,c)[1], θ, data(300,short=[1,10]), opt; cb=()->callback(data_x,data_y,c),lower,upper)
    my_custom_train!((x,y) -> lossf(x,y,c)[1], θ, data(300,short=[1,25]), opt; cb=()->callback(data_x,data_y,c),lower,upper)
    my_custom_train!((x,y) -> lossf(x,y,c)[1], θ, data(3000), opt; cb=()->callback(data_x,data_y,c),lower,upper)
    # train!((x,y) -> lossf(x,y,c)[1], θ, data(1000), ADAM(0.001f0),cb=()->callback(data_x,data_y,c))
    # train!((x,y) -> lossf(x,y,c)[1], θ, data(1000), ADAM(0.0001f0),cb=()->callback(data_x,data_y,c))


    f = OptimizationFunction((x0,p,x,y)->loss_galactic(x0,re,x,y),GalacticOptim.AutoZygote())
    #f = GalacticOptim.instantiate_function(optf,ps,GalacticOptim.AutoZygote(),nothing)
    prob = OptimizationProblem(f,ps)
    # sol = solve(prob, ParticleSwarm(), maxiters=3, cb = cbs)
    # sol = solve(prob, ParticleSwarm(n_particles=16), maxiters=300, cb = cbs)
    sol = GalacticOptim.solve(prob, opt, data(2), maxiters=3)



    # f = OptimizationFunction((x,p)->loss_galactic(x,x,re,data_x,data_y),GalacticOptim.AutoZygote())
    # prob = OptimizationProblem(f,ps,SciMLBase.NullParameters())
    # sol = solve(prob, NelderMead(), maxiters=3, cb = cbs)
    # @time sol = solve(prob,NelderMead(), maxiters=100, cb = cbs)
end
@time train()
