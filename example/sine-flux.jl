using LTC
using Plots
gr()
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
#using DiffEqFlux
#using GalacticOptim
using Juno
using Cthulhu
#using Profile
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

    data_x, data_y
end

function data(iter; data_x=nothing, data_y=nothing, short=false, noisy=false)
    #noisy_data = Vector{Tuple{Vector{Matrix{Float64}}, Vector{Vector{Float64}}}}([])
    if data_y === nothing
      data_x, data_y = generate_data()
    end
    noisy_data = Vector{Tuple{Vector{Matrix{eltype(data_x[1])}}, Vector{Vector{eltype(data_y[1])}}}}([])
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


function loss(x,y,m)
  Flux.reset!(m)
  #m = re(θ)
  #ŷ = m.(x)
  ŷ = map(xi -> m(xi)[end-m.cell.wiring.n_motor+1:end, :], x)
  #ŷ = [m(xi)[end-m.cell.wiring.n_motor+1:end, :] for xi in x]
  #ŷ = m.(x)
  #length(ŷ) < length(y) && return Inf
  #sum(Flux.Losses.mse.(ŷ, y; agg=mean))
  sum(sum([(ŷ[i][end,:] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y)#, ŷ
  #sum([sum((ŷ[i] .- y[i]) .^ 2) for i in 1:length(y)])/length(y)
end
function cb(x,y,l,m)
  println(l)
  # pred = m.(x)
  # # isnan(l) && return false
  # fig = plot([ŷ[size(ŷ,1),1] for ŷ in pred])
  # plot!(fig, [yi[size(yi,1),1] for yi in y])
  # display(fig)
  return false
end

function traintest(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

  function lg(p,model)
    d = train_data[1]
    x,y = d[1], d[2]
    m = model
    ŷ = [m(xi,p) for xi in x]
    #ŷ = m.(x,[p])
    #losses = [Flux.Losses.mse(ŷ[i][end,:], y[i]) for i in 1:length(y)]
    #sum(losses)/length(losses), ŷ
    sum(sum([(ŷ[i][end,:] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y), ŷ
  end
  cbg = function (p,l,pred;doplot=true) #callback function to observe training
    display(l)
    # plot current prediction against data
    if doplot
      fig = plot([ŷ[end,1] for ŷ in pred])
      plot!(fig, [yi[end,1] for yi in y])
      display(fig)
    end
    return false
  end

  x,y = generate_data()
  model = LTC.LTCNet(Wiring(2,1), solver, sensealg)
  #pp, re = Flux.destructure(model)
  # pp = DiffEqFlux.initial_params(model)
  # lower,upper = get_bounds(model)
  lower,upper = [],[]
  θ = Flux.params(model)
  #θ = Flux.params(pp)


  @show sum(length.(θ))
  @show length(lower)

  train_data = data(n)

  opt = Flux.Optimiser(ClipValue(1), ADAM(0.03))


  # Juno.@profiler gs = Flux.Zygote.gradient(θ) do
  #   loss(first(train_data)...,model)
  # end
  # @time  gs = Flux.Zygote.gradient(θ) do
  #   loss(first(train_data)...,model)
  # end
  # @time  gs = Flux.Zygote.gradient(θ) do
  #   loss(first(train_data)...,model)
  # end
  #
  #
  # Juno.@profiler train_loss, back = Flux.Zygote.pullback(() -> loss(x,y,model), θ)
  # Juno.@profiler gs = back(one(train_loss))
  # train_loss, back = Flux.Zygote.pullback(() -> loss(x,y,model), θ)
  # gs = back(one(train_loss))
  # @time train_loss, back = Flux.Zygote.pullback(() -> loss(x,y,model), θ)
  # @time gs = back(one(train_loss))

  # use GalacticOptim.jl to solve the problem
  #adtype = GalacticOptim.AutoZygote()
  #
  #optf = GalacticOptim.OptimizationFunction((x, p) -> lg(x,model), adtype)
  #optfunc = GalacticOptim.instantiate_function(optf, model.cell.p, adtype, nothing)
  #optprob = GalacticOptim.OptimizationProblem(optfunc, model.cell.p, lb=lower, ub=upper)
  #
  #result_neuralode = GalacticOptim.solve(optprob,
  #                                     ParticleSwarm(;lower,upper,n_particles=6),
  #                                     cb = cbg,
  #                                     maxiters = 300)



  ##
  #optfun = OptimizationFunction((x,p,dx,dy)->lg(x,dx,dy,model), GalacticOptim.AutoZygote())
  #optprob = OptimizationProblem(optfun, θ[1], lb=lower, ub=upper)
  ##using IterTools: ncycle
  #res1 = GalacticOptim.solve(optprob, opt, train_data, cb = cbg, maxiters = n)
  #return res1

  #Flux.train!((x,y) -> loss(x,y,model), θ, train_data, opt; cb)
  my_custom_train!(model, (x,y) -> loss(x,y,model), θ, data(3), opt; cb, lower, upper)
  # Juno.@profiler my_custom_train!(model, (x,y) -> loss(x,y,model), θ, train_data, opt; cb, lower, upper)
  my_custom_train!(model, (x,y) -> loss(x,y,model), θ, train_data, opt; cb, lower, upper)
end



function cthulu_test()

  m = LTC.LTCNet(Wiring(2,1), VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
  x = rand(Float32,2,1)
  m(x)
  #d = Dense(2,2)
  # r = RNN(2,2)
  # @descend_code_warntype r(x)
  Flux.reset!(m)
  @descend_code_warntype m(x)
end
cthulu_test()


# import Base: *
# *(::Flux.Zeros, b::AbstractArray) = -b
# *(a::AbstractArray, ::Flux.Zeros) = a
function gtest(g,e,m)
  loss(p) = sum(m .* g)
  gs = Flux.Zygote.gradient(Flux.params(g)) do
    sum([sum((g[i,i] .- e[i,i]) .* m[i,i]) for i in 1:size(g,1)])
  end
  @show gs[g][1,1]
  @show gs[g][1,2]
end
function ztest()
  g = ones(Float32, 100,100)
  e = rand(Float32,100,100)
  m = zeros(Float32, 100,100)
  for i = 1:size(m,1)
    m[i,i] = 1
  end
  # m = sparse(m)
  #m = m - LinearAlgebra.I
  # m = Array(m)
  # Diagonal(m)
  @show typeof(m)
  @show typeof(g)
  @show typeof(e)
  #m[1,1] = 1f0
  @time gtest(g,e,m)

end
ztest()

# model = NCP(Wiring(2,1), VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
# ppp = Flux.params(model)
# sum(length.(ppp))
# Flux.trainable(model)

#@time traintest(10, AutoTsit5(Rosenbrock23()))
@time traintest(10)
@time traintest(300)
#00:53 - 01:02 = 9 min compilation time


#@time traintest(300, VCABM(), ForwardDiffSensitivity())
@time traintest(300, VCABM(), ReverseDiffAdjoint())
# @time traintest(300, Euler(), ForwardDiffSensitivity())
@time traintest(300, VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
@time traintest(300, VCABM(), InterpolatingAdjoint(autojacvec=false))
@time traintest(300, VCABM(), InterpolatingAdjoint(autojacvec=ZygoteVJP()))
@time traintest(300, VCABM(), InterpolatingAdjoint(checkpointing=true))
traintest(300, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
@btime traintest(3, CVODE_BDF(linear_solver=:GMRES), InterpolatingAdjoint(autojacvec=ZygoteVJP()))
@time traintest(300, CVODE_BDF(linear_solver=:GMRES), InterpolatingAdjoint(autojacvec=ZygoteVJP()))
@time traintest(300, CVODE_BDF(linear_solver=:GMRES), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

#  601.084 ms (1153411 allocations: 98.26 MiB)
#  8.510164 seconds (35.02 M allocations: 3.450 GiB, 5.96% gc time)

#@time traintest(300, VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

#@time traintest(3, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
#@time traintest(300, AutoTsit5(Rosenbrock23()), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))


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
# Flux.train!((x,y)->lossf(x,y)[1],Flux.params(ltc),data(200),ADAM(0.02); cb = ()->cbf(data_x,data_y,ltc))


const testMD = Flux.Dense(2,1)
const testMM = Mapper(2)
const testMG = gNN(1,1)

const testM = NCP(Wiring(2,1), VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
const x2 = rand(Float32,2,1)
@time testM(x2)
@descend_code_warntype testM(x2)
@descend testM(x2)

@code_warntype testM(x2)
@code_warntype testM.cell(testM.state,x2)

const testLTC = LTC.LTCCell(Wiring(2,1), VCABM(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
const testhh = rand(Float32,8,1)
testLTC(testhh,x2)
@code_warntype testLTC(testhh,x2,testLTC.p)

const x1 = rand(Float32,2)

function warntest(x, m)
  m(x)
end

@code_warntype testMD(x1)
testMM(x1,p=testMM.p)



const td = data(1)
const xx, yy = first(td)
const pp = Flux.params(testM)
Juno.@profiler train_loss, back = Flux.Zygote.pullback(() -> loss(x,y,testM), Flux.Params(θ))
Juno.@profiler gs = back(one(train_loss))
@time train_loss, back = Flux.Zygote.pullback(() -> loss(x,y,testM), Flux.Params(θ))
@time gs = back(one(train_loss))


@profile train_loss, back = Flux.Zygote.pullback(() -> loss(x,y,testM), Flux.Params(θ))
pprof(;webport=58599)

@time my_custom_train!(testM, (x,y) -> loss(x,y,testM), θ, train_data, ADAM(); cb)

Juno.@profiler my_custom_train!(testM, (x,y) -> loss(x,y,testM), θ, train_data, ADAM(); cb)
