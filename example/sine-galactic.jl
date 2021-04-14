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

    # data_x = [repeat(x,1,20) for x in data_x]
    # data_y = [repeat(x,1,20) for x in data_y]

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
lower, upper = get_bounds(ltc)

#ltcp, ltcre = Functors.functor(ltc)
ltcp,ltcre = Flux.destructure(ltc)
ltcre(ltcp).([rand(Float32,2,1) for i in 1:50])

function loss_galactic(θ,p,x,y)
  m = ltcre(θ)
  ŷ = m.(x)
  sum(sum([(ŷ[i] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y), ŷ
end
@time loss_galactic(ltcp,[],data_x,data_y,ltcre)


function cbs(θ,l,pred;doplot=true) #callback function to observe training
  display(l)
  if doplot
    #plot([x[1] for x in pred])
    #fig = scatter([(isinf(x[end,1]) || isnan(x[end,1])) ? missing : x[end,1] for x in pred])
    #scatter!(fig, [x[1,1] for x in y])
    #display(fig)
  end
  return false
end
cbs(ltcp,loss_galactic(ltcp,[],data_x,data_y,ltcre)...)

opt = GalacticOptim.Flux.Optimiser(GalacticOptim.Flux.ExpDecay(0.001, 0.1, 500, 1e-4), ADAM(0.001))

optfunc = GalacticOptim.OptimizationFunction((x, p, tx,ty) -> loss_galactic(x,p,tx,ty), GalacticOptim.AutoZygote())
optprob = GalacticOptim.OptimizationProblem(optfunc, ltcp; lb = lower, ub = upper)

res = GalacticOptim.solve(optprob, ADAM(0.1), data(2), cb = cbs, maxiters = 100)
res = GalacticOptim.solve(optprob, Fminbox(BFGS(initial_stepnorm = 0.01)), cb = cbs, allow_f_increases=false)
res = GalacticOptim.solve(optprob, Opt(:LN_BOBYQA, length(ltcp)), maxiters=100, cb = cbs)

sol = GalacticOptim.solve(optprob, NelderMead(), data(3), maxiters=3000, cb = cbs)
