using LTC
using Plots
gr()
using DiffEqFlux
using BenchmarkTools
using DiffEqSensitivity
using OrdinaryDiffEq
using GalacticOptim
using Juno
using Cthulhu
using Profile
using BlackBoxOptim
using ModelingToolkit
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

function loss(p,x::Vector{Matrix{T}},y::Vector{Matrix{T}}, m) where T

  # ŷ = bcm(m,x,p)
  # ŷ = m.(x, [p])

  ŷb = GalacticOptim.Flux.Zygote.Buffer(y[1], size(y,1), size(y[1])[1],size(y[1])[2])
  for (i, xi) in enumerate(x)
    ŷi = m(xi, p)#::Matrix{T}
    ŷb[i,:,:] = ŷi
    Inf32 ∈ ŷi && return Inf32, Flux.unstack(copy(ŷb)::Array{T,3},1)::Vector{Matrix{T}}, y # TODO: what if a layer after MTKRecur can handle Infs?
  end

  # ŷ = copy(ŷb)
  # ŷ = [ŷ[i,:,:] for i in 1:size(ŷ)[1]]
  ŷ = Flux.unstack(copy(ŷb)::Array{T,3},1)::Vector{Matrix{T}}

  LTC.reset_state!(m, p)
  # mean(bcl(GalacticOptim.Flux.Losses.mse, ŷ, y)), ŷ, y
  # GalacticOptim.Flux.Losses.mse(ŷ, y, agg=mean)[1], ŷ, y
  mean(sum.(abs2, (ŷ .- y))), ŷ, y
  # bcl(GalacticOptim.Flux.Losses.mse, ŷ, y), ŷ, y
end

function cbg(p,l,pred,y;doplot=false)
  display(l)
  if doplot
    fig = plot([ŷ[end,1] for ŷ in pred], label="ŷ")
    plot!(fig, [yi[end,1] for yi in y], label="y")
    #frame(anim)
    display(fig)
  end
  return false
end


function test()
  n = 5
  solver=Tsit5()
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
  batchsize = 1

  wiring = LTC.FWiring(2,1)
  net = LTC.Net(wiring; name=:net)
  recurmtk = LTC.RecurMTK(LTC.MTKCell(wiring.n_in, wiring.n_out, net, solver, sensealg))

  model = DiffEqFlux.FastChain(LTC.Mapper(wiring.n_in),
                               recurmtk,
                               LTC.Mapper(wiring.n_out),
                               )

  train_dl = data(n)
  opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.01))
  # opt = BBO()
  AD = GalacticOptim.AutoZygote()
  # AD = GalacticOptim.AutoReverseDiff()

  x1,y1 = first(train_dl)

  p_mtk = initial_params(recurmtk)
  lb_mtk, ub_mtk = get_bounds(recurmtk)

  # @code_warntype loss(p_mtk,x1,y1,recurmtk)
  # @descend_code_warntype loss(p_mtk,x1,y1,recurmtk)
  # Juno.@profiler loss(p_mtk,x1,y1,recurmtk)
  # loss(p_mtk,x1,y1,recurmtk)

  optfun = GalacticOptim.OptimizationFunction((θ, p, x, y) -> loss(θ,x,y, recurmtk), AD)
  optfunc = GalacticOptim.instantiate_function(optfun, p_mtk, AD, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, p_mtk, lb=lb_mtk, ub=ub_mtk,
                                #grad = true, hess = true, sparse = true,
                                #parallel=ModelingToolkit.MultithreadedForm()
                                )


  # optfun = GalacticOptim.OptimizationFunction((θ, p, x, y) -> loss(θ,x,y, recurmtk),AutoModelingToolkit(),p_mtk,DiffEqBase.NullParameters(),
  #                    #grad = true, hess = true, sparse = true,
  #                    #checkbounds = false,
  #                    #parallel=ModelingToolkit.MultithreadedForm(),
  #                    )
  # optfunc = GalacticOptim.instantiate_function(optfun, p_mtk, AutoModelingToolkit(), DiffEqBase.NullParameters())
  # optprob = GalacticOptim.OptimizationProblem(optfunc, p_mtk, lb=lb_mtk, ub=ub_mtk,
  #                               #grad = true, hess = true, sparse = true,
  #                               #parallel=ModelingToolkit.MultithreadedForm()
  #                               )

  # GalacticOptim.solve(optprob, opt, train_dl, cb = cbg)

  res = Vector{}(undef,length(p_mtk))
  @time optfunc.grad(res,p_mtk,x1,y1)
  @time optfunc.grad(res,p_mtk,x1,y1)
  @time optfunc.grad(res,p_mtk,x1,y1)
  @time optfunc.grad(res,p_mtk,x1,y1)
end
test()

n = 5
solver=Tsit5()
sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
batchsize = 1

wiring = LTC.FWiring(2,1)
net = LTC.Net(wiring; name=:net)
# sys = ModelingToolkit.structural_simplify(net)
recurmtk = LTC.RecurMTK(LTC.MTKCell(wiring.n_in, wiring.n_out, net, solver, sensealg))

model = DiffEqFlux.FastChain(LTC.Mapper(wiring.n_in),
                             recurmtk,
                             LTC.Mapper(wiring.n_out),
                             )

train_dl = data(n)
opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.01))
# opt = BBO()
AD = GalacticOptim.AutoZygote()
# AD = GalacticOptim.AutoReverseDiff()

x1,y1 = first(train_dl)

p_mtk = initial_params(recurmtk)
lb_mtk, ub_mtk = get_bounds(recurmtk)
h_mtk = recurmtk.state

# h_mtkcell, ŷ_mtkcell = recurmtk.cell(h_mtk, x1[1], p_mtk)
# ŷ_mtkcell = recurmtk(x1[1], p_mtk)
# LTC.reset_state!(recurmtk, p_mtk)


@code_warntype recurmtk.(x1, [p_mtk])

# brc(m, x, p) = m.(x, [p])
# @code_warntype brc(recurmtk, x1, p_mtk)

@code_warntype loss(p_mtk,x1,y1,recurmtk)
@descend_code_warntype loss(p_mtk,x1,y1,recurmtk)

ŷ_mtk = recurmtk.(x1, [p_mtk])
@code_warntype mean(GalacticOptim.Flux.Losses.mse.(ŷ_mtk, y1))
@code_warntype GalacticOptim.Flux.Losses.mse(ŷ_mtk, y1)[1]
@code_warntype LTC.reset_state!(recurmtk, p_mtk)

@code_warntype LTC.solve_ensemble(recurmtk.cell, h_mtk, x1[1], p_mtk)
@descend_code_warntype LTC.solve_ensemble(recurmtk.cell, h_mtk, x1[1], p_mtk)

@descend_code_warntype LTC.optimize(recurmtk, loss, cbg, opt, AD, train_dl)

optfun = GalacticOptim.OptimizationFunction((θ, p, x, y) -> loss(θ,x,y, recurmtk), AD)
optfunc = GalacticOptim.instantiate_function(optfun, p_mtk, AD, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, p_mtk, lb=lb_mtk, ub=ub_mtk,
                              #grad = true, hess = true, sparse = true,
                              #parallel=ModelingToolkit.MultithreadedForm()
                              )
# GalacticOptim.solve(optprob, opt, train_dl, cb = cbg)

res = Vector{}(undef,length(p_mtk))
@time optfunc.grad(res,p_mtk,x1,y1)
@code_warntype LTC.optimize(recurmtk, loss, cbg, opt, AD, train_dl)

ps = Flux.params(pp)
gs = Flux.Zygote.gradient(ps) do
    x = prob.f(θ,prob.p, d...)
    first(x)
  end

# p = initial_params(model)
# lb, ub = get_bounds(model)
#
# ŷ = model.(x1, [p])
# loss(p, x1, y1, model)
#
# LTC.optimize(model, loss, cbg, opt, AD, train_dl)
