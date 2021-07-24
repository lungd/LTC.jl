using LTC
using ModelingToolkit
import Flux: Data.DataLoader
using OrdinaryDiffEq
using DiffEqSensitivity
using DiffEqFlux
using GalacticOptim
using IterTools: ncycle

using Plots

@variables t
D = Differential(t)

function Predator(;name)
  vars = @variables population(t)=1.0f0 prey(t)
  ps = @parameters α=1.0f0 β=1.0f0
  ODESystem([D(population) ~ α*population - β*population*prey], t, vars, ps, name=name)
end
function Prey(;name)
  vars = @variables population(t)=1.0f0 predator(t)
  ps = @parameters δ=1.0f0 γ=1.0f0
  ODESystem([D(population) ~ -δ*population + γ*predator*population], t, vars, ps, name=name)
end
function PPRel(;name)
  @named predator = Predator()
  @named prey = Prey()
  eqs = [
    predator.prey ~ prey.population
    prey.predator ~ predator.population
  ]
  ODESystem(eqs,t,Num[],Num[],systems=[predator,prey],name=name)
end



function get_data(seq_len=20,width=1,batchsize=10)
  @named net = PPRel()
  @nonamespace predator = net.predator
  @nonamespace prey = net.prey
  u0 = [predator.population => 1.2f0, prey.population => 1.5f0]
  p = [predator.α => 1.5f0, predator.β => 1.2f0, prey.δ => 3.0f0, prey.γ => 0.8f0]
  sys = ModelingToolkit.structural_simplify(net)
  prob = ODEProblem(sys,u0,(0.0f0,10.0f0),p)
  sol = solve(prob,Tsit5(),saveat=0.1)
  @show size(sol)
  display(plot(sol))

  data_x = [sol[:,s:s+seq_len-1] for s in 1:width:size(sol,2)-seq_len]
  data_x = [data_x[s:s+batchsize-1] for s in 1:length(data_x)-batchsize+1]
  data_x = [Flux.unstack(permutedims(Flux.stack(b,3),[1,3,2]),3) for b in data_x]
  @show size(data_x)
  @show length(data_x)
  @show size(data_x[1])
  @show size(data_x[1][1])
  data_y = [sol[:,s:s+seq_len-1] for s in 2:width:size(sol,2)-seq_len+1]
  data_y = [data_y[s:s+batchsize-1] for s in 1:length(data_y)-batchsize+1]
  data_y = [Flux.unstack(permutedims(Flux.stack(b,3),[1,3,2]),3) for b in data_y]

  dl = DataLoader((data_x, data_y), batchsize=1, shuffle=true)
  fx,fy = first(dl)
  @show size(fx)
  @show size(fx[1])
  @show size(fx[1][1])
  fig = plot([x[1,1] for x in fx[1]], label="x1")
  plot!(fig, [x[2,1] for x in fx[1]], label="x2")
  plot!(fig, [y[1,1] for y in fy[1]], label="y1")
  plot!(fig, [y[2,1] for y in fy[1]], label="y2")
  display(fig)
  dl
end
dl = get_data()
tx = first(dl)[1]

function train_lv(n, solver=VCABM(), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true));)

  cbg = function (p,l,pred,y;doplot=true)
    display(l)
    if doplot
      fig = plot([ŷ[1,1] for ŷ in pred], label="ŷ1")
      plot!(fig, [ŷ[2,1] for ŷ in pred], label="ŷ2")
      plot!(fig, [yi[1,1] for yi in y], label="y1")
      plot!(fig, [yi[2,1] for yi in y], label="y2")
      display(fig)
    end
    return false
  end

  seq_len=20
  width=1
  batchsize=10
  train_dl = get_data(seq_len,width,batchsize)

  wiring = LTC.NCPWiring(2,2;
    n_sensory=2, n_inter=5, n_command=4, n_motor=2,
    sensory_in=-1, rec_sensory=4, sensory_inter=2, sensory_command=1, sensory_motor=0,
    inter_in=2, rec_inter=2, inter_command=3, inter_motor=1,                       # inter_in = sensory_out
    command_in=0, rec_command=4, command_motor=4,                   # command_in = inter_out
    motor_in=0, rec_motor=2)


    wiring = LTC.FWiring(2,2)
    net = LTC.Net(wiring, name=:net)
    sys = ModelingToolkit.structural_simplify(net)

  model = DiffEqFlux.FastChain(#(x,p) -> x[1],
                               LTC.Mapper(wiring.n_in),
                               LTC.RecurMTK(LTC.MTKCell(wiring.n_in, wiring.n_out, net, sys, solver, sensealg)),
                               LTC.Mapper(wiring.n_out),
                               # (x,p) -> [x],
                               )

  opt = Flux.Optimiser(ClipValue(1.00f0), ExpDecay(1f0, 0.01f0, 200, 0.00001f0), ADAM())
  # opt = Optim.LBFGS()
  # opt = BBO()
  # opt = ParticleSwarm(;lower=lb, upper=ub)
  # opt = Fminbox(GradientDescent())
  AD = GalacticOptim.AutoZygote()
  # AD = GalacticOptim.AutoModelingToolkit()
  LTC.optimize(model, (p, m, x, y)->LTC.loss_seq(p,m,x[1],y[1]), cbg, opt, AD, ncycle(train_dl,n)), model
end

@time res1,model = train_lv(100)
