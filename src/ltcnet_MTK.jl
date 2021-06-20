#using Dates

# TODO: outside NCP net?
struct Mapper{V}
  W::V
  b::V
  p::V
  paramlength::Int
end
function Mapper(in::Integer)
  W = ones(Float32,in)
  b = zeros(Float32,in)
  p = vcat(W,b)
  Mapper(W, b, p, length(p))
end
Flux.@functor Mapper (p,)

function (m::Mapper)(x::AbstractVecOrMat, p=m.p) where T
  Wl = size(m.W,1)
  W = @view p[1 : Wl]
  b = @view p[Wl + 1 : end]
  W .* x .+ b
end
Base.show(io::IO, m::Mapper) = print(io, "Mapper(", length(m.W), ")")

initial_params(m::Mapper) = m.p
paramlength(m::Mapper) = m.paramlength


struct LTCCell{W<:Wiring,NET,SYS,PROB,DEFS,KS,I,U0,S<:AbstractMatrix,SOLVER,SENSEALG,V}
  wiring::W
  net::NET
  sys::SYS
  prob::PROB
  defs::DEFS
  ks::KS
  input_idxs::I
  u0_idxs::U0
  state0::S
  solver::SOLVER
  sensealg::SENSEALG
  p::V
  paramlength::Int
  param_names::Vector{String}
end

Flux.@functor LTCCell (p,)

function LTCCell(wiring, solver, sensealg; state0r=Float32.((0.01)))
  n_in = wiring.n_in
  out = wiring.out
  n_total = wiring.n_total

  @named net = Net(wiring)
  sys = structural_simplify(net)

  defs = ModelingToolkit.get_defaults(sys)
  # ks, ps = get_params(sys)
  prob = ODEProblem(sys, defs, Float32.((0,1)), jac=true, sparse=true) # TODO: jac, sparse ???

  # param_names = collect(parameters(net))
  param_names = collect(parameters(sys))

  input_idxs = Int8[findfirst(x->string(Symbol("x_$(i)_ExternalInput₊val"))==string(x), param_names) for i in 1:n_in]
  param_names = param_names[n_in+1:end]

  u0_idxs = Int8[]
  # for i in 1:n_total
  #   s = Symbol("n$(i)₊v(t)")
  #   idx = findall(x->string(s)==string(x), ks)
  #   push!(u0_idxs, idx[1])
  # end

  state0 = reshape(prob.u0, :,1)
  # state0 = prob.u0

  p_ode = prob.p[n_in+1:end]
  p = Float32.(vcat(p_ode, prob.u0))
  # p = p_ode

  # initial_params() = Float32.(prob.p)
  # initial_params = p


  @show param_names
  @show prob.u0
  @show prob.f.syms
  @show length(prob.p)
  @show length(prob.p[n_in+1:end])
  @show input_idxs

  LTCCell(wiring, net, sys, prob, defs, param_names, input_idxs, u0_idxs, state0, solver, sensealg, p, length(p), string.(param_names))
end


Base.show(io::IO, m::LTCCell) = print(io, "LTCCell(", m.wiring.n_sensory, ",", m.wiring.n_inter, ",", m.wiring.n_command, ",", m.wiring.n_motor, ")")
initial_params(m::LTCCell) = m.p
paramlength(m::LTCCell) = m.paramlength


function (m::LTCCell)(h::AbstractVecOrMat, x::AbstractVecOrMat, p=m.p)
  h = repeat(h, 1, size(x,2)-size(h,2)+1)#::AbstractMatrix
  p_ode_l = size(p,1) - size(h,1)
  p_ode = @view p[1:p_ode_l] # without x, without u0
  # @show now(Dates.UTC)
  h = solve_ensemble(m,h,x,p_ode)        # EnsembleProblem
  # h = solve_ode(m,h,x,p_ode)           # Single solve (unbatched)
  # h = solve_mapreduce(m,h,x,p_ode)     # Parallel solve_ode
  # h = solve_stacked(m,h,x,p_ode)       # Big system with batchsize independent subsystems
  # @show now(Dates.UTC)
  h, h
end

function solve_ensemble(m,h,x,p)#::AbstractMatrix

  function prob_func(prob,i,repeat)
    xi = @view x[:,i]
    u0i = @view h[:,i]
    pp = vcat(xi,p)
    remake(prob, p=pp, u0=u0i)
  end

  # function output_func(sol,i)
  #   # @show size(sol)
  #   sol[:,1], false
  # end
  infs = fill(Inf32, size(h,1))
  function output_func(sol,i)
    sol.retcode != :Success && return infs, false
    sol[:, end], false
  end
  #
  # function reduction(u,data,I)
  #   u = append!(u,data)
  #   finished = (var(u) / sqrt(last(I))) / mean(u) < 0.5
  #   u, finished
  # end

  _u0 = @view h[:,1]
  _p = vcat((@view x[:,1]), p)
  _prob = remake(m.prob, u0=_u0, p=_p)
  ensemble_prob = EnsembleProblem(_prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sol = solve(ensemble_prob, m.solver, EnsembleThreads(), trajectories=size(x,2),
              sensealg=m.sensealg, save_everystep=false, save_start=false) # TODO: saveat ?
  # @show size(sol)
  Array(sol)
end

function solve_stacked(m,h,x,p)
  pp = reduce(vcat, [vcat((@view x[:,i]),p) for i in 1:size(x,2)])
  u0 = reduce(vcat, [(@view h[:,i]) for i in 1:size(h,2)])
  prob = remake(m.stacked_prob, p=pp, u0=u0)
  sol = solve(prob, m.solver; sensealg=m.sensealg, save_everystep=false, save_start=false)[:,:,end]
  sol = reshape(sol,size(h))
  @show size(sol)
  sol
end


function solve_mapreduce(m,h,x,p)#::AbstractMatrix
  f = (hb,xb) -> solve_ode(m,hb,xb,p)
  collections = [GalacticOptim.Flux.unstack(h,2),GalacticOptim.Flux.unstack(x,2)]
  # ThreadsX.mapreduce(f, hcat, collections...) # nested task error: this intrinsic must be compiled to be called. worked some time ago!?
  mapreduce(f, hcat, collections...)
end


function solve_ode(m,h,x,p)
  pp = vcat(vec(x),p)
  prob = remake(m.prob, p=pp, u0=vec(h))
  sol = Array(solve(prob, m.solver; sensealg=m.sensealg, save_everystep=false, save_start=false))
  # @show size(sol)
  sol
end


mutable struct LTCNet{MI<:Mapper,MO<:Mapper,T<:LTCCell,S,V}
  mapin::MI
  mapout::MO
  cell::T
  state::S
  p::V
  paramlength::Int
  #LTCNet(mapin,mapout,cell,state) = new{typeof(mapin),typeof(mapout),typeof(cell),typeof(state)}(mapin,mapout,cell,state)
end
function LTCNet(wiring,solver,sensealg)
  mapin = Mapper(wiring.n_in)
  mapout = Mapper(wiring.n_total)
  cell = LTCCell(wiring,solver,sensealg)

  p = Float32.(vcat(DiffEqFlux.initial_params(mapin), DiffEqFlux.initial_params(mapout), DiffEqFlux.initial_params(cell)))
  #p = DiffEqFlux.initial_params(cell)
  # p = vcat(DiffEqFlux.initial_params(cell), DiffEqFlux.initial_params(mapout))
  # @show length(p)
  # initial_params = p

  LTCNet(mapin,mapout,cell,cell.state0,p,length(p))
end

Flux.@functor Mapper (p,)


function (m::LTCNet{MI,MO,T,<:AbstractMatrix{T2}})(x::AbstractVecOrMat{T2}, p=m.p) where {MI,MO,T,T2}
  mapin_pl = paramlength(m.mapin)
  mapout_pl = paramlength(m.mapout)
  cell_pl = paramlength(m.cell)

  p_mapin  = @view p[1 : mapin_pl]
  p_mapout = @view p[mapin_pl + 1 : mapin_pl + mapout_pl]
  p_cell   = @view p[mapin_pl + mapout_pl + 1 : mapin_pl + mapout_pl + cell_pl]
  #p_cell = p
  # p_cell   = @view p[1 : cell_pl]
  # p_mapout = @view p[cell_pl + 1 : cell_pl + mapout_pl]

  x = m.mapin(x, p_mapin)
  m.state, y = m.cell(m.state, x, p_cell)
  # Inf32 ∈ m.state && return m.state
  m.mapout(y, p_mapout)
  #y
end

initial_params(m::LTCNet) = m.p
paramlength(m::LTCNet) = m.paramlength

Base.show(io::IO, m::LTCNet) = print(io, "LTCNet(", m.mapin, ",", m.mapout, ",", m.cell, ")")

# reset!(m::LTCNet) = (m.state = m.cell.state0)
reset!(m::LTCNet, p=m.p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))
reset!(m::DiffEqFlux.FastChain) = map(l -> reset!(l), m.layers)
# reset!(m) = nothing

reset_state!(m::LTCNet, p=m.p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))
# reset_state!(m::DiffEqFlux.FastChain, p) = map(l -> reset_state!(l, initial_params(l)), m.layers)
function reset_state!(m::DiffEqFlux.FastChain, p)
  start_idx = 1
  for l in m.layers
    pl = paramlength(l)
    p_layer = @view p[start_idx:start_idx+pl-1]
    reset_state!(l,p_layer)
    start_idx += pl
  end
end
reset_state!(m::Flux.Chain) = map(l -> reset_state!(l), m.layers)
reset_state!(m,p=[]) = nothing



get_bounds(m::Function) = Float32[],Float32[]

function get_bounds(m::DiffEqFlux.FastChain)
  lb = vcat([get_bounds(layer)[1] for layer in m.layers]...)
  ub = vcat([get_bounds(layer)[2] for layer in m.layers]...)
  return lb, ub
end

function get_bounds(m::DiffEqFlux.FastDense)
  lb = [[-100.1 for _ in 1:m.out*m.in]...,
        [-100.1 for _ in 1:m.out]...] |> f32
  ub = [[100.1 for _ in 1:m.out*m.in]...,
        [100.1 for _ in 1:m.out]...] |> f32
  return lb, ub
end

function get_bounds(m::Mapper)
  lb = [[-100.1 for i in 1:length(m.W)]...,
        [-100.1 for i in 1:length(m.b)]...] |> f32

  ub = [[100.1 for i in 1:length(m.W)]...,
        [100.1 for i in 1:length(m.b)]...] |> f32
  lb, ub
end

function get_bounds(m::LTCNet)
  cell_lb = Float32[]
  cell_ub = Float32[]
  for pn in m.cell.param_names
    if contains(pn, "Cm")
      push!(cell_lb, 0.8)
      push!(cell_ub, 4)
    elseif contains(pn, "leak₊G")
      push!(cell_lb, 1e-5)
      push!(cell_ub, 1.1)
    elseif contains(pn, "leak₊E")
      push!(cell_lb, -1.1)
      push!(cell_ub, 1.1)
    elseif contains(pn, "SigmoidSynapse₊G")
      push!(cell_lb, 1e-5)
      push!(cell_ub, 1.1)
    elseif contains(pn, "SigmoidSynapse₊E")
      push!(cell_lb, -1.1)
      push!(cell_ub, 1.1)
    elseif contains(pn, "SigmoidSynapse₊σ")
      push!(cell_lb, 1)
      push!(cell_ub, 10)
    elseif contains(pn, "SigmoidSynapse₊μ")
      push!(cell_lb, 0.1)
      push!(cell_ub, 1)
    end
  end
  push!(cell_lb, [-2 for i in 1:length(m.cell.state0)]...)
  push!(cell_ub, [2 for i in 1:length(m.cell.state0)]...)
  lb = Float32.([
      get_bounds(m.mapin)[1]...,
      get_bounds(m.mapout)[1]...,
      cell_lb...,
  ])
  ub = Float32.([
    get_bounds(m.mapin)[2]...,
    get_bounds(m.mapout)[2]...,
    cell_ub...,
  ])
  lb, ub
end











####################################
# OBSOLETE CODE
####################################


# function init_model(m,h,x,p)
#   h = repeat(h, 1, size(x,2)-size(h,2)+1)
#   # @parameters t
#   # @variables xxx[1:size(wiring.sens_mask,1)](t)
#   # for i in 1:size(xxx,1)
#   #   push!(m.net.eqs, xxx[i] ~ x[i,1])
#   # end
#   # m.sys = structural_simplify(m.net)
#   # m.defs = ModelingToolkit.get_defaults(sys)
#   # ks, ps = get_params(sys)
#   # m.paramlength = length(ps)
#   # m.prob = ODEProblem(m.sys, m.defs, Float32.((0,1)), jac=true, sparse=true)
#   h
# end
# Zygote.@nograd init_model

# function (m::LTCCell)(h::AbstractVector, x::AbstractVecOrMat, p)
#   h = init_model(m,h,x,p)
#   m(h,x,p)
# end



function LTCCell(wiring, solver, sensealg, batchsize; state0r=Float32.((0.01)))
  n_in = wiring.n_in
  out = wiring.out
  n_total = wiring.n_total

  # @named net = Net(wiring, batchsize)
  @named net = Net(wiring)
  sys = structural_simplify(net)
  defs = ModelingToolkit.get_defaults(sys)
  ks, ps = get_params(sys)
  prob = ODEProblem(sys, defs, Float32.((0,1)))
  param_names = collect(parameters(sys))

  @named stacked_net = ODESystem(Equation[],ModelingToolkit.get_iv(sys),systems=[ModelingToolkit.rename(sys,Symbol(:net,i)) for i in 1:batchsize])
  stacked_sys = structural_simplify(stacked_net)
  stacked_defs = ModelingToolkit.get_defaults(stacked_sys)
  stacked_prob = ODEProblem(stacked_sys, stacked_defs, Float32.((0,1)))
  stacked_param_names = collect(parameters(stacked_sys))

  @show param_names

  @show prob.u0
  @show prob.f.syms
  input_idxs = Int8[]
  #input_idxs = Int8[findfirst(x->string(Symbol("x_$(i)₊val"))==string(x), param_names) for i in 1:n_in]
  @show input_idxs
  u0_idxs = Int8[]

  state0 = reshape(prob.u0, :, 1)

  p_ode = prob.p[n_in+1:end]
  p = Float32.(vcat(p_ode, prob.u0))
  # p = p_ode
  initial_params = p

  LTCCell(wiring, net, sys, prob, stacked_prob, defs, param_names, input_idxs, u0_idxs, state0, solver, sensealg, initial_params, length(p), param_names)
end

function LTCNet(wiring,solver,sensealg, batchsize)
  mapin = Mapper(wiring.n_in)
  mapout = Mapper(wiring.n_total)
  cell = LTCCell(wiring,solver,sensealg, batchsize)

  p = Float32.(vcat(DiffEqFlux.initial_params(mapin), DiffEqFlux.initial_params(mapout), DiffEqFlux.initial_params(cell)))
  #p = DiffEqFlux.initial_params(cell)
  # p = vcat(DiffEqFlux.initial_params(cell), DiffEqFlux.initial_params(mapout))
  # @show length(p)
  initial_params = p

  LTCNet(mapin,mapout,cell,cell.state0,initial_params,length(p))
end
