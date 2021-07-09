# abstract type MTKLayer <: Function end

mutable struct RecurMTK{T,V,S}# <:MTKLayer
  cell::T
  p::V
  paramlength::Int
  state::S
end
function RecurMTK(cell; seq_len=1)
  p = DiffEqFlux.initial_params(cell)
  RecurMTK(cell, p, length(p), cell.state0)
end
function (m::RecurMTK)(x, p=m.p)
  m.state, y = m.cell(m.state, x, p)
  return y
end
Base.show(io::IO, m::RecurMTK) = print(io, "RecurMTK(", m.cell, ")")
initial_params(m::RecurMTK) = m.p
paramlength(m::RecurMTK) = m.paramlength

Flux.@functor RecurMTK (p,)#(cell,)
Flux.trainable(m::RecurMTK) = (m.p,)#Flux.trainable(m.cell)


# struct MTKCell{NET,SYS,PROB,DEFS,KS,I,U0,SOLVER,SENSEALG,V,S}
struct MTKCell{SYS,PROB,SOLVER,SENSEALG,V,S}
  in::Int
  out::Int
  sys::SYS
  prob::PROB
  solver::SOLVER
  sensealg::SENSEALG
  p::V
  paramlength::Int
  param_names::Vector{String}
  state0::S
end
function MTKCell(in::Int, out::Int, sys, solver, sensealg; seq_len=1)


  defs = ModelingToolkit.get_defaults(sys)
  prob = ODEProblem(sys, defs, Float32.((0,1))) # TODO: jac, sparse ???

  param_names = collect(parameters(sys))
  input_idxs = Int8[findfirst(x->contains(string(x), string(Symbol("x_$(i)_ExternalInput₊val"))), param_names) for i in 1:in]
  param_names = param_names[in+1:end]

  u0_idxs = Int8[]

  state0 = reshape(prob.u0, :, 1)


  p_ode = prob.p[in+1:end]
  p = Float32.(vcat(p_ode, prob.u0))
  # p = p_ode

  @show param_names
  @show prob.u0
  @show size(state0)
  @show prob.f.syms
  @show length(prob.p)
  @show length(prob.p[in+1:end])
  @show input_idxs

  #MTKCell(in, out, net, sys, prob, defs, param_names, input_idxs, u0_idxs, solver, sensealg, p, length(p), string.(param_names), state0)
  MTKCell{typeof(sys),typeof(prob),typeof(solver),typeof(sensealg),typeof(p),typeof(state0)}(in, out, sys, prob, solver, sensealg, p, length(p), string.(param_names), state0)
end
function (m::MTKCell)(h::Matrix{T}, xs::Matrix{T}, p) where {PROB,SOLVER,SENSEALG,V,T}
  # size(h) == (N,1) at the first MTKCell invocation. Need to duplicate batchsize times
  num_reps = size(xs)[2]-size(h)[2]+1
  h = repeat(h, 1, num_reps)#::Matrix{T}
  p_ode_l = size(p)[1] - size(h)[1]
  p_ode = @view p[1:p_ode_l]
  h = solve_ensemble(m,h,xs,p_ode)
  h, h[end-m.out+1:end, :]
end

#function solve_ensemble(m,h,x,p, tspan=(0f0,1f0))#::Matrix{Float32}
function solve_ensemble(m, u0s::Matrix{T}, xs, p_ode, tspan=(0f0,1f0))::Matrix{T} where T

  # infs = fill(Inf32, size(h)[1])

  function prob_func(prob,i,repeat)
    x = @view xs[:,i]
    u0 = @view u0s[:,i]
    p = vcat(x, p_ode)
    remake(prob; tspan, p, u0)
  end
  function output_func(sol,i)::Tuple{Vector{T},Bool}
    sol.retcode != :Success && return h[:,1], false
    sol[:, end], false
  end


  batchsize = size(xs)[2]
  ensemble_prob = EnsembleProblem(m.prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sol = solve(ensemble_prob, m.solver, EnsembleThreads(), trajectories=batchsize,
              sensealg=m.sensealg, save_everystep=false, save_start=false) # TODO: saveat ?
  # @show size(sol)
  # Array(sol)
  sol[:,:]::Matrix{T}
end

Base.show(io::IO, m::MTKCell) = print(io, "MTKCell(", m.in, ",", m.out, ")")
initial_params(m::MTKCell) = m.p
paramlength(m::MTKCell) = m.paramlength

Flux.@functor MTKCell (p,)
Flux.trainable(m::MTKCell) = (m.p,)


reset!(m::RecurMTK, p=m.p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))

function reset_state!(m::RecurMTK, p=m.p)
  trained_state0 = p[end-size(m.cell.state0,1)+1:end]
  m.state = reshape(trained_state0, :, 1)
end

function get_bounds(m::RecurMTK)
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
  cell_lb, cell_ub
end
