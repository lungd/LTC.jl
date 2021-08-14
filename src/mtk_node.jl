mutable struct MTKNODE{T,V,S}
  cell::T
  p::V
  paramlength::Int
  state::S
  MTKNODE(cell, p, paramlength, state) = new{typeof(cell),typeof(p),typeof(state)}(cell,p,paramlength,state)
end
function MTKNODE(cell;)
  p = initial_params(cell)
  MTKNODE(cell, p, length(p), cell.state0)
end
function (m::MTKNODE)(x, p=m.p)
  m.state, y = m.cell(m.state, x, p)
  return y
end

Base.show(io::IO, m::MTKNODE) = print(io, "MTKNODE(", m.cell, ")")
initial_params(m::MTKNODE) = m.p
paramlength(m::MTKNODE) = m.paramlength

Flux.@functor MTKNODE (p,)
Flux.trainable(m::MTKNODE) = (m.p,)

function get_bounds(m::MTKNODE{C,<:AbstractArray{T},S}, _T::Type=nothing; default_lb = -Inf, default_ub = Inf) where {C,T,S}
  # T = eltype(m.wiring.sens_mask)
  cell_lb = T[]
  cell_ub = T[]

  params = collect(parameters(m.cell.sys))
  states = collect(ModelingToolkit.states(m.cell.sys))[m.cell.in+1:end]
  for v in vcat(params,states)
    contains(string(v), "InPin") && continue
		contains(string(v), "OutPin") && continue
		hasmetadata(v, ModelingToolkit.VariableOutput) && continue
    lower = hasmetadata(v, VariableLowerBound) ? getmetadata(v, VariableLowerBound) : default_lb
    upper = hasmetadata(v, VariableUpperBound) ? getmetadata(v, VariableUpperBound) : default_ub
    push!(cell_lb, lower)
    push!(cell_ub, upper)
  end
  return cell_lb, cell_ub
end

reset!(m::MTKNODE, p=m.p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))
reset_state!(m::MTKNODE, p=m.p) = (m.state = reshape(p[end-size(m.cell.state0,1)+1:end], :, 1))


struct MTKNODECell{NET,SYS,PROB,PROBF,SOLVER,SENSEALG,V,OP,S}
  in::Int
  out::Int
  net::NET
  sys::SYS
  prob::PROB
  prob_f::PROBF
  solver::SOLVER
  sensealg::SENSEALG
  tspan::Tuple
  p::V
  paramlength::Int
  param_names::Vector{String}
  outpins::OP
  infs::Vector
  state0::S

  function MTKNODECell(in, out, net, sys, prob, prob_f, solver, sensealg, tspan, p, paramlength, param_names, outpins, infs, state0)
    new{typeof(net),typeof(sys),typeof(prob),typeof(prob_f),typeof(solver),typeof(sensealg),typeof(p),typeof(outpins),typeof(state0)}(
                       in, out, net, sys, prob, prob_f, solver, sensealg, tspan, p, paramlength, param_names, outpins, infs, state0)
  end
end
function MTKNODECell(wiring::Wiring{T}, net, sys, solver, sensealg) where T

  in::Int = wiring.n_in
  out::Int = wiring.n_out

  tspan = (T(0), T(1))
  defs = ModelingToolkit.get_defaults(sys)
  prob = ODEProblem(sys, defs, tspan) # TODO: jac, sparse ???

  prob_f = ODEFunction(sys, states(sys), parameters(sys), tgrad=true)

  _states = collect(states(sys))
  input_idxs = Int8[findfirst(x->contains(string(x), string(Symbol("x$(i)_InPin"))), _states) for i in 1:in]
  param_names = ["placeholder"]
  outpins = 1f0

  p_ode = prob.p
  u0 = prob.u0[in+1:end]
  state0 = reshape(u0, :, 1)
  p = vcat(p_ode, u0)
  infs = fill(T(Inf), size(state0,1))

  @show param_names
  @show prob.u0
  @show size(state0)
  @show prob.f.syms
  @show length(prob.p)
  @show input_idxs
  @show outpins
  @show typeof(p)

  MTKNODECell(in, out, net, sys, prob, prob_f, solver, sensealg, tspan, p, length(p), param_names, outpins, infs, state0)
end

function (m::MTKNODECell{NET,SYS,PROB,PROBF,SOLVER,SENSEALG,V,OP,<:AbstractMatrix{T}})(h, inputs::AbstractArray{T,3}, p) where {NET,SYS,PROB,PROBF,SOLVER,SENSEALG,V,OP,T}
  # size(h) == (N,1) at the first MTKNODECell invocation. Need to duplicate batchsize times
  num_reps = size(inputs,3)-size(h,2)+1
  hr = repeat(h, 1, num_reps)
  p_ode = @view p[1:end-size(hr,1)]
  solve_ensemble_full_seq(m,hr,inputs,p_ode)
end

function solve_ensemble_full_seq(m::MTKNODECell{NET,SYS,PROB,PROBF,SOLVER,SENSEALG,V,OP,<:AbstractMatrix{T}}, u0s, xs::AbstractArray{T,3}, p_ode) where {NET,SYS,PROB,PROBF,SOLVER,SENSEALG,V,OP,T}

  nfs = size(xs,1)
  nobs = size(xs,2)
  batchsize = size(xs,3)
  nout = size(u0s,1)
  tspan = (T(0), T(nobs))

  infs = fill(T(Inf), nout,nobs)

  dosetimes = collect(1f0:T(nobs))

  ts1 = collect(T(0):T(nobs))

  prob_f = m.prob_f

  # basic_tgrad(u,p,t) = zero(u)

  function _f(u,p,t,I)
    du = similar(u)
    _f(du,u,p,t,I)
    return du
  end

  function _f(du,u,p,t,I)
    u_mtk = u
    prob_f.f(du,u_mtk,p,t)
    du[1:m.in] .= ForwardDiff.derivative(t->I(t), t)
    return nothing
  end
  f = ODEFunction(_f)
  # f = ODEFunction(_f,tgrad=basic_tgrad)

  function prob_func(prob,i,repeat)
    ϵ = Float32(1e-5)
    x = @view xs[:,:,i]
    x1 = hcat(x, x[:,end].+ϵ)
    I = LinearInterpolation(x1,ts1)

    u0 = vcat((@view x[:,1]),(@view u0s[:,i]))
    p = p_ode
    ODEProblem((du,u,p,t)->f(du,u,p,t,I), u0, tspan, p, saveat=1f0, save_everystep=false, save_end=true, save_start=true)
  end

  function output_func(sol,i)
    sol.retcode != :Success && return infs, false
    sol, false
  end

  ensemble_prob = EnsembleProblem(m.prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sol = solve(ensemble_prob, m.solver, EnsembleThreads(), trajectories=batchsize,
              sensealg=m.sensealg, saveat=1f0,
              save_everystep=false, save_start=true, save_end=true, reltol=1e-3, abstol=1e-3,
		)
  s = Array(sol)
  get_quantities_of_interest(s, m, Int.(dosetimes))
end



function get_quantities_of_interest(sol::AbstractArray, m::MTKNODECell, dosetimes)
  h = sol[m.in+1:end,end,:]
  out = @view sol[end-m.out+1:end, end-length(dosetimes)+1:end, :]
  return h, out
end

Base.show(io::IO, m::MTKNODECell) = print(io, "MTKNODECell(", m.in, ",", m.out, ")")
initial_params(m::MTKNODECell) = m.p
paramlength(m::MTKNODECell) = m.paramlength
Flux.@functor MTKNODECell (p,)
Flux.trainable(m::MTKNODECell) = (m.p,)


function MTKNODEMapped(chainf, wiring, net, sys, solver, sensealg)
  chainf(LTC.MapperIn(wiring),
          MTKNODE(MTKNODECell(wiring, net, sys, solver, sensealg)),
          LTC.MapperOut(wiring))
end
