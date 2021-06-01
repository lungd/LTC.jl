abstract type Component end
abstract type CurrentComponent <:Component end

struct Mapper{V,F<:Function}
  W::V
  b::V
  initial_params::F
  paramlength::Int
end
function Mapper(in::Integer)
  W = ones(Float32,in)
  b = zeros(Float32,in)
  p = vcat(W,b)
  initial_params() = p
  Mapper(W, b, initial_params, length(p))
end

function (m::Mapper)(x::AbstractVecOrMat, p) where T
  Wl = size(m.W,1)
  W = @view p[1 : Wl]
  b = @view p[Wl + 1 : end]
  W .* x .+ b
end
Base.show(io::IO, m::Mapper) = print(io, "Mapper(", length(m.W), ")")

initial_params(m::Mapper) = m.initial_params()
paramlength(m::Mapper) = m.paramlength


struct LTCCell{W<:Wiring,NET,SYS,PROB,DEFS,KS,I,U0,S<:AbstractMatrix,SOLVER,SENSEALG,F<:Function}
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
  initial_params::F
  paramlength::Int
end


function LTCCell(wiring, solver, sensealg; state0r=Float32.((0.01)))
  n_in = wiring.n_in
  out = wiring.out
  n_total = wiring.n_total

  # sys = generate_sys(wiring, rand(Float32,2))
  @named net = Net(wiring)
  #inputs = getproperty(net, :inputs)
  # @variables x_input[1:n_in](t)
  # push!(net.eqs, [xi ~ rand(Float32) for xi in x_input]...)

  sys = structural_simplify(net)

  defs = ModelingToolkit.get_defaults(sys)
  #defs_with_input = merge(defs, Dict(x_input[i] => 0 for i in 1:n_in))
  ks, ps = get_params(sys)
  #defs = vcat([ks[i] => ps[i] for i in 1:length(ps)])
  prob = ODEProblem(sys, defs, Float32.((0,1)))

  param_names = collect(parameters(sys))

  # indexof(sym,syms) = findfirst(isequal(sym),syms)


  input_idxs = Int8[findfirst(x->string(Symbol("x_$(i)₊val"))==string(x), param_names) for i in 1:n_in]
  @show input_idxs

  u0_idxs = Int8[]
  # for i in 1:n_total
  #   s = Symbol("n$(i)₊v(t)")
  #   idx = findall(x->string(s)==string(x), ks)
  #   push!(u0_idxs, idx[1])
  # end

  state0 = reshape(prob.u0, :,1)

  initial_params() = Float32.(prob.p)
  #initial_params() = vcat(Float32.(prob.p), Float32.(prob.u0))

  LTCCell(wiring, net, sys, prob, defs, param_names, input_idxs, u0_idxs, state0, solver, sensealg, initial_params, length(ps))
end
Base.show(io::IO, m::LTCCell) = print(io, "LTCCell(", m.wiring.n_sensory, ",", m.wiring.n_inter, ",", m.wiring.n_command, ",", m.wiring.n_motor, ")")
initial_params(m::LTCCell) = m.initial_params()
paramlength(m::LTCCell) = m.paramlength

function (m::LTCCell)(h::AbstractVecOrMat, x::AbstractVecOrMat, p)
  h = repeat(h, 1, size(x,2)-size(h,2)+1)
  # p_ode = @view p[1:size(m.prob.p,1)]
  p_ode = p
  h = solve_ode(m,h,x, p_ode)
  h, h
end

function solve_ode(m,h,x,p)

  # _p = Zygote.Buffer(p, size(p,1))
  # _p .= p
  # _p[m.input_idxs] .= x[:,1]
  # pp = copy(_p)

  pp = vcat((@view x[:,1]), (@view p[3:end]))

  prob = remake(m.prob, p=pp, u0=h)
  # prob = remake(m.prob, p=pp)
  solve(prob, m.solver; sensealg=m.sensealg, save_everystep=false, save_start=false)[:,:,end]
end



mutable struct LTCNet{MI<:Mapper,MO<:Mapper,T<:LTCCell,S,F<:Function}
  mapin::MI
  mapout::MO
  cell::T
  state::S
  initial_params::F
  paramlength::Int
  #LTCNet(mapin,mapout,cell,state) = new{typeof(mapin),typeof(mapout),typeof(cell),typeof(state)}(mapin,mapout,cell,state)
end
function LTCNet(wiring,solver,sensealg)
  mapin = Mapper(wiring.n_in)
  mapout = Mapper(wiring.n_total)
  cell = LTCCell(wiring,solver,sensealg)

  p = vcat(DiffEqFlux.initial_params(mapin), DiffEqFlux.initial_params(mapout), DiffEqFlux.initial_params(cell))
  initial_params() = p

  LTCNet(mapin,mapout,cell,cell.state0,initial_params,length(p))
end


function (m::LTCNet{MI,MO,T,<:AbstractMatrix{T2}})(x::AbstractVecOrMat{T2}, p) where {MI,MO,T,T2}
  @show p
  mapin_pl = paramlength(m.mapin)
  mapout_pl = paramlength(m.mapout)
  cell_pl = paramlength(m.cell)
  p_mapin  = @view p[1 : mapin_pl]
  p_mapout = @view p[mapin_pl + 1 : mapin_pl + mapout_pl]
  p_cell   = @view p[mapin_pl + mapout_pl + 1 : end]

  x = m.mapin(x, p_mapin)
  m.state, y = m.cell(m.state, x, p_cell)
  m.mapout(y, p_mapout)
end

reset!(m::LTCNet) = (m.state = m.cell.state0)
reset_state!(m::LTCNet, p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))
reset_state!(m::DiffEqFlux.FastChain, p) = map(l -> reset_state!(l,p), m.layers)
reset_state!(m,p) = nothing

initial_params(m::LTCNet) = m.initial_params()
paramlength(m::LTCNet) = m.paramlength

Base.show(io::IO, m::LTCNet) = print(io, "LTCNet(", m.mapin, ",", m.mapout, ",", m.cell, ")")
