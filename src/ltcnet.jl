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
  Wl = length(m.W)
  W = @view p[1 : Wl]
  b = @view p[Wl + 1 : end]
  W .* x .+ b
end
Base.show(io::IO, m::Mapper) = print(io, "Mapper(", length(m.W), ")")

initial_params(m::Mapper) = m.initial_params()
paramlength(m::Mapper) = m.paramlength

import Base: +, -
struct ZeroSynapse <: CurrentComponent end
ZeroSynapse(args...) = ZeroSynapse()
(m::ZeroSynapse)(h,x,p=nothing) = zeros(eltype(x),size(x,1))#fill(Zeros(), size(x,1))##Flux.Zeros()#zeros(I,size(h,2))
+(::Zeros, b::Real) = b
+(a::Real, ::Zeros) = a
-(::Zeros, b::Real) = -b
-(a::Real, ::Zeros) = -a
initial_params(m::ZeroSynapse) = Float32[]
paramlength(m::ZeroSynapse) = 0

struct SigmoidSynapse{V,F<:Function} <: CurrentComponent
  μ::V
  σ::V
  G::V
  E::V
  initial_params::F
end
function SigmoidSynapse(;
             μr=Float32.((0.3, 0.8)), σr=Float32.((3, 8)), Gr=Float32.((0.001, 1)), Er=Float32.((-0.3, 0.3)))
  μ = rand_uniform(eltype(μr),μr..., 1, 1)[1]#[1,:]
  σ = rand_uniform(eltype(σr),σr..., 1, 1)[1]#[1,:]
  G = rand_uniform(eltype(Gr),Gr..., 1, 1)[1]#[1,:]
  E = rand_uniform(eltype(Er),Er..., 1, 1)[1]#[1,:]
  p = vcat(μ,σ,G,E)
  initial_params() = p
  SigmoidSynapse(μ,σ,G,E,initial_params)
end
function (m::SigmoidSynapse{V})(h::AbstractVecOrMat, x::AbstractVecOrMat{T}, p) where {V,T}
  μ, σ, G, E = p
  @. G * sigmoid((x - μ) * σ) * (h - E)
end

Base.show(io::IO, m::SigmoidSynapse) = print(io, "SigmoidSynapse")
initial_params(m::SigmoidSynapse) = m.initial_params()
paramlength(m::SigmoidSynapse) = 4

struct LeakChannel{V,F<:Function} <: CurrentComponent
  G::V
  E::V
  initial_params::F
  paramlength::Int
end
function LeakChannel(n::Int; Gr=Float32.((0.001,1)), Er=Float32.((-0.3,0.3)))
  G = rand_uniform(eltype(Gr),Gr..., n)
  E = rand_uniform(eltype(Er),Er..., n)
  p = vcat(G,E)
  initial_params() = p
  LeakChannel(G, E, initial_params, length(p))
end
function (m::LeakChannel{V})(h::AbstractVecOrMat{T}, p) where {V,T}
  Gl = Int(size(p,1) / 2)
  G = @view p[1:Gl]
  E = @view p[Gl+1:end]
  @. G * (h - E)
end
function (m::LeakChannel{V})(h::AbstractVecOrMat{T}, p, out) where {V,T}
  Gl = Int(size(p,1) / 2)
  G = @view p[1:Gl]
  E = @view p[Gl+1:end]
  for n in 1:size(h,1)
    out[n,:] += G[n] .* (h[n,:] .- E[n])
  end
end
Base.show(io::IO, m::LeakChannel) = print(io, "LeakChannel(", size(m.G,1), ")")
initial_params(m::LeakChannel) = m.initial_params()
paramlength(m::LeakChannel) = m.paramlength


struct ComponentContainer{C,F<:Function}
  components::C
  initial_params::F
  paramlength::Int
end
function ComponentContainer(components)
  p = reduce(vcat, DiffEqFlux.initial_params.(components))
  initial_params() = p
  ComponentContainer(components, initial_params, length(p))
end
# (m::ComponentContainer{C})(x::AbstractVecOrMat, I::AbstractVecOrMat) where C = doComponentContainer(m.components, x, I)
# doComponentContainer(components, x, I) = reshape(mapreduce(dst -> mapreduce(src -> components[src,dst](x[dst,:],I[src,:]), +, 1:size(I,1)), vcat, 1:size(x,1)), size(x))
# Flux.@functor ComponentContainer
function (m::ComponentContainer{C})(x::AbstractVecOrMat, I::AbstractVecOrMat, p) where C
  out = Zygote.Buffer(x, size(x,1), size(x,2))
  for i in eachindex(out)
    out[i] = 0
  end
  comps = m.components
  i = 1
  for dst in 1:size(x,1)

    buf = Zygote.Buffer(x, size(comps,1), size(x,2))
    for j in eachindex(buf)
      buf[j] = 0
    end

    for src in 1:size(comps,1)
      c = comps[src,dst]
      cpl = paramlength(c)
      p_comp = @view p[i:i-1+cpl]
      c_out = c(x[dst,:],I[src,:],p_comp)
      buf[src,:] = c_out
      i = i + cpl
    end

    bb = copy(buf)
    out[dst,:] += reshape(sum(bb,dims=1),:)
  end
  return copy(out)
end






function (m::ComponentContainer{C})(x::AbstractVecOrMat, I::AbstractVecOrMat, p, out) where C
  # out = Zygote.Buffer(x, size(x,1), size(x,2))
  # for i in eachindex(out)
  #   out[i] = 0
  # end
  comps = m.components
  i = 1
  for dst in 1:size(x,1)

    buf = Zygote.Buffer(x, size(comps,1), size(x,2))
    for j in eachindex(buf)
      buf[j] = 0
    end

    for src in 1:size(comps,1)
      c = comps[src,dst]
      cpl = paramlength(c)
      p_comp = @view p[i:i-1+cpl]
      c_out = c(x[dst,:],I[src,:],p_comp)
      buf[src,:] = c_out
      i = i + cpl
    end

    bb = copy(buf)
    out[dst,:] += reshape(sum(bb,dims=1), :)
  end
  #return copy(out)
end



initial_params(m::ComponentContainer) = m.initial_params()
paramlength(m::ComponentContainer) = m.paramlength


struct LTCCell{W<:Wiring,SE<:ComponentContainer,SY<:ComponentContainer,LE<:LeakChannel,V<:AbstractArray,S<:AbstractMatrix,SOLVER,SENSEALG,F<:Function}
  wiring::W
  sens::SE
  syns::SY
  leaks::LE
  # cc::CC
  # cc_p::CCP
  # cc_re::CCRE
  cm::V
  state0::S
  solver::SOLVER
  sensealg::SENSEALG
  initial_params::F
  paramlength::Int
end


function LTCCell(wiring, solver, sensealg; cmr=Float32.((1.6,2.5)), state0r=Float32.((0.01)))
  n_in = wiring.n_in
  out = wiring.out
  n_total = wiring.n_total
  cm    = rand_uniform(eltype(cmr),cmr..., n_total)
  state0 = fill(state0r[1], n_total, 1)

  sens = Union{ZeroSynapse,SigmoidSynapse}[]
  syns = Union{ZeroSynapse,SigmoidSynapse}[]
  # sens = SigmoidSynapse[]
  # syns = SigmoidSynapse[]
  # sens = []
  # syns = []

  for dst in 1:size(wiring.sens_mask,2)
    for src in 1:size(wiring.sens_mask,1)
      toadd = wiring.sens_mask[src,dst]
      s = toadd == 0 ? ZeroSynapse() : SigmoidSynapse()
      push!(sens, s)
    end
  end
  for dst in 1:size(wiring.syn_mask,2)
    for src in 1:size(wiring.syn_mask,1)
      toadd = wiring.syn_mask[src,dst]
      s = toadd == 0 ? ZeroSynapse() : SigmoidSynapse()
      push!(syns, s)
    end
  end

  sens = reshape(sens, size(wiring.sens_mask))
  syns = reshape(syns, size(wiring.syn_mask))

  #sens = StructArray(sens)
  #syns = StructArray(syns)

  sensc = ComponentContainer(sens)
  synsc = ComponentContainer(syns)
  leaks = LeakChannel(n_total)

  # cc = Flux.Parallel(+, sensc, synsc, leaks)
  # cc_p, cc_re = Flux.destructure(cc)

  p = vcat(DiffEqFlux.initial_params(sensc),DiffEqFlux.initial_params(synsc),DiffEqFlux.initial_params(leaks),cm,vec(state0))
  #p = vcat(DiffEqFlux.initial_params(sensc),DiffEqFlux.initial_params(synsc),DiffEqFlux.initial_params(leaks),cm)
  initial_params() = p

  LTCCell(wiring, sensc, synsc, leaks, cm, state0, solver, sensealg, initial_params, length(p))
end
Base.show(io::IO, m::LTCCell) = print(io, "LTCCell(", m.wiring.n_sensory, ",", m.wiring.n_inter, ",", m.wiring.n_command, ",", m.wiring.n_motor, ")")
initial_params(m::LTCCell) = m.initial_params()
paramlength(m::LTCCell) = m.paramlength

function (m::LTCCell)(h::AbstractVecOrMat, x::AbstractVecOrMat, p)
  h = repeat(h, 1, size(x,2)-size(h,2)+1)
  p_ode = @view p[1:end-length(m.state0)]
  # p_ode = p
  h = solve_ode(m,h,x, p_ode)
  h, h
end

function solve_ode(m,h,x,p)#::Matrix{Float32} where T

  function dltcdt!(dh,h,p,t)
    #cc_p = @view p[1 : cc_p_l]
    sens_p  = @view p[1 : sens_pl]
    syns_p  = @view p[sens_pl + 1 : sens_pl + syns_pl]
    leaks_p = @view p[sens_pl + syns_pl + 1 : sens_pl + syns_pl + leaks_pl]
    cm      = @view p[sens_pl + syns_pl + leaks_pl + 1 : end]
    #cc = cc_re(cc_p)
    #argsv = ((h,x), (h,h), (h,h))
    #I_components = mapreduce(i -> cc.layers[i](argsv[i][1],argsv[i][2]), cc.connection, 1:length(cc.layers))

    # out = Zygote.Buffer(h, size(h,1), size(h,2))
    # for i in eachindex(out)
    #   out[i] = 0
    # end
    # m.sens(h, x, sens_p, out)
    # m.syns(h, h, syns_p, out)
    # m.leaks(h, leaks_p, out)
    # I_components = copy(out)

    I_components = m.sens(h,x,sens_p) .+ m.syns(h,h,syns_p) .+ m.leaks(h,leaks_p)
    @. dh = - cm * I_components
    nothing
  end
  function oop(h,p,t)
    dh = similar(h)
    dltcdt!(dh,h,p,t)
    dh
  end

  sens_pl = paramlength(m.sens)
  syns_pl = paramlength(m.syns)
  leaks_pl = paramlength(m.leaks)

  prob = ODEProblem{true}(dltcdt!,h,Float32.((0,1)),p)
  # prob = ODEProblem{false}(oop,h,Float32.((0,1)),p)

  # de = ModelingToolkit.modelingtoolkitize(prob)
  # jac = eval(ModelingToolkit.generate_jacobian(de)[2])
  # f = ODEFunction((dh,h,p,t)->dltcdt!(dh,h,p,t,x,cc_p_l), jac=jac)
  # prob_jac = ODEProblem(f,h,tspan,p)

  solve(prob,m.solver; sensealg=m.sensealg, save_everystep=false, save_start=false, abstol=1e-2, reltol=1e-2)[:,:,end]
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

  p = vcat(DiffEqFlux.initial_params(mapin), DiffEqFlux.initial_params(cell), DiffEqFlux.initial_params(mapout))
  initial_params() = p

  LTCNet(mapin,mapout,cell,cell.state0,initial_params,length(p))
end


function (m::LTCNet{MI,MO,T,<:AbstractMatrix{T2}})(x::AbstractVecOrMat{T2}, p) where {MI,MO,T,T2}
  mapin_pl = paramlength(m.mapin)
  cell_pl = paramlength(m.cell)
  mapout_pl = paramlength(m.mapout)
  p_mapin  = @view p[1 : mapin_pl]
  p_cell   = @view p[mapin_pl + 1 : mapin_pl + cell_pl]
  p_mapout = @view p[mapin_pl + cell_pl + 1 : mapin_pl + cell_pl + mapout_pl]

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



get_bounds(m::Function) = Float32[],Float32[]
get_bounds(m::Flux.Chain) = [reduce(vcat, [get_bounds(l)[1] for l in m.layers]), reduce(vcat, [get_bounds(l)[2] for l in m.layers])]
function get_bounds(m::Flux.Dense)
  lb = [[-20.0 for _ in 1:length(m.weight)]...,
        [-20.0 for _ in 1:length(m.bias)]...] |> f32
  ub = [[20.0 for _ in 1:length(m.weight)]...,
        [20.0 for _ in 1:length(m.bias)]...] |> f32
  return lb, ub
end

function get_bounds(m::DiffEqFlux.FastChain)
  lb = vcat([get_bounds(layer)[1] for layer in m.layers]...)
  ub = vcat([get_bounds(layer)[2] for layer in m.layers]...)
  return lb, ub
end

function get_bounds(m::DiffEqFlux.FastDense)
  lb = [[-2.0 for _ in 1:m.out*m.in]...,
        [-2.0 for _ in 1:m.out]...] |> f32
  ub = [[2.0 for _ in 1:m.out*m.in]...,
        [2.0 for _ in 1:m.out]...] |> f32
  return lb, ub
end

get_bounds(m::ZeroSynapse) = Float32[], Float32[]

function get_bounds(m::Mapper)
  lb = [[-20.1 for i in 1:length(m.W)]...,
        [-20.1 for i in 1:length(m.b)]...] |> f32

  ub = [[20.1 for i in 1:length(m.W)]...,
        [20.1 for i in 1:length(m.b)]...] |> f32
  lb, ub
end
function get_bounds(m::LTCCell)
  lb = [
    reduce(vcat,[get_bounds(s)[1] for s in m.sens.components])...,
    reduce(vcat,[get_bounds(s)[1] for s in m.syns.components])...,
    get_bounds(m.leaks)[1]...,
    [1 for _ in 1:length(m.cm)]...,
    [-20 for _ in 1:length(m.state0)]...,
  ]
  ub = [
    reduce(vcat,[get_bounds(s)[2] for s in m.sens.components])...,
    reduce(vcat,[get_bounds(s)[2] for s in m.syns.components])...,
    get_bounds(m.leaks)[2]...,
    [3 for _ in 1:length(m.cm)]...,
    [20 for _ in 1:length(m.state0)]...,
  ]
  lb, ub
end
function get_bounds(m::LeakChannel)
  lb = [
    [0 for _ in 1:length(m.G)]...,
    [-2 for _ in 1:length(m.E)]...,
  ] |> f32
  ub = [
    [2 for _ in 1:length(m.G)]...,
    [2 for _ in 1:length(m.E)]...,
  ] |> f32
  lb, ub
end
function get_bounds(m::SigmoidSynapse)
  lb = [0.1, 1, 0.001, -1] |> f32
  ub = [0.9, 10, 1, 1] |> f32
  lb, ub
end


function get_bounds(m::LTCNet)
  lb = [
      get_bounds(m.mapin)[1]...,
      get_bounds(m.cell)[1]...,
      get_bounds(m.mapout)[1]...,
  ]
  ub = [
    get_bounds(m.mapin)[2]...,
    get_bounds(m.cell)[2]...,
    get_bounds(m.mapout)[2]...,
  ]
  lb, ub
end
