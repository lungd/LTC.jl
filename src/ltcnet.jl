abstract type Component end
abstract type CurrentComponent <:Component end

struct Mapper{V<:AbstractVector,B}
  W::V
  b::B
  Mapper(W::V,b) where {V<:AbstractVector} = new{V,typeof(b)}(W,b)
end
Mapper(in::Integer) = Mapper(ones(Float32,in), zeros(Float32,in))
(m::Mapper)(x::AbstractVecOrMat) = m.W .* x .+ m.b
Flux.@functor Mapper
Flux.trainable(m::Mapper) = (m.W, m.b,)
Base.show(io::IO, m::Mapper) = print(io, "Mapper(", length(m.W), ")")


import Base: +, -
struct ZeroSynapse <: CurrentComponent end
ZeroSynapse(args...) = ZeroSynapse()
(m::ZeroSynapse)(h,x) = fill(Flux.Zeros(), size(x,1))#zeros(eltype(x),size(x,1))#Flux.Zeros()#zeros(I,size(h,2))
Flux.@functor ZeroSynapse
Flux.trainable(m::ZeroSynapse) = Float32[]
+(::Flux.Zeros, b::Real) = b
+(a::Real, ::Flux.Zeros) = a

-(::Flux.Zeros, b::Real) = -b
-(a::Real, ::Flux.Zeros) = -a


struct SigmoidSynapse{V} <: CurrentComponent
  μ::V
  σ::V
  G::V
  E::V
end
function SigmoidSynapse(;
             μr=Float32.((0.3, 0.8)), σr=Float32.((3, 8)), Gr=Float32.((0.001, 1)), Er=Float32.((-0.3, 0.3)))
  μ = rand_uniform(eltype(μr),μr..., 1, 1)[1,:]
  σ = rand_uniform(eltype(σr),σr..., 1, 1)[1,:]
  G = rand_uniform(eltype(Gr),Gr..., 1, 1)[1,:]
  E = rand_uniform(eltype(Er),Er..., 1, 1)[1,:]
  SigmoidSynapse(μ,σ,G,E)
end
function (m::SigmoidSynapse{V})(h::AbstractVecOrMat, x::AbstractVecOrMat{T}) where {V,T}
  #@show size(@fastmath @. m.G * sigmoid((x - m.μ) * m.σ) * (h - m.E))    
  @fastmath @. m.G * sigmoid((x - m.μ) * m.σ) * (h - m.E)
end
Flux.@functor SigmoidSynapse
Flux.trainable(m::SigmoidSynapse) = (m.μ, m.σ, m.G, m.E,)
Base.show(io::IO, m::SigmoidSynapse) = print(io, "SigmoidSynapse")


struct LeakChannel{V} <: CurrentComponent
  G::V
  E::V
end
LeakChannel(n::Int; Gr=Float32.((0.001,1)), Er=Float32.((-0.3,0.3))) =
  LeakChannel(rand_uniform(eltype(Gr),Gr..., n), rand_uniform(eltype(Er),Er..., n))
(m::LeakChannel{V})(h::AbstractVecOrMat{T}) where {V,T} = @fastmath m.G .* (h .- m.E)
Flux.@functor LeakChannel
Flux.trainable(m::LeakChannel) = (m.G, m.E,)
Base.show(io::IO, m::LeakChannel) = print(io, "LeakChannel(", size(m.G,1), ")")


struct ComponentContainer{C}
   components::C
 end
(m::ComponentContainer{C})(x::AbstractVecOrMat, I::AbstractVecOrMat) where C = doComponentContainer(m.components, x, I)
doComponentContainer(components, x, I) = reshape(mapreduce(dst -> mapreduce(src -> components[src,dst](x[dst,:],I[src,:]), +, 1:size(I,1)), vcat, 1:size(x,1)), size(x))
Flux.@functor ComponentContainer


struct LTCCell{W<:Wiring,SE<:ComponentContainer,SY<:ComponentContainer,LE<:LeakChannel,CC<:Flux.Parallel,CCP<:AbstractArray,CCRE<:Function,V<:AbstractArray,S<:AbstractMatrix,SOLVER,SENSEALG}
  wiring::W
  sens::SE
  syns::SY
  leaks::LE
  cc::CC
  cc_p::CCP
  cc_re::CCRE
  cm::V
  state0::S
  solver::SOLVER
  sensealg::SENSEALG
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

  # sens = StructArray(sens)
  # syns = StructArray(syns)

  sensc = ComponentContainer(sens)
  synsc = ComponentContainer(syns)
  leaks = LeakChannel(n_total)

  cc = Flux.Parallel(+, sensc, synsc, leaks)
  cc_p, cc_re = Flux.destructure(cc)

  LTCCell(wiring, sensc, synsc, leaks, cc, cc_p, cc_re, cm, state0, solver, sensealg)
end
Flux.@functor LTCCell
Flux.trainable(m::LTCCell) = (m.cc_p, m.cm, m.state0,)
Base.show(io::IO, m::LTCCell) = print(io, "LTCCell(", m.wiring.n_sensory, ",", m.wiring.n_inter, ",", m.wiring.n_command, ",", m.wiring.n_motor, ")")
# TODO remove in v0.13
function Base.getproperty(m::LTCCell, sym::Symbol)
  if sym === :h
    Zygote.ignore() do
      @warn "LTCCell field :h has been deprecated. Use m::LTCCell.state0 instead."
    end
    return getfield(m, :state0)
  else
    return getfield(m, sym)
  end
end

function (m::LTCCell{W,SE,SY,LE,CC,CCP,CCRE,V,<:AbstractMatrix{T},SOLVER,SENSEALG})(h::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}) where {W,SE,SY,LE,CC,CCP,CCRE,V,T,SOLVER,SENSEALG}
  h = repeat(h, 1, size(x,2)-size(h,2)+1)::Matrix{T}
  h = solve_ode(m,h,x)
  h, h
end
function solve_ode(m,h::Matrix{T},x::Matrix)::Matrix{Float32} where T

  function dltcdt!(dh,h,p,t)
    cc_p = @view p[1 : cc_p_l]
    cm   = @view p[cc_p_l+1 : pl]
    cc = cc_re(cc_p)
    
    argsv = ((h,x), (h,h), (h,))

    I_components = mapreduce(i -> cc.layers[i](argsv[i]...), cc.connection, 1:length(cc.layers))
    @. dh = - cm * I_components
    nothing
  end
  function oop(h,p,t)
    dh = similar(h)
    dltcdt!(dh,h,p,t)
    dh
  end
  
  cc_p_l = size(m.cc_p,1)::Int
  cc_re = m.cc_re

  p = vcat(m.cc_p, m.cm)::Vector{Float32}
  pl = size(p,1)::Int
  prob = ODEProblem{true}(dltcdt!,h,Float32.((0,1)),p)
  # prob = ODEProblem{false}(oop,h,tspan,p)

  # de = ModelingToolkit.modelingtoolkitize(prob)
  # jac = eval(ModelingToolkit.generate_jacobian(de)[2])
  # f = ODEFunction((dh,h,p,t)->dltcdt!(dh,h,p,t,x,cc_p_l), jac=jac)
  # prob_jac = ODEProblem(f,h,tspan,p)

  solve(prob,m.solver; sensealg=m.sensealg, save_everystep=false, save_start=false, abstol=1e-3, reltol=1e-3)[:,:,end]
end




mutable struct LTCNet{MI<:Mapper,MO<:Mapper,T<:LTCCell,S}
  mapin::MI
  mapout::MO
  cell::T
  state::S
  LTCNet(mapin,mapout,cell,state) = new{typeof(mapin),typeof(mapout),typeof(cell),typeof(state)}(mapin,mapout,cell,state)
end
function LTCNet(wiring,solver,sensealg)
  mapin = Mapper(wiring.n_in)
  mapout = Mapper(wiring.n_total)
  cell = LTCCell(wiring,solver,sensealg)
  LTCNet(mapin,mapout,cell,cell.state0)
end

function (m::LTCNet{MI,MO,T,<:AbstractMatrix{T2}})(x::AbstractVecOrMat{T2}) where {MI,MO,T,T2}
  x = m.mapin(x)
  m.state, y = m.cell(m.state, x)
  y = m.mapout(y)
  return y
end

Flux.@functor LTCNet
Flux.trainable(m::LTCNet) = (m.mapin, m.mapout, m.cell,)
Flux.reset!(m::LTCNet) = (m.state = m.cell.state0)
Base.show(io::IO, m::LTCNet) = print(io, "LTCNet(", m.mapin, ",", m.mapout, ",", m.cell, ")")
function Base.getproperty(m::LTCNet, sym::Symbol)
  if sym === :init
    Zygote.ignore() do
      @warn "LTCNet field :init has been deprecated. To access initial state weights, use m::LTCNet.cell.state0 instead."
    end
    return getfield(m.cell, :state0)
  else
    return getfield(m, sym)
  end
end




function get_bouds(m::LeakChannel)
end
function get_bounds(m::SigmoidSynapse)
  [0.1, 1, 0.001, -1] |> f32, [0.9, 10, 1, 1] |> f32
end
function get_bounds(m::Mapper)
  lb = [[-20.1 for i in 1:length(m.W)]...,
        [-20.1 for i in 1:length(m.b)]...] |> f32

  ub = [[20.1 for i in 1:length(m.W)]...,
        [20.1 for i in 1:length(m.b)]...] |> f32
  return lb, ub
end

function get_bounds(m::LTCNet)
  lower = [
      get_bounds(m.mapin)[1]...,
      get_bounds(m.mapout)[1]...,
      reduce(vcat,[get_bounds(s)[1] for s in [m.cell.sens.components;m.cell.syns.components]])...,
      [0.01 for _ in m.cell.state0]...,            #
      [-2 for _ in m.cell.state0]...,
      [0.001 for _ in m.cell.cm]...,
      [-50 for _ in m.cell.state0]...,
  ] |> f32
  upper = [
    get_bounds(m.mapin)[2]...,
    get_bounds(m.mapout)[2]...,
    reduce(vcat,[get_bounds(s)[1] for s in [m.cell.sens.components;m.cell.syns.components]])...,
    [10.0 for _ in m.cell.state0]...,
    [-0.1 for _ in m.cell.state0]...,
    [5 for _ in m.cell.cm]...,
    [20 for _ in m.cell.state0]...,
  ] |> f32

  lower, upper
end

