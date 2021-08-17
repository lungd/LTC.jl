const AbstractFluxLayer = Union{Flux.Recur,
                                Flux.Conv,
                                Flux.ConvTranspose,
                                Flux.DepthwiseConv,
                                Flux.CrossCor,
                                Flux.Dropout,
                                Flux.AlphaDropout,
                                Flux.LayerNorm,
                                Flux.BatchNorm,
                                Flux.InstanceNorm,
                                Flux.GroupNorm,
                                Flux.Embedding,
                                Flux.Diagonal,
                                Flux.Bilinear,
                                Flux.Parallel,}


struct Mapper{V,F}
  W::V
  b::V
  p::V
  σ::F
  paramlength::Int

  function Mapper(W, b, p, σ, paramlength)
    new{typeof(W),typeof(σ)}(W, b, p, σ, paramlength)
  end
end
MapperIn(wiring::Wiring{<:AbstractFloat},σ::Function=identity) = Mapper(wiring, wiring.n_in,σ)
MapperOut(wiring::Wiring{<:AbstractFloat},σ::Function=identity) = Mapper(wiring, wiring.n_out,σ)
function Mapper(wiring::Wiring{T}, n::Integer,σ::Function=identity; init=dims->ones(T,dims...), bias=dims->zeros(T,dims...)) where T
  W = init(n)
  b = bias(n) #.+ T(1e-5) # scaling needs initial guess != 0
  p = vcat(W,b)::Vector{T}
  Mapper(W, b, p, σ, length(p))
end
function (m::Mapper{<:AbstractArray{T},F})(x::AbstractArray{T}, p=m.p) where {T,F}
  Wl = size(m.W,1)
  W = @view p[1 : Wl]
  b = @view p[Wl + 1 : end]
  m.σ.(W .* x .+ b)
end
(m::Mapper{<:AbstractMatrix{T},F})(x::AbstractArray{T}, p=m.p) where {T,F} = reshape(m(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
Base.show(io::IO, m::Mapper) = print(io, "Mapper(", length(m.W), ", ", m.σ, ")")
initial_params(m::Mapper) = m.p
paramlength(m::Mapper) = m.paramlength
Flux.@functor Mapper (p,)
Flux.trainable(m::Mapper) = (m.p,)
function get_bounds(m::Mapper{<:AbstractArray{T},F}, ::DataType=nothing) where {T,F}
  lb = T[]
  ub = T[]
  for _ in 1:length(m.W)
    push!(lb, -10.1)
    push!(ub, 10.1)
  end
  for _ in 1:length(m.b)
    push!(lb, -10.1)
    push!(ub, 10.1)
  end
  lb, ub
end



struct FluxLayerWrapper{FL,P,RE}
  layer::FL
  p::P
  re::RE
  paramlength::Int

  FluxLayerWrapper(layer,p,re,paramlength) = new{typeof(layer),typeof(p),typeof(re)}(layer,p,re,paramlength)
end
function FluxLayerWrapper(layer, T::DataType=Float32) #where T <: AbstractFloat
  p::Vector{T}, re = LTC.destructure(layer)
  FluxLayerWrapper(layer, p, re, length(p))
end
(m::FluxLayerWrapper)(x, p) = m.re(p)(x)
Base.show(io::IO, m::FluxLayerWrapper) = print(io, "FluxLayerWrapper(", m.layer, ")")
initial_params(m::FluxLayerWrapper) = m.p
paramlength(m::FluxLayerWrapper) = m.paramlength
get_bounds(m::FluxLayerWrapper, T::DataType=Float32) = get_bounds(m.layer, T)
reset_state!(m::FluxLayerWrapper, p) = reset_state!(m.layer, p)


# WIP
struct Broadcaster{M,P}
  model::M
  p::P
  paramlength::Int
end
function Broadcaster(model)
  p = initial_params(model)
  paramlength = length(p)
  Broadcaster(model,p,paramlength)
end
(m::Broadcaster)(xs, p) = [m.model(x, p) for x in xs] # mapfoldl(x -> m.model(x, p), vcat, xs)
Base.show(io::IO, m::Broadcaster) = print(io, "Broadcaster(", m.model, ")")
initial_params(m::Broadcaster) = m.p
paramlength(m::Broadcaster) = m.paramlength
# Flux.@functor Broadcaster #(model,)
Flux.trainable(m::Broadcaster) = (m.model,)
# get_bounds(m::Broadcaster; T=Float32) = get_bounds(m.model; T)


# paramlength() needed for Flux layers
paramlength(m::Flux.Dense) = length(m.weight) + length(m.bias)
paramlength(m::Union{Flux.Chain,FastChain}) = sum([paramlength(l) for l in m.layers])
paramlength(m::AbstractFluxLayer) = length(Flux.destructure(m)[1])

get_bounds(l, T::DataType) = T[], T[] # For anonymous functions as layer
function get_bounds(m::Union{Flux.Chain, FastChain}, T::DataType)
  lb = vcat([get_bounds(layer, T)[1] for layer in m.layers]...)
  ub = vcat([get_bounds(layer, T)[2] for layer in m.layers]...)
  # lb = reduce(vcat, [get_bounds(layer)[1] for layer in m.layers])
  # ub = reduce(vcat, [get_bounds(layer)[2] for layer in m.layers])
  lb, ub
end
function get_bounds(m::FastDense{F,<:AbstractMatrix{T}}, ::DataType=nothing) where {F,T}
  # T = eltype(m.initial_params())
  lb = T[]
  ub = T[]
  for _ in 1:m.out*m.in # weights
    push!(lb, -10.1)
    push!(ub, 10.1)
  end
  if m.bias
    for _ in 1:m.out
      push!(lb, -10.1)
      push!(ub, 10.1)
    end
  end
  lb, ub
end

function get_bounds(m::Flux.Dense{F, <:AbstractMatrix{T}, B}, ::DataType=nothing) where {F,T,B}
  lb = T[]
  ub = T[]
  for _ in 1:length(m.weight)
    push!(lb, -10.1)
    push!(ub, 10.1)
  end
  for _ in 1:length(m.bias)
    push!(lb, -10.1)
    push!(ub, 10.1)
  end
  lb, ub
end


# TODO: overload reset!
reset_state!(m,p) = nothing

function reset_state!(m::Union{Flux.Chain, FastChain}, p)
  start_idx = 1
  for l in m.layers
    pl = paramlength(l)
    p_layer = p[start_idx:start_idx+pl-1]
    reset_state!(l, p_layer)
    start_idx += pl
  end
end




# https://github.com/SciML/DiffEqFlux.jl/issues/432#issuecomment-708079051
function destructure(m; cache = IdDict())
  xs = GalacticOptim.Zygote.Buffer([])
  Flux.fmap(m) do x
    if x isa AbstractArray
      push!(xs, x)
    else
      cache[x] = x
    end
    return x
  end
  return vcat(vec.(copy(xs))...), p -> _restructure(m, p, cache = cache)
end
function _restructure(m, xs; cache = IdDict())
  i = 0
  Flux.fmap(m) do x
    x isa AbstractArray || return cache[x]
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
end

# GalacticOptim.Zygote.@adjoint function (f::Mapper)(x,p)
#   n_in = size(f.W,1)
#   W = @view p[1:n_in]
#   b = @view p[n_in+1:end]
#   r = W .* x .+ b
#   # ifgpufree(b)
#
#   y = f.σ.(r)
#
#   function Mapper_adjoint(ȳ)
#     if typeof(f.σ) <: typeof(tanh)
#       zbar = ȳ .* (1 .- y.^2)
#     elseif typeof(f.σ) <: typeof(identity)
#       zbar = ȳ
#     else
#       zbar = ȳ .* ForwardDiff.derivative.(f.σ,r)
#     end
#     Wbar = zbar * x'
#     bbar = zbar
#     xbar = W' * zbar
#     pbar = if f.bias == true
#         tmp = typeof(bbar) <: AbstractVector ?
#                          vec(vcat(vec(Wbar),bbar)) :
#                          vec(vcat(vec(Wbar),sum(bbar,dims=2)))
#         # ifgpufree(bbar);
#         tmp
#     else
#         vec(Wbar)
#     end
#     ifgpufree(Wbar); ifgpufree(r)
#     nothing,xbar,pbar
#   end
#   y,Mapper_adjoint
# end
