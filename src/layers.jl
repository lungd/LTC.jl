# https://github.com/SciML/DiffEqFlux.jl/issues/432#issuecomment-708079051
function destructure(m; cache = IdDict())
  xs = Zygote.Buffer([])
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

paramlength(m::Flux.Dense) = length(m.weight) + length(m.bias)



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
# function (m::Mapper{<:AbstractVector{T}})(x::Matrix{T}, p=m.p) where T
function (m::Mapper)(x, p=m.p)
  Wl = size(m.W,1)
  W = @view p[1 : Wl]
  b = @view p[Wl + 1 : end]
  W .* x .+ b
end
Base.show(io::IO, m::Mapper) = print(io, "Mapper(", length(m.W), ")")
initial_params(m::Mapper) = m.p
paramlength(m::Mapper) = m.paramlength

Flux.@functor Mapper (p,)
Flux.trainable(m::Mapper) = (m.p,)


struct FluxLayerWrapper{FL,P,RE}
  layer::FL
  p::P
  re::RE
  paramlength::Int
end
function FluxLayerWrapper(layer)
  p, re = Flux.destructure(layer)
  FluxLayerWrapper(layer, p, re, length(p))
end
(m::FluxLayerWrapper)(x, p) = m.re(p)(x)
Base.show(io::IO, m::FluxLayerWrapper) = print(io, "FluxLayerWrapper(", layer, ")")
initial_params(m::FluxLayerWrapper) = m.p
paramlength(m::FluxLayerWrapper) = m.paramlength


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

get_bounds(m) = Float32[],Float32[]
get_bounds(m::Broadcaster) = get_bounds(m.model)
get_bounds(m::FluxLayerWrapper) = get_bounds(m.layer)

function get_bounds(m::Mapper)
  lb = vcat(fill(-10.1, length(m.W)),
            fill(-10.1, length(m.b))) |> f32
  ub = vcat(fill(-10.1, length(m.W)),
            fill(-10.1, length(m.b))) |> f32
  lb, ub
end

function get_bounds(m::Union{Flux.Chain, FastChain})
  lb = vcat([get_bounds(layer)[1] for layer in m.layers]...)
  ub = vcat([get_bounds(layer)[2] for layer in m.layers]...)
  lb, ub
end

function get_bounds(m::FastDense)
  lb = vcat(fill(-10.1, m.out*m.in),
            fill(-10.1, m.out)) |> f32
  ub = vcat(fill(10.1, m.out*m.in),
            fill(10.1, m.out)) |> f32
  lb, ub
end

function get_bounds(m::Flux.Dense)
  lb = vcat(fill(-10.1, length(m.weight)),
            fill(-10.1, length(m.bias))) |> f32
  ub = vcat(fill(10.1, length(m.weight)),
            fill(10.1, length(m.bias))) |> f32
  lb, ub
end
