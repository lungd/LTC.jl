function get_model(re, p)
  m = re(p)
  LTC.reset_state!(m, p)
  (x, p) -> m(x)
end
function get_model(m::FastChain, p)
  LTC.reset_state!(m, p)
  (x, p) -> m(x, p)
end


function loss_seq(p, m, x, y)
  m = get_model(m,p)

  ŷb = Flux.Zygote.Buffer([y[1]], size(y,1))
  for i in 1:size(x,1)
    xi = x[i]
    ŷi = m(xi, p)
    Inf32 ∈ ŷi && return Inf32, copy(ŷb), y # TODO: what if a layer after MTKRecur can handle Infs?
    ŷb[i] = ŷi
  end
  ŷ = copy(ŷb)

  return mean(Flux.Losses.mse.(ŷ,y, agg=mean)), ŷ, y
end


function loss_seq_node(p, m, x, y)

  m = get_model(m,p)
  ŷ = m(x, p)

  Inf ∈ ŷ && return Inf32, ŷ, y
  NaN ∈ ŷ && return Inf32, ŷ, y

  ŷ = ndims(ŷ) < 3 ? Flux.stack(ŷ,2) : ŷ
  y = ndims(y) < 3 ? Flux.stack(y,2) : y

  return mean(Flux.Losses.mse.([ŷi for ŷi in Flux.unstack(ŷ,2)],[yi for yi in Flux.unstack(y,2)], agg=mean)), ŷ, y
end
