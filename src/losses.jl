function loss_seq(p, re, x, y)
  m = re(p)
  LTC.reset_state!(m, p) # use initial conditions from current params

  ŷb = Flux.Zygote.Buffer([y[1]], size(y,1))
  for i in 1:size(x,1)
    xi = x[i]
    ŷi = m(xi)
    Inf32 ∈ ŷi && return Inf32, copy(ŷb), y # TODO: what if a layer after MTKRecur can handle Infs?
    ŷb[i] = ŷi
  end
  ŷ = copy(ŷb)

  # mean(sum.(abs2, (ŷ .- y))), ŷ, y
  return mean(Flux.Losses.mse.(ŷ,y, agg=mean)), ŷ, y
end

function loss_seq(p, m::FastChain, x, y)
  # ŷ = m.(x, [p])

  LTC.reset_state!(m, p)

  ŷb = Flux.Zygote.Buffer([y[1]], size(y,1))
  for i in 1:size(x,1)
    xi = x[i]
    ŷi = m(xi, p)
    Inf32 ∈ ŷi && return Inf32, copy(ŷb), y # TODO: what if a layer after MTKRecur can handle Infs?
    ŷb[i] = ŷi
  end
  ŷ = copy(ŷb)

  # mean(sum.(abs2, (ŷ .- y))), ŷ, y
  return mean(Flux.Losses.mse.(ŷ,y, agg=mean)), ŷ, y
end


function loss_full_seq(p, m::FastChain, x, y)
  # ŷ = m.(x, [p])

  LTC.reset_state!(m, p)

  ŷ = m(x, p)
  Inf32 ∈ ŷ && return Inf32, ŷ, y

  # mean(sum.(abs2, (ŷ .- y))), ŷ, y
  return mean(Flux.Losses.mse.(ŷ,y, agg=mean)), ŷ, y
end
