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

function loss_seq(p, m::FastChain, _x, _y)
  # ŷ = m.(x, [p])
  # x = _x[1]
  # y = _y[1]

  x = _x
  y = _y

  LTC.reset_state!(m, p)

  ŷb = Flux.Zygote.Buffer([y[1]], size(y,1))
  # ŷb = Array{typeof(y[1])}(undef,size(y,1))
  for i in 1:size(x,1)
    xi = x[i]
    ŷi = m(xi, p)
    Inf32 ∈ ŷi && return Inf32, copy(ŷb), y # TODO: what if a layer after MTKRecur can handle Infs?
    # Inf32 ∈ ŷi && return Inf32, ŷb, y # TODO: what if a layer after MTKRecur can handle Infs?
    ŷb[i] = ŷi
  end
  ŷ = copy(ŷb)
  # ŷ = ŷb
  # @show size(ŷ)
  # @show size(ŷ[1])
  # mean(sum.(abs2, (ŷ .- y))), ŷ, y

  # ŷ_probs = [ŷ[i][1,1] for i in 1:length(ŷ)]
  # y_probs = [y[i][1,1] for i in 1:length(y)]
  # return Flux.Losses.logitbinarycrossentropy(ŷ_probs, y_probs, agg=mean), ŷ, y

  return mean(Flux.Losses.mse.(ŷ,y, agg=mean)), ŷ, y
  # return Flux.Losses.mse(ŷ[end],y[end], agg=mean), ŷ, y
end




function loss_seq_node(p, m::FastChain, _x, _y)
  # ŷ = m.(x, [p])
  # x = _x[1]
  # y = _y[1]

  # x = Flux.stack(_x, 2)
  x = _x
  y = _y

  LTC.reset_state!(m, p)

  _ŷ = m(x, p)

  Inf ∈ _ŷ && return Inf32, _ŷ, y
  NaN ∈ _ŷ && return Inf32, _ŷ, y

  # ŷ = Flux.unstack(_ŷ, 2)[2:end]
  ŷ = _ŷ#[2:end]

  # ŷ = ŷb
  # @show size(ŷ)
  # @show size(ŷ[1])
  # mean(sum.(abs2, (ŷ .- y))), ŷ, y

  # ŷ_probs = [ŷ[i][1,1] for i in 1:length(ŷ)]
  # y_probs = [y[i][1,1] for i in 1:length(y)]
  # return Flux.Losses.logitbinarycrossentropy(ŷ_probs, y_probs, agg=mean), ŷ, y

  return mean(Flux.Losses.mse.(Flux.unstack(ŷ,2),Flux.unstack(y,2), agg=mean)), ŷ, y
  # return Flux.Losses.mse(ŷ[end],y[end], agg=mean), ŷ, y
end


function loss_new(p, m::FastChain, _x, _y, tspan)
  # ŷ = m.(x, [p])

  LTC.reset_state!(m, p)

  y = _x[1]
  saveat = _y[1]
  # @show size(y)
  # @show size(y[1])
  # @show size(saveat)
  # @show saveat

  ŷb = Flux.Zygote.Buffer([y[1]], size(saveat,1))
  t = 1
  for i in tspan[1]:tspan[end]
    # @show i
    #xi = x[i]
    ŷi = m(nothing, p)
    Inf ∈ ŷi && return Inf, copy(ŷb), y # TODO: what if a layer after MTKRecur can handle Infs?
    i ∉ saveat && continue
    # @show i
    # @show t
    # @show size(ŷi)
    # @show size(ŷb)
    ŷb[t] = ŷi
    t += 1
  end
  ŷ = copy(ŷb)

  # mean(sum.(abs2, (ŷ .- y))), ŷ, y
  return mean(Flux.Losses.mse.(ŷ,y, agg=mean)), ŷ, y
end


function loss_full_seq(p, m::FastChain, y, saveat)
  # ŷ = m.(x, [p])

  LTC.reset_state!(m, p)
  # @show size(saveat)
  # @show "<LOSS>"
  # @show size(saveat)
  # @show size(saveat[1])
  # @show "</LOSS>"

  ŷ = m(saveat, p)
  # @show size(ŷ)
  # @show size(ŷ[1])
  # @show size(y)
  # @show size(y[1])
  Inf ∈ ŷ && return Inf, ŷ, y
  NaN ∈ ŷ && return Inf, ŷ, y

  # mean(sum.(abs2, (ŷ .- y))), ŷ, y
  return mean(Flux.Losses.mse.(ŷ,y, agg=mean)), ŷ, y
end
