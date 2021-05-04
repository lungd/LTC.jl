struct LTCCell{W,SENS,SYN,SENSP,SENSRE,SYNP,SYNRE,MI,MO,V,S,SOLVER,SENSEALG}

  wiring::W

  mapin::MI
  mapout::MO

  sens_f::SENS
  sens_p::SENSP
  sens_pl::Int
  sens_re::SENSRE
  syn_f::SYN
  syn_p::SYNP
  syn_pl::Int
  syn_re::SYNRE

  cm::V
  Gleak::V
  Eleak::V

  state0::S

  solver::SOLVER
  sensealg::SENSEALG


  function LTCCell(wiring, mapin, mapout, sens_f, sens_p, sens_pl, sens_re, syn_f, syn_p, syn_pl, syn_re, cm, Gleak, Eleak, state0, solver, sensealg)
    new{typeof(wiring),typeof(sens_f),typeof(syn_f),typeof(sens_p),typeof(sens_re),typeof(syn_p),typeof(syn_re),typeof(mapin),typeof(mapout),typeof(cm),typeof(state0),typeof(solver),typeof(sensealg)}(
      wiring,mapin,mapout,sens_f,sens_p,sens_pl,sens_re,syn_f,syn_p,syn_pl,syn_re,cm,Gleak,Eleak,state0,solver,sensealg)
  end
end


function LTCCell(wiring, solver, sensealg)
  n_in = wiring.n_in
  out = wiring.out
  n_total = wiring.n_total
  cm    = rand_uniform(1.6, 2.5, n_total)
  Gleak = rand_uniform(0.001, 1, n_total)
  Eleak = rand_uniform(-0.2, 0.2, n_total)
  state0 = fill(Float32(0.01), n_total, 1)

  mapin = Mapper(n_in)
  mapout = Mapper(n_total)

  sens_f = gNN(n_in, n_total)
  syn_f = gNN(n_total, n_total)
  sens_p, sens_re = Flux.destructure(sens_f)
  syn_p, syn_re = Flux.destructure(syn_f)

  LTCCell(wiring,mapin,mapout,sens_f,sens_p,length(sens_p),sens_re,syn_f,syn_p,length(syn_p),syn_re, cm,Gleak,Eleak,state0,solver,sensealg)
end



Flux.@functor LTCCell (mapin, mapout, sens_p, syn_p, cm, Gleak, Eleak, state0,)
Flux.trainable(m::LTCCell)=(Flux.trainable(m.mapin), Flux.trainable(m.mapout), m.sens_p, m.syn_p, m.cm, m.Gleak, m.Eleak, m.state0)

NCP(args...) = Flux.Recur(LTCCell(args...))
Flux.Recur(m::LTCCell) = Flux.Recur(m, m.state0)

#(m::LTCCell)(h::AbstractVector, x::AbstractVecOrMat) == m(repeat(h, 1, size(x,2)-size(h,2)+1), x)

function (m::LTCCell)(h::AbstractVecOrMat, x::AbstractVecOrMat)

  x = m.mapin(x)
  I = x#::Matrix{Float32}

  # TODO: sens_f() here?

  u0 = repeat(h, 1, size(x,2)-size(h,2)+1)

  # condition(u,t,integrator) = any(x -> (x < -100) || (x > 100), u)
  # affect!(integrator) = terminate!(integrator)
  # cb = DiscreteCallback(condition,affect!)

  # sens_f = m.sens_re(m.sens_p)
  # syn_f  = m.syn_re(m.syn_p)
  # p = [m.cm; m.Gleak; m.Eleak; m.sens_p; m.syn_p]::Vector{Float32}
  prob = ODEProblem{true}((dx,x,p,t)->dltcdt!(dx,x,p,t, m, I),u0,Float32.((0, 1)),[m.cm; m.Gleak; m.Eleak; m.sens_p; m.syn_p])
  #prob = ODEProblem((x,p,t)->oop(x,p,t, m, m.sens_f, m.syn_f, I),u0,Float32.((0, 1)),[m.cm; m.Gleak; m.Eleak; m.sens_p; m.syn_p])
  #prob = ODEProblem((x,p,t)->oop(x,p,t, m),u0,Float32.((0, 1)),[m.cm; m.Gleak; m.Eleak; m.sens_p; m.syn_p; x[:]])
  sol = solve(prob, m.solver; dt=0.1, adaptive=false, sensealg=m.sensealg, save_everystep=false, save_start=false)#, isoutofdomain=(u,p,t)->any(x -> (x < -1e40) || (x > 1e40), u))
  #solve(prob, m.solver, alias_u0=true, dtmax=1.0, sensealg=m.sensealg, save_everystep=false, save_start=false)#, isoutofdomain=(u,p,t)->any(x -> (x < -10) || (x > 10), u))
  #sol = solve(prob, m.solver; sensealg=m.sensealg, save_everystep=false, save_start=false)

  #hh = copy(u0)#[:,:]
  hh = sol[:,:,end]#::typeof(h)
  #h =
  out = m.mapout(hh)
  return hh, @view out[end-m.wiring.n_motor+1:end, :]
end




#function dltcdt!(dx,x,p,t, m, I::Matrix{Float32}, sens_f, syn_f)
function dltcdt!(dx,x,p,t, m, I)

  cm_pl = size(x,1)::Int
  Gleak_pl = size(x,1)::Int
  Eleak_pl = size(x,1)::Int
  sens_pl = m.sens_pl::Int
  syn_pl = m.syn_pl::Int

  cm    = @view p[1 : cm_pl]#::AbstractVecOrMat{eltype(x)}
  Gleak = @view p[cm_pl + 1 : cm_pl + Gleak_pl]#::AbstractVecOrMat{eltype(x)}
  Eleak = @view p[cm_pl + Gleak_pl + 1 : cm_pl + Gleak_pl + Eleak_pl]#::AbstractVecOrMat{eltype(x)}

  sens_p = @view p[cm_pl + Gleak_pl + Eleak_pl + 1 : cm_pl + Gleak_pl + Eleak_pl + sens_pl]#::AbstractVecOrMat{eltype(x)}
  syn_p = @view p[cm_pl + Gleak_pl + Eleak_pl + sens_pl + 1 : cm_pl + Gleak_pl + Eleak_pl + sens_pl + syn_pl]#::AbstractVecOrMat{eltype(x)}

  # I = reshape((@view p[cm_pl + Gleak_pl + Eleak_pl + sens_pl + syn_pl + 1 : end]), m.wiring.n_in, :)

  sens_f = m.sens_re(sens_p)
  syn_f  = m.syn_re(syn_p)

  #@unpack cm, Gleak, Eleak, sens_p, syn_p, I = p

  I_sens = sens_f(x, I, m.wiring.sens_mask, m.wiring.sens_pol)#::AbstractVecOrMat{eltype(x)}
  I_syn = syn_f(x, x, m.wiring.syn_mask, m.wiring.syn_pol)#::AbstractVecOrMat{eltype(x)}


  @. dx = cm * (-(Gleak * (x - Eleak)) + I_syn + I_sens)
  nothing
end


# function oop(x,p,t, m, I::Matrix{Float32})
function oop(x,p,t, m, sens_f, syn_f, I)

  cm_pl = size(x,1)::Int
  Gleak_pl = size(x,1)::Int
  Eleak_pl = size(x,1)::Int
  sens_pl = m.sens_pl::Int
  syn_pl = m.syn_pl::Int

  cm    = @view p[1 : cm_pl]#::AbstractVecOrMat{eltype(x)}
  Gleak = @view p[cm_pl + 1 : cm_pl + Gleak_pl]#::AbstractVecOrMat{eltype(x)}
  Eleak = @view p[cm_pl + Gleak_pl + 1 : cm_pl + Gleak_pl + Eleak_pl]#::AbstractVecOrMat{eltype(x)}

  sens_p = @view p[cm_pl + Gleak_pl + Eleak_pl + 1 : cm_pl + Gleak_pl + Eleak_pl + sens_pl]#::AbstractVecOrMat{eltype(x)}
  syn_p = @view p[cm_pl + Gleak_pl + Eleak_pl + sens_pl + 1 : cm_pl + Gleak_pl + Eleak_pl + sens_pl + syn_pl]#::AbstractVecOrMat{eltype(x)}

  # I = reshape(p[cm_pl + Gleak_pl + Eleak_pl + sens_pl + syn_pl + 1 : end], m.wiring.n_in, :)

  sens_f = m.sens_re(sens_p)
  syn_f  = m.syn_re(syn_p)
  #@unpack cm, Gleak, Eleak, sens_p, syn_p, I = p

  I_sens = sens_f(x, I, m.wiring.sens_mask, m.wiring.sens_pol)#::AbstractVecOrMat{eltype(x)}
  I_syn = syn_f(x, x, m.wiring.syn_mask, m.wiring.syn_pol)#::AbstractVecOrMat{eltype(x)}

  @. cm * (-(Gleak * (x - Eleak)) + I_syn + I_sens)
end



struct Mapper{V<:AbstractArray}
  W::V
  b::V
  Mapper(W::V,b::V) where {V<:AbstractArray} = new{V}(W,b)
end
Mapper(in::Integer) = Mapper(ones(Float32,in), zeros(Float32,in))

(m::Mapper)(x::AbstractVecOrMat) = m.W .* x .+ m.b

Flux.@functor Mapper



struct gNN{A<:AbstractArray}
  G::A
  μ::A
  σ::A
  E::A
  gNN(G::A, μ::A, σ::A, E::A) where {A<:AbstractArray} = new{A}(G, μ, σ, E)
end
function gNN(in::Integer, out::Integer)
  G = rand_uniform(0.001, 1, in, out)
  μ = rand_uniform(0.3, 0.8, in, out)
  σ = rand_uniform(3, 8, in, out)
  E = rand_uniform(-0.3, 0.3, in, out)
  gNN(G,μ,σ,E)
end


function (m::gNN)(h::AbstractVecOrMat, x::AbstractVecOrMat, bitmask::Matrix{Float32}, polmask::Matrix{Float32})
  G, μ, sig, E = m.G, m.μ, m.σ, m.E

  #fun4e(bitmask, G, E, x, polmask, h, μ, σ) = @tullio out[j] := bitmask[i,j] * G[i,j] * sigmoid((x[i] - μ[i,j]) * σ[i,j]) * (polmask[i,j] * E[i,j] - h[j])

  #fun1(bitmask, G, E, x, polmask, h, μ, σ) = reshape(reduce(vcat, [reshape(sum(bitmask .* G .* Flux.sigmoid.(((x[:,b]) .- μ) .* σ) .* (polmask .* E .- reshape((h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)]), size(h,1),:)
  function fun1(bitmask, G, E, x, polmask, h, μ, σ)
    out = reduce(hcat, [reshape(sum(bitmask .* G .* sigmoid.((x[:,b] .- μ) .* σ) .* ( polmask .* E .- reshape(h[:,b], 1,:)),dims=1),:) for b in 1:size(x,2)])
    return out
  end

  function fun2(bitmask, G, E, x, polmask, h, μ, σ)
    @views out = hcat([reshape(sum(bitmask .* G .* sigmoid.((x[:,b] .- μ) .* σ) .* (polmask .* E .- reshape(h[:,b], 1,:)),dims=1),:) for b in 1:size(x,2)]...)
    return out
  end

  # First function call produced NaNs. Exiting.
  function fun3(bitmask, G, E, x, polmask, h, μ, σ)
  	@views out = hcat([reshape(sum(bitmask .* G .* (1 ./ (1 .+ exp.(-((x[:,b] .- μ) .* σ)))) .* (polmask .* E .- reshape(h[:,b], 1,:)),dims=1),:) for b in 1:size(x,2)]...)
  	return out
  end


  # First function call produced NaNs. Exiting.
  # function fun4(bitmask, G, E, x, polmask, h, μ, σ)
  #   @tullio out[j,b] := bitmask[i,j] * G[i,j] * (1 / (1 + exp(-((x[i,b] - μ[i,j]) * σ[i,j])))) * (polmask[i,j] * E[i,j] - h[j,b]) #threads=false avx=false verbose=true
  #   return out
  # end
  # no gradient
  function fun5(bitmask, G, E, x, polmask, h, μ, sig)
    @tullio et[i,j,b] := (x[i,b] - μ[i,j]) * sig[i,j] verbose=true
    @tullio t[i,j,b] := exp(-abs(et[i,j,b])) verbose=true
    @tullio s[i,j,b] := ifelse(et[i,j,b] ≥ 0, inv(1 + t[i,j,b]), t[i,j,b] / (1 + t[i,j,b])) verbose=true

    @tullio out[j,b] := bitmask[i,j] * G[i,j] * s[i,j,b] * (polmask[i,j] * E[i,j] - h[j,b]) verbose=true #threads=false avx=false
    return out
  end
  function fun6(bitmask, G, E, x, polmask, h, μ, sig)
    @tullio out[j,b] := bitmask[i,j] * G[i,j] * sigmoid(-((x[i,b] - μ[i,j]) * sig[i,j])) * (polmask[i,j] * E[i,j] - h[j,b]) grad=Dual #threads=false avx=false verbose=true
    return out
  end

  return fun1(bitmask, G, E, x, polmask, h, μ, sig)

  # @views reshape(reduce(vcat, [reshape(sum(bitmask .* G .* Flux.sigmoid.(((x[:,b]) .- μ) .* sig) .* (polmask .* E .- reshape((h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)]), size(h,1),:)
  # Flux.stack([reshape(sum(bitmask .* G .* Flux.sigmoid.(((x[:,b]) .- μ) .* sig) .* (polmask .* E .- reshape((h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)], 2)
  #reshape(reduce(vcat, [reshape(sum(bitmask .* G .* Flux.sigmoid.(((@view x[:,b]) .- μ) .* σ) .* (polmask .* E .- reshape((@view h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)]), size(h,1),:)

  #Flux.stack([reshape(sum(bitmask .* G .* Flux.sigmoid.(((@view x[:,b]) .- μ) .* σ) .* (polmask .* E .- reshape((@view h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)], 2)
end






function (m::gNN)(p::AbstractVector, h::AbstractVecOrMat, x::AbstractVecOrMat, bitmask::Matrix{Float32}, polmask::Matrix{Float32})
  Gl = length(m.G)
  μl = length(m.μ)
  σl = length(m.σ)
  El = length(m.E)
  @views G = p[1 : Gl]
  @views μ = p[Gl + 1 : Gl + μl]
  @views σ = p[Gl + μl + 1 : Gl + μl + σl]
  @views E = p[Gl + μl + σl + 1 : Gl + μl + σl + El]

  G = reshape(G, size(m.G))
  μ = reshape(μ, size(m.μ))
  σ = reshape(σ, size(m.σ))
  E = reshape(E, size(m.E))

  #fun4e(bitmask, G, E, x, polmask, h, μ, σ) = @tullio out[j] := bitmask[i,j] * G[i,j] * sigmoid((x[i] - μ[i,j]) * σ[i,j]) * (polmask[i,j] * E[i,j] - h[j])

  #fun1(bitmask, G, E, x, polmask, h, μ, σ) = reshape(reduce(vcat, [reshape(sum(bitmask .* G .* Flux.sigmoid.(((x[:,b]) .- μ) .* σ) .* (polmask .* E .- reshape((h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)]), size(h,1),:)
  function fun1(bitmask, G, E, x, polmask, h, μ, σ)
    @views out = reduce(hcat, [reshape(sum(bitmask .* G .* sigmoid.((x[:,b] .- μ) .* σ) .* ( polmask .* E .- reshape(h[:,b], 1,:)),dims=1),:) for b in 1:size(x,2)])
    return out
  end

  return fun1(bitmask, G, E, x, polmask, h, μ, σ)

end






Flux.@functor gNN
