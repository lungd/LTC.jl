# init_ranges = Dict(
#   "cm" => (1, 1000),
#   "Gleak" => (0.001, 1.0),
#   "Eleak" => (-0.2, 0.2),
#   "G" => (0.001, 1.0),
#   "μ" => (0.3, 0.8),
#   "σ" => (3, 8),
#   "E" => (1, 1.00001), # pol_mask
# )

struct LTCCellE{W,SENS,SYN,SENSP,SENSRE,SYNP,SYNRE,MI,MO,V,S,SOLVER,SENSEALG}

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


  function LTCCellE(wiring, mapin, mapout, sens_f, sens_p, sens_pl, sens_re, syn_f, syn_p, syn_pl, syn_re, cm, Gleak, Eleak, state0, solver, sensealg)
    new{typeof(wiring),typeof(sens_f),typeof(syn_f),typeof(sens_p),typeof(sens_re),typeof(syn_p),typeof(syn_re),typeof(mapin),typeof(mapout),typeof(cm),typeof(state0),typeof(solver),typeof(sensealg)}(
      wiring,mapin,mapout,sens_f,sens_p,sens_pl,sens_re,syn_f,syn_p,syn_pl,syn_re,cm,Gleak,Eleak,state0,solver,sensealg)
  end
end


function LTCCellE(wiring, solver, sensealg)
  n_in = wiring.n_in
  out = wiring.out
  n_total = wiring.n_total
  # cm    = rand_uniform(init_ranges["cm"][1], init_ranges["cm"][2], n_total)
  # Gleak = rand_uniform(init_ranges["Gleak"][1], init_ranges["Gleak"][2], n_total)
  # Eleak = rand_uniform(init_ranges["Eleak"][1], init_ranges["Eleak"][2], n_total)
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

  LTCCellE(wiring,mapin,mapout,sens_f,sens_p,length(sens_p),sens_re,syn_f,syn_p,length(syn_p),syn_re, cm,Gleak,Eleak,state0,solver,sensealg)
end



Flux.@functor LTCCellE (mapin, mapout, sens_p, syn_p, cm, Gleak, Eleak, state0,)
#Flux.@functor LTCCellE (mapin, mapout, sens_p, syn_p, cm, Gleak, Eleak)
Flux.trainable(m::LTCCellE)=(Flux.trainable(m.mapin), Flux.trainable(m.mapout), m.sens_p, m.syn_p, m.cm, m.Gleak, m.Eleak, m.state0)

NCP(args...) = Flux.Recur(LTCCellE(args...))
Flux.Recur(m::LTCCellE) = Flux.Recur(m, m.state0)


#function (m::LTCCellE{W,SENS,SYN,SENSP,SENSRE,SYNP,SYNRE,MI,MO,V,<:AbstractArray{T}})(h::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}) where {W,SENS,SYN,SENSP,SENSRE,SYNP,SYNRE,MI,MO,V,T}
function (m::LTCCellE)(h, x)


  _p = vcat(m.cm, m.Gleak, m.Eleak, m.sens_p, m.syn_p)
  #_p = vcat(m.cm, m.Gleak, m.Eleak)

  function prob_func(prob,i,repeat)
    #f = (du,u,p,t) -> dudt!(du,u,p,t,i)
    Ii = @view I[:,i]
    hi = @view h[:,i]
    #f = ODEFunction((x,p,t)->oop(x,p,t, m))
    #f = ODEFunction((x,p,t)->oop(x,p,t, m, Ii, sens_f,syn_f))
    f = ODEFunction((dx, x,p,t)->densltcdt!(dx, x,p,t, m, m.sens_f, m.syn_f))
    remake(prob, f=f, p=[_p;Ii[:]], u0=hi)
  end
  # function output_func(sol,i)
  #
  #   #i == 1 && @show sol.t
  #
  #   #@show size(sol)
  #   #sol_b[:,1,:] .= sol[:,end]
  #   #sol[:,end], false
  #   out = [x[:,1] for x in sol]
  #   return (DiffEqArray(out,sol.t), false)
  # end


  x = m.mapin(x)
  h = repeat(h, 1, size(x,2)-size(h,2)+1)
  # @show size(h)

  # h_new = Flux.Zygote.Buffer(h, size(h))
  # for i in eachindex(h_new)
  #   h_new[i] = h[i]
  # end

  # h_new = Flux.Zygote.Buffer(h, size(h,1),size(h,2))
  # for i in 1:size(h,1)
  #   h_new[i,:] .= h[i,:]
  # end

  #sol_b = Flux.Zygote.Buffer(h, size(h,1), 1, size(h,2))

  I = x
  sens_f = m.sens_re(m.sens_p)
  syn_f  = m.syn_re(m.syn_p)
  prob = ODEProblem((dx,x,p,t)->densltcdt!(dx,x,p,t, m, m.sens_f, m.syn_f), h[:,1], Float32.((0, 1)), _p)
  #prob = ODEProblem(ODEFunction{false}((x,p,t)->oop(x,p,t, m)), h[:,1], Float32.((0, 1)), [_p;I[:,1][:]])
  ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
  #sol = Array(solve(ensemble_prob,m.solver,EnsembleThreads(),trajectories=size(x,2), sensealg=m.sensealg, save_everystep=false, save_start=false))[:,end,:]
  sol = solve(ensemble_prob,m.solver,EnsembleThreads(),trajectories=size(x,2), dt=0.1, adaptive=false, sensealg=m.sensealg, save_everystep=false, save_start=false)[:,end,:]
  #sol = copy(sol_b)

  h = sol#[:,:,end]
  #@show size(h)

  # @show size(sol)
  # @show size(sol[1])
  # @show size(Array(sol))
  # @show size(Array(sol)[1])
  # for b in size(sol,1)
  #   h_new[:,b] .= sol[b]
  # end
  #h = Array(sol)[:,1,:]
  # h = copy(h_new)
  #h = reshape(sol, size(h,1),:)

  #sol = solve(prob, solver; sensealg, save_everystep=false, save_start=false, reltol=1e-3, abstol=1e-3)
  #sol = solve(prob, solver; sensealg, save_everystep=false, save_start=false, reltol=1e-3, abstol=1e-3)
  #sol = solve(prob, m.solver; sensealg=m.sensealg, dtmax=1.0)
  #@show length(sol.t)
  #sol = Array(solve(prob, solver; sensealg, save_everystep=false, save_start=false, reltol=1e-2, abstol=1e-2))
  #@show size(sol)
  #@show size(sol,3)


  out = m.mapout(h)
  return h, @view out[end-m.wiring.n_motor+1:end, :]
end


function densltcdt!(dx,x,p,t, m, sens_f, syn_f)

  cm_pl = size(x,1)
  Gleak_pl = size(x,1)
  Eleak_pl = size(x,1)
  sens_pl = m.sens_pl
  syn_pl = m.syn_pl

  @views cm    = p[1 : cm_pl]
  @views Gleak = p[cm_pl + 1 : cm_pl + Gleak_pl]
  @views Eleak = p[cm_pl + Gleak_pl + 1 : cm_pl + Gleak_pl + Eleak_pl]

  @views sens_p = p[cm_pl + Gleak_pl + Eleak_pl + 1 : cm_pl + Gleak_pl + Eleak_pl + sens_pl]
  @views syn_p = p[cm_pl + Gleak_pl + Eleak_pl + sens_pl + 1 : cm_pl + Gleak_pl + Eleak_pl + sens_pl + syn_pl]

  @views I = p[cm_pl + Gleak_pl + Eleak_pl + sens_pl + syn_pl + 1 : end]
  I = reshape(I, m.wiring.n_in, :)

  #@unpack cm, Gleak, Eleak, sens_p, syn_p, I = p


  # sens_f = m.sens_re(sens_p)
  # syn_f  = m.syn_re(syn_p)

  I_sens = sens_f(sens_p, x, I, m.wiring.sens_mask, m.wiring.sens_pol)
  I_syn = syn_f(sens_p, x, x, m.wiring.syn_mask, m.wiring.syn_pol)

  #tout(cm, Gleak, x, Eleak, I_syn, I_sens) = @tullio out[j,b] := cm[j] * (-(Gleak[j] * (x[j,b] - Eleak[j])) + I_syn[j,b] + I_sens[j,b])
  #dx .= tout(cm, Gleak, x, Eleak, I_syn, I_sens)
  @. dx = cm * (-(Gleak * (x - Eleak)) + I_syn + I_sens)
  nothing
end


function oop(x,p,t, m)

  cm_pl = size(x,1)
  Gleak_pl = size(x,1)
  Eleak_pl = size(x,1)
  sens_pl = m.sens_pl
  syn_pl = m.syn_pl

  @views cm    = p[1 : cm_pl]
  @views Gleak = p[cm_pl + 1 : cm_pl + Gleak_pl]
  @views Eleak = p[cm_pl + Gleak_pl + 1 : cm_pl + Gleak_pl + Eleak_pl]

  @views sens_p = p[cm_pl + Gleak_pl + Eleak_pl + 1 : cm_pl + Gleak_pl + Eleak_pl + sens_pl]
  @views syn_p = p[cm_pl + Gleak_pl + Eleak_pl + sens_pl + 1 : cm_pl + Gleak_pl + Eleak_pl + sens_pl + syn_pl]

  I = p[cm_pl + Gleak_pl + Eleak_pl + sens_pl + syn_pl + 1 : end]
  I = reshape(I, m.wiring.n_in, :)

  #@unpack cm, Gleak, Eleak, sens_p, syn_p, I = p


  sens_f = m.sens_re(sens_p)
  syn_f  = m.syn_re(syn_p)

  I_sens = sens_f(x, I, m.wiring.sens_mask, m.wiring.sens_pol)
  I_syn = syn_f(x, x, m.wiring.syn_mask, m.wiring.syn_pol)

  #tout(cm, Gleak, x, Eleak, I_syn, I_sens) = @tullio out[j,b] := cm[j] * (-(Gleak[j] * (x[j,b] - Eleak[j])) + I_syn[j,b] + I_sens[j,b])
  #dx .= tout(cm, Gleak, x, Eleak, I_syn, I_sens)
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
  #gNN(G::A, μ::A, σ::A, E::A)  where {A} = new{typeof(G)}(G, μ, σ, E)
  gNN(G::A, μ::A, σ::A, E::A) where {A<:AbstractArray} = new{A}(G, μ, σ, E)
end
function gNN(in::Integer, out::Integer)
  # G = rand_uniform(init_ranges["G"][1], init_ranges["G"][2], in, out)
  # μ = rand_uniform(init_ranges["μ"][1], init_ranges["μ"][2], in, out)
  # σ = rand_uniform(init_ranges["σ"][1], init_ranges["σ"][2], in, out)
  # E = rand_uniform(init_ranges["E"][1], init_ranges["E"][2], in, out)
  G = rand_uniform(0.001, 1, in, out)
  μ = rand_uniform(0.3, 0.8, in, out)
  σ = rand_uniform(3, 8, in, out)
  E = rand_uniform(-0.3, 0.3, in, out)
  gNN(G,μ,σ,E)
end



function (m::gNN)(h::AbstractVector, x, bitmask::Matrix{Float32}, polmask::Matrix{Float32})
  G, μ, σ, E = m.G, m.μ, m.σ, m.E

  function fun1e(bitmask, G, E, x, polmask, h, μ, σ)
    #sigmoid((x .- μ) .* σ)
    reshape(sum(bitmask .* G .* sigmoid.((x .- μ) .* σ) .* (polmask .* E .- reshape(h, 1,:)),dims=1),:)
  end

  function fun2e(bitmask, G, E, x, polmask, h, μ, σ)
    @tullio out[j] := bitmask[i,j] * G[i,j] * 1 / (1 + exp(-((x[i] - μ[i,j]) * σ[i,j]))) * (h[j] - polmask[i,j] * E[i,j]) #verbose=true
  end

  #fun4e(bitmask, G, E, x, polmask, h, μ, σ) = @tullio out[j] := bitmask[i,j] * G[i,j] * sigmoid((x[j] - μ[i,j]) * σ[i,j]) * (polmask[i,j] * E[i,j] - h[j])

  return fun1e(bitmask, G, E, x, polmask, h, μ, σ)
end




function (m::gNN)(p::AbstractVector, h::AbstractVector, x, bitmask::Matrix{Float32}, polmask::Matrix{Float32})
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


  function fun1e(bitmask, G, E, x, polmask, h, μ, σ)
    #sigmoid((x .- μ) .* σ)
    reshape(sum(bitmask .* G .* sigmoid.((x .- μ) .* σ) .* (polmask .* E .- reshape(h, 1,:)),dims=1),:)
  end

  function fun2e(bitmask, G, E, x, polmask, h, μ, σ)
    @tullio out[j] := bitmask[i,j] * G[i,j] * 1 / (1 + exp(-((x[i] - μ[i,j]) * σ[i,j]))) * (h[j] - polmask[i,j] * E[i,j]) #verbose=true
  end

  #fun4e(bitmask, G, E, x, polmask, h, μ, σ) = @tullio out[j] := bitmask[i,j] * G[i,j] * sigmoid((x[j] - μ[i,j]) * σ[i,j]) * (polmask[i,j] * E[i,j] - h[j])

  return fun1e(bitmask, G, E, x, polmask, h, μ, σ)
end




Flux.@functor gNN
