# init_ranges = Dict(
#   "cm" => (1, 1000),
#   "Gleak" => (0.001, 1.0),
#   "Eleak" => (-0.2, 0.2),
#   "G" => (0.001, 1.0),
#   "μ" => (0.3, 0.8),
#   "σ" => (3, 8),
#   "E" => (1, 1.00001), # pol_mask
# )

rand_uniform(lb,ub,dims...) = Float32.(rand(Uniform(lb,ub),dims...))# |> f32

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
  cm    = rand_uniform(1, 2, n_total)
  Gleak = rand_uniform(0.5, 3, n_total)
  Eleak = rand_uniform(-0.2, 0.2, n_total)
  state0 = zeros(eltype(cm), n_total, 1)

  mapin = Mapper(n_in)
  mapout = Mapper(out)

  sens_f = gNN(n_in, n_total)
  syn_f = gNN(n_total, n_total)
  sens_p, sens_re = Flux.destructure(sens_f)
  syn_p, syn_re = Flux.destructure(syn_f)

  LTCCellE(wiring,mapin,mapout,sens_f,sens_p,length(sens_p),sens_re,syn_f,syn_p,length(syn_p),syn_re, cm,Gleak,Eleak,state0,solver,sensealg)
end



Flux.@functor LTCCellE (mapin, mapout, sens_p, syn_p, cm, Gleak, Eleak, state0,)
#Flux.@functor LTCCellE (mapin, mapout, sens_p, syn_p, cm, Gleak, Eleak)
Flux.trainable(m::LTCCellE)=(Flux.trainable(m.mapin), Flux.trainable(m.mapout), m.sens_p, m.syn_p, m.cm, m.Gleak, m.Eleak, m.state0)

NCPE(args...) = Flux.Recur(LTCCellE(args...))
Flux.Recur(m::LTCCellE) = Flux.Recur(m, m.state0)


#function (m::LTCCellE{W,SENS,SYN,SENSP,SENSRE,SYNP,SYNRE,MI,MO,V,<:AbstractArray{T}})(h::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}) where {W,SENS,SYN,SENSP,SENSRE,SYNP,SYNRE,MI,MO,V,T}
function (m::LTCCellE)(h, x)


  _p = vcat(m.cm, m.Gleak, m.Eleak, m.sens_p, m.syn_p)

  function prob_func(prob,i,repeat)
    #f = (du,u,p,t) -> dudt!(du,u,p,t,i)
    f = ODEFunction{true}((dx,x,p,t)->densltcdt!(dx,x,p,t, m, I[:,i]))
    remake(prob, f=f, u0=h[:,i])
  end
  function output_func(sol,i)

    #i == 1 && @show sol.t

    #@show size(sol)
    #sol_b[:,1,:] .= sol[:,end]
    #sol[:,end], false
    out = [x[:,1] for x in sol]
    return (DiffEqArray(out,sol.t), false)
  end


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
  prob = ODEProblem((dx,x,p,t)->densltcdt!(dx,x,p,t, m, I[:,1]), h[:,1], (0f0, 1f0), _p)
  ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
  sol = Array(solve(ensemble_prob,m.solver,EnsembleThreads(),trajectories=size(x,2), sensealg=m.sensealg, save_everystep=false, save_start=false))[:,end,:]
  #sol = copy(sol_b)

  h = sol

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


  out = m.mapout(h[end-m.wiring.out+1:end])

  return h, out
end


function densltcdt!(dx,x,p,t, m, I)

  cm_pl = size(x,1)
  Gleak_pl = size(x,1)
  Eleak_pl = size(x,1)
  sens_pl = m.sens_pl
  syn_pl = m.syn_pl

  cm    = p[1 : cm_pl]
  Gleak = p[cm_pl + 1 : cm_pl + Gleak_pl]
  Eleak = p[cm_pl + Gleak_pl + 1 : cm_pl + Gleak_pl + Eleak_pl]

  sens_p = p[cm_pl + Gleak_pl + Eleak_pl + 1 : cm_pl + Gleak_pl + Eleak_pl + sens_pl]
  syn_p = p[cm_pl + Gleak_pl + Eleak_pl + sens_pl + 1 : cm_pl + Gleak_pl + Eleak_pl + sens_pl + syn_pl]

  #I = p[cm_pl + Gleak_pl + Eleak_pl + sens_pl + syn_pl + 1 : end]
  #I = reshape(I, m.wiring.n_in, :)

  #@unpack cm, Gleak, Eleak, sens_p, syn_p, I = p


  sens_f = m.sens_re(sens_p)
  syn_f  = m.syn_re(syn_p)

  I_sens = sens_f(x, I, m.wiring.sens_mask, m.wiring.sens_pol)
  I_syn = syn_f(x, x, m.wiring.syn_mask, m.wiring.syn_pol)

  #tout(cm, Gleak, x, Eleak, I_syn, I_sens) = @tullio out[j,b] := cm[j] * (-(Gleak[j] * (x[j,b] - Eleak[j])) + I_syn[j,b] + I_sens[j,b])
  #dx .= tout(cm, Gleak, x, Eleak, I_syn, I_sens)
  @. dx = cm * (-(Gleak * (x - Eleak)) + I_syn + I_sens)
  nothing
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
  G = rand_uniform(0.1, 1, in, out)
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

  fun4e(bitmask, G, E, x, polmask, h, μ, σ) = @tullio out[j] := bitmask[i,j] * G[i,j] * sigmoid((x[j] - μ[i,j]) * σ[i,j]) * (polmask[i,j] * E[i,j] - h[j])

  return fun1e(bitmask, G, E, x, polmask, h, μ, σ)
end




function (m::gNN)(h, x, bitmask::Matrix{Float32}, polmask::Matrix{Float32})
  G, μ, σ, E = m.G, m.μ, m.σ, m.E

  fun4e(bitmask, G, E, x, polmask, h, μ, σ) = @tullio out[j] := bitmask[i,j] * G[i,j] * sigmoid((x[i] - μ[i,j]) * σ[i,j]) * (polmask[i,j] * E[i,j] - h[j])

  fun1(bitmask, G, E, x, polmask, h, μ, σ) = reshape(reduce(vcat, [reshape(sum(bitmask .* G .* Flux.sigmoid.(((@view x[:,b]) .- μ) .* σ) .* (polmask .* E .- reshape((@view h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)]), size(h,1),:)

  fun4(bitmask, G, E, x, polmask, h, μ, σ) =
    @tullio out[j,b] := bitmask[i,j] * G[i,j] * sigmoid(((x[i,b]) - μ[i,j]) * σ[i,j]) * (polmask[i,j] * E[i,j] - h[j,b])


  function fun6(bitmask, G, E, x, polmask, h, μ, σ)
    @tullio tmp[i,b] := sigmoid(((x[i,b]) - μ[i,j]) * σ[i,j]) #avx=4
    @tullio out[j,b] := bitmask[i,j] * G[i,j] * tmp[i,b] * (polmask[i,j] * E[i,j] - h[j,b]) #avx=4
  end

  return fun4(bitmask, G, E, x, polmask, h, μ, σ)

  # @views reshape(reduce(vcat, [reshape(sum(bitmask .* G .* Flux.sigmoid.(((x[:,b]) .- μ) .* sig) .* (polmask .* E .- reshape((h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)]), size(h,1),:)
  # Flux.stack([reshape(sum(bitmask .* G .* Flux.sigmoid.(((x[:,b]) .- μ) .* sig) .* (polmask .* E .- reshape((h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)], 2)
  #reshape(reduce(vcat, [reshape(sum(bitmask .* G .* Flux.sigmoid.(((@view x[:,b]) .- μ) .* σ) .* (polmask .* E .- reshape((@view h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)]), size(h,1),:)

  #Flux.stack([reshape(sum(bitmask .* G .* Flux.sigmoid.(((@view x[:,b]) .- μ) .* σ) .* (polmask .* E .- reshape((@view h[:,b]),1,:)),dims=1), :) for b in 1:size(h,2)], 2)
end

Flux.@functor gNN


function get_bounds(m::Mapper)
  lb = [[-10.1f0 for i in 1:length(m.W)]...,
        [-10.1f0 for i in 1:length(m.b)]...] |> f32

  ub = [[10.1f0 for i in 1:length(m.W)]...,
        [10.1f0 for i in 1:length(m.b)]...] |> f32
  return lb, ub
end

function get_bounds(m::gNN)
  lb = [[0.001f0 for _ in m.G]...,          #
        [-1e2 for _ in m.μ]...,
        [-1e2 for _ in m.σ]...,
        [-1e2 for _ in m.E]...,] |> f32

  ub = [[1f0 for _ in m.G]...,          #
        [1e2 for _ in m.μ]...,
        [1e2 for _ in m.σ]...,
        [1e2 for _ in m.E]...,] |> f32
  return lb, ub
end

get_bounds(m::Flux.Recur) = get_bounds(m.cell)
function get_bounds(m::LTCCellE)
  lower = [
      get_bounds(m.mapin)[1]...,
      get_bounds(m.mapout)[1]...,
      get_bounds(m.sens_f)[1]...,
      get_bounds(m.syn_f)[1]...,
      [1 for _ in m.cm]...,               #
      [0.001 for _ in m.Gleak]...,            #
      [-1e3 for _ in m.Eleak]...,
      [0 for _ in m.state0]...,
  ] |> f32
  upper = [
    get_bounds(m.mapin)[2]...,
    get_bounds(m.mapout)[2]...,
    get_bounds(m.sens_f)[2]...,
    get_bounds(m.syn_f)[2]...,
    [1000 for _ in m.cm]...,
    [1.0 for _ in m.Gleak]...,
    [1e3 for _ in m.Eleak]...,
    [0.001 for _ in m.state0]...,
  ] |> f32

  lower, upper
end
