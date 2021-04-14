abstract type WiringT <: Function end

struct Wiring{M1,M2,M3,M4} <: WiringT
    n_in::Int
    n_sensory::Int
    n_inter::Int
    n_command::Int
    n_motor::Int
    n_total::Int

    sensory_out::Int
    inter_out::Int
    rec_command_out::Int
    motor_in::Int

    sens_mask::M1
    syn_mask::M2

    sens_pol::M3
    syn_pol::M4

    function Wiring(n_in,n_sensory,n_inter,n_command,n_motor,n_total,sensory_out,inter_out,rec_command_out,motor_in,sens_mask,syn_mask,sens_pol,syn_pol)
        new{typeof(sens_mask),typeof(syn_mask),typeof(sens_pol),typeof(syn_pol)}(
                    n_in,n_sensory,n_inter,n_command,n_motor,n_total,sensory_out,inter_out,rec_command_out,motor_in,sens_mask,syn_mask,sens_pol,syn_pol)
    end
end

function Wiring(in::Int, out::Int;
                n_sensory=2, n_inter=5, n_command=0, n_motor=1,
                sensory_in=-1, rec_sensory=-1, sensory_out=-1,
                rec_inter=-1, inter_out=-1,                       # inter_in = sensory_out
                rec_command=-1, command_out=-1,                   # command_in = inter_out
                rec_motor=-1)                                     # motor_in = command_out, motor_out = out
  sensory_s = 1
  inter_s   = n_sensory + 1
  command_s = n_sensory + n_inter + 1
  motor_s   = n_sensory + n_inter + n_command + 1
  n_total   = n_sensory + n_inter + n_command + n_motor

  sens_mask = ones(Int8, n_total, in)
  syn_mask  = ones(Int8, n_total, n_total)

  sens_pol = zeros(Float32, n_total, in)
  syn_pol  = zeros(Float32, n_total, n_total)
  for i in eachindex(sens_pol)
    sens_pol[i] = [-1,1,1][rand(1:3)]
  end
  for i in eachindex(syn_pol)
    syn_pol[i] = [-1,1,1][rand(1:3)]
  end

  Wiring(in,n_sensory, n_inter,n_command,n_motor,n_total,sensory_out,inter_out,rec_command,command_out,sens_mask,syn_mask,sens_pol,syn_pol)
end


Flux.trainable(m::Wiring) = ()


struct LTCCell{W,SENS,SYN,SENSP,SENSRE,SYNP,SYNRE,MI,MO,V,S}

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

  function LTCCell(wiring, mapin, mapout, sens_f, sens_p, sens_pl, sens_re, syn_f, syn_p, syn_pl, syn_re, cm, Gleak, Eleak, state0)
    new{typeof(wiring),typeof(sens_f),typeof(syn_f),typeof(sens_p),typeof(sens_re),typeof(syn_p),typeof(syn_re),typeof(mapin),typeof(mapout),typeof(cm),typeof(state0)}(
      wiring,mapin,mapout,sens_f,sens_p,sens_pl,sens_re,syn_f,syn_p,syn_pl,syn_re,cm,Gleak,Eleak,state0)
  end
end


function LTCCell(wiring)
  n_in = wiring.n_in
  out = wiring.n_motor
  n_total = wiring.n_total
  cm    = rand_uniform(3,   5, n_total)
  Gleak = rand_uniform(0.001, 1,   n_total)
  Eleak = rand_uniform(-0.2,  0.2, n_total)
  state0 = zeros(eltype(cm), n_total, 1)

  sens_f = gNN(n_in, n_total)
  syn_f = gNN(n_total, n_total)
  sens_p, sens_re = Flux.destructure(sens_f)
  syn_p, syn_re = Flux.destructure(syn_f)

  LTCCell(wiring,Mapper(n_in),Mapper(out),sens_f,sens_p,length(sens_p),sens_re,syn_f,syn_p,length(syn_p),syn_re, cm,Gleak,Eleak,state0)
end



Flux.@functor LTCCell (mapin, mapout, sens_p, syn_p, cm, Gleak, Eleak, state0,)
#Flux.@functor LTCCell (mapin, mapout, sens_p, syn_p, cm, Gleak, Eleak)
Flux.trainable(m::LTCCell)=(Flux.trainable(m.mapin), Flux.trainable(m.mapout), m.sens_p, m.syn_p, m.cm, m.Gleak, m.Eleak, m.state0)

NCP(args...) = Flux.Recur(LTCCell(args...))
Flux.Recur(m::LTCCell) = Flux.Recur(m, m.state0)


#function (m::LTCCell{W,SENS,SYN,SENSP,SENSRE,SYNP,SYNRE,MI,MO,V,<:AbstractArray{T}})(h::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}) where {W,SENS,SYN,SENSP,SENSRE,SYNP,SYNRE,MI,MO,V,T}
function (m::LTCCell)(h, x)
  x = m.mapin(x)
  #m.mapin(x,x)

  # TODO: sens_f() here?

  u0 = repeat(h, 1, size(x,2)-size(h,2)+1)

  sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
  #sensealg = InterpolatingAdjoint()
  #sensealg = BacksolveAdjoint()
  #sensealg = TrackerAdjoint()
  solver = VCABM()
  #solver = Tsit5()
  #solver = AutoTsit5(Rosenbrock23())
  #solver = TRBDF2()
  tspan = (0f0, 1f0)


  p = [m.cm[:]; m.Gleak[:]; m.Eleak[:]; m.sens_p[:]; m.syn_p[:]; x[:]]

  #p = ComponentArray(cm=m.cm, Gleak=m.Gleak, Eleak=m.Eleak, sens_p=m.sens_p, syn_p=m.syn_p, I=x)
  f = ODEFunction{true}((dx,x,p,t)->dltcdt!(dx,x,p,t, m))
  #f = ODEFunction((x,p,t)->dltcdt(x,p,t, m))
  prob = ODEProblem(f,u0,tspan,p)

  sol = Array(solve(prob, solver; sensealg, save_everystep=false, save_start=false))
  #@show size(sol)

  h = sol[:,:,end]
  out = sol[end-m.wiring.n_motor+1:end, : ,end]

  out = m.mapout(out)
  #m.mapout(out,out)
  # out = h
  # out = m.mapout(out)

  return h, out
end



function dltcdt!(dx,x,p,t, m)

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

  sens_f = m.sens_re(sens_p)
  syn_f  = m.syn_re(syn_p)

  curr_buf = Flux.Zygote.Buffer(x, (size(x,1), size(x,2)))
  for i in eachindex(curr_buf)
    curr_buf[i] = 0
  end
  sens_f(x, I, m.wiring.sens_mask, m.wiring.sens_pol, curr_buf)
  syn_f(x, x, m.wiring.syn_mask, m.wiring.syn_pol, curr_buf)
  I_syn = copy(curr_buf)

  # I_sens = sens_f(x, I, m.wiring.sens_mask, m.wiring.sens_pol)
  # I_syn = syn_f(x, x, m.wiring.syn_mask, m.wiring.syn_pol)

  #dx = Flux.Zygote.Buffer(x, (size(x,1), size(x,2)))
  @inbounds for b in 1:size(I,2)
    for n in 1:size(x,1)
      dx[n,b] = (cm[n]) * (-(Gleak[n] * (Eleak[n] - x[n,b])) + I_syn[n,b])
    end
  end
  #return copy(dx)
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
  G = rand_uniform(0.001, 1, out,in)
  μ = rand_uniform(0.3, 0.8, out,in)
  σ = rand_uniform(3, 8, out,in)
  E = rand_uniform(0.001, 0.3, out,in)
  gNN(G,μ,σ,E)
end

function (m::gNN)(h, x, bitmask, polmask)
  G, μ, σ, E = m.G, m.μ, m.σ, m.E

  curr_buf = Flux.Zygote.Buffer(h, size(h,1),size(h,2))
  for i in eachindex(curr_buf)
    curr_buf[i] = 0
  end

  for b in 1:size(h,2)
    for s in 1:size(x,1)
      for n in 1:size(h,1)
        tmp = bitmask[n,s] * G[n,s] * Flux.sigmoid((x[s,b] - μ[n,s]) * σ[n,s])
        curr_buf[n,b] += tmp * (polmask[n,s] * E[n,s] - h[n,b])
        #curr_buf[n,b] += mask[n,s] * G[n,s] * Flux.sigmoid((x[s,b] - μ[n,s]) * σ[n,s]) * (E[n,s] - h[n,b])
      end
    end
  end
  return copy(curr_buf)
  #nothing
end



function (m::gNN)(h, x, bitmask, polmask, curr_buf)
  G, μ, σ, E = m.G, m.μ, m.σ, m.E

  for b in 1:size(h,2)
    for s in 1:size(x,1)
      for n in 1:size(h,1)
        tmp = bitmask[n,s] * G[n,s] * Flux.sigmoid((x[s,b] - μ[n,s]) * σ[n,s])
        curr_buf[n,b] += tmp * (polmask[n,s] * E[n,s] - h[n,b])
        #curr_buf[n,b] += mask[n,s] * G[n,s] * Flux.sigmoid((x[s,b] - μ[n,s]) * σ[n,s]) * (E[n,s] - h[n,b])
      end
    end
  end
  #nothing
end

Flux.@functor gNN






function get_bounds(m::Mapper)
  lb = [[-1.1f0 for i in 1:length(m.W)]...,
        [-1.1f0 for i in 1:length(m.b)]...]

  ub = [[1.1f0 for i in 1:length(m.W)]...,
        [1.1f0 for i in 1:length(m.b)]...]
  return lb, ub
end

function get_bounds(m::gNN)
  lb = [[0f0 for _ in m.G]...,
        [0f0 for _ in m.μ]...,
        [2.9f0 for _ in m.σ]...,
        [0f0 for _ in m.E]...,]

  ub = [[1f0 for _ in m.G]...,
        [0.9f0 for _ in m.μ]...,
        [10f0 for _ in m.σ]...,
        [1f0 for _ in m.E]...,]
  return lb, ub
end

get_bounds(m::Flux.Recur) = get_bounds(m.cell)
function get_bounds(m::LTCCell)
  lower = [
      get_bounds(m.mapin)[1]...,
      get_bounds(m.mapout)[1]...,
      get_bounds(m.sens_f)[1]...,
      get_bounds(m.syn_f)[1]...,
      [1 for _ in m.cm]...,
      [0 for _ in m.Gleak]...,
      [-0.3 for _ in m.Eleak]...,
      [0 for _ in m.state0]...,
  ] |> f32
  upper = [
    get_bounds(m.mapin)[2]...,
    get_bounds(m.mapout)[2]...,
    get_bounds(m.sens_f)[2]...,
    get_bounds(m.syn_f)[2]...,
    [10 for _ in m.cm]...,
    [2.0 for _ in m.Gleak]...,
    [0.3 for _ in m.Eleak]...,
    [0.001 for _ in m.state0]...,
  ] |> f32

  lower, upper
end
