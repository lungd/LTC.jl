abstract type WiringT <: Function end

struct Wiring{T} <:WiringT
    n_in::Int
    n_out::Int
    n_sensory::Int
    n_inter::Int
    n_command::Int
    n_motor::Int
    n_total::Int

    sensory_out::Int
    inter_out::Int
    rec_command_out::Int
    motor_in::Int

    sens_mask::Matrix{T}
    syn_mask::Matrix{T}

    sens_pol::Matrix{T}
    syn_pol::Matrix{T}

    function Wiring(n_in,out,n_sensory,n_inter,n_command,n_motor,n_total,sensory_out,inter_out,rec_command_out,motor_in,sens_mask,syn_mask,sens_pol,syn_pol)
        new{eltype(sens_mask)}(
                    n_in,out,n_sensory,n_inter,n_command,n_motor,n_total,sensory_out,inter_out,rec_command_out,motor_in,sens_mask,syn_mask,sens_pol,syn_pol)
    end
end

random_polarity(p=[-1,1,1]) = p[rand(1:length(p))]

# get_n_in(m::Wiring) = m.n_in
# get_n_total(m::Wiring) = m.n_total

function FWiring(in::Int, out::Int, T::DataType=Float32;) #where TYPE <: AbstractFloat
  n_total   = out

  sens_mask = ones(T, in, n_total)
  syn_mask  = ones(T, n_total, n_total)

  sens_pol = ones(T, in, n_total)
  syn_pol  = ones(T, n_total, n_total)
  for i in eachindex(sens_pol)
    sens_pol[i] = random_polarity()
  end
  for i in eachindex(syn_pol)
    syn_pol[i] = random_polarity()
  end

  Wiring(in,out,-1,-1,-1,n_total,n_total,-1,-1,-1,-1,sens_mask,syn_mask,sens_pol,syn_pol)
end


function add_synapses!(srcs,dsts,n_out,bitmask,polmask)
    n_out == 0 && return
    n_out = n_out == -1 ? length(dsts) : n_out
    n_out = min(n_out,length(dsts))
    for src in srcs
        for _ in n_out
            dst = dsts[rand(1:length(dsts))]
            bitmask[src,dst] = 1
            polmask[src,dst] = random_polarity()
        end
    end
end

function add_missing_synapses!(srcs, dsts, bitmask, polmask; n=1)
    for dst in dsts
        bm = @view bitmask[:,dst]
        pm = @view polmask[:,dst]

        if 1 ∉ filter(x -> x != dst, bm[srcs])
        #if 1 ∉ bm[srcs][bm[srcs] .!= dst]
            for i in 1:n
                random_src = srcs[rand(1:length(srcs))]
                bm[random_src] = 1
                pm[random_src] = random_polarity()
            end
        end
    end
end


function create_wiring(n_sensory, n_inter, n_command, n_motor,
  sensory_in, rec_sensory, sensory_inter, sensory_command, sensory_motor,
  inter_in, rec_inter, inter_command, inter_motor,                       # inter_in = sensory_out
  command_in, rec_command, command_motor,                   # command_in = inter_out
  motor_in, rec_motor,
  sens_mask, sens_pol, create_input_syns=0, T::DataType=Float32;)# where TYPE <: AbstractFloat

  n_total   = n_sensory + n_inter + n_command + n_motor

  syn_mask  = zeros(T, n_total, n_total)
  syn_pol  = ones(T, n_total, n_total)

  sensory_s = 1
  inter_s   = n_sensory + 1
  command_s = n_sensory + n_inter + 1
  motor_s   = n_sensory + n_inter + n_command + 1

  sens_range = range(sensory_s, length=n_sensory)
  inter_range = range(inter_s, length=n_inter)
  command_range = range(command_s, length=n_command)
  motor_range = range(motor_s, length=n_motor)

  if create_input_syns != 0
    input_range = range(1, length=create_input_syns)
    add_synapses!(input_range,    sens_range,    sensory_in, sens_mask, sens_pol)
    add_synapses!(input_range,    inter_range,    inter_in, sens_mask, sens_pol)
    add_synapses!(input_range,    command_range,    command_in, sens_mask, sens_pol)
    add_synapses!(input_range,    motor_range,    motor_in, sens_mask, sens_pol)
  end

  add_synapses!(sens_range,    sens_range,    rec_sensory, syn_mask, syn_pol)
  add_synapses!(sens_range,    inter_range,   sensory_inter, syn_mask, syn_pol)
  add_synapses!(sens_range,    command_range, sensory_command, syn_mask, syn_pol)
  add_synapses!(sens_range,    motor_range,   sensory_motor, syn_mask, syn_pol)
  add_synapses!(inter_range,   inter_range,   rec_inter, syn_mask, syn_pol)
  add_synapses!(inter_range,   command_range, inter_command, syn_mask, syn_pol)
  add_synapses!(inter_range,   motor_range,   inter_motor, syn_mask, syn_pol)
  add_synapses!(command_range, command_range, rec_command, syn_mask, syn_pol)
  add_synapses!(command_range, motor_range,   command_motor, syn_mask, syn_pol)
  add_synapses!(motor_range,   motor_range,   rec_motor, syn_mask, syn_pol)

  add_missing_synapses!(sens_range, inter_range, syn_mask, syn_pol)
  add_missing_synapses!(inter_range, command_range, syn_mask, syn_pol)
  add_missing_synapses!(command_range, motor_range, syn_mask, syn_pol)

  return sens_mask, syn_mask, sens_pol, syn_pol
end

function DiagSensNCPWiring(n_in::Int, n_out::Int, T::DataType=Float32;
                n_sensory=2, n_inter=3, n_command=5, n_motor=1,
                sensory_in=-1, rec_sensory=0, sensory_inter=2, sensory_command=0, sensory_motor=0,
                inter_in=0, rec_inter=2, inter_command=2, inter_motor=0,                       # inter_in = sensory_out
                command_in=0, rec_command=1, command_motor=3,                   # command_in = inter_out
                motor_in=0, rec_motor=1, orig=false)

  @assert n_in == n_sensory
  n_total   = n_sensory + n_inter + n_command + n_motor
  sens_mask = zeros(T, n_in, n_total)
  sens_pol = copy(sens_mask)

  for s in 1:n_in
    for t in 1:n_sensory
      s != t && continue
      polarity = 1
      add_synapse!(s, t, sens_mask, sens_pol, polarity)
    end
  end

  sens_mask, syn_mask, sens_pol, syn_pol = orig ? create_ncp_wiring(n_sensory, n_inter, n_command, n_motor,
    sensory_inter,
    inter_command,
    rec_command, command_motor,
    sens_mask, sens_pol, T) : create_wiring(n_sensory, n_inter, n_command, n_motor,
    sensory_in, rec_sensory, sensory_inter, sensory_command, sensory_motor,
    inter_in, rec_inter, inter_command, inter_motor,                       # inter_in = sensory_out
    command_in, rec_command, command_motor,                   # command_in = inter_out
    motor_in, rec_motor,
    sens_mask, sens_pol, T)
  Wiring(n_in,n_out,n_sensory, n_inter,n_command,n_motor,n_total,sensory_inter,inter_command,rec_command,command_motor,sens_mask,syn_mask,sens_pol,syn_pol)
end

function FullSensNCPWiring(n_in::Int, n_out::Int, T::DataType=Float32;
                n_sensory=2, n_inter=3, n_command=5, n_motor=1,
                sensory_in=-1, rec_sensory=0, sensory_inter=2, sensory_command=0, sensory_motor=0,
                inter_in=0, rec_inter=2, inter_command=2, inter_motor=0,                       # inter_in = sensory_out
                command_in=0, rec_command=1, command_motor=3,                   # command_in = inter_out
                motor_in=0, rec_motor=1, orig=false) where TYPE <: AbstractFloat

  @assert n_in == n_sensory
  n_total   = n_sensory + n_inter + n_command + n_motor
  sens_mask = zeros(T, n_in, n_total)
  sens_pol = copy(sens_mask)

  for s in 1:n_in
    for t in 1:n_sensory
      add_synapse!(s, t, sens_mask, sens_pol)
    end
  end

  sens_mask, syn_mask, sens_pol, syn_pol = orig ? create_ncp_wiring(n_sensory, n_inter, n_command, n_motor,
    sensory_inter,
    inter_command,
    rec_command, command_motor,
    sens_mask, sens_pol, T) : create_wiring(n_sensory, n_inter, n_command, n_motor,
    sensory_in, rec_sensory, sensory_inter, sensory_command, sensory_motor,
    inter_in, rec_inter, inter_command, inter_motor,                       # inter_in = sensory_out
    command_in, rec_command, command_motor,                   # command_in = inter_out
    motor_in, rec_motor,
    sens_mask, sens_pol, T)
  Wiring(n_in,n_out,n_sensory, n_inter,n_command,n_motor,n_total,sensory_inter,inter_command,rec_command,command_motor,sens_mask,syn_mask,sens_pol,syn_pol)
end

function DiagFullNCPWiring(n_in::Int, n_out::Int, T::DataType=Float32;
                n_sensory=2, n_inter=3, n_command=5, n_motor=1,
                sensory_in=-1, rec_sensory=0, sensory_inter=2, sensory_command=0, sensory_motor=0,
                inter_in=0, rec_inter=2, inter_command=2, inter_motor=0,                       # inter_in = sensory_out
                command_in=0, rec_command=1, command_motor=3,                   # command_in = inter_out
                motor_in=0, rec_motor=1) where TYPE <: AbstractFloat

  n_total   = n_sensory + n_inter + n_command + n_motor
  @assert n_in == n_total
  @assert n_out == n_total
  sens_mask = zeros(T,n_total,n_total)
  sens_pol = copy(sens_mask)

  for s in 1:n_in
    for t in 1:n_total
      s != t && continue
      add_synapse!(s, t, sens_mask, sens_pol)
    end
  end

  sens_mask, syn_mask, sens_pol, syn_pol = orig ? create_ncp_wiring(n_sensory, n_inter, n_command, n_motor,
    sensory_inter,
    inter_command,
    rec_command, command_motor,
    sens_mask, sens_pol, T) : create_wiring(n_sensory, n_inter, n_command, n_motor,
    sensory_in, rec_sensory, sensory_inter, sensory_command, sensory_motor,
    inter_in, rec_inter, inter_command, inter_motor,                       # inter_in = sensory_out
    command_in, rec_command, command_motor,                   # command_in = inter_out
    motor_in, rec_motor,
    sens_mask, sens_pol, T)
  Wiring(n_in,n_out,n_sensory, n_inter,n_command,n_motor,n_total,sensory_inter,inter_command,rec_command,command_motor,sens_mask,syn_mask,sens_pol,syn_pol)
end


function NCPWiring(n_in::Int, n_out::Int, T::DataType=Float32;
                n_sensory=2, n_inter=3, n_command=5, n_motor=1,
                sensory_in=-1, rec_sensory=0, sensory_inter=2, sensory_command=0, sensory_motor=0,
                inter_in=0, rec_inter=2, inter_command=2, inter_motor=0,                       # inter_in = sensory_out
                command_in=0, rec_command=1, command_motor=3,                   # command_in = inter_out
                motor_in=0, rec_motor=1) where TYPE <: AbstractFloat                                     # motor_in = command_out, motor_out = out

  n_total   = n_sensory + n_inter + n_command + n_motor
  sens_mask = zeros(T, n_in, n_total)
  sens_pol = ones(T, n_in, n_total)

  sens_mask, syn_mask, sens_pol, syn_pol = create_wiring(n_sensory, n_inter, n_command, n_motor,
    sensory_in, rec_sensory, sensory_inter, sensory_command, sensory_motor,
    inter_in, rec_inter, inter_command, inter_motor,                       # inter_in = sensory_out
    command_in, rec_command, command_motor,                   # command_in = inter_out
    motor_in, rec_motor,
    sens_mask, sens_pol, n_in, T)

  Wiring(n_in,n_out,n_sensory, n_inter,n_command,n_motor,n_total,sensory_inter,inter_command,rec_command,command_motor,sens_mask,syn_mask,sens_pol,syn_pol)
end


Flux.trainable(m::Wiring) = ()



function create_ncp_wiring(n_sensory, n_inter, n_command, n_motor,
  sensory_inter,
  inter_command,                       # inter_in = sensory_out
  rec_command, command_motor,
  sens_mask, sens_pol, T::DataType=Float32) #where TYPE <: AbstractFloat

  n_total   = n_sensory + n_inter + n_command + n_motor

  syn_mask  = zeros(T, n_total, n_total)
  syn_pol   = copy(syn_mask)

  sensory_s = 1
  inter_s   = n_sensory + 1
  command_s = n_sensory + n_inter + 1
  motor_s   = n_sensory + n_inter + n_command + 1

  sens_range = range(sensory_s, length=n_sensory)
  inter_range = range(inter_s, length=n_inter)
  command_range = range(command_s, length=n_command)
  motor_range = range(motor_s, length=n_motor)

  connect_layers!(sens_range, inter_range, sensory_inter, syn_mask, syn_pol)
  connect_layers!(inter_range, command_range, inter_command, syn_mask, syn_pol)
  add_recurrent_connections!(command_range, rec_command, syn_mask, syn_pol)
  connect_command_motor!(command_range, motor_range, command_motor, syn_mask, syn_pol)

  return sens_mask, syn_mask, sens_pol, syn_pol
end

function connect_command_motor!(src_range, dst_range, k, syn_mask, syn_pol)
  k = min(k, length(src_range))
  for t in dst_range
    S = Random.shuffle(src_range)
    for i in 1:k
      s = S[i]
      add_synapse!(s, t, syn_mask, syn_pol)
    end
  end
end

function add_synapse!(s, t, mask, pol, p = random_polarity())
  mask[s, t] = one(eltype(mask))
  pol[s, t] = p
end

function add_recurrent_connections!(layer_range, k, syn_mask, syn_pol)
  for i in 1:k
    s = rand(layer_range)
    t = rand(layer_range)
    add_synapse!(s, t, syn_mask, syn_pol)
  end
end

function connect_layers!(src_range, dst_range, k, syn_mask, syn_pol)
  k = min(k, length(dst_range))
  for s in src_range
    T = Random.shuffle(dst_range)
    for i in 1:k
      t = T[i]
      add_synapse!(s, t, syn_mask, syn_pol)
    end
  end

  # handle dst neurons without connections
  μᵢ = round(Int, sum(syn_mask[:,dst_range]) / length(dst_range))  # Compute mean fan-in of dst neurons
  for t in dst_range
    1 ∈ syn_mask[:, t] && continue
    S = Random.shuffle(src_range)
    for i in 1:μᵢ
      s = S[i]
      add_synapse!(s, t, syn_mask, syn_pol)
    end
  end
end










# using Plots
# w = LTC.DiagSensNCPWiring(8,8,orig=true,
#   n_sensory=8, n_inter=5, n_command=5, n_motor=8,
#   sensory_inter=3, inter_command=2, command_motor=3)
# w.syn_mask
# heatmap(w.syn_mask)
# heatmap(w.syn_pol)
# heatmap(w.sens_mask)
# heatmap(w.sens_pol)
#
#
#
# w = LTC.FWiring(2,1)
# heatmap(w.syn_mask)
# heatmap(w.syn_pol)
# heatmap(w.sens_mask)
# heatmap(w.sens_pol)
